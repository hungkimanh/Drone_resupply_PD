/**
 * @file pdp_reader.cpp
 * @brief Parse PDP instance files and build distance matrices.
 *
 * NODE ID CONVENTIONS:
 *   Physical array (0-based): index 0 = depot, 1..N-1 = customers
 *   Truck--Drone Resupply encoding (per MD specs):
 *     0[P]  -> depot load P
 *     i     -> serve customer i
 *     i[P]  -> drone resupply P at customer i (before serving i)
 *     0     -> return to depot and end route
 */

#include "pdp_reader.h"
#include "pdp_utils.h"   // euclideanDistance
#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <map>
#include <cmath>
#include <algorithm>
#include <stdexcept>

using namespace std;

// ============================================================
// === DISTANCE HELPERS =======================================
// ============================================================

double manhattanDistance(double x1, double y1, double x2, double y2) {
    return abs(x1 - x2) + abs(y1 - y2);
}

void buildAllDistanceMatrices(PDPData& data) {
    int n = data.numNodes;
    // Allocate n×n matrices (row/col 0 = physical depot, 1..n-1 = customers)
    data.truckDistMatrix.assign(n, vector<double>(n, 0.0));
    data.droneDistMatrix.assign(n, vector<double>(n, 0.0));

    for (int i = 0; i < n; ++i) {
        for (int j = i + 1; j < n; ++j) {
            const auto& c1 = data.coordinates[i];
            const auto& c2 = data.coordinates[j];

            // Drone: Euclidean minutes from the input coordinates, without rounding.
            double eu   = euclideanDistance(c1.first, c1.second, c2.first, c2.second);
            double tDrn = (eu / data.droneSpeed) * 60.0;
            data.droneDistMatrix[i][j] = data.droneDistMatrix[j][i] = tDrn;

            // Truck: Manhattan → exact minutes (no rounding)
            double mh   = manhattanDistance(c1.first, c1.second, c2.first, c2.second);
            double tTrk = (mh / data.truckSpeed) * 60.0;
            data.truckDistMatrix[i][j] = data.truckDistMatrix[j][i] = tTrk;
        }
    }
    cout << "[READER] Distance matrices built: "
         << n << "x" << n
         << " | Truck=Manhattan->min | Drone=Euclidean->min (exact float)\n";
}

// ============================================================
// === readPDPFile ============================================
// ============================================================

bool readPDPFile(const string& filename, PDPData& data) {
    ifstream file(filename);
    if (!file.is_open()) {
        cerr << "[READER] ERROR: Cannot open file: " << filename << "\n";
        return false;
    }
    cout << "[READER] Reading: " << filename << "\n";

    // Preserve runtime settings set before calling readPDPFile
    int savedDepotMode = data.depotMode;
    data = PDPData();
    data.depotMode = savedDepotMode;

    // ---- Raw parse: read all customer rows into temp buffers ----
    string line;

    vector<pair<double,double>> tempCoords;
    vector<string>              tempTypes;
    vector<int>                 tempReadyTimes;
    vector<int>                 tempPairIds;

    int line_number = 0;
    while (getline(file, line)) {
        ++line_number;
        if (line_number == 1 && line.compare(0, 3, "\xEF\xBB\xBF") == 0) line.erase(0, 3);
        const auto first = line.find_first_not_of(" \t\r");
        if (first == string::npos || line[first] == '#') continue;
        istringstream fields(line);
        vector<string> tokens;
        string token;
        while (fields >> token) tokens.push_back(token);
        if (tokens[0] == "ri" || tokens[0] == "id") continue;
        try {
            int id, readyTime, pairId = 0;
            double x, y;
            string type = "D";
            if (tokens.size() == 3) {
                id = static_cast<int>(tempCoords.size()) + 1;
                readyTime = stoi(tokens[0]); x = stod(tokens[1]); y = stod(tokens[2]);
            } else if (tokens.size() == 6) {
                id = stoi(tokens[0]); x = stod(tokens[1]); y = stod(tokens[2]);
                type = tokens[3]; readyTime = stoi(tokens[4]); pairId = stoi(tokens[5]);
                if (id == 0) {
                    if (!tempCoords.empty() || type != "D" || readyTime != 0 || pairId != 0)
                        throw runtime_error("invalid depot row");
                    continue;
                }
            } else throw runtime_error("expected 3 or 6 columns");
            if (id != static_cast<int>(tempCoords.size()) + 1 || readyTime < 0 ||
                !isfinite(x) || !isfinite(y) || (type != "D" && type != "P" && type != "DL"))
                throw runtime_error("invalid customer fields");
            tempCoords.push_back({x, y}); tempTypes.push_back(type);
            tempReadyTimes.push_back(readyTime); tempPairIds.push_back(pairId);
        } catch (const exception& ex) {
            cerr << "[READER] Invalid row " << line_number << ": " << ex.what() << "\n";
            return false;
        }
    }
    if (tempCoords.empty()) return false;
    data.allowC1CarryAcrossDepot = int(tempTypes.size()) <= data.truckCapacity &&
        all_of(tempTypes.begin(), tempTypes.end(), [](const string& t) { return t == "D"; });
    file.close();

    // ================================================================
    // STEP 1: Insert DEPOT at physical array index 0
    //         depotIndex is always 0 (0-based, never changes)
    // ================================================================
    pair<double,double> selectedDepot;
    switch (data.depotMode) {
        case 1:  selectedDepot = data.depotBorder;  break;
        case 2:  selectedDepot = data.depotOutside; break;
        default: selectedDepot = data.depotCenter;  break;  // mode 0 = center
    }

    data.coordinates.push_back(selectedDepot);  // index 0 = depot
    data.nodeTypes.push_back("D");
    data.readyTimes.push_back(0);
    data.pairIds.push_back(0);
    data.demands.push_back(0);   // depot demand = 0

    data.depotIndex = 0;  // Always 0-based, always index 0

    // ================================================================
    // STEP 2: Append customer nodes at indices 1 .. N-1
    //         These are the only nodes that can appear as Gene(node_id > 0)
    // ================================================================
    for (size_t i = 0; i < tempCoords.size(); ++i) {
        data.coordinates.push_back(tempCoords[i]);
        data.nodeTypes.push_back(tempTypes[i]);
        data.readyTimes.push_back(tempReadyTimes[i]);
        data.pairIds.push_back(tempPairIds[i]);

        // Truck--Drone Resupply: each customer has exactly 1 package.
        // Use readyTime as release date r_i. Default demand q_i = 1.
        data.demands.push_back(1);
    }

    data.numNodes = (int)data.coordinates.size();

    // ================================================================
    // STEP 3: Build distance matrices (0-based, depot at [0][*])
    // ================================================================
    buildAllDistanceMatrices(data);

    // ================================================================
    // STEP 4: Count customers (all indices > 0)
    // ================================================================
    data.numCustomers = std::max(0, data.numNodes - 1);

    cout << "[READER] Loaded " << data.numNodes << " nodes total | "
         << "Depot: index 0 | Customers: " << data.numCustomers << "\n";
    return true;
}

// ============================================================
// === showPDPInfo ============================================
// ============================================================

void showPDPInfo(const PDPData& data) {
    cout << "\n+-------------------------------------------------------+\n"
         << "|              PDP Problem Summary                      |\n"
         << "+-------------------------------------------------------+\n";

        cout << "  Total nodes      : " << data.numNodes      << "\n"
            << "  Customer nodes   : " << data.numCustomers  << "  (all nodes with index > 0)\n"
         << "  Depot array idx  : " << data.depotIndex    << "  (always 0-based)\n"
         << "  Trucks           : " << data.numTrucks
         << "  (capacity=" << data.truckCapacity << ")\n"
         << "  Drones           : " << data.numDrones
            << "  (capacity=" << data.getDroneCapacity()
         << ", endurance=" << data.droneEndurance << " min)\n"
         << "  Truck speed      : " << data.truckSpeed    << " km/h\n"
         << "  Drone speed      : " << data.droneSpeed    << " km/h\n";

    // Node type breakdown (kept for dataset introspection)
    map<string,int> typeCount;
    for (int i = 0; i < data.numNodes; ++i) typeCount[data.nodeTypes[i]]++;
    cout << "\n  Node type breakdown (raw instance labels):\n";
    for (const auto& kv : typeCount) {
        cout << "    " << setw(6) << kv.first << " : " << kv.second << "\n";
    }

        // ---- Encoding mapping block ----
    cout << "\n+-------------------------------------------------------+\n"
         << "|  [ENCODING MAPPING]                                   |\n"
            << "|  0[P]   -> Depot loads packages P                     |\n"
            << "|  i      -> Truck serves customer i                    |\n"
            << "|  i[P]   -> Drone resupplies P at customer i (pre-serve)|\n"
            << "|  0      -> Truck returns depot and ends route         |\n"
         << "+-------------------------------------------------------+\n\n";
}
