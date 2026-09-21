#include "pdp_fitness.h"
#include "pdp_reader.h"
#include "pdp_validation.h"

#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>

int main(int argc, char* argv[]) {
    if (argc < 3 || argc > 5) {
        std::cerr << "Usage: " << argv[0]
                  << " <instance> <encoding.txt> [depot_mode] [verbose]\n";
        return 1;
    }

    PDPData data;
    data.depotMode = argc == 4 ? std::stoi(argv[3]) : 1;
    if (!readPDPFile(argv[1], data)) return 1;

    std::ifstream input(argv[2]);
    if (!input) {
        std::cerr << "Cannot open encoding file: " << argv[2] << "\n";
        return 1;
    }

    SolutionEncoding encoding;
    Route* current = nullptr;
    std::string line;
    while (std::getline(input, line)) {
        std::istringstream row(line);
        std::string tag;
        row >> tag;
        if (tag.empty() || tag[0] == '#') continue;
        if (tag == "MILP_CMAX") continue;
        if (tag == "ROUTE") {
            encoding.emplace_back();
            current = &encoding.back();
            continue;
        }
        if (tag == "END_ROUTE") {
            current = nullptr;
            continue;
        }
        if (tag == "STOP" && current != nullptr) {
            int node = 0;
            row >> node;
            std::vector<int> packages;
            int fixedDrone = -1;
            int fixedTrip = -1;
            double fixedDepart = -1.0;
            double fixedArrive = -1.0;
            double fixedReturn = -1.0;
            std::string token;
            while (row >> token) {
                if (token == "DRONE") {
                    row >> fixedDrone >> fixedTrip;
                    row >> fixedDepart >> fixedArrive >> fixedReturn;
                    break;
                }
                packages.push_back(std::stoi(token));
            }
            current->emplace_back(node, std::move(packages));
            current->back().fixed_drone_id = fixedDrone;
            current->back().fixed_drone_trip = fixedTrip;
            current->back().fixed_drone_depart = fixedDepart;
            current->back().fixed_drone_arrive = fixedArrive;
            current->back().fixed_drone_return = fixedReturn;
            continue;
        }
        std::cerr << "Malformed encoding line: " << line << "\n";
        return 1;
    }

    const PDPSolution solution = decode_solution(encoding, data, false);
    const bool valid = validateSolution(solution, data, false);
    if (argc == 5 && std::stoi(argv[4]) != 0) {
        print_decoded_routes(solution, data);
    }
    std::cout << std::fixed << std::setprecision(9)
              << "DECODE_RESULT"
              << " valid=" << (valid ? 1 : 0)
              << " feasible=" << (solution.isFeasible ? 1 : 0)
              << " cmax=" << solution.totalCost
              << " penalty=" << solution.totalPenalty << "\n";
    return 0;
}
