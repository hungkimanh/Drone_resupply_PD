#include "pdp_validation.h"

#include "pdp_utils.h"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <unordered_map>
#include <unordered_set>

using namespace std;

static double packageRelease(const PDPData& data, int pkg) {
    if (pkg <= 0 || pkg >= (int)data.readyTimes.size()) return 0.0;
    return (double)data.readyTimes[pkg];
}

static bool validateC1TripBoundaries(const SolutionEncoding& encoded,
                                     const PDPData& data,
                                     bool verbose) {
    bool ok = true;

    for (int truck = 0; truck < (int)encoded.size(); ++truck) {
        unordered_set<int> tripC1;
        const auto& route = encoded[truck];

        for (int idx = 0; idx < (int)route.size(); ++idx) {
            const auto& st = route[idx];
            if (st.node == 0) {
                if (idx > 0 && !tripC1.empty()) {
                    ok = false;
                    if (verbose) {
                        cout << "  Truck " << truck
                             << " returns to depot with undelivered C1 packages:";
                        for (int p : tripC1) cout << " " << p;
                        cout << "\n";
                    }
                }
                tripC1.clear();
                for (int p : st.packages) {
                    if (p > 0 && p < (int)data.nodeTypes.size() &&
                        data.nodeTypes[p] != "P" && data.nodeTypes[p] != "DL") {
                        tripC1.insert(p);
                    }
                }
                continue;
            }

            for (int p : st.packages) {
                if (p > 0 && p < (int)data.nodeTypes.size() &&
                    data.nodeTypes[p] != "P" && data.nodeTypes[p] != "DL") {
                    tripC1.insert(p);
                }
            }
            if (st.node > 0 && st.node < (int)data.nodeTypes.size() &&
                data.nodeTypes[st.node] != "P" && data.nodeTypes[st.node] != "DL") {
                tripC1.erase(st.node);
            }
        }
    }
    return ok;
}

bool validateSolution(const PDPSolution& solution, const PDPData& data, bool verbose) {
    bool ok = true;

    if (verbose) {
        cout << "\n+========================================================+\n"
             << "|               SOLUTION VALIDATION                     |\n"
             << "+========================================================+\n";
    }

    // [0] Decoder/simulator feasibility
    bool simOK = solution.isFeasible && solution.totalPenalty <= 1e-9;
    if (verbose) {
        cout << "\n[0] DECODED FEASIBILITY: "
             << (simOK ? "OK" : "FAILED")
             << " | flag=" << (solution.isFeasible ? "YES" : "NO")
             << " penalty=" << solution.totalPenalty << "\n";
    }
    if (!simOK) ok = false;

    // [1] Encoding constraints
    bool encOK = validateEncodingConstraints(solution.encoded_routes, data);
    if (verbose) {
        cout << "\n[1] ENCODING CONSTRAINTS: " << (encOK ? "✓" : "❌") << "\n";
    }
    if (!encOK) ok = false;

    // [2] Customer coverage
    unordered_map<int, int> served;
    for (const auto& r : solution.encoded_routes) {
        for (const auto& st : r) {
            if (st.node > 0) served[st.node]++;
        }
    }
    int missing = 0;
    int duplicated = 0;
    for (int i = 1; i < data.numNodes; ++i) {
        if (!data.isCustomer(i)) continue;
        if (served[i] == 0) missing++;
        if (served[i] > 1) duplicated++;
    }
    if (verbose) {
        cout << "\n[2] CUSTOMER COVERAGE: missing=" << missing << " duplicated=" << duplicated << "\n";
    }
    if (missing != 0 || duplicated != 0) ok = false;

    // [2b] Every depot-to-depot truck trip must finish all C1 packages
    // loaded at its starting depot or received from a drone during that trip.
    bool c1TripsOK = validateC1TripBoundaries(solution.encoded_routes, data, verbose);
    if (verbose) {
        cout << "\n[2b] C1 TRIP BOUNDARIES: " << (c1TripsOK ? "OK" : "FAILED") << "\n";
    }
    if (!c1TripsOK) ok = false;

    // [3] Drone endurance + waiting
    if (verbose) {
        cout << "\n[3] DRONE FEASIBILITY (endurance=" << fixed << setprecision(1) << data.droneEndurance
             << " wait<=" << data.allowedWait << "):\n";
    }

    for (const auto& ev : solution.resupply_events) {
        const double truck_wait = max(0.0, ev.drone_arrive_time - ev.truck_arrive_time);
        const bool enduranceOK = (ev.total_flight_time <= data.droneEndurance + 1e-9);
        const bool waitOK = (truck_wait <= data.allowedWait + 1e-9);
        if (verbose) {
            cout << "  Drone " << ev.drone_id << " h=" << ev.resupply_point
                 << " flight=" << ev.total_flight_time
                 << " wait=" << truck_wait
                 << " => " << ((enduranceOK && waitOK) ? "✓" : "❌") << "\n";
        }
        if (!enduranceOK || !waitOK) ok = false;
    }

    // [3b] Drone timing consistency and per-drone sequencing
    vector<vector<const ResupplyEvent*>> eventsByDrone(max(0, data.numDrones));
    for (const auto& ev : solution.resupply_events) {
        bool idOK = (ev.drone_id >= 0 && ev.drone_id < data.numDrones);
        if (idOK) eventsByDrone[ev.drone_id].push_back(&ev);

        double maxRelease = 0.0;
        for (int pkg : ev.customer_ids) {
            maxRelease = max(maxRelease, packageRelease(data, pkg));
        }

        bool nodeOK = data.isCustomer(ev.resupply_point);
        double flyOut = nodeOK ? data.droneDistMatrix[data.depotIndex][ev.resupply_point] : 0.0;
        double flyBack = nodeOK ? data.droneDistMatrix[ev.resupply_point][data.depotIndex] : 0.0;
        double expectedArrive = ev.drone_depart_time + flyOut;
        double expectedStart = max(ev.drone_arrive_time, ev.truck_arrive_time);
        double expectedEnd = expectedStart + data.resupplyTime;
        double expectedReturn = expectedEnd + flyBack;

        bool releaseOK = (ev.drone_depart_time + 1e-9 >= maxRelease + data.depotDroneLoadTime);
        bool timingOK =
            idOK && nodeOK &&
            fabs(ev.drone_arrive_time - expectedArrive) <= 0.1 &&
            fabs(ev.resupply_start_time - expectedStart) <= 0.1 &&
            fabs(ev.resupply_end_time - expectedEnd) <= 0.1 &&
            fabs(ev.drone_return_time - expectedReturn) <= 0.1;

        if (verbose) {
            cout << "  Drone " << ev.drone_id
                 << " timing=" << (timingOK ? "OK" : "BAD")
                 << " release=" << (releaseOK ? "OK" : "BAD") << "\n";
        }
        if (!releaseOK || !timingOK) ok = false;
    }

    for (int d = 0; d < (int)eventsByDrone.size(); ++d) {
        auto& evs = eventsByDrone[d];
        sort(evs.begin(), evs.end(), [](const ResupplyEvent* a, const ResupplyEvent* b) {
            return a->drone_depart_time < b->drone_depart_time;
        });

        for (int idx = 1; idx < (int)evs.size(); ++idx) {
            const ResupplyEvent* prev = evs[idx - 1];
            const ResupplyEvent* cur = evs[idx];
            bool sequenceOK = (cur->drone_depart_time + 1e-9 >=
                               prev->drone_return_time + data.depotDroneLoadTime);
            if (verbose && !sequenceOK) {
                cout << "  Drone " << d
                     << " sequence BAD: dep=" << cur->drone_depart_time
                     << " prev_return=" << prev->drone_return_time
                     << " load=" << data.depotDroneLoadTime << "\n";
            }
            if (!sequenceOK) ok = false;
        }
    }

    // [5] C2 Pairing and Precedence constraints
    bool c2OK = true;
    for (int i = 1; i < data.numNodes; ++i) {
        if (data.nodeTypes[i] == "P") {
            int pairId = data.pairIds[i];
            int d = -1;
            for (int j = 1; j < data.numNodes; ++j) {
                if (data.pairIds[j] == pairId && data.nodeTypes[j] == "DL") {
                    d = j;
                    break;
                }
            }
            if (d == -1) continue;

            int route_i = -1, pos_i = -1;
            int route_d = -1, pos_d = -1;
            for (int rIdx = 0; rIdx < (int)solution.encoded_routes.size(); ++rIdx) {
                const auto& r = solution.encoded_routes[rIdx];
                for (int sIdx = 0; sIdx < (int)r.size(); ++sIdx) {
                    if (r[sIdx].node == i) {
                        route_i = rIdx;
                        pos_i = sIdx;
                    }
                    if (r[sIdx].node == d) {
                        route_d = rIdx;
                        pos_d = sIdx;
                    }
                }
            }

            bool sameRoute = (route_i == route_d && route_i != -1);
            bool orderOK = (pos_i < pos_d);
            
            bool droneResupplied = false;
            for (const auto& r : solution.encoded_routes) {
                for (const auto& st : r) {
                    for (int pkg : st.packages) {
                        if (pkg == i || pkg == d) {
                            droneResupplied = true;
                        }
                    }
                }
            }

            if (!sameRoute || !orderOK || droneResupplied) {
                c2OK = false;
                if (verbose) {
                    cout << "  ❌ C2 constraint violated for pair " << pairId 
                         << " (Pickup " << i << ", Delivery " << d << "): "
                         << "SameRoute=" << (sameRoute ? "yes" : "no") 
                         << ", OrderOK=" << (orderOK ? "yes" : "no")
                         << ", DroneResupplied=" << (droneResupplied ? "yes" : "no") << "\n";
                }
            }
        }
    }
    if (verbose) {
        cout << "\n[5] C2 PAIRING CONSTRAINTS: " << (c2OK ? "✓" : "❌") << "\n";
    }
    if (!c2OK) ok = false;

    // [4] C_max verification
    double calc = 0.0;
    for (const auto& td : solution.truck_details) calc = max(calc, td.completion_time);
    for (double t : solution.drone_completion_times) calc = max(calc, t);

    if (verbose) {
        cout << "\n[4] C_MAX CHECK: reported=" << fixed << setprecision(2) << solution.totalCost
             << " calculated=" << calc
             << " => " << (fabs(calc - solution.totalCost) <= 0.1 ? "✓" : "❌") << "\n";
    }

    if (fabs(calc - solution.totalCost) > 0.1) ok = false;

    if (verbose) {
        cout << "\n[RESULT] " << (ok ? "VALID" : "INVALID") << "\n";
    }

    return ok;
}

void printSolutionSummary(const PDPSolution& solution, double costBeforeLS, double costAfterLS) {
    cout << "\n+---------------- SUMMARY ----------------+\n";
    cout << "C_max (before): " << fixed << setprecision(2) << costBeforeLS << "\n";
    cout << "C_max (after) : " << fixed << setprecision(2) << costAfterLS << "\n";
    cout << "Penalty       : " << solution.totalPenalty << "\n";
    cout << "Feasible      : " << (solution.isFeasible ? "YES" : "NO") << "\n";
    cout << "+-----------------------------------------+\n";
}
