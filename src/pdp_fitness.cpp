#include "pdp_fitness.h"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <set>
#include <unordered_set>

using namespace std;

static double truckTravelMinutes(const PDPData& data, int a, int b) {
    if (a < 0 || a >= data.numNodes || b < 0 || b >= data.numNodes) {
        return numeric_limits<double>::infinity();
    }
    // Dist matrices are already in minutes in pdp_reader.cpp.
    return data.truckDistMatrix[a][b];
}

static double droneTravelMinutes(const PDPData& data, int a, int b) {
    if (a < 0 || a >= data.numNodes || b < 0 || b >= data.numNodes) {
        return numeric_limits<double>::infinity();
    }
    // Dist matrices are already in minutes in pdp_reader.cpp.
    return data.droneDistMatrix[a][b];
}

static double packageWeight(const PDPData& data, int pkg) {
    if (pkg <= 0 || pkg >= (int)data.demands.size()) return 0.0;
    return (double)data.demands[pkg];
}

static double packageRelease(const PDPData& data, int pkg) {
    if (pkg <= 0 || pkg >= (int)data.readyTimes.size()) return 0.0;
    return (double)data.readyTimes[pkg];
}

static bool containsPackage(const vector<int>& packages, int pkg) {
    return find(packages.begin(), packages.end(), pkg) != packages.end();
}

static int findPickupForDelivery(const PDPData& data, int dl) {
    if (dl <= 0 || dl >= data.numNodes) return -1;
    for (int p = 1; p < data.numNodes; ++p) {
        if (data.pairIds[p] == data.pairIds[dl] && data.nodeTypes[p] == "P") {
            return p;
        }
    }
    return -1;
}

struct DroneTry {
    bool feasible = false;
    int drone_id = -1;
    ResupplyEvent ev;
    double truck_wait = 0.0;
};

static DroneTry tryAssignDroneTrip(const PDPData& data,
                                  const vector<double>& droneAvail,
                                  int truck_id,
                                  int rendezvous,
                                  const vector<int>& packages,
                                  double truck_sync_time,
                                  const RouteStop* fixed_stop = nullptr) {
    DroneTry best;
    best.feasible = false;

    double totalDemand = 0.0;
    double maxRelease = 0.0;
    for (int p : packages) {
        totalDemand += packageWeight(data, p);
        maxRelease = max(maxRelease, packageRelease(data, p));
    }
    if (totalDemand > (double)data.getDroneCapacity() + 1e-9) {
        return best;
    }

    const double flyOut = droneTravelMinutes(data, data.depotIndex, rendezvous);
    const double flyBack = droneTravelMinutes(data, rendezvous, data.depotIndex);
    if (!isfinite(flyOut) || !isfinite(flyBack)) {
        return best;
    }

    for (int d = 0; d < data.numDrones; ++d) {
        if (fixed_stop != nullptr && fixed_stop->fixed_drone_id >= 0 &&
            d != fixed_stop->fixed_drone_id) {
            continue;
        }
        ResupplyEvent ev;
        ev.customer_ids = packages;
        ev.resupply_point = rendezvous;
        ev.drone_id = d;
        ev.truck_id = truck_id;

        const bool fixedTiming = fixed_stop != nullptr &&
            fixed_stop->fixed_drone_depart >= 0.0;
        const double earliestDepart = max(droneAvail[d], maxRelease) + data.depotDroneLoadTime;
        const double latestNoWaitDepart = truck_sync_time - flyOut;
        ev.drone_depart_time = fixedTiming
            ? fixed_stop->fixed_drone_depart
            : max(earliestDepart, latestNoWaitDepart);
        ev.drone_arrive_time = fixedTiming
            ? fixed_stop->fixed_drone_arrive
            : ev.drone_depart_time + flyOut;

        // Store the time when the truck is actually ready to synchronize:
        // arrival time for "receive before service", or service completion time
        // for "serve before receive".
        ev.truck_arrive_time = truck_sync_time;
        ev.resupply_start_time = max(ev.drone_arrive_time, truck_sync_time);
        ev.resupply_end_time = ev.resupply_start_time + data.resupplyTime;

        const double drone_wait = ev.resupply_start_time - ev.drone_arrive_time;
        // Endurance is measured from drone departure until return, including
        // waiting at rendezvous and the physical handover duration.
        ev.total_flight_time = flyOut + drone_wait + data.resupplyTime + flyBack;
        ev.drone_return_time = fixedTiming
            ? fixed_stop->fixed_drone_return
            : ev.resupply_end_time + flyBack;

        const double truck_wait = max(0.0, ev.drone_arrive_time - truck_sync_time);

        const bool enduranceOK = (ev.total_flight_time <= data.droneEndurance + 1e-9);
        const bool waitOK = (truck_wait <= data.allowedWait + 1e-9);

        const bool timingOK = !fixedTiming || (
            ev.drone_depart_time + 1e-6 >= maxRelease + data.depotDroneLoadTime &&
            fabs(ev.drone_arrive_time - ev.drone_depart_time - flyOut) <= 1e-5 &&
            ev.drone_return_time + 1e-6 >= ev.resupply_end_time + flyBack);
        const bool feasible = enduranceOK && waitOK && timingOK;
        if (!feasible) {
            continue;
        }

        if (!best.feasible ||
            ev.resupply_end_time < best.ev.resupply_end_time ||
            (fabs(ev.resupply_end_time - best.ev.resupply_end_time) <= 1e-9 &&
             ev.drone_return_time < best.ev.drone_return_time)) {
            best.feasible = true;
            best.drone_id = d;
            best.ev = ev;
            best.truck_wait = truck_wait;
        }
    }

    return best;
}

static vector<int> extractCustomersFromRoute(const Route& route) {
    vector<int> customers;
    customers.reserve(route.size());
    for (const auto& st : route) {
        if (st.node > 0) customers.push_back(st.node);
    }
    return customers;
}

PDPSolution decode_solution(const SolutionEncoding& encoded, const PDPData& data,
                            bool throw_on_infeasible) {
    PDPSolution sol;
    sol.encoded_routes = encoded;
    sol.totalCost = 0.0;
    sol.totalPenalty = 0.0;
    sol.isFeasible = true;

    const int nTrucks = (int)encoded.size();
    sol.customer_routes.resize(nTrucks);
    sol.truck_details.clear();
    sol.resupply_events.clear();
    sol.drone_completion_times.assign(data.numDrones, 0.0);
    sol.drone_order.clear();

    vector<double> droneAvail(data.numDrones, 0.0);

    auto addPenalty = [&](double p, const string& msg) {
        sol.totalPenalty += p;
        sol.isFeasible = false;
        if (throw_on_infeasible) {
            throw InfeasibleException(msg);
        }
    };

    for (int k = 0; k < nTrucks; ++k) {
        const Route& route = encoded[k];
        sol.customer_routes[k] = extractCustomersFromRoute(route);

        TruckRouteInfo info;
        info.truck_id = k;
        info.route.clear();
        info.arrival_times.clear();
        info.departure_times.clear();

        double t = 0.0;
        int pos = data.depotIndex;

        auto logVisit = [&](int node, double arrival, double depart) {
            info.route.push_back(node);
            info.arrival_times.push_back(arrival);
            info.departure_times.push_back(depart);
        };

        logVisit(data.depotIndex, 0.0, 0.0);

        unordered_set<int> onboard;
        double onboardLoad = 0.0;

        // Basic structural checks
        if (route.empty()) {
            addPenalty(1e6, "Empty route");
            continue;
        }
        if (route.front().node != 0) {
            addPenalty(1e4, "Route must start at depot node 0");
        }
        if (route.back().node != 0 || !route.back().packages.empty()) {
            addPenalty(1e4, "Route must end with depot node 0 (no packages)");
        }

        for (size_t idx = 0; idx < route.size(); ++idx) {
            const RouteStop& st = route[idx];

            if (st.node == 0) {
                // Move to depot if needed
                if (pos != data.depotIndex) {
                    const double travel = truckTravelMinutes(data, pos, data.depotIndex);
                    if (!isfinite(travel)) {
                        addPenalty(1e6, "Invalid truck travel to depot");
                        break;
                    }
                    const double arr = t + travel;
                    t = arr;
                    pos = data.depotIndex;
                    logVisit(data.depotIndex, arr, arr);
                }

                // A depot visit closes the current truck trip. C1 packages
                // supplied at a depot or by a drone must be delivered before
                // that boundary; C2 pickup loads may remain onboard.
                if (idx > 0) {
                    for (int p : onboard) {
                        if (p > 0 && p < (int)data.nodeTypes.size() &&
                            data.nodeTypes[p] != "P" && data.nodeTypes[p] != "DL") {
                            addPenalty(1e4, "C1 package still onboard at end of truck trip");
                        }
                    }
                }

                if (!st.packages.empty()) {
                    double maxRel = 0.0;
                    double loadW = 0.0;
                    for (int p : st.packages) {
                        if (p > 0 && p < (int)data.nodeTypes.size()) {
                            if (data.nodeTypes[p] == "P" || data.nodeTypes[p] == "DL") {
                                addPenalty(1e5, "Cannot load C2 Pickup/Delivery packages at Depot");
                            }
                        }
                        maxRel = max(maxRel, packageRelease(data, p));
                        loadW += packageWeight(data, p);
                    }

                    double arr = t;
                    double dep = 0.0;
                    if (idx == 0 && info.route.size() == 1 && info.route[0] == data.depotIndex) {
                        dep = maxRel;
                        info.departure_times[0] = dep;
                    } else {
                        dep = max(arr + data.depotReceiveTime, maxRel);
                        if (!info.route.empty() && info.route.back() == data.depotIndex)
                            info.departure_times.back() = dep;
                        else
                            logVisit(data.depotIndex, arr, dep);
                    }

                    if (onboardLoad + loadW > (double)data.truckCapacity + 1e-9) {
                        addPenalty(1e4, "Truck capacity exceeded at depot load");
                    }

                    for (int p : st.packages) {
                        onboard.insert(p);
                    }
                    onboardLoad += loadW;

                    t = dep;
                } else if (idx > 0 && idx + 1 < route.size()) {
                    // An intermediate depot stop starts a new truck trip. The
                    // final return has no handling time.
                    t += data.depotReceiveTime;
                    if (!info.departure_times.empty())
                        info.departure_times.back() = t;
                }

                // End depot node (0 without packages) handled implicitly.
                continue;
            }

            // Customer node
            const int i = st.node;
            if (!data.isCustomer(i)) {
                addPenalty(1e4, "Non-customer node appears in route");
            }

            const double travel = truckTravelMinutes(data, pos, i);
            if (!isfinite(travel)) {
                addPenalty(1e6, "Invalid truck travel");
                break;
            }

            const double arrive = t + travel;
            t = arrive;
            pos = i;

            auto serveCustomer = [&]() {
                if (i > 0 && i < (int)data.nodeTypes.size()) {
                    string type = data.nodeTypes[i];
                    if (type == "P") {
                        onboard.insert(i);
                        onboardLoad += packageWeight(data, i);
                    } else if (type == "DL") {
                        int pickupNode = findPickupForDelivery(data, i);
                        if (pickupNode == -1 || !onboard.count(pickupNode)) {
                            addPenalty(1e4, "Corresponding C2 Pickup package not onboard before C2 Delivery service");
                        } else {
                            onboard.erase(pickupNode);
                            onboardLoad -= packageWeight(data, pickupNode);
                        }
                    } else { // "D" or other types: Delivery C1
                        if (!onboard.count(i)) {
                            addPenalty(1e4, "Package not available before C1 service");
                        } else {
                            onboard.erase(i);
                            onboardLoad -= packageWeight(data, i);
                        }
                    }
                } else {
                    addPenalty(1e4, "Invalid customer node index");
                }

                if (onboardLoad > (double)data.truckCapacity + 1e-9) {
                    addPenalty(1e4, "Truck capacity exceeded after service");
                }

                t += data.truckServiceTime;
            };

            auto receiveDrone = [&](double truck_sync_time) {
                // Disallow supplying packages that are served earlier in the route (position rule)
                // This is a structural check; detailed repair happens in GA operators.
                // Here we only ensure packages are valid customer ids.
                for (int p : st.packages) {
                    if (!data.isCustomer(p)) {
                        addPenalty(1e4, "Resupply contains invalid package id");
                    }
                    if (p > 0 && p < (int)data.nodeTypes.size()) {
                        if (data.nodeTypes[p] == "P" || data.nodeTypes[p] == "DL") {
                            addPenalty(1e5, "Drone cannot resupply C2 Pickup/Delivery packages");
                        }
                    }
                }

                const RouteStop* fixedStop = st.fixed_drone_id >= 0 ? &st : nullptr;
                DroneTry best = tryAssignDroneTrip(
                    data, droneAvail, k, i, st.packages, truck_sync_time, fixedStop);
                if (!best.feasible) {
                    addPenalty(1e4, "Infeasible drone trip (capacity/endurance/wait)");
                    return;
                }

                t = best.ev.resupply_end_time;
                droneAvail[best.drone_id] = max(
                    droneAvail[best.drone_id], best.ev.drone_return_time);

                const int evIndex = (int)sol.resupply_events.size();
                sol.resupply_events.push_back(best.ev);
                sol.drone_order.push_back(evIndex);

                for (int p : st.packages) {
                    if (!onboard.count(p)) {
                        onboard.insert(p);
                        onboardLoad += packageWeight(data, p);
                    }
                }
                if (onboardLoad > (double)data.truckCapacity + 1e-9) {
                    addPenalty(1e4, "Truck capacity exceeded after resupply");
                }
            };

            bool serveBeforeDrone = false;
            if (!st.packages.empty() && i > 0 && i < (int)data.nodeTypes.size()) {
                const string& type = data.nodeTypes[i];
                const bool currentPkgFromDrone = containsPackage(st.packages, i);
                if (type == "P") {
                    serveBeforeDrone = true;
                } else if (type == "DL") {
                    int pickupNode = findPickupForDelivery(data, i);
                    serveBeforeDrone = (pickupNode != -1 && onboard.count(pickupNode) > 0);
                } else {
                    serveBeforeDrone = (onboard.count(i) > 0 && !currentPkgFromDrone);
                }
            }

            if (!st.packages.empty() && serveBeforeDrone) {
                serveCustomer();
                receiveDrone(t);
            } else {
                if (!st.packages.empty()) {
                    receiveDrone(t);
                }
                serveCustomer();
            }

            logVisit(i, arrive, t);
        }

        // Ensure truck returns to depot at end
        if (pos != data.depotIndex) {
            const double travel = truckTravelMinutes(data, pos, data.depotIndex);
            if (!isfinite(travel)) {
                addPenalty(1e6, "Invalid truck travel to depot at end");
            } else {
                const double arr = t + travel;
                logVisit(data.depotIndex, arr, arr);
                t = arr;
            }
        }

        info.completion_time = t;
        sol.truck_details.push_back(info);
    }

    for (int d = 0; d < data.numDrones; ++d) {
        sol.drone_completion_times[d] = droneAvail[d];
    }

    double makespan = 0.0;
    for (const auto& td : sol.truck_details) makespan = max(makespan, td.completion_time);
    for (double dt : sol.drone_completion_times) makespan = max(makespan, dt);
    sol.totalCost = makespan;

    return sol;
}

void print_decoded_routes(const PDPSolution& sol, const PDPData& data) {
    (void)data;
    cout << "\n[DECODED SUMMARY]\n";
    cout << "  C_max: " << fixed << setprecision(2) << sol.totalCost
         << " | penalty: " << sol.totalPenalty
         << " | feasible: " << (sol.isFeasible ? "YES" : "NO") << "\n";

    cout << "\n[TRUCK ROUTES]\n";
    for (const auto& tr : sol.truck_details) {
        cout << "Truck " << tr.truck_id << ": ";
        for (size_t i = 0; i < tr.route.size(); ++i) {
            cout << tr.route[i];
            if (i + 1 < tr.route.size()) cout << " -> ";
        }
        cout << " | finish=" << fixed << setprecision(2) << tr.completion_time << "\n";
        for (size_t i = 0; i < tr.route.size(); ++i) {
            cout << "  stop " << tr.route[i]
                 << " arr=" << fixed << setprecision(2) << tr.arrival_times[i]
                 << " dep=" << tr.departure_times[i] << "\n";
        }
    }

    cout << "\n[DRONE TRIPS]\n";
    for (const auto& ev : sol.resupply_events) {
        cout << "Drone " << ev.drone_id << " -> h=" << ev.resupply_point << " packages={";
        for (size_t i = 0; i < ev.customer_ids.size(); ++i) {
            cout << ev.customer_ids[i] << (i + 1 < ev.customer_ids.size() ? "," : "");
        }
        cout << "} dep=" << fixed << setprecision(2) << ev.drone_depart_time
             << " arr=" << ev.drone_arrive_time
             << " start=" << ev.resupply_start_time
             << " end=" << ev.resupply_end_time
             << " ret=" << ev.drone_return_time << "\n";
    }
}
