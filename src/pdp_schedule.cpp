#include "pdp_fitness.h"

#include <algorithm>
#include <cmath>
#include <map>
#include <tuple>
#include <vector>

bool retimeFixedDroneSchedule(SolutionEncoding& enc, const PDPData& data) {
    struct Edge { int from, to; double lag; };
    struct Sortie { int event, node, truck, stop, order; };
    std::vector<Edge> edges;
    std::vector<double> earliest(1, 0.0);
    auto variable = [&]() { earliest.push_back(0.0); return int(earliest.size()) - 1; };
    auto edge = [&](int a, int b, double lag) { edges.push_back({a, b, lag}); };
    std::map<int, std::vector<Sortie>> drones;
    std::vector<Sortie> all;

    for (int v = 0; v < int(enc.size()); ++v) {
        if (enc[v].empty() || enc[v].front().node != 0 || enc[v].back().node != 0)
            return false;
        int previous = 0, previous_node = 0;
        for (int k = 0; k < int(enc[v].size()); ++k) {
            auto& stop = enc[v][k];
            const int n = stop.node;
            if (n < 0 || n >= data.numNodes) return false;
            const int depart = variable();
            const double travel = data.truckDistMatrix[previous_node][n];
            if (n == 0) {
                const bool intermediate = k > 0 && k + 1 < int(enc[v].size());
                edge(previous, depart, travel + (intermediate ? data.depotReceiveTime : 0.0));
                for (int pkg : stop.packages) {
                    if (!data.isCustomer(pkg)) return false;
                    edge(0, depart, data.readyTimes[pkg]);
                }
            } else if (stop.packages.empty()) {
                edge(previous, depart, travel + data.truckServiceTime);
            } else {
                if (stop.fixed_drone_id < 0 || stop.fixed_drone_id >= data.numDrones ||
                    stop.fixed_drone_trip < 1) return false;
                const bool own = std::find(stop.packages.begin(), stop.packages.end(), n)
                                   != stop.packages.end();
                const double fly = data.droneDistMatrix[0][n];
                const int launch = variable();
                // Dispatch just in time: no hovering and service before handover
                // whenever the rendezvous customer's package is already onboard.
                edge(previous, launch, travel + (own ? 0.0 : data.truckServiceTime) - fly);
                for (int pkg : stop.packages) {
                    if (!data.isCustomer(pkg)) return false;
                    edge(0, launch, data.readyTimes[pkg]);
                }
                edge(launch, depart, fly + data.resupplyTime + (own ? data.truckServiceTime : 0.0));
                Sortie sortie{launch, n, v, k, stop.fixed_drone_trip};
                drones[stop.fixed_drone_id].push_back(sortie);
                all.push_back(sortie);
            }
            previous = depart; previous_node = n;
        }
    }
    for (auto& item : drones) {
        auto& sorties = item.second;
        std::sort(sorties.begin(), sorties.end(), [](const Sortie& a, const Sortie& b) {
            return a.order < b.order;
        });
        for (int k = 0; k < int(sorties.size()); ++k) {
            if (sorties[k].order != k + 1) return false;
            if (k == 0) continue; // first sortie is prepared at time zero
            const auto& p = sorties[k - 1];
            edge(p.event, sorties[k].event, data.droneDistMatrix[0][p.node] +
                 data.resupplyTime + data.droneDistMatrix[p.node][0] + data.depotDroneLoadTime);
        }
    }
    // Difference constraints. Unlike a greedy truck-by-truck allocation, this
    // accounts for all shared drones and both truck routes simultaneously.
    for (int pass = 0; pass < int(earliest.size()); ++pass) {
        bool changed = false;
        for (const auto& e : edges) {
            if (earliest[e.to] + 1e-9 < earliest[e.from] + e.lag) {
                if (e.to == 0) return false;
                earliest[e.to] = earliest[e.from] + e.lag;
                changed = true;
            }
        }
        if (!changed) {
            for (const auto& s : all) {
                auto& stop = enc[s.truck][s.stop];
                stop.fixed_drone_depart = earliest[s.event];
                stop.fixed_drone_arrive = earliest[s.event] + data.droneDistMatrix[0][s.node];
                stop.fixed_drone_return = stop.fixed_drone_arrive + data.resupplyTime +
                                          data.droneDistMatrix[s.node][0];
            }
            return true;
        }
    }
    return false;
}
