#include "pdp_lb.h"
#include <algorithm>
#include <cmath>
#include <limits>

// Distance matrices are already in minutes (see pdp_reader.cpp comments).
static double truckTravelTime(const PDPData& d, int from, int to) {
    return d.truckDistMatrix[from][to];
}
static double droneTravelTime(const PDPData& d, int from, int to) {
    return d.droneDistMatrix[from][to];
}

// a_i = r_i + min(depotDroneLoadTime + flyTime(depot,i), t_{0i}) + s_i
double computeNodeLB(const PDPData& data, int customer) {
    if (customer <= 0 || customer >= data.numNodes) return 0.0;

    double r_i        = static_cast<double>(data.readyTimes[customer]);
    double s_i        = data.truckServiceTime;
    double t_0i       = truckTravelTime(data, 0, customer);
    double drone_reach = data.resupplyTime
                       + droneTravelTime(data, 0, customer);

    // P/DL requests originate at a customer, not at the depot. Do not apply
    // the delivery-only release/depot bound to those nodes.
    if (data.nodeTypes[customer] == "P" || data.nodeTypes[customer] == "DL")
        return s_i;

    return r_i + std::min(drone_reach, t_0i) + s_i;
}

LBInfo computeRouteLB(const PDPData& data, const std::vector<int>& route) {
    LBInfo info;
    if (route.empty()) {
        info.lb_completion = 0.0;
        return info;
    }

    int m = static_cast<int>(route.size());
    info.li.resize(m);

    double a_first = computeNodeLB(data, route[0]);
    double t_first = truckTravelTime(data, 0, route[0]);
    info.li[0] = std::max(a_first, t_first);

    for (int k = 1; k < m; ++k) {
        int prev = route[k - 1];
        int curr = route[k];
        double a_k = computeNodeLB(data, curr);
        double t_prev_curr = truckTravelTime(data, prev, curr);
        info.li[k] = std::max(a_k, info.li[k - 1] + t_prev_curr);
    }

    int last = route[m - 1];
    double t_last_depot = truckTravelTime(data, last, 0);
    info.lb_completion = info.li[m - 1] + t_last_depot;

    return info;
}

double computeVRPLB(const PDPData& data, const VRPSolution& vrp) {
    double lb = 0.0;
    for (const auto& route : vrp) {
        if (route.empty()) continue;
        LBInfo info = computeRouteLB(data, route);
        lb = std::max(lb, info.lb_completion);
    }
    return lb;
}

double computeThreshold(double z_star, double lb_star, double phi) {
    return (1.0 - phi) * z_star + phi * lb_star;
}

int findLastTruckByLB(const PDPData& data, const VRPSolution& vrp) {
    int last_truck = 0;
    double max_lb = -1.0;

    for (int v = 0; v < static_cast<int>(vrp.size()); ++v) {
        if (vrp[v].empty()) continue;
        LBInfo info = computeRouteLB(data, vrp[v]);
        if (info.lb_completion > max_lb) {
            max_lb = info.lb_completion;
            last_truck = v;
        }
    }
    return last_truck;
}
