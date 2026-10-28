#include "pdp_fitness.h"
#include "pdp_lb.h"
#include "pdp_reader.h"

#include <cmath>
#include <cstdlib>
#include <iostream>

static void require(bool condition, const char* message) {
    if (!condition) { std::cerr << message << '\n'; std::exit(1); }
}

static PDPData dataFor(int n) {
    PDPData d;
    d.numCustomers = n; d.numNodes = n + 1; d.numDrones = 1; d.numTrucks = 2;
    d.nodeTypes.assign(n + 1, "D"); d.readyTimes.assign(n + 1, 0);
    d.pairIds.assign(n + 1, 0); d.demands.assign(n + 1, 1); d.demands[0] = 0;
    d.truckDistMatrix.assign(n + 1, std::vector<double>(n + 1, 10.0));
    d.droneDistMatrix.assign(n + 1, std::vector<double>(n + 1, 5.0));
    for (int i = 0; i <= n; ++i) d.truckDistMatrix[i][i] = d.droneDistMatrix[i][i] = 0;
    return d;
}

int main() {
    auto twenty = dataFor(20);
    require(twenty.getDroneCapacity() == 2, "20 customers must use drone capacity 2");
    twenty.droneCapacity = 3;
    require(twenty.getDroneCapacity() == 3, "explicit drone capacity must take precedence");

    auto d = dataFor(1); d.readyTimes[1] = 50;
    SolutionEncoding enc{{RouteStop(0), RouteStop(1, {1}), RouteStop(0)}};
    auto sol = decode_solution(enc, d, false);
    require(sol.isFeasible && std::abs(sol.resupply_events[0].drone_depart_time - 50) < 1e-9,
            "release does not incur another depot turnaround");
    require(std::abs(sol.totalCost - 73) < 1e-9, "explicit single-customer timeline must equal 73");

    d = dataFor(2); d.readyTimes[1] = 50;
    enc = {{RouteStop(0), RouteStop(1, {1}), RouteStop(0)},
           {RouteStop(0), RouteStop(2, {2}), RouteStop(0)}};
    enc[0][1].fixed_drone_id = enc[1][1].fixed_drone_id = 0;
    enc[0][1].fixed_drone_trip = 2; enc[1][1].fixed_drone_trip = 1;
    require(retimeFixedDroneSchedule(enc, d), "joint shared-drone schedule must exist");
    sol = decode_solution(enc, d, false);
    require(sol.isFeasible && std::abs(sol.totalCost - 73) < 1e-9,
            "truck iteration order must not destroy a fixed shared-drone schedule");
    require(std::abs(enc[1][1].fixed_drone_depart - 5) < 1e-9,
            "earlier sortie may belong to the second truck");

    enc = {{RouteStop(0), RouteStop(1, {1}), RouteStop(2, {2}), RouteStop(0)}};
    enc[0][1].fixed_drone_id = enc[0][2].fixed_drone_id = 0;
    enc[0][1].fixed_drone_trip = 2; enc[0][2].fixed_drone_trip = 1;
    require(!retimeFixedDroneSchedule(enc, d), "inconsistent truck/drone order must be rejected");

    d = dataFor(2); d.allowC1CarryAcrossDepot = true;
    enc = {{RouteStop(0, {1, 2}), RouteStop(1), RouteStop(0), RouteStop(2), RouteStop(0)}};
    require(decode_solution(enc, d, false).isFeasible, "paper benchmark permits C1 carry across depot");
    d.depotDroneLoadTime = 19; d.resupplyTime = 2;
    require(std::abs(computeNodeLB(d, 1) - 10) < 1e-9,
            "node bound uses rendezvous handover, not depot turnaround");
    return 0;
}
