#include "pdp_fitness.h"

#include <cstdlib>
#include <iostream>
#include <string>

namespace {

PDPData makeData(std::initializer_list<std::string> types) {
    PDPData data;
    data.numNodes = static_cast<int>(types.size());
    data.numCustomers = data.numNodes - 1;
    data.numTrucks = 1;
    data.numDrones = 1;
    data.truckCapacity = 10;
    data.nodeTypes.assign(types);
    data.readyTimes.assign(data.numNodes, 0);
    data.pairIds.assign(data.numNodes, -1);
    data.demands.assign(data.numNodes, 1);
    data.truckDistMatrix.assign(data.numNodes, std::vector<double>(data.numNodes, 0.0));
    data.droneDistMatrix.assign(data.numNodes, std::vector<double>(data.numNodes, 0.0));
    return data;
}

void expectFeasible(const char* name, const SolutionEncoding& encoding,
                    const PDPData& data, bool expected) {
    const PDPSolution solution = decode_solution(encoding, data, false);
    if (solution.isFeasible != expected) {
        std::cerr << name << ": expected feasible=" << expected
                  << ", got feasible=" << solution.isFeasible
                  << ", penalty=" << solution.totalPenalty << "\n";
        std::exit(EXIT_FAILURE);
    }
}

} // namespace

int main() {
    PDPData c1 = makeData({"DEPOT", "D", "D"});

    expectFeasible("depot-loaded C1 crosses depot",
                   {Route{RouteStop(0, {1}), RouteStop(0), RouteStop(1), RouteStop(0)}},
                   c1, false);

    expectFeasible("drone-supplied C1 crosses depot",
                   {Route{RouteStop(0, {1}), RouteStop(1, {2}), RouteStop(0),
                          RouteStop(2), RouteStop(0)}},
                   c1, false);

    expectFeasible("C1 delivered in its source trip",
                   {Route{RouteStop(0, {1, 2}), RouteStop(1), RouteStop(2), RouteStop(0)}},
                   c1, true);

    PDPData c2 = makeData({"DEPOT", "P", "DL"});
    c2.pairIds[1] = 7;
    c2.pairIds[2] = 7;
    expectFeasible("C2 pickup crosses depot",
                   {Route{RouteStop(0), RouteStop(1), RouteStop(0), RouteStop(2),
                          RouteStop(0)}},
                   c2, true);

    PDPData endurance = makeData({"DEPOT", "D"});
    endurance.droneDistMatrix[0][1] = 43.0;
    endurance.droneDistMatrix[1][0] = 43.0;
    endurance.droneEndurance = 90.0;
    endurance.resupplyTime = 5.0;
    expectFeasible("drone endurance includes handover",
                   {Route{RouteStop(0), RouteStop(1, {1}), RouteStop(0)}},
                   endurance, false);

    return EXIT_SUCCESS;
}
