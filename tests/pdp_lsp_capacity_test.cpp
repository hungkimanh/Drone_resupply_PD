#include "pdp_lsp.h"
#include "pdp_fitness.h"
#include <cmath>
#include <cstdlib>
#include <iostream>

int main() {
    PDPData data;
    data.numCustomers=2; data.numNodes=3;
    data.numTrucks=1; data.numDrones=1; data.truckCapacity=1;
    data.droneCapacity=2; data.droneEndurance=7;
    data.lspWorkers=1;
    data.nodeTypes={"D","D","D"};
    data.readyTimes={0,0,0}; data.pairIds={0,0,0};data.demands={0,1,1};
    data.truckDistMatrix={{0,2,4},{2,0,2},{4,2,0}};
    data.droneDistMatrix={{0,1,2},{1,0,1},{2,1,0}};
    // Only node 1 is within drone endurance. Its own package is already
    // onboard, so service frees capacity before parcel 2 is handed over.
    SolutionEncoding warm{{RouteStop(0,{1}),RouteStop(1,{2}),RouteStop(2),RouteStop(0)}};
    auto warm_decoded=decode_solution(warm,data,false);
    auto result=solveLSP(data,VRPSolution{{1,2}},warm,3.0);
    auto decoded=decode_solution(result.encoding,data,false);
    if (!warm_decoded.isFeasible || !result.feasible || !decoded.isFeasible ||
        std::abs(result.objective-19.0)>1e-6 || std::abs(result.model_objective-19.0)>1e-6) {
        std::cerr<<"Capacity must follow service/handover chronology: "
                 <<result.status<<" model="<<result.model_objective<<" physical="<<result.objective<<'\n';
        return 1;
    }
    return 0;
}
