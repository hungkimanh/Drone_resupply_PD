#include "pdp_init.h"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <unordered_set>

using namespace std;

// ============================================================
// Helper functions
// ============================================================

static double truckTravelMinutes(const PDPData& data, int a, int b) {
    if (a < 0 || a >= data.numNodes || b < 0 || b >= data.numNodes) return 1e18;
    return data.truckDistMatrix[a][b];
}

static double droneTravelMinutes(const PDPData& data, int a, int b) {
    if (a < 0 || a >= data.numNodes || b < 0 || b >= data.numNodes) return 1e18;
    return data.droneDistMatrix[a][b];
}

static double pkgRelease(const PDPData& data, int p) {
    if (p <= 0 || p >= (int)data.readyTimes.size()) return 0.0;
    return (double)data.readyTimes[p];
}

static double pkgWeight(const PDPData& data, int p) {
    if (p <= 0 || p >= (int)data.demands.size()) return 0.0;
    return (double)data.demands[p];
}

static vector<int> listCustomers(const PDPData& data) {
    vector<int> c;
    for (int i = 1; i < data.numNodes; ++i) {
        if (data.isCustomer(i)) c.push_back(i);
    }
    return c;
}

/**
 * @brief Check if a customer needs its package loaded on the truck before visiting.
 *
 * - Type "P"  (Pickup): truck picks up at customer location → no pre-loading needed
 * - Type "DL" (Delivery Location): needs paired pickup → handled by route ordering
 * - Type "D"  (Direct delivery): needs package from depot or drone resupply
 */
static bool customerNeedsPackage(const PDPData& data, int j) {
    if (j <= 0 || j >= (int)data.nodeTypes.size()) return true;
    if (data.nodeTypes[j] == "P")  return false;
    if (data.nodeTypes[j] == "DL") return false;
    return true;  // Type "D" or unknown
}

static bool isDirectDelivery(const PDPData& data, int j) {
    return j > 0 && j < (int)data.nodeTypes.size() && data.nodeTypes[j] == "D";
}

// ============================================================
// Drone Schedule tracking during init construction
// (Shared across all trucks in one individual)
// ============================================================

struct DroneScheduleInit {
    vector<double> droneAvail;  // droneAvail[d] = time when drone d returns to depot

    explicit DroneScheduleInit(int numDrones, double turnaround)
        : droneAvail(numDrones, -turnaround) {}

    /// Find the earliest-available drone
    int findBestDrone() const {
        int best = -1;
        double bestTime = 1e18;
        for (int d = 0; d < (int)droneAvail.size(); ++d) {
            if (droneAvail[d] < bestTime) {
                bestTime = droneAvail[d];
                best = d;
            }
        }
        return best;
    }

    void update(int droneId, double returnTime) {
        if (droneId >= 0 && droneId < (int)droneAvail.size())
            droneAvail[droneId] = returnTime;
    }
};

// ============================================================
// Drone resupply feasibility result
// ============================================================

struct DroneResupplyOption {
    bool   feasible = false;
    vector<int> packages;        // packages the drone carries
    int    drone_id = -1;
    double drone_arrive_time = 0.0;
    double drone_return_time = 0.0;
    double cost = 1e18;          // estimated truck delay caused by drone
};

// ============================================================
// Algorithm 11 (ga_encoding_init_solution.md §11)
// SELECT_DEPOT_LOAD
//
// Greedily load ready packages of upcoming customers,
// subject to truck remaining capacity.
// ============================================================

static vector<int> selectDepotLoad(
    const vector<int>& customers,
    int pos,
    double& current_time,
    const unordered_set<int>& onboard,
    const PDPData& data)
{
    vector<int> P;
    double remaining = (double)data.truckCapacity;
    for (int p : onboard) remaining -= pkgWeight(data, p);

    // Phase 1: Load all currently-ready packages (original behavior)
    for (int h = pos; h < (int)customers.size(); ++h) {
        int i = customers[h];
        if (!isDirectDelivery(data, i)) continue;
        if (onboard.count(i)) continue;
        if (pkgRelease(data, i) > current_time + 1e-9) continue;

        double w = pkgWeight(data, i);
        if (w > remaining + 1e-9) continue;

        P.push_back(i);
        remaining -= w;
        if (remaining <= 1e-9) break;
    }

    // Phase 2: Batch Wait at Depot
    // Consider waiting for not-yet-ready packages if waiting is cheaper
    // than a future depot return trip.
    //   wait_cost   = release_time - current_time
    //   return_cost = 2 × truck_travel(depot, customer) + depotReceiveTime
    // If wait_cost < return_cost → wait and load now to avoid costly return.
    // This is especially effective when depot is far (outside/border).
    if (remaining > 1e-9) {
        unordered_set<int> inP(P.begin(), P.end());
        double batchWaitUntil = current_time;

        for (int h = pos; h < (int)customers.size(); ++h) {
            int i = customers[h];
            if (onboard.count(i)) continue;
            if (inP.count(i)) continue;

            double ri = pkgRelease(data, i);
            if (ri <= current_time + 1e-9) continue;  // already handled in Phase 1

            // Only type "D" packages need depot/drone supply
            if (!isDirectDelivery(data, i)) continue;

            double w = pkgWeight(data, i);
            if (w > remaining + 1e-9) continue;

            double waitCost = ri - current_time;
            // Conservative return cost: round-trip depot ↔ customer + receive time
            double returnCost = 2.0 * truckTravelMinutes(data, data.depotIndex, i)
                              + data.depotReceiveTime;

            if (waitCost < returnCost) {
                P.push_back(i);
                inP.insert(i);
                remaining -= w;
                batchWaitUntil = max(batchWaitUntil, ri);
                if (remaining <= 1e-9) break;
            }
        }

        // Advance time to reflect waiting at depot
        if (batchWaitUntil > current_time + 1e-9) {
            current_time = batchWaitUntil;
        }
    }

    return P;
}

// ============================================================
// Algorithm 12 (ga_encoding_init_solution.md §12)
// FIND_DRONE_RESUPPLY_SET
//
// Finds a feasible set of packages that a drone can carry
// from the depot to rendezvous_node, respecting:
//   - drone capacity Q_D
//   - drone endurance (total flight time)
//   - truck allowed wait time
//   - package release dates
//   - only type "D" packages (not P or DL)
// ============================================================

static DroneResupplyOption findDroneResupplySet(
    const vector<int>& customers,
    int pos,
    int rendezvous_node,
    double truck_time,
    const unordered_set<int>& onboard,
    const PDPData& data,
    const DroneScheduleInit& schedule)
{
    DroneResupplyOption result;

    // Cannot do drone resupply at depot
    if (rendezvous_node <= 0) return result;

    const double QD      = (double)data.getDroneCapacity();
    const double flyOut  = droneTravelMinutes(data, data.depotIndex, rendezvous_node);
    const double flyBack = droneTravelMinutes(data, rendezvous_node, data.depotIndex);

    if (!isfinite(flyOut) || !isfinite(flyBack)) return result;

    // Basic endurance: even an empty round-trip must be feasible
    if (flyOut + flyBack > data.droneEndurance + 1e-9) return result;

    double remaining_cap = QD;
    vector<int> P_drone;
    double maxRelease = 0.0;

    for (int h = pos; h < (int)customers.size(); ++h) {
        int i = customers[h];
        if (i <= 0) continue;
        if (onboard.count(i)) continue;

        // Only type "D" packages can be drone-resupplied
        if (i > 0 && i < (int)data.nodeTypes.size()) {
            if (data.nodeTypes[i] == "P" || data.nodeTypes[i] == "DL") continue;
        }

        double w = pkgWeight(data, i);
        if (w > remaining_cap + 1e-9) continue;

        double ri          = pkgRelease(data, i);
        double testMaxRel  = max(maxRelease, ri);

        // Find earliest-available drone
        int droneId = schedule.findBestDrone();
        if (droneId < 0) break;

        double droneDept = max(schedule.droneAvail[droneId] + data.depotDroneLoadTime, testMaxRel);
        double droneArr  = droneDept + flyOut;

        // Check truck wait constraint
        double truckWait = max(0.0, droneArr - truck_time);
        if (truckWait > data.allowedWait + 1e-9) continue;

        // Check endurance constraint (flyOut + wait-at-rendezvous + flyBack)
        double resStart    = max(droneArr, truck_time);
        double droneWait   = resStart - droneArr;
        double totalFlight = flyOut + droneWait + flyBack;
        if (totalFlight > data.droneEndurance + 1e-9) continue;

        // Feasible candidate
        P_drone.push_back(i);
        maxRelease = testMaxRel;
        remaining_cap -= w;
        if (remaining_cap <= 1e-9) break;
    }

    if (P_drone.empty()) return result;

    // Compute final timing with the chosen package set
    int droneId = schedule.findBestDrone();
    if (droneId < 0) return result;

    double droneDept = max(schedule.droneAvail[droneId] + data.depotDroneLoadTime, maxRelease);
    double droneArr  = droneDept + flyOut;
    double resStart  = max(droneArr, truck_time);
    double resEnd    = resStart + data.resupplyTime;
    double droneRet  = resEnd + flyBack;
    double truckWait = max(0.0, droneArr - truck_time);

    result.feasible          = true;
    result.packages          = P_drone;
    result.drone_id          = droneId;
    result.drone_arrive_time = droneArr;
    result.drone_return_time = droneRet;
    result.cost              = truckWait + data.resupplyTime;

    return result;
}

// ============================================================
// Algorithm 13 (ga_encoding_init_solution.md §13)
// FIND_DEPOT_RETURN cost
//
// Estimate total time penalty if the truck returns to depot
// to pick up package j, then drives to customer j.
// ============================================================

static double findDepotReturnCost(
    int pkg_j,
    int current_node,
    double current_time,
    const PDPData& data)
{
    double toDepot   = truckTravelMinutes(data, current_node, data.depotIndex);
    double arrDepot  = current_time + toDepot;
    double rj        = pkgRelease(data, pkg_j);
    double waitTime  = max(0.0, rj - arrDepot);
    double toJ       = truckTravelMinutes(data, data.depotIndex, pkg_j);

    return toDepot + waitTime + data.depotReceiveTime + toJ;
}

// ============================================================
// Algorithm 14 (ga_encoding_init_solution.md §14)
// SELECT_RESUPPLY_OPTION
//
// Probabilistic selection between drone resupply and depot
// return, providing diversity in the initial population.
//   - If drone is cheaper: 70% drone / 30% depot
//   - If depot is cheaper: 30% drone / 70% depot
// ============================================================

static bool selectDroneOverDepot(
    const DroneResupplyOption& optDrone,
    double costReturn,
    mt19937& gen)
{
    if (!optDrone.feasible) return false;

    uniform_real_distribution<double> prob(0.0, 1.0);

    if (optDrone.cost < costReturn)
        return prob(gen) < 0.7;   // Drone is cheaper → prefer drone
    else
        return prob(gen) < 0.3;   // Depot is cheaper → still try drone 30%
}

// ============================================================
// Main route construction with drone resupply
// Implements Algorithm INIT_TRUCK_ROUTE (Section 10)
//
// Replaces the former buildDepotLoadsForRoute which only
// supported depot returns.  Now the algorithm considers
// both drone resupply and depot return at every point
// where a package is missing, and also proactively inserts
// drone resupply after serving each customer (Section 15).
// ============================================================

static Route buildRouteWithResupply(
    const vector<int>& customers,
    int truckId,
    const PDPData& data,
    DroneScheduleInit& droneSchedule,
    mt19937& gen)
{
    (void)truckId;  // Available for future truck-specific logic

    Route enc;
    int    cur  = data.depotIndex;
    double t    = 0.0;
    unordered_set<int> onboard;
    double load = 0.0;

    // ---- Step 1: Initial depot load  ----
    vector<int> P0 = selectDepotLoad(customers, 0, t, onboard, data);
    enc.push_back(RouteStop(0, P0));
    for (int p : P0) {
        onboard.insert(p);
        load += pkgWeight(data, p);
    }
    if (!P0.empty()) t += data.depotReceiveTime;

    uniform_real_distribution<double> prob(0.0, 1.0);

    // ---- Step 2: Process each customer ----
    for (int pos = 0; pos < (int)customers.size(); ++pos) {
        int j = customers[pos];

        // --- If j needs its package and it's not onboard ---
        if (customerNeedsPackage(data, j) && !onboard.count(j)) {

            // Option A: Drone resupply at current node
            DroneResupplyOption optD = findDroneResupplySet(
                customers, pos, cur, t, onboard, data, droneSchedule);

            // The drone must actually include package j to resolve
            // the missing-package issue; if it only carries future
            // packages, it doesn't help here.
            if (optD.feasible) {
                bool includesJ = false;
                for (int p : optD.packages)
                    if (p == j) { includesJ = true; break; }
                if (!includesJ) optD.feasible = false;
            }

            // Option B: Depot return
            double costRet = findDepotReturnCost(j, cur, t, data);

            // Select (Algorithm 14)
            bool canAttachDroneHere = (!enc.empty() && enc.back().node == cur &&
                                       enc.back().packages.empty());

            if (canAttachDroneHere && selectDroneOverDepot(optD, costRet, gen)) {

                // ========= DRONE RESUPPLY =========
                // Attach packages to the bracket of current node
                // (which is the last stop in enc)
                for (int p : optD.packages)
                    enc.back().packages.push_back(p);

                for (int p : optD.packages) {
                    onboard.insert(p);
                    load += pkgWeight(data, p);
                }

                droneSchedule.update(optD.drone_id, optD.drone_return_time);

                double resStart = max(optD.drone_arrive_time, t);
                t = resStart + data.resupplyTime;

            } else {

                // ========= DEPOT RETURN =========
                t   += truckTravelMinutes(data, cur, data.depotIndex);
                cur  = data.depotIndex;

                // Load packages greedily at depot
                vector<int> P = selectDepotLoad(customers, pos, t, onboard, data);

                // Ensure j is loaded (wait for release if needed)
                bool jInP = false;
                for (int p : P) if (p == j) { jInP = true; break; }
                if (!jInP) {
                    double rj = pkgRelease(data, j);
                    if (rj > t + 1e-9) t = rj;
                    P.push_back(j);
                }

                enc.push_back(RouteStop(0, P));
                for (int p : P) {
                    if (!onboard.count(p)) {
                        onboard.insert(p);
                        load += pkgWeight(data, p);
                    }
                }
                if (!P.empty()) t += data.depotReceiveTime;
            }
        }

        // ---- Travel to customer j ----
        t   += truckTravelMinutes(data, cur, j);
        cur  = j;

        // ---- Append j to encoded route ----
        enc.push_back(RouteStop(j));

        // ---- Serve j: update onboard tracking ----
        if (j > 0 && j < (int)data.nodeTypes.size()) {
            const string& tp = data.nodeTypes[j];
            if (tp == "P") {
                // Pickup: truck picks up package at customer location
                onboard.insert(j);
                load += pkgWeight(data, j);
            } else if (tp == "DL") {
                // Delivery Location: remove paired pickup package
                for (int p = 1; p < data.numNodes; ++p) {
                    if (data.pairIds[p] == data.pairIds[j]
                        && data.nodeTypes[p] == "P") {
                        onboard.erase(p);
                        load = max(0.0, load - pkgWeight(data, p));
                        break;
                    }
                }
            } else {
                // Type "D": standard delivery, remove from truck
                onboard.erase(j);
                load = max(0.0, load - pkgWeight(data, j));
            }
        } else {
            onboard.erase(j);
            load = max(0.0, load - pkgWeight(data, j));
        }

        t += data.truckServiceTime;

        // ---- Optional proactive drone resupply (Algorithm 15) ----
        // After serving j, proactively call drone for future packages
        // at node j.  Done with probability p_resupply ≈ 0.35 to
        // balance solution quality and population diversity.
        if (pos + 1 < (int)customers.size() && prob(gen) < 0.35) {
            DroneResupplyOption optF = findDroneResupplySet(
                customers, pos + 1, cur, t, onboard, data, droneSchedule);

            if (optF.feasible && !optF.packages.empty() && enc.back().packages.empty()) {
                // Add future packages to bracket of node j
                for (int p : optF.packages)
                    enc.back().packages.push_back(p);

                for (int p : optF.packages) {
                    onboard.insert(p);
                    load += pkgWeight(data, p);
                }

                droneSchedule.update(optF.drone_id, optF.drone_return_time);

                double resStart = max(optF.drone_arrive_time, t);
                t = resStart + data.resupplyTime;
            }
        }
    }

    // ---- End: truck returns to depot ----
    enc.push_back(RouteStop(0));

    return enc;
}

static void repairC2CustomerRoutes(vector<vector<int>>& routes, const PDPData& data) {
    auto findNode = [&](int node) {
        for (int r = 0; r < (int)routes.size(); ++r) {
            for (int p = 0; p < (int)routes[r].size(); ++p) {
                if (routes[r][p] == node) return pair<int,int>{r, p};
            }
        }
        return pair<int,int>{-1, -1};
    };

    for (int pickup = 1; pickup < data.numNodes; ++pickup) {
        if (pickup >= (int)data.nodeTypes.size() || data.nodeTypes[pickup] != "P") {
            continue;
        }

        int delivery = -1;
        for (int node = 1; node < data.numNodes; ++node) {
            if (node != pickup &&
                node < (int)data.nodeTypes.size() &&
                data.nodeTypes[node] == "DL" &&
                data.pairIds[node] == data.pairIds[pickup]) {
                delivery = node;
                break;
            }
        }
        if (delivery < 0) continue;

        auto [pRoute, pPos] = findNode(pickup);
        auto [dRoute, dPos] = findNode(delivery);
        if (pRoute < 0 || dRoute < 0) continue;

        routes[dRoute].erase(routes[dRoute].begin() + dPos);

        auto [newPRoute, newPPos] = findNode(pickup);
        if (newPRoute < 0) continue;
        routes[newPRoute].insert(routes[newPRoute].begin() + newPPos + 1, delivery);
    }
}

// ============================================================
// Build one individual (complete solution encoding)
// ============================================================

static SolutionEncoding buildIndividual(const PDPData& data, mt19937& gen) {
    vector<int> customers = listCustomers(data);
    shuffle(customers.begin(), customers.end(), gen);

    const int nTrucks = max(1, data.numTrucks);
    SolutionEncoding sol;
    sol.assign(nTrucks, Route{});

    const int N    = (int)customers.size();
    const int base = N / nTrucks;
    const int rem  = N % nTrucks;

    vector<vector<int>> customerRoutes(nTrucks);
    int offset = 0;
    for (int k = 0; k < nTrucks; ++k) {
        const int len = base + (k < rem ? 1 : 0);

        customerRoutes[k].assign(customers.begin() + offset,
                                 customers.begin() + offset + len);
        offset += len;
    }
    repairC2CustomerRoutes(customerRoutes, data);

    // Shared drone schedule: drones are shared across all trucks
    DroneScheduleInit droneSchedule(data.numDrones, data.depotDroneLoadTime);
    for (int k = 0; k < nTrucks; ++k) {
        sol[k] = buildRouteWithResupply(customerRoutes[k], k, data, droneSchedule, gen);
    }

    return sol;
}

// ============================================================
// Public API: Build initial population
// ============================================================

std::vector<SolutionEncoding> initStructuredPopulationPDP(int populationSize,
                                                         const PDPData& data,
                                                         int runNumber) {
    std::vector<SolutionEncoding> pop;
    pop.reserve((size_t)max(0, populationSize));

    std::seed_seq seed{runNumber, data.numNodes, data.numTrucks, data.numDrones};
    std::mt19937 gen(seed);

    for (int i = 0; i < populationSize; ++i) {
        pop.push_back(buildIndividual(data, gen));
    }

    return pop;
}
