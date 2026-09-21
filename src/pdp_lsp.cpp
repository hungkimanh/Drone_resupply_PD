#include "pdp_lsp.h"
#include "pdp_fitness.h"

#include "ortools/sat/cp_model.h"
#include "ortools/sat/cp_model_solver.h"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <unordered_map>
#include <vector>

using namespace operations_research::sat;

// ============================================================
// Internal helpers
// ============================================================
namespace {

static constexpr int64_t SCALE = 100;  // 0.01 min precision (was 10 = 0.1 min)
static int64_t sc(double minutes) {
    return static_cast<int64_t>(std::round(minutes * SCALE));
}
static bool isC1(const PDPData& d, int n) {
    return n > 0 && n < d.numNodes && d.nodeTypes[n] == "D";
}
static bool isP(const PDPData& d, int n) {
    return n > 0 && n < d.numNodes && d.nodeTypes[n] == "P";
}
static bool isDL(const PDPData& d, int n) {
    return n > 0 && n < d.numNodes && d.nodeTypes[n] == "DL";
}
static int findPickup(const PDPData& d, int dl) {
    for (int p = 1; p < d.numNodes; ++p)
        if (d.nodeTypes[p] == "P" && d.pairIds[p] == d.pairIds[dl]) return p;
    return -1;
}

} // namespace

// ============================================================
// greedyLoading
// ============================================================
// Load ALL C1 packages at initial depot (always feasible since trucks are uncapacitated,
// per paper §3.1). Truck waits at depot until max(release_j) + depotReceiveTime.
// CP-SAT will then reassign some packages to drone resupply or depot return to reduce C_max.

SolutionEncoding greedyLoading(const PDPData& data, const VRPSolution& vrp) {
    int numTrucks = static_cast<int>(vrp.size());
    SolutionEncoding enc(numTrucks);

    for (int v = 0; v < numTrucks; ++v) {
        const auto& route = vrp[v];
        Route& r = enc[v];

        std::vector<int> depot_pkgs;
        for (int node : route)
            if (isC1(data, node)) depot_pkgs.push_back(node);

        r.push_back(RouteStop(0, depot_pkgs));
        for (int node : route) r.push_back(RouteStop(node));
        r.push_back(RouteStop(0));
    }
    return enc;
}

// ============================================================
// solveLSP — CP-SAT model
//
// Three loading mechanisms (paper §4.2):
//   Slot 0          : initial depot load   (γ_j in paper)
//   RS slot (v,k)   : drone resupply at position k on truck v   (y^drone in paper)
//   DR slot (v,k)   : intermediate depot return before position k (k≥1) (ỹ in paper)
//
// Key: trucks are uncapacitated (paper §3.1). Only release dates constrain
// when packages can be loaded.
// ============================================================

LSPResult solveLSP(const PDPData& data,
                   const VRPSolution& vrp,
                   const SolutionEncoding& warm_start,
                   double cutoff_sec,
                   double ub_cutoff) {
    LSPResult result;

    int nT = static_cast<int>(vrp.size());
    const int64_t INF_T     = sc(1e5);
    const int64_t DRONE_CAP = static_cast<int64_t>(data.getDroneCapacity());

    // ---- Collect C1 packages ----
    struct PkgInfo { int node, truck, pos, demand; int64_t release; };
    std::vector<PkgInfo> pkgs;
    for (int v = 0; v < nT; ++v)
        for (int k = 0; k < (int)vrp[v].size(); ++k) {
            int n = vrp[v][k];
            if (isC1(data, n))
                pkgs.push_back({n, v, k, data.demands[n], sc(data.readyTimes[n])});
        }
    int nP = static_cast<int>(pkgs.size());

    // ---- Slot numbering ----
    // RS slots : gslot_rs(v,k) = rs_base[v] + k + 1          k=0..m_v-1
    // DR slots : gslot_dr(v,k) = total_rs + dr_base[v]+(k-1)+1  k=1..m_v-1
    std::vector<int> rs_base(nT, 0);
    int total_rs = 0;
    for (int v = 0; v < nT; ++v) { rs_base[v] = total_rs; total_rs += (int)vrp[v].size(); }

    std::vector<int> dr_base(nT, 0);
    int total_dr = 0;
    for (int v = 0; v < nT; ++v) {
        dr_base[v] = total_dr;
        total_dr += std::max(0, (int)vrp[v].size() - 1);
    }

    auto gslot_rs = [&](int v, int k) -> int { return rs_base[v] + k + 1; };
    auto gslot_dr = [&](int v, int k) -> int { return total_rs + dr_base[v] + (k-1) + 1; };
    // Inverse: given flat slot s, determine type/truck/pos
    auto is_rs = [&](int s) { return s >= 1 && s <= total_rs; };
    auto is_dr = [&](int s) { return s > total_rs && s <= total_rs + total_dr; };

    // ---- Warm start hint for assign ----
    std::vector<int64_t> warm_assign(nP, 0);
    if (!warm_start.empty()) {
        for (int v = 0; v < nT && v < (int)warm_start.size(); ++v) {
            int cust_pos = 0;
            for (const auto& st : warm_start[v]) {
                if (st.node == 0) continue; // depot stop
                for (int pkg : st.packages) {
                    for (int pi = 0; pi < nP; ++pi) {
                        if (pkgs[pi].node == pkg && pkgs[pi].truck == v) {
                            warm_assign[pi] = gslot_rs(v, cust_pos);
                            break;
                        }
                    }
                }
                ++cust_pos;
            }
        }
    }

    // ---- Build CP-SAT model ----
    CpModelBuilder cp;

    // assign[i]: 0=depot, RS slot, or DR slot
    std::vector<IntVar> assign(nP);
    for (int i = 0; i < nP; ++i) {
        int v  = pkgs[i].truck;
        int ki = pkgs[i].pos;
        std::vector<int64_t> vals = {0};
        for (int j = 0; j <= ki; ++j) vals.push_back(gslot_rs(v, j));  // RS
        for (int j = 1; j <= ki; ++j) vals.push_back(gslot_dr(v, j));  // DR (k>=1)
        assign[i] = cp.NewIntVar(operations_research::Domain::FromValues(vals));
        cp.AddHint(assign[i], warm_assign[i]);
    }

    // ---- Truck timing variables ----
    std::vector<IntVar> T0(nT);
    std::vector<std::vector<IntVar>> Tv(nT), TA(nT);
    for (int v = 0; v < nT; ++v) {
        T0[v] = cp.NewIntVar({0, INF_T});
        int m  = (int)vrp[v].size();
        Tv[v].resize(m); TA[v].resize(m);
        for (int k = 0; k < m; ++k) {
            Tv[v][k] = cp.NewIntVar({0, INF_T});
            TA[v][k] = cp.NewIntVar({0, INF_T});
        }
    }

    // ---- depot_return[v][k] BoolVar (k=1..m-1) ----
    // 1 = truck v returns to depot before serving customer at position k
    std::vector<std::vector<BoolVar>> depot_return(nT);
    for (int v = 0; v < nT; ++v) {
        int m = (int)vrp[v].size();
        depot_return[v].resize(m);
        for (int k = 1; k < m; ++k)
            depot_return[v][k] = cp.NewBoolVar();
    }

    // ---- Drone (RS) slot variables ----
    struct SlotVars { BoolVar active; IntVar dr_dep, dr_arr, dr_ret; };
    std::vector<SlotVars> svars(total_rs + 1); // indexed by gslot_rs
    for (int v = 0; v < nT; ++v) {
        int m = (int)vrp[v].size();
        for (int k = 0; k < m; ++k) {
            int s         = gslot_rs(v, k);
            int node      = vrp[v][k];
            int64_t fly_out  = sc(data.droneDistMatrix[0][node]);
            int64_t fly_back = sc(data.droneDistMatrix[node][0]);
            int64_t load_t   = sc(data.depotDroneLoadTime);
            int64_t resup_t  = sc(data.resupplyTime);
            int64_t endur    = sc(data.droneEndurance);

            svars[s].active  = cp.NewBoolVar();
            svars[s].dr_dep  = cp.NewIntVar({0, INF_T});
            svars[s].dr_arr  = cp.NewIntVar({0, INF_T});
            svars[s].dr_ret  = cp.NewIntVar({0, INF_T});

            cp.AddEquality(svars[s].dr_arr, svars[s].dr_dep + fly_out);
            cp.AddEquality(svars[s].dr_ret, svars[s].dr_arr + resup_t + fly_back);

            if (fly_out + resup_t + fly_back > endur)
                cp.AddEquality(svars[s].active, 0);
        }
    }

    // ---- Drone serialization: at most numDrones trips simultaneously (cumulative) ----
    // Each active RS slot occupies 1 drone from dr_dep to dr_ret.
    // Using optional fixed-size interval vars + cumulative constraint.
    {
        std::vector<IntervalVar> drone_ivars;
        for (int v = 0; v < nT; ++v) {
            int m = (int)vrp[v].size();
            for (int k = 0; k < m; ++k) {
                int s         = gslot_rs(v, k);
                int node      = vrp[v][k];
                int64_t dur   = sc(data.droneDistMatrix[0][node])
                              + sc(data.resupplyTime)
                              + sc(data.droneDistMatrix[node][0])
                              + sc(data.depotDroneLoadTime);
                // Optional interval: [dr_dep, dr_dep+dur]  present iff active
                IntervalVar iv = cp.NewOptionalFixedSizeIntervalVar(
                    svars[s].dr_dep, dur, svars[s].active);
                drone_ivars.push_back(iv);
            }
        }
        if (!drone_ivars.empty()) {
            // Demand = 1 per trip; capacity = numDrones
            auto cumul = cp.AddCumulative(data.numDrones);
            for (auto& iv : drone_ivars) cumul.AddDemand(iv, 1);
        }
    }

    int64_t svc        = sc(data.truckServiceTime);
    int64_t allow_wait = sc(data.allowedWait);
    int64_t depot_recv = sc(data.depotReceiveTime);

    // ============================================================
    // CONSTRAINTS
    // ============================================================

    // ---- 1. Initial depot departure (T0) ----
    // Paper constraint (31): τ_0^v >= r_j * γ_j^v — NO σ for first departure [paper §3.2]
    // σ (depot_recv) only applies to INTERMEDIATE returns (handled in DR slot constraints).
    for (int v = 0; v < nT; ++v) {
        if (vrp[v].empty()) continue;
        for (int i = 0; i < nP; ++i) {
            if (pkgs[i].truck != v) continue;
            BoolVar at_depot = cp.NewBoolVar();
            cp.AddEquality(assign[i], 0).OnlyEnforceIf(at_depot);
            cp.AddNotEqual(assign[i], 0).OnlyEnforceIf(at_depot.Not());
            // T0 >= release_j only (no depot_recv for first departure)
            cp.AddGreaterOrEqual(T0[v], pkgs[i].release).OnlyEnforceIf(at_depot);
        }
    }

    // Precompute: for each (v,k), find the package index for the customer AT position k
    // Used to distinguish Case A (customer's order on truck) vs Case B (order from drone)
    std::vector<std::vector<int>> self_pkg(nT);
    for (int v = 0; v < nT; ++v)
        self_pkg[v].assign(vrp[v].size(), -1);
    for (int i = 0; i < nP; ++i)
        self_pkg[pkgs[i].truck][pkgs[i].pos] = i;

    // ---- 2. Timing chain ----
    for (int v = 0; v < nT; ++v) {
        int m = (int)vrp[v].size();
        if (m == 0) continue;

        // TA[v][0] = T0[v] + travel(depot → c_0)
        int64_t t_first = sc(data.truckDistMatrix[0][vrp[v][0]]);
        cp.AddEquality(TA[v][0], T0[v] + t_first);

        // TA is the actual arrival time. Returning to the depot changes both
        // travel time and the rendezvous synchronization time.
        for (int k = 1; k < m; ++k) {
            int64_t direct = sc(data.truckDistMatrix[vrp[v][k-1]][vrp[v][k]]);
            int64_t via_depot = sc(data.truckDistMatrix[vrp[v][k-1]][0])
                              + depot_recv
                              + sc(data.truckDistMatrix[0][vrp[v][k]]);
            cp.AddEquality(TA[v][k], Tv[v][k-1] + direct)
              .OnlyEnforceIf(depot_return[v][k].Not());
            cp.AddEquality(TA[v][k], Tv[v][k-1] + via_depot)
              .OnlyEnforceIf(depot_return[v][k]);
        }

        for (int k = 0; k < m; ++k) {
            int s_rs = gslot_rs(v, k);

            // Base departure: Tv[v][k] >= TA[v][k] + svc
            cp.AddGreaterOrEqual(Tv[v][k], TA[v][k] + svc);

            // RS drone constraint [paper constraint (30)]:
            // Case A (customer k's order on truck, drone brings lookahead packages):
            //   Tv >= dr_arr + δ_m  (svc already counted via TA+svc above)
            // Case B (customer k's order FROM drone):
            //   Tv >= dr_arr + δ_m + svc  (serve AFTER receiving drone)
            int64_t resup_t = sc(data.resupplyTime);
            cp.AddGreaterOrEqual(Tv[v][k], svars[s_rs].dr_arr + resup_t)
              .OnlyEnforceIf(svars[s_rs].active);

            // Add svc only when current customer's own package is at this RS slot (Case B)
            int i_self = self_pkg[v][k];
            if (i_self >= 0) {
                BoolVar own_at_rs = cp.NewBoolVar();
                cp.AddEquality(assign[i_self], (int64_t)gslot_rs(v, k)).OnlyEnforceIf(own_at_rs);
                cp.AddNotEqual(assign[i_self], (int64_t)gslot_rs(v, k)).OnlyEnforceIf(own_at_rs.Not());
                cp.AddGreaterOrEqual(Tv[v][k], svars[s_rs].dr_arr + resup_t + svc)
                  .OnlyEnforceIf({svars[s_rs].active, own_at_rs});
            }

            // Truck can't wait more than allow_wait for drone
            cp.AddLessOrEqual(svars[s_rs].dr_arr, TA[v][k] + allow_wait)
              .OnlyEnforceIf(svars[s_rs].active);

        }
    }

    // ---- 3. Link RS slots: active flag, release date, drone capacity ----
    for (int v = 0; v < nT; ++v) {
        int m = (int)vrp[v].size();
        for (int k = 0; k < m; ++k) {
            int s_rs = gslot_rs(v, k);

            std::vector<BoolVar> at_rs;
            for (int i = 0; i < nP; ++i) {
                if (pkgs[i].truck != v || pkgs[i].pos < k) continue;
                // gslot_rs(v,k) is in domain of assign[i] iff k <= ki (guaranteed above)

                BoolVar b = cp.NewBoolVar();
                cp.AddEquality(assign[i], s_rs).OnlyEnforceIf(b);
                cp.AddNotEqual(assign[i], s_rs).OnlyEnforceIf(b.Not());
                at_rs.push_back(b);

                cp.AddImplication(b, svars[s_rs].active);
                cp.AddGreaterOrEqual(svars[s_rs].dr_dep,
                    pkgs[i].release).OnlyEnforceIf(b);
            }

            if (at_rs.empty()) {
                cp.AddEquality(svars[s_rs].active, 0);
            }

            // Drone capacity constraint at this slot
            LinearExpr wt_sum;
            for (int i = 0; i < nP; ++i) {
                if (pkgs[i].truck != v || pkgs[i].pos < k) continue;
                BoolVar b2 = cp.NewBoolVar();
                cp.AddEquality(assign[i], s_rs).OnlyEnforceIf(b2);
                cp.AddNotEqual(assign[i], s_rs).OnlyEnforceIf(b2.Not());
                wt_sum += LinearExpr::Term(b2, pkgs[i].demand);
            }
            cp.AddLessOrEqual(wt_sum, DRONE_CAP);
        }
    }

    // ---- 4. Link DR slots: depot_return flag + release date constraints ----
    // Paper constraint (61): Tv[v][k] >= r_j + travel(depot→c_k) + svc  [if DR pkg j at k]
    // Paper constraint (55): Tv[v][k] >= Tv[v][k-1] + via_depot + svc   [if depot_return[v][k]]
    //   (second part already added in timing chain above)
    for (int v = 0; v < nT; ++v) {
        int m = (int)vrp[v].size();
        for (int k = 1; k < m; ++k) {
            int s_dr        = gslot_dr(v, k);
            int64_t t_d_ck  = sc(data.truckDistMatrix[0][vrp[v][k]]);

            for (int i = 0; i < nP; ++i) {
                if (pkgs[i].truck != v || pkgs[i].pos < k) continue;
                // gslot_dr(v,k) in domain iff k in [1..ki] — guaranteed since pos>=k and k>=1

                BoolVar b = cp.NewBoolVar();
                cp.AddEquality(assign[i], s_dr).OnlyEnforceIf(b);
                cp.AddNotEqual(assign[i], s_dr).OnlyEnforceIf(b.Not());

                // Release constraint (paper eqs. 61-62):
                // Truck arrives at c_k from depot, must have collected pkg by then:
                // Tv[v][k] >= r_j + depot_recv + travel(depot,c_k) + svc
                cp.AddGreaterOrEqual(Tv[v][k],
                    pkgs[i].release + t_d_ck + svc).OnlyEnforceIf(b);

                // If this package uses DR(v,k), truck must return to depot before k
                cp.AddImplication(b, depot_return[v][k]);
            }
        }
    }

    // ---- 5. C1 packages must be delivered within their source trip ----
    // A depot return is a trip boundary. Initial-depot and drone-supplied C1
    // packages cannot cross one; a package loaded by DR(v,j) may cross that
    // boundary j, but no later boundary before its delivery.
    for (int i = 0; i < nP; ++i) {
        const int v = pkgs[i].truck;
        const int delivery_pos = pkgs[i].pos;

        auto forbid_later_returns = [&](int source, int first_boundary) {
            BoolVar selected = cp.NewBoolVar();
            cp.AddEquality(assign[i], source).OnlyEnforceIf(selected);
            cp.AddNotEqual(assign[i], source).OnlyEnforceIf(selected.Not());
            for (int k = first_boundary; k <= delivery_pos; ++k)
                cp.AddImplication(selected, depot_return[v][k].Not());
        };

        forbid_later_returns(0, 1); // Initial depot load
        for (int j = 0; j <= delivery_pos; ++j)
            forbid_later_returns(gslot_rs(v, j), j + 1);
        for (int j = 1; j <= delivery_pos; ++j)
            forbid_later_returns(gslot_dr(v, j), j + 1);
    }

    // ---- 6. Truck capacity before and after every customer ----
    // C1 packages contribute after their selected source is reached and until
    // delivery. C2 pickup loads persist across intermediate depot returns.
    const int64_t truck_cap = static_cast<int64_t>(data.truckCapacity);
    for (int v = 0; v < nT; ++v) {
        const int m = static_cast<int>(vrp[v].size());
        if (m == 0) continue;

        std::unordered_map<int, int> route_pos;
        for (int k = 0; k < m; ++k) route_pos[vrp[v][k]] = k;

        for (int k = 0; k < m; ++k) {
            LinearExpr load_before;
            LinearExpr load_after;

            for (int p = 1; p < data.numNodes; ++p) {
                if (!isP(data, p)) continue;
                auto p_it = route_pos.find(p);
                if (p_it == route_pos.end()) continue;

                int dl = -1;
                for (int n = 1; n < data.numNodes; ++n) {
                    if (isDL(data, n) && data.pairIds[n] == data.pairIds[p]) {
                        dl = n;
                        break;
                    }
                }
                auto d_it = route_pos.find(dl);
                if (dl < 0 || d_it == route_pos.end()) continue;

                const int kp = p_it->second;
                const int kd = d_it->second;
                const int64_t q = static_cast<int64_t>(data.demands[p]);
                if (kp < k && k <= kd) load_before += q;
                if (kp <= k && k < kd) load_after += q;
            }

            for (int i = 0; i < nP; ++i) {
                if (pkgs[i].truck != v || pkgs[i].pos < k) continue;

                const bool in_before = pkgs[i].pos >= k;
                const bool in_after  = pkgs[i].pos > k;
                auto add_source = [&](int source) {
                    BoolVar selected = cp.NewBoolVar();
                    cp.AddEquality(assign[i], source).OnlyEnforceIf(selected);
                    cp.AddNotEqual(assign[i], source).OnlyEnforceIf(selected.Not());
                    if (in_before)
                        load_before += LinearExpr::Term(selected, pkgs[i].demand);
                    if (in_after)
                        load_after += LinearExpr::Term(selected, pkgs[i].demand);
                };

                add_source(0);
                for (int j = 0; j <= k; ++j) add_source(gslot_rs(v, j));
                for (int j = 1; j <= k; ++j) add_source(gslot_dr(v, j));
            }

            cp.AddLessOrEqual(load_before, truck_cap);
            cp.AddLessOrEqual(load_after, truck_cap);
        }
    }

    // ---- 7. CMAX ----
    IntVar cmax = cp.NewIntVar({0, INF_T});

    for (int v = 0; v < nT; ++v) {
        int m = (int)vrp[v].size();
        if (m == 0) continue;
        int64_t ret_t = sc(data.truckDistMatrix[vrp[v][m-1]][0]);
        cp.AddGreaterOrEqual(cmax, Tv[v][m-1] + ret_t);
    }


    if (ub_cutoff < 1e17)
        cp.AddLessOrEqual(cmax, sc(ub_cutoff) - 1);

    // Warm start hint for cmax
    {
        PDPSolution ws = decode_solution(warm_start, data, false);
        if (ws.isFeasible)
            cp.AddHint(cmax, sc(ws.totalCost));
    }

    // Minimize makespan first, then the number of truck trips.
    LinearExpr objective = LinearExpr::Term(cmax, total_dr + 1);
    for (int v = 0; v < nT; ++v)
        for (int k = 1; k < (int)vrp[v].size(); ++k)
            objective += LinearExpr::Term(depot_return[v][k], 1);
    cp.Minimize(objective);

    // ---- Solve ----
    SatParameters params;
    params.set_max_time_in_seconds(cutoff_sec);
    params.set_num_workers(4);
    params.set_log_search_progress(false);

    CpSolverResponse response = SolveWithParameters(cp.Build(), params);

    bool cp_solved = (response.status() == CpSolverStatus::OPTIMAL ||
                      response.status() == CpSolverStatus::FEASIBLE);

    if (!cp_solved) {
        PDPSolution gs  = decode_solution(warm_start, data, false);
        result.solved   = true;
        result.feasible = gs.isFeasible;
        result.encoding = warm_start;
        result.objective = gs.isFeasible ? gs.totalCost : std::numeric_limits<double>::infinity();
        return result;
    }

    result.solved   = true;
    result.feasible = true;
    result.objective = static_cast<double>(SolutionIntegerValue(response, cmax)) / SCALE;

    // ---- Reconstruct encoding ----
    // Three categories per truck: depot_pkgs, rs_at[k], dr_before[k]
    struct TruckBuild {
        std::vector<int> depot_pkgs;
        std::unordered_map<int, std::vector<int>> rs_at;      // pos -> drone resupply pkgs
        std::unordered_map<int, std::vector<int>> dr_before;  // pos -> depot-return pkgs
    };
    std::vector<TruckBuild> tb(nT);

    for (int i = 0; i < nP; ++i) {
        int64_t s = SolutionIntegerValue(response, assign[i]);
        int v     = pkgs[i].truck;

        if (s == 0) {
            tb[v].depot_pkgs.push_back(pkgs[i].node);
        } else if (is_rs(static_cast<int>(s))) {
            // RS slot: s = rs_base[v] + k + 1  →  k = s - rs_base[v] - 1
            int k = static_cast<int>(s) - rs_base[v] - 1;
            tb[v].rs_at[k].push_back(pkgs[i].node);
        } else { // DR slot
            // DR slot: s = total_rs + dr_base[v] + (k-1) + 1  →  k = s - total_rs - dr_base[v]
            int k = static_cast<int>(s) - total_rs - dr_base[v];
            tb[v].dr_before[k].push_back(pkgs[i].node);
        }
    }

    SolutionEncoding enc(nT);
    for (int v = 0; v < nT; ++v) {
        Route& r = enc[v];
        r.push_back(RouteStop(0, tb[v].depot_pkgs));

        for (int k = 0; k < (int)vrp[v].size(); ++k) {
            int node = vrp[v][k];

            // Insert every split selected by LSP. A split may carry no newly
            // loaded packages; pickup loads already onboard are preserved.
            if (k >= 1) {
                auto dit = tb[v].dr_before.find(k);
                std::vector<int> depot_pkgs;
                if (dit != tb[v].dr_before.end()) depot_pkgs = dit->second;
                if (SolutionBooleanValue(response, depot_return[v][k]))
                    r.push_back(RouteStop(0, std::move(depot_pkgs)));
            }

            // Customer visit, with possible drone resupply
            auto rit = tb[v].rs_at.find(k);
            if (rit != tb[v].rs_at.end() && !rit->second.empty())
                r.push_back(RouteStop(node, rit->second));
            else
                r.push_back(RouteStop(node));
        }

        r.push_back(RouteStop(0)); // final depot return
    }
    result.encoding = enc;

    // Decode the encoding through the simulator to get the true objective.
    // CP-SAT is a relaxation; the scheduler-verified time is authoritative.
    {
        PDPSolution actual = decode_solution(enc, data, false);
        if (actual.isFeasible) {
            result.feasible = true;
            result.objective = actual.totalCost;
        } else {
            // Fall back to greedy if CP-SAT produced an infeasible schedule
            PDPSolution gs = decode_solution(warm_start, data, false);
            result.encoding  = warm_start;
            result.feasible  = gs.isFeasible;
            result.objective = gs.isFeasible ? gs.totalCost : std::numeric_limits<double>::infinity();
        }
    }

    return result;
}

// ============================================================
// evaluateRoutes
// ============================================================

LSPResult evaluateRoutes(const PDPData& data, const VRPSolution& vrp,
                         double cutoff_sec, double ub_cutoff) {
    SolutionEncoding gs = greedyLoading(data, vrp);
    return solveLSP(data, vrp, gs, cutoff_sec, ub_cutoff);
}
