#include "pdp_tabu_main.h"
#include "pdp_lb.h"
#include "pdp_alns.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <iostream>
#include <limits>
#include <iomanip>
#include <random>
#include <unordered_map>
#include <vector>

using namespace std;

// ============================================================
// Helpers
// ============================================================

VRPSolution extractVRPSolution(const SolutionEncoding& enc) {
    VRPSolution vrp(enc.size());
    for (int v = 0; v < (int)enc.size(); ++v)
        for (const auto& st : enc[v])
            if (st.node > 0) vrp[v].push_back(st.node);
    return vrp;
}

static int totalCustomers(const VRPSolution& vrp) {
    int n = 0;
    for (const auto& r : vrp) n += (int)r.size();
    return n;
}

// Tabu tenure: θ = round(5 × √n) [Taillard 1993 — larger tenure for VRP-D avoids cycling]
static int tabuTenure(int n) {
    if (n <= 1) return 5;
    return (int)round(5.0 * sqrt((double)n));
}

// C2 feasibility: every DL has its P on the same truck, before it
static bool checkC2Feasibility(const PDPData& data, const VRPSolution& vrp) {
    for (int v = 0; v < (int)vrp.size(); ++v) {
        const auto& route = vrp[v];
        for (int k = 0; k < (int)route.size(); ++k) {
            int n = route[k];
            if (n <= 0 || n >= data.numNodes || data.nodeTypes[n] != "DL") continue;
            int p_node = -1;
            for (int m = 1; m < data.numNodes; ++m)
                if (data.pairIds[m] == data.pairIds[n] && data.nodeTypes[m] == "P")
                    { p_node = m; break; }
            if (p_node < 0) continue;
            bool found = false;
            for (int j = 0; j < k; ++j)
                if (route[j] == p_node) { found = true; break; }
            if (!found) return false;
        }
    }
    return true;
}

// ============================================================
// Move types
// ============================================================

enum class MoveType { RELOCATE = 0, SWAP = 1, TWO_OPT = 2 };
static const char* MOVE_NAMES[] = { "RELOC", "SWAP", "2OPT" };

struct Move {
    MoveType type     = MoveType::RELOCATE;
    int v1 = -1, k1 = -1;
    int v2 = -1, k2 = -1;
    int customer  = -1;  // first customer (RELOCATE) / SWAP c1 / 2OPT left endpoint
    int customer2 = -1;  // SWAP c2 / 2OPT right endpoint

    pair<int,int> tabuKey() const {
        switch (type) {
            case MoveType::RELOCATE: return {customer, v1};
            case MoveType::SWAP:     return {min(customer,customer2), max(customer,customer2)};
            case MoveType::TWO_OPT:  return {customer, customer2};
        }
        return {-1,-1};
    }
};

static VRPSolution applyMove(const VRPSolution& vrp, const Move& m) {
    switch (m.type) {
        case MoveType::RELOCATE: {
            VRPSolution nv = vrp;
            nv[m.v1].erase(nv[m.v1].begin() + m.k1);
            int ins = m.k2;
            if (m.v1 == m.v2 && m.k2 > m.k1) ins--;
            ins = max(0, min(ins, (int)nv[m.v2].size()));
            nv[m.v2].insert(nv[m.v2].begin() + ins, m.customer);
            return nv;
        }
        case MoveType::SWAP: {
            VRPSolution nv = vrp;
            swap(nv[m.v1][m.k1], nv[m.v2][m.k2]);
            return nv;
        }
        case MoveType::TWO_OPT: {
            VRPSolution nv = vrp;
            reverse(nv[m.v1].begin() + m.k1, nv[m.v1].begin() + m.k2 + 1);
            return nv;
        }
    }
    return vrp;
}

static bool moveKeepsC2Feasibility(const PDPData& data,
                                    const VRPSolution& vrp, const Move& m) {
    return checkC2Feasibility(data, applyMove(vrp, m));
}

// ============================================================
// Tabu list
// ============================================================

using TabuList = unordered_map<int, int>; // key → expiry iter

static int encodeTabuKey(const Move& m) {
    auto [a, b] = m.tabuKey();
    switch (m.type) {
        case MoveType::RELOCATE: return           a * 10000 + b;
        case MoveType::SWAP:     return 2000000 + a * 1000  + b;
        case MoveType::TWO_OPT:  return 5000000 + a * 1000  + b;
    }
    return 0;
}

// ============================================================
// Adaptive operator weights (roulette wheel + exponential smoothing)
//
// Scores:
//   3 — operator finds a new global best
//   2 — operator improves current solution
//   1 — operator finds any accepted move
//   0 — operator finds no valid move / no improvement
// ============================================================

static constexpr int N_OPS = 3;

struct TSOpWeights {
    array<double, N_OPS> w   = {1.0, 1.0, 1.0};  // weights
    array<int,    N_OPS> use = {0,   0,   0};     // times selected
    array<int,    N_OPS> win = {0,   0,   0};     // times improved
    double rho = 0.1;                              // smoothing factor

    int select(mt19937& rng) const {
        double total = 0;
        for (double wi : w) total += wi;
        uniform_real_distribution<double> dist(0.0, total);
        double r = dist(rng), cum = 0;
        for (int i = 0; i < N_OPS; ++i) {
            cum += w[i];
            if (r <= cum) return i;
        }
        return N_OPS - 1;
    }

    // score: 0..3 — see above
    void update(int op, int score) {
        w[op] = max(0.05, (1.0 - rho) * w[op] + rho * score);
    }

    void record(int op, int score) {
        use[op]++;
        if (score >= 2) win[op]++;
    }

    void print() const {
        cout << "[TSop] weights: ";
        for (int i = 0; i < N_OPS; ++i)
            cout << MOVE_NAMES[i] << "=" << fixed << setprecision(3) << w[i]
                 << "(u=" << use[i] << ",w=" << win[i] << ") ";
        cout << "\n";
    }
};

// ============================================================
// Neighbourhood generators (return true iff any move evaluated)
// ============================================================

// Check if a node is a "DL" (C2 delivery) — don't RELOCATE DL; move its P instead
static bool isDL(const PDPData& d, int n) {
    return n > 0 && n < d.numNodes && d.nodeTypes[n] == "DL";
}

// Evaluate a candidate move against current best; update best_move if better.
// Returns the score contribution: 1 if move is accepted, 0 otherwise.
static bool evalCandidate(const PDPData& data,
                           const VRPSolution& chi_curr,
                           const Move& m,
                           double z_star, double threshold,
                           const TabuList& tabu, int iter,
                           double& best_lb, Move& best_move) {
    if (!moveKeepsC2Feasibility(data, chi_curr, m)) return false;
    VRPSolution chi_prime = applyMove(chi_curr, m);
    double lb = computeVRPLB(data, chi_prime);
    int  tkey    = encodeTabuKey(m);
    bool is_tabu = (tabu.count(tkey) && tabu.at(tkey) > iter);

    bool accept = (lb < z_star) ||                   // aspiration
                  (!is_tabu && lb < threshold);       // normal acceptance
    if (accept && lb < best_lb) {
        best_lb   = lb;
        best_move = m;
        return true;
    }
    return false;
}

// Search RELOCATE neighbourhood for a single source truck
static bool searchRelocate(const PDPData& data, const VRPSolution& chi,
                            int v1, int numTrucks,
                            double z_star, double threshold,
                            const TabuList& tabu, int iter,
                            double& best_lb, Move& best_move) {
    bool found = false;
    const auto& src = chi[v1];
    for (int k1 = 0; k1 < (int)src.size(); ++k1) {
        int cust = src[k1];
        if (isDL(data, cust)) continue;
        for (int v2 = 0; v2 < numTrucks; ++v2)
            for (int k2 = 0; k2 <= (int)chi[v2].size(); ++k2) {
                if (v2 == v1 && (k2 == k1 || k2 == k1+1)) continue;
                Move m;
                m.type = MoveType::RELOCATE;
                m.v1 = v1; m.k1 = k1;
                m.v2 = v2; m.k2 = k2;
                m.customer = cust;
                found |= evalCandidate(data, chi, m, z_star, threshold,
                                       tabu, iter, best_lb, best_move);
            }
    }
    return found;
}

// Search SWAP neighbourhood for a single source truck
// Only swaps where one endpoint is on v1 (avoids double-counting via start_k2 offset)
static bool searchSwap(const PDPData& data, const VRPSolution& chi,
                        int v1, int numTrucks,
                        double z_star, double threshold,
                        const TabuList& tabu, int iter,
                        double& best_lb, Move& best_move) {
    bool found = false;
    const auto& src = chi[v1];
    for (int k1 = 0; k1 < (int)src.size(); ++k1) {
        int c1 = src[k1];
        if (isDL(data, c1)) continue;
        for (int v2 = 0; v2 < numTrucks; ++v2) {
            const auto& dst = chi[v2];
            int start = (v2 == v1) ? k1 + 1 : 0;
            for (int k2 = start; k2 < (int)dst.size(); ++k2) {
                int c2 = dst[k2];
                if (isDL(data, c2)) continue;
                Move m;
                m.type = MoveType::SWAP;
                m.v1 = v1; m.k1 = k1;
                m.v2 = v2; m.k2 = k2;
                m.customer  = c1;
                m.customer2 = c2;
                found |= evalCandidate(data, chi, m, z_star, threshold,
                                       tabu, iter, best_lb, best_move);
            }
        }
    }
    return found;
}

// Search 2-OPT neighbourhood within a single truck
static bool search2Opt(const PDPData& data, const VRPSolution& chi,
                        int v,
                        double z_star, double threshold,
                        const TabuList& tabu, int iter,
                        double& best_lb, Move& best_move) {
    bool found = false;
    const auto& route = chi[v];
    int sz = (int)route.size();
    for (int k1 = 0; k1 < sz - 1; ++k1)
        for (int k2 = k1 + 1; k2 < sz; ++k2) {
            Move m;
            m.type = MoveType::TWO_OPT;
            m.v1 = v; m.k1 = k1;
            m.v2 = v; m.k2 = k2;
            m.customer  = route[k1];
            m.customer2 = route[k2];
            found |= evalCandidate(data, chi, m, z_star, threshold,
                                   tabu, iter, best_lb, best_move);
        }
    return found;
}

// Dispatch to the right search function for a given operator type and truck
static bool searchOp(MoveType op, const PDPData& data, const VRPSolution& chi,
                     int v, int numTrucks,
                     double z_star, double threshold,
                     const TabuList& tabu, int iter,
                     double& best_lb, Move& best_move) {
    switch (op) {
        case MoveType::RELOCATE:
            return searchRelocate(data, chi, v, numTrucks,
                                  z_star, threshold, tabu, iter, best_lb, best_move);
        case MoveType::SWAP:
            return searchSwap(data, chi, v, numTrucks,
                              z_star, threshold, tabu, iter, best_lb, best_move);
        case MoveType::TWO_OPT:
            return search2Opt(data, chi, v,
                              z_star, threshold, tabu, iter, best_lb, best_move);
    }
    return false;
}

// ============================================================
// tabuSearch — main loop with adaptive operator selection
// ============================================================

TSResult tabuSearch(const PDPData& data,
                    const VRPSolution& init_vrp,
                    const TSConfig& cfg) {
    mt19937 rng(cfg.seed);

    VRPSolution chi_star = init_vrp;
    VRPSolution chi_curr = init_vrp;

    LSPResult best_lsp  = evaluateRoutes(data, chi_star, cfg.lsp_cutoff);
    double z_star       = best_lsp.objective;
    double lb_star      = computeVRPLB(data, chi_star);

    if (cfg.verbose)
        cout << "[TS] init z=" << z_star << " LB=" << lb_star << "\n";

    TabuList    tabu;
    ALNSWeights alns_w;
    TSOpWeights ts_ops;              // adaptive TS operator weights

    int n_cust    = totalCustomers(init_vrp);
    int theta_base = tabuTenure(n_cust);
    int theta      = theta_base;
    int numTrucks = (int)init_vrp.size();
    int stagnation = 0;
    int global_no_improve = 0;  // [Sacramento+ 2019] global early-stop counter
    int iter       = 0;

    // Tenure randomization bounds [Taillard 1993]
    int theta_lo = max(3, theta_base - 3);
    int theta_hi = theta_base + 5;
    uniform_int_distribution<int> tenure_dist(theta_lo, theta_hi);

    // Print operator weight summary every N_PRINT iterations
    const int N_PRINT = 100;

    // FIX 3: Time-based termination [paper §5.2]
    auto ts_start = chrono::steady_clock::now();
    auto elapsed_sec = [&]() -> double {
        return chrono::duration<double>(chrono::steady_clock::now() - ts_start).count();
    };

    for (iter = 0; iter < cfg.max_iter; ++iter) {

        // FIX 3: Check wall-clock time limit instead of early-stopping
        if (cfg.time_limit_sec > 0 && elapsed_sec() >= cfg.time_limit_sec) {
            if (cfg.verbose)
                cout << "[TS] Time limit " << fixed << setprecision(1)
                     << cfg.time_limit_sec << "s reached at iter=" << iter << "\n";
            break;
        }

        // FIX 3: Convert "no global improvement" from early-stop → forced restart
        if (cfg.no_improve_limit > 0 && global_no_improve >= cfg.no_improve_limit) {
            if (cfg.verbose)
                cout << "[TS] Restart at iter=" << iter
                     << " (no global improvement for " << global_no_improve << " iters)\n";
            // Force ALNS escape by pushing stagnation over threshold
            stagnation        = cfg.eta_max;
            global_no_improve = 0;
            // (ALNS escape fires at end of this iteration)
        }

        // [Taillard 1993] Randomize tenure each iteration to break cycling
        if (cfg.randomize_tenure)
            theta = tenure_dist(rng);

        // FIX 1: φ grows naturally to 1.0 — no phi_max cap [paper §4.3.2]
        double phi       = min(1.0, (double)stagnation / cfg.eta_max);
        double threshold = computeThreshold(z_star, lb_star, phi);

        // ---- Adaptive operator selection ----
        int      sel_idx  = ts_ops.select(rng);
        MoveType sel_type = static_cast<MoveType>(sel_idx);

        double best_lb   = numeric_limits<double>::infinity();
        Move   best_move;
        best_move.v1 = -1;

        int last_truck = findLastTruckByLB(data, chi_curr);

        // Phase 1: selected operator, last truck (bottleneck focus)
        searchOp(sel_type, data, chi_curr, last_truck, numTrucks,
                 z_star, threshold, tabu, iter, best_lb, best_move);

        // Phase 2: selected operator, all other trucks
        if (best_move.v1 < 0) {
            for (int v = 0; v < numTrucks; ++v) {
                if (v == last_truck) continue;
                searchOp(sel_type, data, chi_curr, v, numTrucks,
                         z_star, threshold, tabu, iter, best_lb, best_move);
            }
        }

        bool found_with_selected = (best_move.v1 >= 0);

        // Phase 3: fallback — try remaining operators if selected found nothing
        if (best_move.v1 < 0) {
            for (int op_idx = 0; op_idx < N_OPS; ++op_idx) {
                if (op_idx == sel_idx) continue;
                MoveType op = static_cast<MoveType>(op_idx);
                for (int v = 0; v < numTrucks; ++v)
                    searchOp(op, data, chi_curr, v, numTrucks,
                             z_star, threshold, tabu, iter, best_lb, best_move);
                if (best_move.v1 >= 0) break;
            }
        }

        // ---- Apply best move ----
        if (best_move.v1 < 0) {
            // No valid move found — penalize selected operator
            ts_ops.update(sel_idx, 0);
            ts_ops.record(sel_idx, 0);
            stagnation++;
        } else {
            chi_curr = applyMove(chi_curr, best_move);

            int tkey = encodeTabuKey(best_move);
            tabu[tkey] = iter + theta;

            // FIX 2: Cutoff ε — CP-SAT only searches for solutions < z*−ε [paper §4.3.2]
            LSPResult lsp = evaluateRoutes(data, chi_curr, cfg.lsp_cutoff,
                                           z_star - cfg.cutoff_eps);

            // Score: 3 = new best, 1 = accepted no improvement, 0 = fallback did the work
            int score;
            if (lsp.feasible && lsp.objective < z_star) {
                z_star   = lsp.objective;
                chi_star = chi_curr;
                best_lsp = lsp;
                lb_star  = computeVRPLB(data, chi_star);
                stagnation = 0;
                global_no_improve = 0;  // reset global counter
                score = found_with_selected ? 3 : 0;

                if (cfg.verbose) {
                    cout << "[TS] iter=" << iter
                         << " NEW BEST z=" << z_star
                         << " LB=" << lb_star
                         << " mv=" << MOVE_NAMES[(int)best_move.type]
                         << " (sel=" << MOVE_NAMES[sel_idx] << ")"
                         << "\n";
                }
            } else {
                stagnation++;
                global_no_improve++;
                score = found_with_selected ? 1 : 0;
            }

            ts_ops.update(sel_idx, score);
            ts_ops.record(sel_idx, score);
        }

        // Print operator stats periodically
        if (cfg.verbose && iter > 0 && iter % N_PRINT == 0)
            ts_ops.print();

        // ---- ALNS escape on stagnation ----
        if (stagnation >= cfg.eta_max) {
            if (cfg.verbose)
                cout << "[TS] ALNS escape at iter=" << iter << "\n";

            // [Ropke & Pisinger 2006] Randomize destroy size each escape
            int d_min = max(2, (int)(0.15 * n_cust));
            int d_max = max(3, (int)(0.35 * n_cust));
            uniform_int_distribution<int> remove_dist(d_min, d_max);
            int n_remove = remove_dist(rng);
            for (int attempt = 0; attempt < cfg.n_alns_escape; ++attempt) {
                int d_op = alns_w.selectDestroy(rng);
                int r_op = alns_w.selectRepair(rng);
                VRPSolution chi_alns = alnsIteration(data, chi_star, alns_w,
                                                      d_op, r_op, n_remove, rng);
                if (!checkC2Feasibility(data, chi_alns)) continue;

                LSPResult lsp_alns = evaluateRoutes(data, chi_alns, cfg.lsp_cutoff,
                                                    z_star - cfg.cutoff_eps);
                int ascore = 0;
                if (lsp_alns.feasible && lsp_alns.objective < z_star) {
                    z_star   = lsp_alns.objective;
                    chi_star = chi_alns;
                    best_lsp = lsp_alns;
                    lb_star  = computeVRPLB(data, chi_star);
                    ascore   = 2;
                    if (cfg.verbose)
                        cout << "[ALNS] NEW BEST z=" << z_star << "\n";
                } else if (lsp_alns.feasible && lsp_alns.objective < best_lsp.objective * 1.05) {
                    ascore = 1;
                }
                alns_w.updateDestroy(d_op, ascore);
                alns_w.updateRepair(r_op, ascore);
            }

            chi_curr   = chi_star;
            tabu.clear();
            stagnation = 0;
        }
    }

    if (cfg.verbose)
        ts_ops.print();

    TSResult res;
    res.best_vrp        = chi_star;
    res.best_lsp        = best_lsp;
    res.best_obj        = z_star;
    res.iterations_done = iter;
    return res;
}
