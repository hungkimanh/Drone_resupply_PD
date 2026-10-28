#ifndef PDP_TABU_MAIN_H
#define PDP_TABU_MAIN_H

#include "pdp_types.h"
#include "pdp_lsp.h"
#include "pdp_search.h"
#include <random>
#include <vector>

// ============================================================
// Layer 1: Tabu Search main loop
// Operates on VRPSolution (customer sequences only).
// Layer 2 (CP-SAT LSP) evaluates each candidate.
// ALNS escape when stagnated for eta_max iterations.
// ============================================================

struct TSConfig {
    int max_iter          = 100000; // safety cap on iterations (termination via time_limit_sec)
    int no_improve_limit  = 500;    // iters without global improvement → force restart [paper §4.4.4]
    int eta_max           = 150;    // iters without improvement → ALNS escape [Sacramento+ 2019]
    int n_alns_escape     = 5;      // ALNS attempts per escape event [Ropke & Pisinger 2006]
    // phi_max removed: φ = stagnation/eta_max grows naturally to 1.0 [paper §4.3.2]
    double cutoff_eps     = 0.5;    // ε: only accept LSP if < z*−ε [paper §4.3.2]
    double time_limit_sec = 150.0;  // total TS wall-clock budget [paper §5.2: 2.5min small, 60min large]
    double lsp_cutoff     = 3.0;    // CP-SAT time limit per LSP call (seconds)
    int seed              = 42;
    bool verbose          = true;
    bool randomize_tenure = true;   // randomize tabu tenure each iter [Taillard 1993]
    SearchContext* context = nullptr;
    bool pair_moves = true;
};

struct TSResult {
    VRPSolution best_vrp;
    LSPResult   best_lsp;  // full encoding + objective for best solution found
    double      best_obj;
    int         iterations_done;
};

VRPSolution waitingDrivenEscape(const PDPData& data, const VRPSolution& routes,
                                 const LSPResult& solution, int count, std::mt19937& rng);

// Run TS-ALNS-LSP solver
// Layer 0 (init) must be done before calling this; pass the starting VRPSolution.
TSResult tabuSearch(const PDPData& data,
                    const VRPSolution& init_vrp,
                    const TSConfig& cfg = TSConfig{});

// Extract VRPSolution from a SolutionEncoding (strips loading info)
// (mirrors extractCustomerRoutes in pdp_ga.cpp)
VRPSolution extractVRPSolution(const SolutionEncoding& enc);

#endif
