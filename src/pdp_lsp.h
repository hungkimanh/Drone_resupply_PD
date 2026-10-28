#ifndef PDP_LSP_H
#define PDP_LSP_H

#include "pdp_types.h"
#include <vector>

// ============================================================
// Layer 2: Loading Subproblem (LSP)
// Given fixed truck routes (VRPSolution), decide:
//   - Which C1 packages go in initial depot load
//   - Which go via drone resupply (at which customer node)
// Objective: minimize C_max (makespan)
// ============================================================

// Initial all-at-depot loading hint. It must be decoded/validated before being
// used as a feasible UB, especially when truck capacity is binding.
SolutionEncoding greedyLoading(const PDPData& data, const VRPSolution& vrp);

// CP-SAT grid-model search followed by exact-distance fixed-order scheduling.
//   vrp        : fixed customer routes (no loading info)
//   warm_start : greedy solution used as initial hint (AddHint)
//   cutoff_sec : CP-SAT time limit in seconds (default 5.0)
//   ub_cutoff  : if known best C_max is ub_cutoff, only accept strictly better
// status preserves FEASIBLE/OPTIMAL/UNKNOWN/INFEASIBLE for the grid model;
// objective is always a physically replayed feasible UB, not an exact LSP value.
LSPResult solveLSP(const PDPData& data,
                   const VRPSolution& vrp,
                   const SolutionEncoding& warm_start,
                   double cutoff_sec = 5.0,
                   double ub_cutoff = 1e18);

// Convenience: run greedy + CP-SAT in one call.
LSPResult evaluateRoutes(const PDPData& data,
                         const VRPSolution& vrp,
                         double cutoff_sec = 5.0,
                         double ub_cutoff = 1e18);

#endif
