#ifndef PDP_SEARCH_H
#define PDP_SEARCH_H
#include "pdp_lsp.h"
#include <chrono>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

bool validRouting(const PDPData& data, const VRPSolution& routes);
std::string routingKey(const VRPSolution& routes);

struct SearchOptions {
    bool cache = false, warm = false, adaptive = false, critical = false;
    double lsp_seconds = 3.0;
    double certificate_bound = -1.0; // independently proven bound, stopping only
};

class SearchContext {
public:
    const PDPData& data;
    SearchOptions options;
    std::chrono::steady_clock::time_point start, deadline;
    LSPResult best;
    LSPResult checkpoint_best;
    std::vector<std::pair<double, double>> history;
    int evaluations = 0, cp_calls = 0, cache_hits = 0, fallbacks = 0;
    int warm_reuses = 0, adaptive_refinements = 0, waiting_escapes = 0;
    std::unordered_map<std::string, int> statuses;
    SearchContext(const PDPData& d, SearchOptions o,
                  std::chrono::steady_clock::time_point begin, double seconds);
    double elapsed() const;
    double remaining() const;
    bool done() const;
    LSPResult evaluate(const VRPSolution& routes, const SolutionEncoding* hint = nullptr);
    void consider(const LSPResult& result);
private:
    std::unordered_map<std::string, LSPResult> memo;
    std::unordered_map<std::string, int> visits;
};
#endif
