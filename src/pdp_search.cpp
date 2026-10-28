#include "pdp_search.h"
#include "pdp_fitness.h"
#include "pdp_lb.h"
#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <unordered_map>

std::string routingKey(const VRPSolution& routes) {
    std::ostringstream out;
    for (const auto& route : routes) {
        for (int n : route) out << n << ',';
        out << ';';
    }
    return out.str();
}

bool validRouting(const PDPData& d, const VRPSolution& routes) {
    if (int(routes.size()) != d.numTrucks) return false;
    std::vector<int> owner(d.numNodes, -1), position(d.numNodes, -1);
    for (int v = 0; v < int(routes.size()); ++v)
        for (int k = 0; k < int(routes[v].size()); ++k) {
            int n = routes[v][k];
            if (!d.isCustomer(n) || owner[n] >= 0) return false;
            owner[n] = v; position[n] = k;
        }
    for (int n = 1; n < d.numNodes; ++n) {
        if (owner[n] < 0) return false;
        if (d.nodeTypes[n] != "DL") continue;
        int pickup = -1;
        for (int p = 1; p < d.numNodes; ++p)
            if (d.nodeTypes[p] == "P" && d.pairIds[p] == d.pairIds[n]) pickup = p;
        if (pickup < 0 || owner[pickup] != owner[n] || position[pickup] >= position[n]) return false;
    }
    return true;
}

SearchContext::SearchContext(const PDPData& d, SearchOptions o,
                            std::chrono::steady_clock::time_point begin, double seconds)
    : data(d), options(o), start(begin),
      deadline(begin + std::chrono::duration_cast<std::chrono::steady_clock::duration>(
                           std::chrono::duration<double>(seconds))) {}
double SearchContext::elapsed() const {
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
}
double SearchContext::remaining() const {
    return std::max(0.0, std::chrono::duration<double>(deadline - std::chrono::steady_clock::now()).count());
}
bool SearchContext::done() const {
    return remaining() <= 0.02 || (options.certificate_bound >= 0 && best.feasible &&
           best.objective <= options.certificate_bound + 1e-6);
}
void SearchContext::consider(const LSPResult& r) {
    if (r.feasible && r.objective + 1e-8 < best.objective) {
        best = r; history.emplace_back(elapsed(), r.objective);
        if (elapsed() <= 150.0) checkpoint_best = r;
        std::cout << "[INCUMBENT] elapsed=" << std::fixed << std::setprecision(6)
                  << elapsed() << " objective=" << r.objective << '\n';
        std::cout.flush();
    }
}

static SolutionEncoding reuseLoading(const PDPData& d, const VRPSolution& routes,
                                     const SolutionEncoding& old) {
    auto enc = greedyLoading(d, routes);
    std::vector<std::unordered_map<int,int>> pos(routes.size());
    for (int v = 0; v < int(routes.size()); ++v)
        for (int k = 0; k < int(routes[v].size()); ++k) pos[v][routes[v][k]] = k;
    for (int v = 0; v < int(routes.size()) && v < int(old.size()); ++v) {
        std::vector<bool> supplied(d.numNodes, false);
        for (const auto& st : old[v]) {
            if (st.node == 0 || !pos[v].count(st.node)) continue;
            auto& target = enc[v][pos[v][st.node] + 1];
            for (int pkg : st.packages)
                if (pos[v].count(pkg) && pos[v][pkg] >= pos[v][st.node] &&
                    d.nodeTypes[pkg] == "D" && !supplied[pkg] &&
                    int(target.packages.size()) < d.getDroneCapacity()) {
                    target.packages.push_back(pkg); supplied[pkg] = true;
                }
        }
        auto& load = enc[v][0].packages;
        load.erase(std::remove_if(load.begin(), load.end(), [&](int n) { return supplied[n]; }), load.end());
    }
    return enc;
}

LSPResult SearchContext::evaluate(const VRPSolution& routes, const SolutionEncoding* hint) {
    ++evaluations;
    if (!validRouting(data, routes)) { LSPResult r; r.status = "INVALID_ROUTING"; return r; }
    const std::string key = routingKey(routes);
    auto found = memo.find(key);
    if (options.cache && found != memo.end()) {
        ++cache_hits;
        // Time-limited UBs are reusable, but not frozen forever as route optima.
        if (found->second.status == "OPTIMAL" || ++visits[key] % 5 != 0 || remaining() < 0.1)
            return found->second;
    }
    SolutionEncoding warm = greedyLoading(data, routes);
    if (options.warm) {
        const auto* source = hint ? hint : (best.feasible ? &best.encoding : nullptr);
        if (source) {
            auto reused = reuseLoading(data, routes, *source);
            auto candidate = decode_solution(reused, data, false);
            auto simple = decode_solution(warm, data, false);
            if (candidate.isFeasible && (!simple.isFeasible || candidate.totalCost < simple.totalCost))
                { warm = std::move(reused); ++warm_reuses; }
        }
    }
    LSPResult r;
    auto initial = decode_solution(warm, data, false);
    r.encoding = warm; r.feasible = initial.isFeasible;
    r.objective = initial.isFeasible ? initial.totalCost : INFINITY;
    r.status = "GREEDY"; r.used_fallback = true;
    consider(r);
    if (remaining() > 0.03 && !done()) {
        double allowance = std::min(options.lsp_seconds, remaining() - 0.01);
        double first = options.adaptive ? std::min(0.15, allowance) : allowance;
        r = solveLSP(data, routes, warm, first);
        ++cp_calls; ++statuses[r.status]; fallbacks += r.used_fallback;
        consider(r);
        // A short solve is only a screening pass. UNKNOWN is never an infeasibility proof.
        if (options.adaptive && r.status != "OPTIMAL" && !done() && remaining() > 0.1 &&
            (!r.feasible || !best.feasible || r.objective <= best.objective * 1.02 ||
             computeVRPLB(data, routes) < best.objective - 0.5)) {
            double extra = std::min(allowance - first, remaining() - 0.01);
            if (extra > 0.02) {
                ++adaptive_refinements;
                auto improved = solveLSP(data, routes, r.feasible ? r.encoding : warm, extra);
                ++cp_calls; ++statuses[improved.status]; fallbacks += improved.used_fallback;
                if (improved.feasible && (!r.feasible || improved.objective < r.objective)) r = improved;
                consider(improved);
            }
        }
    }
    if (options.cache && r.feasible) {
        if (found != memo.end() && found->second.objective < r.objective) r = found->second;
        if (memo.size() >= 30000) { memo.clear(); visits.clear(); }
        memo[key] = r; // feasible UB only; never described as exact continuous route optimum
    }
    return r;
}
