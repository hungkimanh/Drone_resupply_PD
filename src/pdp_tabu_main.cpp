#include "pdp_tabu_main.h"
#include "pdp_lb.h"
#include "pdp_alns.h"
#include "pdp_fitness.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <memory>
#include <random>
#include <sstream>
#include <unordered_map>
#include <unordered_set>

using namespace std;

VRPSolution extractVRPSolution(const SolutionEncoding& enc) {
    VRPSolution routes(enc.size());
    for (int v = 0; v < int(enc.size()); ++v)
        for (const auto& stop : enc[v]) if (stop.node > 0) routes[v].push_back(stop.node);
    return routes;
}

struct Candidate {
    VRPSolution routes;
    string key, undo;
    double lb;
    bool tabu;
};

static int partner(const PDPData& d, int n) {
    if (d.nodeTypes[n] != "P" && d.nodeTypes[n] != "DL") return -1;
    for (int i = 1; i < d.numNodes; ++i)
        if (i != n && d.pairIds[i] == d.pairIds[n] &&
            ((d.nodeTypes[n] == "P" && d.nodeTypes[i] == "DL") ||
             (d.nodeTypes[n] == "DL" && d.nodeTypes[i] == "P"))) return i;
    return -1;
}
static string keyFor(const char* type, int a, int b) {
    return string(type) + ':' + to_string(a) + ':' + to_string(b);
}

static vector<Candidate> neighborhood(const PDPData& d, const VRPSolution& current,
                                      const unordered_map<string,int>& tabu, int iteration,
                                      bool pair_moves, int focus) {
    vector<Candidate> pool;
    unordered_set<string> seen;
    auto add = [&](VRPSolution routes, string key, string undo) {
        if (!validRouting(d, routes)) return;
        string representation = routingKey(routes);
        if (representation == routingKey(current) || !seen.insert(representation).second) return;
        auto it = tabu.find(key);
        double lb = computeVRPLB(d, routes);
        pool.push_back({std::move(routes), std::move(key), std::move(undo), lb,
                        it != tabu.end() && it->second > iteration});
    };
    vector<int> trucks;
    if (focus >= 0) trucks.push_back(focus);
    for (int v = 0; v < int(current.size()); ++v) if (v != focus) trucks.push_back(v);
    for (int v : trucks) {
        const auto& route = current[v];
        for (int a = 0; a < int(route.size()); ++a) {
            int n = route[a];
            for (int w = 0; w < int(current.size()); ++w) {
                for (int b = 0; b <= int(current[w].size()); ++b) {
                    auto child = current;
                    child[v].erase(child[v].begin() + a);
                    int insertion = b - (v == w && b > a ? 1 : 0);
                    child[w].insert(child[w].begin() + insertion, n);
                    // Query the destination attribute; prohibit return to the source.
                    add(std::move(child), keyFor("R", n, w), keyFor("R", n, v));
                }
                for (int b = 0; b < int(current[w].size()); ++b) {
                    if (w == v && b <= a) continue;
                    int m = current[w][b]; auto child = current;
                    swap(child[v][a], child[w][b]);
                    string key = keyFor("S", min(n,m), max(n,m));
                    add(std::move(child), key, key);
                }
            }
            for (int b = a + 1; b < int(route.size()); ++b) {
                auto child = current;
                reverse(child[v].begin() + a, child[v].begin() + b + 1);
                string key = keyFor("O", min(route[a],route[b]), max(route[a],route[b]));
                add(std::move(child), key, key);
            }
            if (!pair_moves || d.nodeTypes[n] != "P") continue;
            const int dl = partner(d,n);
            if (dl < 0) continue;
            auto partial = current;
            partial[v].erase(remove_if(partial[v].begin(), partial[v].end(),
                                       [&](int i) { return i == n || i == dl; }), partial[v].end());
            for (int w = 0; w < int(partial.size()); ++w)
                for (int p = 0; p <= int(partial[w].size()); ++p)
                    for (int q = p + 1; q <= int(partial[w].size()) + 1; ++q) {
                        auto child = partial;
                        child[w].insert(child[w].begin() + p, n);
                        child[w].insert(child[w].begin() + q, dl);
                        add(std::move(child), keyFor("PR", n, w), keyFor("PR", n, v));
                    }
            for (int w = v + 1; w < int(current.size()); ++w)
                for (int other : current[w]) if (d.nodeTypes[other] == "P") {
                    int other_dl = partner(d,other); if (other_dl < 0) continue;
                    auto child = current;
                    for (auto& r : child) for (int& node : r) {
                        if (node == n) node = other;
                        else if (node == dl) node = other_dl;
                        else if (node == other) node = n;
                        else if (node == other_dl) node = dl;
                    }
                    string key = keyFor("PS", min(n,other), max(n,other));
                    add(std::move(child), key, key);
                }
        }
    }
    return pool;
}

VRPSolution waitingDrivenEscape(const PDPData& d, const VRPSolution& routes,
                                  const LSPResult& solution, int count, mt19937& rng) {
    auto decoded = decode_solution(solution.encoding, d, false);
    vector<pair<double,int>> scores;
    for (const auto& ev : decoded.resupply_events) {
        double wait = max(0.0, ev.drone_arrive_time - ev.truck_arrive_time);
        for (int n : ev.customer_ids) scores.emplace_back(wait + 0.01*d.readyTimes[n], n);
    }
    if (scores.empty()) return repairRegret2Insert(d, destroyRelatedRemoval(d,routes,count,rng),rng);
    sort(scores.rbegin(),scores.rend());
    unordered_set<int> removed;
    for (const auto& item : scores) {
        removed.insert(item.second);
        int other = partner(d,item.second); if (other > 0) removed.insert(other);
        if (int(removed.size()) >= count) break;
    }
    DestroyResult destroyed{routes,{}};
    for (auto& route : destroyed.partial)
        route.erase(remove_if(route.begin(),route.end(),[&](int n) {
            if (!removed.count(n)) return false;
            destroyed.removed.push_back(n); return true;
        }),route.end());
    return repairRegret2Insert(d,destroyed,rng);
}

TSResult tabuSearch(const PDPData& data, const VRPSolution& initial, const TSConfig& cfg) {
    SearchOptions options; options.lsp_seconds = cfg.lsp_cutoff;
    SearchContext local(data,options,chrono::steady_clock::now(),cfg.time_limit_sec);
    SearchContext& context = cfg.context ? *cfg.context : local;
    const auto local_deadline = chrono::steady_clock::now() +
        chrono::duration_cast<chrono::steady_clock::duration>(chrono::duration<double>(cfg.time_limit_sec));
    auto finished = [&]() { return context.done() || chrono::steady_clock::now() >= local_deadline; };
    mt19937 rng(cfg.seed);
    auto current = initial, best_routes = initial;
    auto best = context.evaluate(initial);
    auto current_result = best;
    double zstar = best.objective, lb_star = computeVRPLB(data,initial);
    unordered_map<string,int> tabu;
    int stagnant = 0, iterations = 0;
    const int base_tenure = max(3, int(round(7.5*log10(max(2,data.numCustomers)))));
    ALNSWeights weights;
    for (; iterations < cfg.max_iter && !finished(); ++iterations) {
        int focus = findLastTruckByLB(data,current);
        if (context.options.critical && current_result.feasible) {
            auto actual = decode_solution(current_result.encoding,data,false);
            double latest = -1;
            for (const auto& tr : actual.truck_details) if (tr.completion_time > latest) {
                latest = tr.completion_time; focus = tr.truck_id;
            }
        }
        auto candidates = neighborhood(data,current,tabu,iterations,cfg.pair_moves,focus);
        shuffle(candidates.begin(), candidates.end(), rng);
        stable_sort(candidates.begin(), candidates.end(), [](const Candidate& a,const Candidate& b) {
            return a.lb < b.lb;
        });
        double threshold = computeThreshold(zstar,lb_star,min(1.0,stagnant/double(max(1,cfg.eta_max))));
        int chosen = -1, fallback = -1, evaluated = 0;
        LSPResult chosen_result;
        for (int i = 0; i < int(candidates.size()) && !finished(); ++i) {
            auto& move = candidates[i];
            if (!move.tabu && fallback < 0) fallback = i;
            if (move.tabu && move.lb >= zstar - 1e-8) continue;
            if (move.lb >= threshold && evaluated > 0) continue;
            auto result = context.evaluate(move.routes, current_result.feasible ? &current_result.encoding : nullptr);
            ++evaluated;
            // Aspiration requires a verified feasible improvement, never a low LB.
            if (result.feasible && (!move.tabu || result.objective < zstar - 1e-8) &&
                (chosen < 0 || result.objective < chosen_result.objective)) {
                chosen = i; chosen_result = result;
            }
            if (result.feasible && result.objective < zstar - 1e-8) {
                zstar = result.objective; best = result; best_routes = move.routes;
                lb_star = move.lb; stagnant = -1;
            }
            if (evaluated >= 5) break;
        }
        if (chosen >= 0) {
            auto& move = candidates[chosen];
            current = move.routes; current_result = chosen_result;
            int tenure = base_tenure + (cfg.randomize_tenure ? int(rng()%5) : 0);
            tabu[move.undo] = iterations + tenure;
        } else if (fallback >= 0) {
            // As in the paper, a discarded non-tabu route can be used for exploration.
            current = candidates[fallback].routes; current_result = LSPResult{};
            tabu[candidates[fallback].undo] = iterations + base_tenure;
        }
        ++stagnant;
        if (stagnant >= cfg.eta_max && !finished()) {
            const int count = max(2,int(data.numCustomers*(0.15+0.2*(rng()%100)/100.0)));
            for (int attempt = 0; attempt < cfg.n_alns_escape && !finished(); ++attempt) {
                if (context.options.critical && attempt == 0) ++context.waiting_escapes;
                int destroy = weights.selectDestroy(rng), repair = weights.selectRepair(rng);
                auto route = context.options.critical && attempt == 0
                    ? waitingDrivenEscape(data,best_routes,best,count,rng)
                    : alnsIteration(data,best_routes,weights,destroy,repair,count,rng);
                auto result = context.evaluate(route, &best.encoding);
                int score = 0;
                if (result.feasible && result.objective < zstar - 1e-8) {
                    zstar = result.objective; best = result; best_routes = route;
                    lb_star = computeVRPLB(data,route); score = 2;
                }
                // Controlled diversification actually changes the current solution.
                if (result.feasible && (score == 2 || result.objective <= zstar*1.05)) {
                    current = route; current_result = result; score = max(score,1);
                }
                weights.updateDestroy(destroy,score); weights.updateRepair(repair,score);
            }
            tabu.clear(); stagnant = 0;
        }
    }
    TSResult result;
    result.best_vrp = best_routes; result.best_lsp = best;
    result.best_obj = best.objective; result.iterations_done = iterations;
    return result;
}
