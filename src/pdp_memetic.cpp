#include "pdp_memetic.h"
#include "pdp_ga.h"
#include "pdp_fitness.h"
#include <algorithm>
#include <cmath>
#include <random>
#include <set>
#include <unordered_set>

namespace {
struct Individual { VRPSolution routes; LSPResult evaluation; };
std::set<std::pair<int,int>> arcs(const VRPSolution& routes) {
    std::set<std::pair<int,int>> result;
    for (const auto& route : routes) {
        int previous = 0;
        for (int n : route) { result.insert({previous,n}); previous = n; }
        result.insert({previous,0});
    }
    return result;
}
double difference(const Individual& a,const Individual& b) {
    auto x = arcs(a.routes), y = arcs(b.routes);
    int shared = 0;
    for (const auto& edge : x) shared += y.count(edge);
    return 1.0 - shared / double(std::max(x.size(),y.size()));
}
}

LSPResult memeticSearch(const PDPData& data, const std::vector<SolutionEncoding>& initial,
                       SearchContext& context, TSConfig config, int seed) {
    std::mt19937 rng(seed);
    std::vector<Individual> population;
    std::unordered_set<std::string> seen;
    for (const auto& encoded : initial) {
        if (context.done()) break;
        auto routes = extractVRPSolution(encoded);
        if (!validRouting(data,routes) || !seen.insert(routingKey(routes)).second) continue;
        auto result = context.evaluate(routes,&encoded);
        if (result.feasible) population.push_back({routes,result});
        if (population.size() >= 12) break;
    }
    if (population.empty()) return context.best;
    auto sort_quality = [&]() {
        std::sort(population.begin(),population.end(),[](const Individual& a,const Individual& b) {
            return a.evaluation.objective < b.evaluation.objective;
        });
    };
    sort_quality();
    int generation = 0;
    while (!context.done()) {
        auto parent = [&]() -> const Individual& {
            int a = rng()%population.size(), b = rng()%population.size();
            return population[a].evaluation.objective <= population[b].evaluation.objective
                     ? population[a] : population[b];
        };
        // Route-segment crossover preserves part of one parent's truck assignment;
        // the other parent supplies the remaining order. Pair repair is request-aware.
        Individual p1 = parent(), p2 = parent();
        VRPSolution child(data.numTrucks);
        std::vector<bool> used(data.numNodes,false);
        int v = rng()%data.numTrucks;
        if (!p1.routes[v].empty()) {
            int a = rng()%p1.routes[v].size(), b = rng()%p1.routes[v].size();
            if (a > b) std::swap(a,b);
            for (int k = a; k <= b; ++k) { int n=p1.routes[v][k]; child[v].push_back(n); used[n]=true; }
        }
        for (int w=0; w<data.numTrucks; ++w)
            for (int n : p2.routes[w]) if (!used[n]) { child[w].push_back(n); used[n]=true; }
        if (rng()%100 < 35) {
            int from=rng()%data.numTrucks, to=rng()%data.numTrucks;
            if (!child[from].empty()) {
                int at=rng()%child[from].size(), n=child[from][at];
                child[from].erase(child[from].begin()+at);
                child[to].insert(child[to].begin()+(rng()%(child[to].size()+1)),n);
            }
        }
        repairC2Pairs(child,data);
        if (context.options.critical && generation > 0 && generation % 5 == 0 && context.best.feasible) {
            ++context.waiting_escapes;
            child = waitingDrivenEscape(data, extractVRPSolution(context.best.encoding), context.best,
                                        std::max(2,data.numCustomers/4), rng);
        }
        if (!validRouting(data,child)) continue;
        auto evaluation=context.evaluate(child,&p1.evaluation.encoding);
        if (evaluation.feasible && !context.done()) {
            config.context=&context; config.verbose=false; config.seed=seed+ ++generation;
            config.time_limit_sec=std::min(2.0,context.remaining());
            config.max_iter=15;
            auto improved=tabuSearch(data,child,config);
            if (improved.best_lsp.feasible && improved.best_obj < evaluation.objective) {
                child=improved.best_vrp; evaluation=improved.best_lsp;
            }
        }
        if (!evaluation.feasible) continue;
        bool clone=false;
        for (auto& individual : population) if (routingKey(individual.routes)==routingKey(child)) {
            if (evaluation.objective < individual.evaluation.objective) individual={child,evaluation};
            clone=true; break;
        }
        if (!clone) population.push_back({child,evaluation});
        sort_quality();
        if (population.size()>16) {
            // Keep the two quality elites; remove a redundant low-quality member.
            int discard=2; double worst=-1;
            for (int i=2;i<int(population.size());++i) {
                double nearest=1.0;
                for (int j=0;j<int(population.size());++j) if (i!=j)
                    nearest=std::min(nearest,difference(population[i],population[j]));
                double score=i/double(population.size()) + (1.0-nearest);
                if (score>worst) { worst=score; discard=i; }
            }
            population.erase(population.begin()+discard);
        }
    }
    return context.best;
}
