#include "pdp_ga.h"

#include "pdp_fitness.h"
#include "pdp_init.h"
#include "pdp_tabu.h"
#include "pdp_utils.h"

#include <algorithm>
#include <chrono>
#include <climits>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <random>
#include <string>
#include <unordered_map>
#include <unordered_set>

using namespace std;

// namespace { // Expose helpers globally

vector<vector<int>> extractCustomerRoutes(const SolutionEncoding& sol) {
    vector<vector<int>> routes;
    routes.reserve(sol.size());
    for (const auto& r : sol) {
        vector<int> cr;
        for (const auto& st : r) {
            if (st.node > 0) cr.push_back(st.node);
        }
        routes.push_back(std::move(cr));
    }
    return routes;
}

vector<int> flatten(const vector<vector<int>>& routes) {
    vector<int> flat;
    size_t total = 0;
    for (const auto& r : routes) total += r.size();
    flat.reserve(total);
    for (const auto& r : routes) flat.insert(flat.end(), r.begin(), r.end());
    return flat;
}

vector<int> routeLengths(const vector<vector<int>>& routes) {
    vector<int> lens;
    lens.reserve(routes.size());
    for (const auto& r : routes) lens.push_back((int)r.size());
    return lens;
}

vector<vector<int>> splitByLengths(const vector<int>& flat, const vector<int>& lengths) {
    vector<vector<int>> out;
    out.reserve(lengths.size());
    size_t off = 0;
    for (int len : lengths) {
        vector<int> r;
        r.reserve((size_t)max(0, len));
        for (int i = 0; i < len && off < flat.size(); ++i, ++off) r.push_back(flat[off]);
        out.push_back(std::move(r));
    }
    return out;
}

SolutionEncoding initializeEmptyEncodedRoutes(const vector<vector<int>>& customerRoutes) {
    SolutionEncoding enc;
    enc.reserve(customerRoutes.size());
    for (const auto& cr : customerRoutes) {
        Route r;
        r.reserve(cr.size() + 2);
        r.push_back(RouteStop(0));
        for (int c : cr) r.push_back(RouteStop(c));
        r.push_back(RouteStop(0));
        r.back().packages.clear();
        enc.push_back(std::move(r));
    }
    return enc;
}

// Trip struct is defined in pdp_ga.h

string tripKey(int h, const vector<int>& pkgs) {
    string key = to_string(h) + ":";
    for (int p : pkgs) {
        key += to_string(p);
        key.push_back(',');
    }
    return key;
}

vector<Trip> extractResupplyTrips(const SolutionEncoding& sol, int originFlag) {
    vector<Trip> trips;
    for (const auto& r : sol) {
        for (const auto& st : r) {
            if (st.node > 0 && !st.packages.empty()) {
                Trip t;
                t.h = st.node;
                t.pkgs = st.packages;
                sort(t.pkgs.begin(), t.pkgs.end());
                t.pkgs.erase(unique(t.pkgs.begin(), t.pkgs.end()), t.pkgs.end());
                t.origin = originFlag;
                trips.push_back(std::move(t));
            }
        }
    }
    return trips;
}


// RouteIndexInfo struct is defined in pdp_ga.h

unordered_map<int, RouteIndexInfo> buildCustomerIndex(const SolutionEncoding& sol) {
    unordered_map<int, RouteIndexInfo> idx;
    for (int rId = 0; rId < (int)sol.size(); ++rId) {
        int pos = 0;
        for (const auto& st : sol[rId]) {
            if (st.node > 0) {
                idx[st.node] = RouteIndexInfo{rId, pos};
                pos++;
            }
        }
    }
    return idx;
}

vector<int> getValidPackageSubset(const Trip& trip,
                                  const unordered_map<int, RouteIndexInfo>& childIndex,
                                  const unordered_set<int>& suppliedAlready) {
    auto itH = childIndex.find(trip.h);
    if (itH == childIndex.end()) return {};
    const int routeH = itH->second.route_id;
    const int posH = itH->second.pos;

    vector<int> valid;
    valid.reserve(trip.pkgs.size());

    for (int p : trip.pkgs) {
        if (suppliedAlready.count(p)) continue;
        auto itP = childIndex.find(p);
        if (itP == childIndex.end()) continue;
        if (itP->second.route_id != routeH) continue;
        if (itP->second.pos < posH) continue;
        valid.push_back(p);
    }

    return valid;
}

static double pkgWeight(const PDPData& data, int p) {
    if (p <= 0 || p >= (int)data.demands.size()) return 0.0;
    return (double)data.demands[p];
}

static void repairInvalidResupplyPositions(SolutionEncoding& child, const PDPData& data) {
    for (auto& r : child) {
        unordered_map<int,int> stopPos;
        for (int i = 0; i < (int)r.size(); ++i)
            if (r[i].node > 0) stopPos[r[i].node] = i;

        for (int sourceIdx = 0; sourceIdx < (int)r.size(); ++sourceIdx) {
            auto& st = r[sourceIdx];
            if (st.node <= 0 || st.packages.empty()) continue;
            vector<int> kept;
            kept.reserve(st.packages.size());
            for (int p : st.packages) {
                if (p <= 0 || p >= (int)data.nodeTypes.size()) continue;
                if (data.nodeTypes[p] == "P" || data.nodeTypes[p] == "DL") continue;

                auto it = stopPos.find(p);
                if (it == stopPos.end() || it->second < sourceIdx) continue;

                bool crossesDepot = false;
                for (int k = sourceIdx + 1; k <= it->second; ++k) {
                    if (r[k].node == 0) {
                        crossesDepot = true;
                        break;
                    }
                }
                if (crossesDepot) continue;
                kept.push_back(p);
            }
            st.packages.swap(kept);
        }
    }
}

static void repairDroneCapacity(SolutionEncoding& child, const PDPData& data) {
    const double QD = (double)data.getDroneCapacity();
    for (auto& r : child) {
        // Precompute pos of each customer in this route
        vector<int> customers;
        for (const auto& st : r) if (st.node > 0) customers.push_back(st.node);
        unordered_map<int,int> pos;
        for (int i = 0; i < (int)customers.size(); ++i) pos[customers[i]] = i;

        for (auto& st : r) {
            if (st.node <= 0 || st.packages.empty()) continue;

            // Prioritize packages served earlier after h
            vector<pair<int,int>> pk;
            pk.reserve(st.packages.size());
            for (int p : st.packages) {
                auto it = pos.find(p);
                pk.push_back({it == pos.end() ? INT_MAX : it->second, p});
            }
            sort(pk.begin(), pk.end());

            vector<int> kept;
            double load = 0.0;
            for (auto& it : pk) {
                const int p = it.second;
                const double w = pkgWeight(data, p);
                if (load + w <= QD + 1e-9) {
                    kept.push_back(p);
                    load += w;
                }
            }
            st.packages.swap(kept);
        }
    }
}

static void removeEmptyResupplyMarkers(SolutionEncoding& child) {
    for (auto& r : child) {
        for (auto& st : r) {
            if (st.node > 0 && st.packages.empty()) {
                // keep as plain customer stop
            }
        }
    }
}

static void dedupPackageSupply(SolutionEncoding& child) {
    // If a package appears in multiple resupplies, keep the earliest resupply (by route order).
    unordered_map<int, pair<int,int>> bestResupply; // pkg -> (routeIdx, stopIdx)

    for (int rId = 0; rId < (int)child.size(); ++rId) {
        for (int sId = 0; sId < (int)child[rId].size(); ++sId) {
            const auto& st = child[rId][sId];
            if (st.node <= 0 || st.packages.empty()) continue;
            for (int p : st.packages) {
                auto it = bestResupply.find(p);
                if (it == bestResupply.end() || make_pair(rId, sId) < it->second) {
                    bestResupply[p] = {rId, sId};
                }
            }
        }
    }

    // Remove package from non-best resupplies
    for (int rId = 0; rId < (int)child.size(); ++rId) {
        for (int sId = 0; sId < (int)child[rId].size(); ++sId) {
            auto& st = child[rId][sId];
            if (st.node <= 0 || st.packages.empty()) continue;

            vector<int> kept;
            kept.reserve(st.packages.size());
            for (int p : st.packages) {
                auto it = bestResupply.find(p);
                if (it != bestResupply.end() && it->second == make_pair(rId, sId)) {
                    kept.push_back(p);
                }
            }
            st.packages.swap(kept);
        }
    }

    // Remove resupplied packages from all depot loads
    unordered_set<int> resupplied;
    for (const auto& kv : bestResupply) resupplied.insert(kv.first);

    for (auto& r : child) {
        for (auto& st : r) {
            if (st.node != 0 || st.packages.empty()) continue;
            vector<int> kept;
            kept.reserve(st.packages.size());
            for (int p : st.packages) {
                if (!resupplied.count(p)) kept.push_back(p);
            }
            st.packages.swap(kept);
        }
    }
}

static double truckTravelMinutes(const PDPData& data, int a, int b) {
    if (a < 0 || a >= data.numNodes || b < 0 || b >= data.numNodes) return 1e18;
    return data.truckDistMatrix[a][b];
}

static double pkgRelease(const PDPData& data, int p) {
    if (p <= 0 || p >= (int)data.readyTimes.size()) return 0.0;
    return (double)data.readyTimes[p];
}

static void mergeConsecutiveDepotStops(Route& r) {
    if (r.size() < 2) return;
    Route out;
    out.reserve(r.size());
    for (size_t i = 0; i < r.size(); ++i) {
        if (r[i].node == 0) {
            if (i + 1 < r.size() && r[i + 1].node == 0) {
                r[i + 1].packages.insert(r[i + 1].packages.begin(), r[i].packages.begin(), r[i].packages.end());
                continue;
            }
        }
        out.push_back(r[i]);
    }
    r.swap(out);
}

static void removeEmptyIntermediateDepotStops(Route& r) {
    if (r.size() <= 2) return;
    // Keep first and last depot stop.
    Route out;
    out.reserve(r.size());
    for (size_t i = 0; i < r.size(); ++i) {
        const auto& st = r[i];
        const bool isFirst = (i == 0);
        const bool isLast = (i + 1 == r.size());
        if (st.node == 0 && st.packages.empty() && !isFirst && !isLast) {
            continue;
        }
        out.push_back(st);
    }
    r.swap(out);
}

static void rebuildDepotLoadsForRoute(Route& r, const PDPData& data) {
    // Preserve any pre-specified depot packages as REQUIRED loads for that depot stop
    // (used for inserted depot-return stops like 0[{i}] when the truck lacks package i).
    vector<vector<int>> requiredAtDepot(r.size());
    for (size_t i = 0; i < r.size(); ++i) {
        if (r[i].node == 0 && !r[i].packages.empty()) {
            requiredAtDepot[i] = r[i].packages;
        }
    }

    // Clear all depot packages; will rebuild (required packages will be re-applied).
    for (auto& st : r) {
        if (st.node == 0) st.packages.clear();
    }
    // Ensure structural depot boundary
    if (r.empty() || r.front().node != 0) r.insert(r.begin(), RouteStop(0));
    if (r.back().node != 0) r.push_back(RouteStop(0));
    r.back().packages.clear();

    // Precompute for this route: which package is resupplied at which stop index.
    unordered_map<int,int> resupplyStopIdx;
    for (int i = 0; i < (int)r.size(); ++i) {
        if (r[i].node <= 0) continue;
        for (int p : r[i].packages) {
            if (!resupplyStopIdx.count(p)) resupplyStopIdx[p] = i;
            else resupplyStopIdx[p] = min(resupplyStopIdx[p], i);
        }
    }

    unordered_set<int> onboard;
    double onboardLoad = 0.0;

    double t = 0.0;
    int pos = data.depotIndex;

    auto greedyLoadAtDepotIdx = [&](int depotIdx) {
        vector<int> P;
        double remaining = (double)data.truckCapacity - onboardLoad;

        // Wait until required packages for this depot stop are released, then load them first.
        double requiredReleaseTime = t;
        if (depotIdx >= 0 && depotIdx < (int)requiredAtDepot.size()) {
            for (int req : requiredAtDepot[(size_t)depotIdx]) {
                requiredReleaseTime = max(requiredReleaseTime, pkgRelease(data, req));
            }
        }
        if (requiredReleaseTime > t + 1e-9) {
            t = requiredReleaseTime;
        }

        if (depotIdx >= 0 && depotIdx < (int)requiredAtDepot.size()) {
            for (int req : requiredAtDepot[(size_t)depotIdx]) {
                if (req <= 0) continue;
                if (onboard.count(req)) continue;
                const double w = pkgWeight(data, req);
                if (w <= remaining + 1e-9) {
                    P.push_back(req);
                    remaining -= w;
                    onboard.insert(req);
                    onboardLoad += w;
                }
            }
        }

        for (int j = depotIdx + 1; j < (int)r.size(); ++j) {
            const int node = r[j].node;
            if (node == 0) break;
            if (node < 0) continue;
            const int pkg = node;
            if (onboard.count(pkg)) continue;

            if (pkg > 0 && pkg < (int)data.nodeTypes.size()) {
                if (data.nodeTypes[pkg] == "P" || data.nodeTypes[pkg] == "DL") {
                    continue;
                }
            }

            // If package will be received by drone before service, skip depot loading.
            auto itRS = resupplyStopIdx.find(pkg);
            if (itRS != resupplyStopIdx.end() && itRS->second <= j) {
                continue;
            }

            if (pkgRelease(data, pkg) > t + 1e-9) continue;

            const double w = pkgWeight(data, pkg);
            if (w <= remaining + 1e-9) {
                P.push_back(pkg);
                remaining -= w;
                onboard.insert(pkg);
                onboardLoad += w;
            }
        }

        r[depotIdx].packages = std::move(P);
        if (!r[depotIdx].packages.empty()) {
            t += data.depotReceiveTime;
        }
    };

    for (int idx = 0; idx < (int)r.size(); ++idx) {
        auto& st = r[idx];

        if (st.node == 0) {
            if (pos != data.depotIndex) {
                t += truckTravelMinutes(data, pos, data.depotIndex);
                pos = data.depotIndex;
            }
            greedyLoadAtDepotIdx(idx);
            continue;
        }

        const int i = st.node;

        // Travel to customer
        t += truckTravelMinutes(data, pos, i);
        pos = i;

        // Receive drone resupply at i[P] before serving i
        for (int p : st.packages) {
            if (!onboard.count(p)) {
                onboard.insert(p);
                onboardLoad += pkgWeight(data, p);
            }
        }

        // Ensure package i is available; else insert depot return 0[i] right before this customer.
        if (i > 0 && i < (int)data.nodeTypes.size()) {
            string type = data.nodeTypes[i];
            if (type == "D" && !onboard.count(i)) {
                RouteStop depot(0, vector<int>{i});
                r.insert(r.begin() + idx, depot);

                // Restart rebuilding after structural change
                rebuildDepotLoadsForRoute(r, data);
                return;
            }
        }

        // Serve i
        if (i > 0 && i < (int)data.nodeTypes.size()) {
            string type = data.nodeTypes[i];
            if (type == "P") {
                onboard.insert(i);
                onboardLoad += pkgWeight(data, i);
            } else if (type == "DL") {
                int pickupNode = -1;
                for (int p = 1; p < data.numNodes; ++p) {
                    if (data.pairIds[p] == data.pairIds[i] && data.nodeTypes[p] == "P") {
                        pickupNode = p;
                        break;
                    }
                }
                if (pickupNode != -1) {
                    onboard.erase(pickupNode);
                    onboardLoad -= pkgWeight(data, i);
                }
            } else { // "D": Delivery C1
                onboard.erase(i);
                onboardLoad -= pkgWeight(data, i);
            }
        }
        t += data.truckServiceTime;
    }

    mergeConsecutiveDepotStops(r);
    removeEmptyIntermediateDepotStops(r);
}

static void rebuildDepotLoads(SolutionEncoding& child, const PDPData& data) {
    for (auto& r : child) rebuildDepotLoadsForRoute(r, data);
}

void repairSolution(SolutionEncoding& child, const PDPData& data) {
    // 1) Repair customer uniqueness is assumed from crossover; we keep it minimal here.

    // 2) Remove duplicated package supply (resupply vs depot load, and multiple resupplies)
    repairInvalidResupplyPositions(child, data);
    repairDroneCapacity(child, data);
    dedupPackageSupply(child);

    // 3) Rebuild depot loads for unsupplied packages
    rebuildDepotLoads(child, data);

    removeEmptyResupplyMarkers(child);
}

static vector<int> onePointCrossoverFlat(const vector<int>& f1, const vector<int>& f2, mt19937& gen) {
    if (f1.size() != f2.size() || f1.size() < 2) return f1;
    uniform_int_distribution<int> dist(1, (int)f1.size() - 1);
    int c = dist(gen);

    vector<int> child;
    child.reserve(f1.size());
    child.insert(child.end(), f1.begin(), f1.begin() + c);

    unordered_set<int> used(child.begin(), child.end());
    for (int x : f2) {
        if (!used.count(x)) {
            child.push_back(x);
            used.insert(x);
        }
    }
    return child;
}

static vector<int> twoPointCrossoverFlat(const vector<int>& f1, const vector<int>& f2, mt19937& gen) {
    if (f1.size() != f2.size() || f1.size() < 3) return f1;
    uniform_int_distribution<int> dist(0, (int)f1.size() - 1);
    int c1 = dist(gen);
    int c2 = dist(gen);
    if (c1 > c2) std::swap(c1, c2);
    if (c1 == c2) return f1;

    vector<int> child(f1.size(), -1);
    for (int i = c1; i <= c2; ++i) child[i] = f1[i];

    unordered_set<int> used;
    for (int x : child) if (x != -1) used.insert(x);

    vector<int> missing;
    missing.reserve(f1.size());
    for (int x : f2) if (!used.count(x)) missing.push_back(x);

    int mi = 0;
    for (int i = 0; i < (int)child.size(); ++i) {
        if (child[i] == -1) child[i] = missing[mi++];
    }
    return child;
}

static SolutionEncoding crossoverWithResupplyReuse(const SolutionEncoding& p1,
                                                  const SolutionEncoding& p2,
                                                  const PDPData& data,
                                                  mt19937& gen,
                                                  bool twoPoint) {
    // Extract customer routes
    auto r1 = extractCustomerRoutes(p1);
    auto r2 = extractCustomerRoutes(p2);

    // Flatten
    auto f1 = flatten(r1);
    auto f2 = flatten(r2);

    // Route lengths from parent 1
    auto lens = routeLengths(r1);

    // Crossover
    vector<int> childFlat = twoPoint ? twoPointCrossoverFlat(f1, f2, gen)
                                     : onePointCrossoverFlat(f1, f2, gen);
    auto childRoutes = splitByLengths(childFlat, lens);

    // Initialize empty encoded routes [0, customers..., 0]
    repairC2Pairs(childRoutes, data);
    SolutionEncoding child = initializeEmptyEncodedRoutes(childRoutes);

    // Extract candidate resupply trips
    auto trips1 = extractResupplyTrips(p1, 1);
    auto trips2 = extractResupplyTrips(p2, 2);

    unordered_map<string, Trip> merged;
    merged.reserve(trips1.size() + trips2.size());

    auto absorb = [&](vector<Trip>& trips) {
        for (auto& t : trips) {
            string key = tripKey(t.h, t.pkgs);
            auto it = merged.find(key);
            if (it == merged.end()) {
                merged.emplace(key, t);
            } else {
                it->second.origin |= t.origin;
            }
        }
    };
    absorb(trips1);
    absorb(trips2);

    vector<Trip> candidates;
    candidates.reserve(merged.size());
    for (auto& kv : merged) candidates.push_back(std::move(kv.second));

    // Sort candidates: prefer appearing in both parents, then by |P| desc
    sort(candidates.begin(), candidates.end(), [](const Trip& a, const Trip& b) {
        const int ca = (a.origin == 3) ? 1 : 0;
        const int cb = (b.origin == 3) ? 1 : 0;
        if (ca != cb) return ca > cb;
        if (a.pkgs.size() != b.pkgs.size()) return a.pkgs.size() > b.pkgs.size();
        if (a.h != b.h) return a.h < b.h;
        return a.pkgs < b.pkgs;
    });

    // Insert resupplies (structure-only); feasibility handled in repair via decode_solution.
    unordered_set<int> supplied;
    auto childIndex = buildCustomerIndex(child);

    const double QD = (double)data.getDroneCapacity();

    for (const auto& trip : candidates) {
        vector<int> valid = getValidPackageSubset(trip, childIndex, supplied);
        vector<int> filteredValid;
        for (int p : valid) {
            if (p > 0 && p < (int)data.nodeTypes.size() && (data.nodeTypes[p] == "P" || data.nodeTypes[p] == "DL")) {
                continue;
            }
            filteredValid.push_back(p);
        }
        valid.swap(filteredValid);
        if (valid.empty()) continue;

        double load = 0.0;
        for (int p : valid) load += pkgWeight(data, p);
        if (load > QD + 1e-9) {
            // Trim by keeping those served soonest after h (repairDroneCapacity will also do this)
            // For now, skip; repair will not recover subset without extra logic.
            continue;
        }

        // Add to resupply marker at node h
        auto itH = childIndex.find(trip.h);
        if (itH == childIndex.end()) continue;
        Route& r = child[itH->second.route_id];

        // Find the actual RouteStop for node h
        for (auto& st : r) {
            if (st.node == trip.h) {
                // Append packages (dedup later)
                st.packages.insert(st.packages.end(), valid.begin(), valid.end());
                break;
            }
        }

        for (int p : valid) supplied.insert(p);
    }

    // Repair + build depot loads for the rest
    repairSolution(child, data);

    return child;
}

static void customerSwapMutation(SolutionEncoding& sol, const PDPData& data, mt19937& gen) {
    auto routes = extractCustomerRoutes(sol);
    auto flat = flatten(routes);
    if (flat.size() < 2) return;

    uniform_int_distribution<int> dist(0, (int)flat.size() - 1);
    int a = dist(gen);
    int b = dist(gen);
    if (a == b) return;
    swap(flat[a], flat[b]);

    auto lens = routeLengths(routes);
    auto mutatedRoutes = splitByLengths(flat, lens);
    repairC2Pairs(mutatedRoutes, data);
    sol = initializeEmptyEncodedRoutes(mutatedRoutes);
}

static void droneResupplyMutation(SolutionEncoding& sol, const PDPData& data, mt19937& gen) {
    if (sol.empty()) return;

    uniform_int_distribution<int> truckDist(0, (int)sol.size() - 1);
    int k = truckDist(gen);
    Route& r = sol[k];

    // Collect customer stops positions
    vector<int> customerPos;
    for (int i = 0; i < (int)r.size(); ++i) if (r[i].node > 0) customerPos.push_back(i);
    if (customerPos.size() < 2) return;

    uniform_int_distribution<int> posDist(0, (int)customerPos.size() - 2);
    int hIdxInCustomerList = posDist(gen);
    int hStopIdx = customerPos[hIdxInCustomerList];

    // Build list of candidate packages from later customers in the same route.
    vector<int> later;
    for (int t = hIdxInCustomerList; t < (int)customerPos.size(); ++t) {
        int stopIdx = customerPos[t];
        int pkg = r[stopIdx].node;
        later.push_back(pkg);
    }

    // Remove packages already in some resupply
    unordered_set<int> already;
    for (const auto& st : r) {
        if (st.node > 0) {
            for (int p : st.packages) already.insert(p);
        }
    }

    vector<int> cand;
    for (int p : later) {
        if (!already.count(p)) {
            if (p > 0 && p < (int)data.nodeTypes.size()) {
                if (data.nodeTypes[p] == "P" || data.nodeTypes[p] == "DL") continue;
            }
            cand.push_back(p);
        }
    }
    if (cand.empty()) return;

    shuffle(cand.begin(), cand.end(), gen);

    const double QD = (double)data.getDroneCapacity();
    vector<int> chosen;
    double load = 0.0;
    for (int p : cand) {
        const double w = pkgWeight(data, p);
        if (load + w <= QD + 1e-9) {
            chosen.push_back(p);
            load += w;
        }
    }
    if (chosen.empty()) return;

    // Insert into h[P]
    r[hStopIdx].packages.insert(r[hStopIdx].packages.end(), chosen.begin(), chosen.end());

    // Repair / dedup / rebuild depot loads
    repairSolution(sol, data);
}

static SolutionEncoding tournamentSelection(const vector<SolutionEncoding>& pop,
                                           const vector<double>& fit,
                                           int tournamentSize,
                                           mt19937& gen) {
    uniform_int_distribution<int> dist(0, (int)pop.size() - 1);
    int best = dist(gen);
    for (int i = 1; i < tournamentSize; ++i) {
        int idx = dist(gen);
        if (fit[idx] < fit[best]) best = idx;
    }
    return pop[best];
}

// } // namespace

PDPSolution geneticAlgorithmPDP(const PDPData& data,
                                const std::vector<SolutionEncoding>& initialPopulation,
                                int populationSize,
                                int maxGenerations,
                                double mutationRate,
                                int runNumber,
                                bool isSmallScale,
                                std::ostream* logStream,
                                int logEvery,
                                double timeLimitSeconds) {
    (void)isSmallScale;
    if (logEvery <= 0) logEvery = 1;

    const auto gaStartTime = chrono::high_resolution_clock::now();
    const double lowerBound = computeLowerBound(data);

    // Init population is provided by caller (pdp_init)
    vector<SolutionEncoding> population = initialPopulation;
    if (population.empty()) {
        PDPSolution empty;
        empty.totalCost = numeric_limits<double>::infinity();
        empty.totalPenalty = numeric_limits<double>::infinity();
        empty.isFeasible = false;
        return empty;
    }
    populationSize = (int)population.size();

    std::seed_seq seed{runNumber, populationSize, maxGenerations, data.numNodes};
    mt19937 gen(seed);
    uniform_real_distribution<double> prob(0.0, 1.0);

    PDPSolution best;
    best.totalCost = numeric_limits<double>::infinity();
    best.totalPenalty = numeric_limits<double>::infinity();
    bool hasBest = false;
    int stagnation_counter = 0;

    struct PopStats {
        double bestFitness = numeric_limits<double>::infinity();
        double bestCost = numeric_limits<double>::infinity();
        double bestPenalty = numeric_limits<double>::infinity();
        double avgFitness = 0.0;
        int feasibleCount = 0;
    };

    auto evaluatePopulation = [&](const vector<SolutionEncoding>& pop,
                                  vector<double>& fit) -> PopStats {
        PopStats st;
        if (pop.empty()) return st;
        fit.assign(pop.size(), 0.0);
        for (size_t i = 0; i < pop.size(); ++i) {
            const SolutionEncoding& ind = pop[i];
            PDPSolution sol = decode_solution(ind, data, false);
            sol.encoded_routes = ind;

            const double fitness = sol.totalCost + sol.totalPenalty;
            fit[i] = fitness;
            st.avgFitness += fitness;
            if (sol.isFeasible) st.feasibleCount++;

            if (fitness < st.bestFitness) {
                st.bestFitness = fitness;
                st.bestCost = sol.totalCost;
                st.bestPenalty = sol.totalPenalty;
            }

            // Global best selection: prefer feasible, then lowest cost; else lowest fitness.
            if (!hasBest || (sol.isFeasible && (!best.isFeasible || sol.totalCost < best.totalCost)) ||
                (!sol.isFeasible && !best.isFeasible && fitness < (best.totalCost + best.totalPenalty))) {
                best = sol;
                hasBest = true;
            }
        }
        st.avgFitness /= (double)pop.size();
        return st;
    };

    vector<double> fitness;
    PopStats initStats = evaluatePopulation(population, fitness);

    // Header
    cout << "\n[GA] pop=" << populationSize
         << " maxGen=" << maxGenerations
         << " mut=" << mutationRate
         << " customers=" << data.numCustomers
         << " trucks=" << data.numTrucks
         << " drones=" << data.numDrones << "\n"
         << "[GA] LowerBound=" << fixed << setprecision(2) << lowerBound
         << (timeLimitSeconds > 0 ? ("  TimeLimit=" + to_string((int)timeLimitSeconds) + "s") : "  TimeLimit=none") << "\n"
         << "[GA] Init  | Feasible=" << initStats.feasibleCount << "/" << populationSize
         << " | BestCmax=" << fixed << setprecision(2) << initStats.bestCost
         << " | BestPenalty=" << initStats.bestPenalty
         << " | AvgFit=" << initStats.avgFitness << "\n"
         << string(90, '-') << "\n";
    cout << setw(6)  << "Gen"
         << setw(10) << "BestCmax"
         << setw(10) << "Penalty"
         << setw(12) << "Feasible"
         << setw(10) << "AvgFit"
         << setw(8)  << "Stag"
         << setw(8)  << "MutC"
         << setw(8)  << "MutD"
         << "  Note\n"
         << string(90, '-') << "\n";
    cout.flush();

    if (logStream) {
        (*logStream) << "[GA] start pop=" << populationSize
                     << " gen=" << maxGenerations
                     << " mut=" << mutationRate
                     << " customers=" << data.numCustomers
                     << " trucks=" << data.numTrucks
                     << " drones=" << data.numDrones << "\n";
        (*logStream) << "[GA] gen=-1 feasible=" << initStats.feasibleCount
                     << "/" << population.size()
                     << " bestFitness=" << initStats.bestFitness
                     << " bestCmax=" << initStats.bestCost
                     << " bestPenalty=" << initStats.bestPenalty
                     << " avgFitness=" << initStats.avgFitness << "\n";
    }

    const int tournamentSize = 5;
    const int eliteCount = 1;
    int consecutiveTSFails = 0;
    int restartSeed = runNumber * 100;

    for (int genIdx = 0; genIdx < maxGenerations; ++genIdx) {
        int madeChildren = 0;
        int usedTwoPointCount = 0;
        int customerMutationCount = 0;
        int droneMutationCount = 0;

        vector<SolutionEncoding> nextPop;
        nextPop.reserve(population.size());

        // Elitism: keep best individual seen so far (encoded_routes from best)
        if (eliteCount > 0 && hasBest) {
            nextPop.push_back(best.encoded_routes);
        }

        while ((int)nextPop.size() < populationSize) {
            SolutionEncoding p1 = tournamentSelection(population, fitness, tournamentSize, gen);
            SolutionEncoding p2 = tournamentSelection(population, fitness, tournamentSize, gen);

            bool useTwoPoint = (prob(gen) < 0.5);
            if (useTwoPoint) usedTwoPointCount++;
            SolutionEncoding child = crossoverWithResupplyReuse(p1, p2, data, gen, useTwoPoint);
            madeChildren++;

            if (prob(gen) < mutationRate) {
                customerSwapMutation(child, data, gen);
                customerMutationCount++;
                // After customer-only mutation, re-add some drone structure with a chance.
                if (prob(gen) < 0.5) {
                    droneResupplyMutation(child, data, gen);
                    droneMutationCount++;
                }
                // Ensure feasibility structure
                // (customerSwapMutation resets to empty encoding; repair adds depot loads)
                // For safety:
                // repairSolution(child, data);
                // rebuildDepotLoads(child, data);
                PDPSolution tmp = decode_solution(child, data, false);
                (void)tmp;
            } else if (prob(gen) < 0.2) {
                // Occasional drone mutation even without customer mutation
                droneResupplyMutation(child, data, gen);
                droneMutationCount++;
            }

            nextPop.push_back(std::move(child));
        }

        population.swap(nextPop);

        double prevBestCost = best.totalCost;
        double prevBestPenalty = best.totalPenalty;
        bool prevBestFeasible = best.isFeasible;

        PopStats genStats = evaluatePopulation(population, fitness);

        bool improved = false;
        if (!hasBest) {
            // Should not happen
        } else {
            if (best.isFeasible != prevBestFeasible) {
                if (best.isFeasible) improved = true;
            } else if (best.isFeasible) {
                if (best.totalCost < prevBestCost - 1e-5) improved = true;
            } else {
                double prevFit = prevBestCost + prevBestPenalty;
                double currentFit = best.totalCost + best.totalPenalty;
                if (currentFit < prevFit - 1e-5) improved = true;
            }
        }

        if (improved) {
            stagnation_counter = 0;
        } else {
            stagnation_counter++;
        }

        // Trigger Tabu Search on stagnation (10% of maxGenerations)
        bool tsTriggered = false;
        if (stagnation_counter >= (int)(0.1 * maxGenerations)) {
            int K = max(1, (int)(0.1 * population.size()));
            vector<pair<double, int>> indexedFitness(population.size());
            for (size_t i = 0; i < population.size(); ++i) {
                indexedFitness[i] = {fitness[i], (int)i};
            }
            sort(indexedFitness.begin(), indexedFitness.end());

            const double bestCostBeforeTS = best.totalCost;
            const bool feasibleBeforeTS   = best.isFeasible;

            cout << "[TS] gen=" << genIdx
                 << " stagnant=" << stagnation_counter
                 << " consecutiveFails=" << consecutiveTSFails
                 << " → running TS on " << K << " elites...\n";
            cout.flush();

            if (logStream) {
                (*logStream) << "[GA-TS] Stagnation reached at gen=" << genIdx
                             << " (" << stagnation_counter << " gens stagnant). Running TS on "
                             << K << " elite individuals...\n";
            }

            double globalBestFit = best.totalCost + best.totalPenalty;
            for (int i = 0; i < K; ++i) {
                int idx = indexedFitness[i].second;
                SolutionEncoding refined = tabuSearchPDP(population[idx], data, 50, globalBestFit, 7);
                population[idx] = refined;
            }

            stagnation_counter = 0;
            tsTriggered = true;

            // Re-evaluate population after Tabu Search updates
            genStats = evaluatePopulation(population, fitness);

            // Track consecutive TS failures (no improvement)
            const bool tsImproved = (best.isFeasible && (!feasibleBeforeTS ||
                                      best.totalCost < bestCostBeforeTS - 0.5));
            if (tsImproved) {
                consecutiveTSFails = 0;
            } else {
                consecutiveTSFails++;
            }

            cout << "[TS] done → BestCmax=" << fixed << setprecision(2) << best.totalCost
                 << " Penalty=" << best.totalPenalty
                 << " Feasible=" << (best.isFeasible ? "YES" : "NO")
                 << (tsImproved ? " *IMPROVED*" : " (no improvement)") << "\n";
            cout.flush();

            if (logStream) {
                (*logStream) << "[GA-TS] TS complete. New global best Cmax=" << best.totalCost
                             << " Penalty=" << best.totalPenalty
                             << " Feasible=" << (best.isFeasible ? 1 : 0)
                             << " improved=" << (tsImproved ? 1 : 0) << "\n";
            }

            // --- RESTART: after 3 consecutive TS failures, diversify population ---
            if (consecutiveTSFails >= 3) {
                const int keepCount  = max(1, populationSize / 10);
                const int freshCount = populationSize - keepCount;

                // Sort population by fitness, keep top keepCount
                vector<SolutionEncoding> newPop;
                newPop.reserve(populationSize);
                for (int i = 0; i < keepCount; ++i)
                    newPop.push_back(population[indexedFitness[i].second]);
                // Always ensure global best is in elite
                newPop[0] = best.encoded_routes;

                // Inject fresh random individuals
                auto freshInds = initStructuredPopulationPDP(freshCount, data, ++restartSeed);
                for (auto& ind : freshInds) newPop.push_back(std::move(ind));

                population.swap(newPop);
                consecutiveTSFails = 0;

                // Re-evaluate after restart
                genStats = evaluatePopulation(population, fitness);

                cout << "[RESTART] gen=" << genIdx
                     << " kept=" << keepCount << " fresh=" << freshCount
                     << " seed=" << restartSeed << "\n";
                cout.flush();
                if (logStream)
                    (*logStream) << "[GA] RESTART gen=" << genIdx
                                 << " kept=" << keepCount << " fresh=" << freshCount << "\n";
            }
        }

        // Per-gen stdout log (every gen)
        {
            string note;
            if (tsTriggered)  note += "[TS]";
            if (improved)     note += " *IMPROVED*";
            if (best.isFeasible && improved) note += " [FEASIBLE]";

            cout << setw(6)  << genIdx
                 << setw(10) << fixed << setprecision(2) << best.totalCost
                 << setw(10) << fixed << setprecision(2) << best.totalPenalty
                 << setw(8)  << genStats.feasibleCount << "/" << left << setw(4) << populationSize
                 << setw(10) << fixed << setprecision(2) << right << genStats.avgFitness
                 << setw(8)  << stagnation_counter
                 << setw(8)  << customerMutationCount
                 << setw(8)  << droneMutationCount
                 << "  " << note << "\n";
            cout.flush();
        }

        if (logStream && (genIdx % logEvery == 0 || genIdx + 1 == maxGenerations)) {
            (*logStream) << "[GA] gen=" << genIdx
                         << " feasible=" << genStats.feasibleCount << "/" << population.size()
                         << " bestFitness=" << genStats.bestFitness
                         << " bestCmax=" << genStats.bestCost
                         << " bestPenalty=" << genStats.bestPenalty
                         << " avgFitness=" << genStats.avgFitness
                         << " children=" << madeChildren
                         << " twoPoint=" << usedTwoPointCount
                         << " mutCustomer=" << customerMutationCount
                         << " mutDrone=" << droneMutationCount
                         << " globalBestCmax=" << best.totalCost
                         << " globalBestPenalty=" << best.totalPenalty
                         << " globalBestFeasible=" << (best.isFeasible ? 1 : 0)
                         << "\n";
        }

        // --- Early stopping ---
        const double elapsed = chrono::duration<double>(
            chrono::high_resolution_clock::now() - gaStartTime).count();

        if (best.isFeasible && best.totalCost <= lowerBound + 0.5) {
            cout << "\n[GA] *** OPTIMAL (LB reached) ***"
                 << "  Cmax=" << fixed << setprecision(2) << best.totalCost
                 << "  LB=" << lowerBound
                 << "  gen=" << genIdx
                 << "  elapsed=" << setprecision(2) << elapsed << "s\n";
            if (logStream)
                (*logStream) << "[GA] STOP early: LB reached. Cmax=" << best.totalCost
                             << " LB=" << lowerBound << " gen=" << genIdx << "\n";
            break;
        }

        if (timeLimitSeconds > 0.0 && elapsed >= timeLimitSeconds) {
            cout << "\n[GA] *** TIME LIMIT reached ***"
                 << "  elapsed=" << fixed << setprecision(2) << elapsed << "s"
                 << "  limit=" << setprecision(0) << timeLimitSeconds << "s"
                 << "  gen=" << genIdx
                 << "  Cmax=" << setprecision(2) << best.totalCost << "\n";
            if (logStream)
                (*logStream) << "[GA] STOP early: time limit. elapsed=" << elapsed
                             << "s gen=" << genIdx << " Cmax=" << best.totalCost << "\n";
            break;
        }
    }

    return best;
}

void repairC2Pairs(vector<vector<int>>& routes, const PDPData& data) {
    for (int i = 1; i < data.numNodes; ++i) {
        if (data.nodeTypes[i] == "P") {
            int pairId = data.pairIds[i];
            int d = -1;
            for (int j = 1; j < data.numNodes; ++j) {
                if (data.pairIds[j] == pairId && data.nodeTypes[j] == "DL") {
                    d = j;
                    break;
                }
            }
            if (d == -1) continue;

            int route_p = -1, pos_p = -1;
            int route_d = -1, pos_d = -1;
            for (int rIdx = 0; rIdx < (int)routes.size(); ++rIdx) {
                for (int sIdx = 0; sIdx < (int)routes[rIdx].size(); ++sIdx) {
                    if (routes[rIdx][sIdx] == i) {
                        route_p = rIdx;
                        pos_p = sIdx;
                    }
                    if (routes[rIdx][sIdx] == d) {
                        route_d = rIdx;
                        pos_d = sIdx;
                    }
                }
            }

            if (route_p == -1 || route_d == -1) continue;

            if (route_p != route_d) {
                routes[route_d].erase(routes[route_d].begin() + pos_d);
                routes[route_p].insert(routes[route_p].begin() + pos_p + 1, d);
            } else if (pos_p > pos_d) {
                swap(routes[route_p][pos_p], routes[route_p][pos_d]);
            }
        }
    }
}
