/**
 * @file main.cpp
 * @brief Entry point for TS-ALNS-LSP solver (3-layer architecture).
 *
 * Layer 0: GA init population → best SolutionEncoding → extractVRPSolution
 * Layer 1: Tabu Search on VRPSolution (routing only), ALNS escape
 * Layer 2: CP-SAT LSP evaluates each candidate routing (loading)
 *
 * Usage: ./pdp_solver <instance_file> [options]
 *   --depot 0|1|2   depot mode (1=border default for bundled instances)
 *   --iter N        optional TS iteration cap (default INT_MAX; time limit binds)
 *   --lsp T         CP-SAT time limit per call in seconds (default 3.0)
 *   --run K         run number / random seed
 *   --time T        total time limit in seconds
 *   --truck-cap Q   override truck capacity for experiments
 *   --drones D      override the number of available drones
 *   --check-diverse  evaluate the built-in U_10 multi-trip test fixture
 */

#include <chrono>
#include <cmath>
#include <thread>
#include <filesystem>
#include "pdp_memetic.h"
#include <algorithm>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>

#include "pdp_ga.h"         // extractCustomerRoutes
#include "pdp_init.h"       // initStructuredPopulationPDP
#include "pdp_reader.h"
#include "pdp_tabu_main.h"  // tabuSearch, TSConfig, TSResult
#include "pdp_types.h"
#include "pdp_utils.h"
#include "pdp_validation.h"
#include "pdp_fitness.h"    // decode_solution, print_decoded_routes

using namespace std;

static void writeEncodingFile(const string& path, const PDPSolution& sol) {
    SolutionEncoding encoded = sol.encoded_routes;
    vector<int> sortie(sol.resupply_events.size(), 0);
    for (int d = 0; d < (int)sol.drone_completion_times.size(); ++d) {
        vector<int> ids;
        for (int e = 0; e < (int)sol.resupply_events.size(); ++e)
            if (sol.resupply_events[e].drone_id == d) ids.push_back(e);
        sort(ids.begin(), ids.end(), [&](int a, int b) {
            return sol.resupply_events[a].drone_depart_time < sol.resupply_events[b].drone_depart_time;
        });
        for (int r = 0; r < (int)ids.size(); ++r) sortie[ids[r]] = r + 1;
    }
    for (int e = 0; e < (int)sol.resupply_events.size(); ++e) {
        const auto& ev = sol.resupply_events[e];
        for (auto& st : encoded[ev.truck_id]) {
            if (st.node == ev.resupply_point && st.packages == ev.customer_ids &&
                st.fixed_drone_id < 0) {
                st.fixed_drone_id = ev.drone_id;
                st.fixed_drone_trip = sortie[e];
                st.fixed_drone_depart = ev.drone_depart_time;
                st.fixed_drone_arrive = ev.drone_arrive_time;
                st.fixed_drone_return = ev.drone_return_time;
                break;
            }
        }
    }

    ofstream out(path);
    out << fixed << setprecision(12) << "HEURISTIC_CMAX " << sol.totalCost << "\n";
    for (int k = 0; k < (int)encoded.size(); ++k) {
        out << "ROUTE " << k << "\n";
        for (const auto& st : encoded[k]) {
            out << "STOP " << st.node;
            for (int pkg : st.packages) out << " " << pkg;
            if (st.fixed_drone_id >= 0) {
                out << " DRONE " << st.fixed_drone_id << " " << st.fixed_drone_trip
                    << " " << st.fixed_drone_depart << " " << st.fixed_drone_arrive
                    << " " << st.fixed_drone_return;
            }
            out << "\n";
        }
        out << "END_ROUTE\n";
    }
}

static SolutionEncoding diverseU10Fixture() {
    return {
        {
            RouteStop(0),
            RouteStop(4, {3}),
            RouteStop(3),
            RouteStop(0, {5, 1}),
            RouteStop(10),
            RouteStop(5),
            RouteStop(1, {2}),
            RouteStop(2),
            RouteStop(0),
        },
        {
            RouteStop(0, {9}),
            RouteStop(9),
            RouteStop(0, {6}),
            RouteStop(6),
            RouteStop(7, {7, 8}),
            RouteStop(8),
            RouteStop(0),
        },
    };
}

int main(int argc, char* argv[]) {
    auto t_start = chrono::steady_clock::now();

    if (argc < 2) {
        cerr << "Usage: " << argv[0] << " <instance_file> [--depot 0|1|2]"
             << " [--iter N] [--lsp T] [--run K] [--time T]\n";
        return 1;
    }

    const string instanceFile = argv[1];
    int depotMode    = 1;
    int maxIter      = std::numeric_limits<int>::max();
    double lspCutoff = 3.0;
    int runNumber    = 1;
    double timeLimit = -1.0;
    int truckCapacityOverride = -1;
    int numDronesOverride = -1;
    string encodingOutput;
    bool checkDiverse = false;
    int droneCapacityOverride = 0;
    int workers = std::max(1, int(std::thread::hardware_concurrency()));
    double certificateBound = -1;
    string algorithm = "adaptive", historyOutput;

    for (int i = 2; i < argc; ++i) {
        string arg = argv[i];
        if (arg == "--check-diverse") {
            checkDiverse = true;
            continue;
        }
        if ((arg == "--depot" || arg == "--iter" || arg == "--lsp" ||
             arg == "--run"   || arg == "--time" || arg == "--truck-cap" ||
             arg == "--drones" || arg == "--encoding-output" || arg == "--algorithm" ||
             arg == "--history-output" || arg == "--threads" || arg == "--drone-cap" ||
             arg == "--certificate-bound") && i + 1 < argc) {
            istringstream ss(argv[++i]);
            if      (arg == "--depot") ss >> depotMode;
            else if (arg == "--iter")  ss >> maxIter;
            else if (arg == "--lsp")   ss >> lspCutoff;
            else if (arg == "--run")   ss >> runNumber;
            else if (arg == "--time")  ss >> timeLimit;
            else if (arg == "--truck-cap") ss >> truckCapacityOverride;
            else if (arg == "--drones") ss >> numDronesOverride;
            else if (arg == "--encoding-output") encodingOutput = argv[i];
            else if (arg == "--history-output") historyOutput = argv[i];
            else if (arg == "--algorithm") algorithm = argv[i];
            else if (arg == "--threads") ss >> workers;
            else if (arg == "--drone-cap") ss >> droneCapacityOverride;
            else if (arg == "--certificate-bound") ss >> certificateBound;
        } else {
            cerr << "Unknown argument: " << arg << "\n";
            return 1;
        }
    }

    if (algorithm != "corrected" && algorithm != "adaptive" && algorithm != "memetic" && algorithm != "full") {
        cerr << "--algorithm must be corrected, adaptive, memetic or full\n"; return 1;
    }
    if (workers < 1 || droneCapacityOverride < 0) { cerr << "Invalid resource setting\n"; return 1; }
    if (depotMode < 0 || depotMode > 2) {
        cerr << "Invalid --depot value: " << depotMode << " (expected 0, 1, or 2)\n";
        return 1;
    }

    // ---- Load instance ----
    PDPData data;
    data.depotMode = depotMode;
    if (!readPDPFile(instanceFile, data)) {
        cerr << "Error: cannot read " << instanceFile << "\n";
        return 1;
    }
    if (truckCapacityOverride > 0) data.truckCapacity = truckCapacityOverride;
    // Match MILP.py's automatic choice: unrestricted C1 carry is the
    // delivery-only/nonbinding-capacity benchmark policy. The extended
    // positional PDP policy requires C1 delivery before the next depot visit.
    data.allowC1CarryAcrossDepot = data.truckCapacity >= data.numCustomers &&
        all_of(data.nodeTypes.begin()+1,data.nodeTypes.end(),[](const string& type) { return type=="D"; });
    if (numDronesOverride >= 0) data.numDrones = numDronesOverride;
    data.droneCapacity = droneCapacityOverride;
    data.lspWorkers = workers;
    showPDPInfo(data);

    if (checkDiverse) {
        if (data.numNodes != 11) {
            cerr << "--check-diverse requires the bundled 10-customer instance\n";
            return 1;
        }
        SolutionEncoding fixture = diverseU10Fixture();
        PDPSolution checked = decode_solution(fixture, data, false);
        printEncodedSolution(fixture);
        print_decoded_routes(checked, data);
        bool valid = validateSolution(checked, data, true);
        cout << "DIVERSE_FIXTURE_OBJECTIVE=" << fixed << setprecision(6)
             << checked.totalCost << "\n";
        return valid ? 0 : 2;
    }

    bool isSmall = (data.numCustomers <= 20);
    if (timeLimit < 0) timeLimit = isSmall ? 150.0 : 3600.0;

    const string depotNames[] = {"center", "border", "outside"};
    cout << "\n+========================================================+\n"
         << "|        PDP SOLVER -- TS-ALNS-LSP (CP-SAT)             |\n"
         << "+========================================================+\n"
         << "Instance : " << instanceFile << "\n"
         << "Depot    : " << depotNames[depotMode] << "\n"
         << "TS iters : " << maxIter << "\n"
         << "LSP cut  : " << lspCutoff << " s\n"
         << "Time lim : " << timeLimit << " s\n\n";

    // ============================================================
    // Layer 0: Init population → extract best VRPSolution
    // ============================================================
    const int POP_SIZE = isSmall ? 40 : 60;  // [Vidal HGS 2022] μ=25, μ+λ=65
    cout << "[L0] Building init population (size=" << POP_SIZE << ")...\n";
    vector<SolutionEncoding> population =
        initStructuredPopulationPDP(POP_SIZE, data, runNumber);

    // Pick best individual from population using greedy decode
    SolutionEncoding best_init_enc;
    double best_init_cost = numeric_limits<double>::infinity();
    for (const auto& enc : population) {
        PDPSolution sol = decode_solution(enc, data, /*throw=*/false);
        if (sol.isFeasible && sol.totalCost < best_init_cost) {
            best_init_cost = sol.totalCost;
            best_init_enc  = enc;
        }
    }
    if (best_init_enc.empty()) {
        // Fallback: take first individual
        best_init_enc = greedyLoading(data, extractVRPSolution(population[0]));
        auto fallback = decode_solution(best_init_enc, data, false);
        if (!fallback.isFeasible) { cerr << "No feasible initial solution\n"; return 2; }
        best_init_cost = fallback.totalCost;
    }

    VRPSolution init_vrp = extractVRPSolution(best_init_enc);
    cout << "[L0] Best init C_max=" << fixed << setprecision(2)
         << best_init_cost << "\n\n";

    // ============================================================
    // Layer 1+2: Tabu Search + CP-SAT LSP
    // ============================================================
    TSConfig cfg;
    cfg.lsp_cutoff = lspCutoff;
    cfg.seed       = runNumber * 137;
    cfg.verbose    = true;

    // [V3] Adaptive parameters based on problem size
    cfg.eta_max = isSmall ? 150 : 200;
    if (data.numCustomers >= 18 && cfg.lsp_cutoff < 5.0)
        cfg.lsp_cutoff = 5.0;

    SearchOptions options;
    options.cache = options.warm = options.adaptive = algorithm != "corrected";
    options.critical = algorithm == "full";
    options.lsp_seconds = cfg.lsp_cutoff;
    options.certificate_bound = certificateBound;
    SearchContext context(data, options, t_start, timeLimit);
    if (std::isfinite(best_init_cost)) {
        LSPResult initial;
        initial.feasible = true; initial.objective = best_init_cost;
        initial.encoding = best_init_enc; initial.status = "INITIAL_VERIFIED";
        context.consider(initial);
    }
    cfg.context = &context;
    cfg.time_limit_sec = context.remaining();
    cfg.max_iter = maxIter;
    TSResult ts;
    ts.iterations_done = 0;
    if (algorithm == "memetic" || algorithm == "full") {
        cout << "[SEARCH] Memetic GA + shared Tabu/LSP evaluator\n";
        memeticSearch(data, population, context, cfg, runNumber * 137);
    } else {
        cout << "[SEARCH] Tabu/LSP algorithm=" << algorithm << "\n";
        ts = tabuSearch(data, init_vrp, cfg);
    }
    ts.best_lsp = context.best;
    ts.best_obj = context.best.objective;
    if (!historyOutput.empty() && context.checkpoint_best.feasible) {
        auto checkpoint_path = std::filesystem::path(historyOutput).parent_path() / "checkpoint_150_encoding.txt";
        writeEncodingFile(checkpoint_path.string(), decode_solution(context.checkpoint_best.encoding, data, false));
    }
    if (!historyOutput.empty()) {
        ofstream out(historyOutput);
        out << std::setprecision(12);
        double checkpoint = std::numeric_limits<double>::infinity();
        for (const auto& h : context.history) if (h.first <= 150.0) checkpoint = h.second;
        out << "{\"algorithm\":\"" << algorithm << "\",\"time_limit_seconds\":" << timeLimit
            << ",\"elapsed_seconds\":" << context.elapsed()
            << ",\"checkpoint_seconds\":150,\"checkpoint_objective\":";
        if (std::isfinite(checkpoint)) out << checkpoint; else out << "null";
        out << ",\"objective\":";
        if (std::isfinite(ts.best_obj)) out << ts.best_obj; else out << "null";
        out << ",\"evaluations\":" << context.evaluations << ",\"cp_calls\":" << context.cp_calls
            << ",\"cache_hits\":" << context.cache_hits << ",\"fallbacks\":" << context.fallbacks
            << ",\"warm_reuses\":" << context.warm_reuses
            << ",\"adaptive_refinements\":" << context.adaptive_refinements
            << ",\"waiting_escapes\":" << context.waiting_escapes
            << ",\"threads\":" << workers << ",\"drone_capacity\":" << data.getDroneCapacity()
            << ",\"certificate_bound\":" << certificateBound
            << ",\"certified_stop\":" << (certificateBound >= 0 && ts.best_obj <= certificateBound + 1e-6 ? "true" : "false")
            << ",\"history\":[";
        bool first = true;
        for (const auto& h : context.history) {
            if (!first) out << ','; first = false;
            out << '[' << h.first << ',' << h.second << ']';
        }
        out << "]}\n";
    }

    // ============================================================
    // Results
    // ============================================================
    cout << "\n+========================================================+\n"
         << "|               TS-ALNS-LSP RESULT                      |\n"
         << "+========================================================+\n"
         << "TS iterations  : " << ts.iterations_done << "\n";

    if (ts.best_lsp.feasible && !ts.best_lsp.encoding.empty()) {
        PDPSolution final_sol = decode_solution(ts.best_lsp.encoding, data, false);
        bool valid = validateSolution(final_sol, data, /*verbose=*/true);
        double final_obj = valid ? final_sol.totalCost : numeric_limits<double>::infinity();

        cout << "C_max (verified): " << fixed << setprecision(2) << final_obj << " min\n";
        cout << "Feasible       : " << (final_sol.isFeasible ? "YES" : "NO") << "\n";
        cout << "Penalty        : " << final_sol.totalPenalty << "\n";

        if (!valid) {
            cout << "No verified feasible solution found.\n";
            double elapsed = chrono::duration<double>(
                chrono::steady_clock::now() - t_start).count();
            cout << "\n[RUNTIME] " << fixed << setprecision(2) << elapsed << " s\n";
            return 2;
        }

        cout << "\n[ENCODING]\n";
        printEncodedSolution(ts.best_lsp.encoding);

        cout << "\n[SOLUTION DETAIL]\n";
        print_decoded_routes(final_sol, data);
        printSolution(final_sol, data);
        printSolutionSummary(final_sol, best_init_cost, final_obj);
        if (!encodingOutput.empty()) writeEncodingFile(encodingOutput, final_sol);
    } else {
        cout << "No feasible solution found.\n";
    }

    double elapsed = chrono::duration<double>(
        chrono::steady_clock::now() - t_start).count();
    cout << "\n[RUNTIME] " << fixed << setprecision(2) << elapsed << " s\n";

    return 0;
}
