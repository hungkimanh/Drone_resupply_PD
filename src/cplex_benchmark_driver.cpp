// Optional CPLEX C-API runner: records real incumbent snapshots without restarting search.
#include <ilcplex/cplex.h>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <mutex>
#include <string>
#include <vector>

struct Recorder {
    int columns;
    double checkpoint=150, best=std::numeric_limits<double>::infinity();
    double checkpoint_objective=std::numeric_limits<double>::infinity();
    std::chrono::steady_clock::time_point start;
    std::vector<double> snapshot;
    std::mutex mutex;
    std::ofstream history;
};
static int CPXPUBLIC incumbent(CPXCALLBACKCONTEXTptr context, CPXLONG, void* handle) {
    auto& record=*static_cast<Recorder*>(handle);
    int is_point=0;
    int error=CPXcallbackcandidateispoint(context,&is_point);
    if (error || !is_point) return error;
    std::vector<double> x(record.columns); double objective;
    error=CPXcallbackgetcandidatepoint(context,x.data(),0,record.columns-1,&objective);
    if (error) return error;
    std::lock_guard<std::mutex> lock(record.mutex);
    double elapsed=std::chrono::duration<double>(std::chrono::steady_clock::now()-record.start).count();
    if (objective < record.best-1e-8) {
        record.best=objective;
        record.history<<std::setprecision(14)<<elapsed<<','<<objective<<'\n'; record.history.flush();
        if (elapsed<=record.checkpoint) {
            record.snapshot=std::move(x);record.checkpoint_objective=objective;
        }
    }
    return 0;
}
int main(int argc,char** argv) {
    if (argc<5) { std::cerr<<"Usage: cplex_benchmark_driver model.lp output_directory seconds threads\n";return 1; }
    const std::string output=argv[2]; double seconds=std::stod(argv[3]);int threads=std::stoi(argv[4]);
    int status=0;CPXENVptr env=CPXopenCPLEX(&status);
    if (!env) { std::cerr<<"CPXopenCPLEX failed "<<status<<'\n';return 2; }
    CPXLPptr lp=CPXcreateprob(env,&status,"benchmark");
    if ((status=CPXreadcopyprob(env,lp,argv[1],nullptr))) { std::cerr<<"CPXreadcopyprob failed "<<status<<'\n';return 3; }
    CPXsetintparam(env,CPX_PARAM_SCRIND,CPX_ON);
    CPXsetintparam(env,CPX_PARAM_THREADS,threads);
    CPXsetintparam(env,CPX_PARAM_CLOCKTYPE,2);
    CPXsetdblparam(env,CPX_PARAM_TILIM,seconds);
    CPXsetdblparam(env,CPX_PARAM_EPGAP,1e-9);
    CPXsetdblparam(env,CPX_PARAM_EPAGAP,1e-7);
    CPXsetdblparam(env,CPX_PARAM_WORKMEM,128);
    Recorder record;record.columns=CPXgetnumcols(env,lp);
    record.history.open(output+"/incumbents.csv");record.history<<"seconds,objective\n";
    status=CPXcallbacksetfunc(env,lp,CPX_CALLBACKCONTEXT_CANDIDATE,incumbent,&record);
    if (status) { std::cerr<<"CPXcallbacksetfunc failed "<<status<<'\n';return 4; }
    int surplus=0;
    CPXgetcolname(env,lp,nullptr,nullptr,0,&surplus,0,record.columns-1);
    std::vector<char> store(-surplus+1);std::vector<char*> names(record.columns);
    status=CPXgetcolname(env,lp,names.data(),store.data(),store.size(),&surplus,0,record.columns-1);
    if (status) { std::cerr<<"CPXgetcolname failed "<<status<<'\n';return 5; }
    record.start=std::chrono::steady_clock::now();
    int solve_error=CPXmipopt(env,lp);
    double elapsed=std::chrono::duration<double>(std::chrono::steady_clock::now()-record.start).count();
    double objective=0,bound=0,gap=0;
    bool feasible=CPXgetobjval(env,lp,&objective)==0;
    int solution_status=CPXgetstat(env,lp);
    CPXgetbestobjval(env,lp,&bound);
    CPXgetmiprelgap(env,lp,&gap);
    if (feasible) CPXsolwrite(env,lp,(output+"/milp_solution.sol").c_str());
    if (!record.snapshot.empty()) {
        std::ofstream sol(output+"/checkpoint_150.sol");sol<<std::setprecision(16);
        sol<<"<?xml version=\"1.0\"?><CPLEXSolution version=\"1.2\"><header objectiveValue=\""
           <<record.checkpoint_objective<<"\" solutionStatusString=\"feasible checkpoint\"/><variables>\n";
        for (int i=0;i<record.columns;++i)
            sol<<"<variable name=\""<<names[i]<<"\" value=\""<<record.snapshot[i]<<"\"/>\n";
        sol<<"</variables></CPLEXSolution>\n";
    }
    std::ofstream metrics(output+"/solver_metrics.json");metrics<<std::setprecision(16);
    metrics<<"{\"solver\":\"CPLEX\",\"version\":\""<<CPXversion(env)<<"\",\"cplex_status\":"<<solution_status
           <<",\"solve_error\":"<<solve_error<<",\"elapsed_seconds\":"<<elapsed<<",\"time_limit_seconds\":"<<seconds
           <<",\"threads\":"<<threads<<",\"objective\":";
    if (feasible) metrics<<objective;else metrics<<"null";
    metrics<<",\"best_bound\":"<<bound<<",\"relative_gap\":"<<gap<<",\"checkpoint_objective\":";
    if (!record.snapshot.empty()) metrics<<record.checkpoint_objective;else metrics<<"null";
    metrics<<"}\n";
    CPXfreeprob(env,&lp);CPXcloseCPLEX(&env);
    return solve_error || !feasible ? 6 : 0;
}
