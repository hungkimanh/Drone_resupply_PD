#!/usr/bin/env python3
"""Reproducible, resumable local benchmark. No git operations or remote workflow runs."""
from __future__ import annotations
import argparse,concurrent.futures,csv,hashlib,importlib.util,json,os,shutil,subprocess,sys,time,zipfile
from decimal import Decimal
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from MILP import read_instance,require_docplex
from paper_milp import build_paper_milp
from benchmark_audit import read_encoding,read_paper_solution,export_audit


def dump(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value,indent=2,ensure_ascii=False));temporary.replace(path)


def references():
    result={}
    for label,file in [('milp','Truck-and-Drone (Limited Drone Fleet) MILP Model Results.csv'),
                       ('mh','Truck-and-Drone (Limited Drone Fleet) Matheuristic Results - 10 customers.csv')]:
        with (ROOT/'Results'/file).open(encoding='utf-8-sig',newline='') as f:
            for r in csv.DictReader(f):
                if r['Customers']=='10' and r['Depot location']=='Border' and r['Drone capacity']=='2' and r['Trucks']=='2' and r['Drones']=='2':
                    key=(str(Decimal(r['Beta']).normalize()),int(r['Instance']))
                    result.setdefault(key,{})[label]=float(r['Objective value'])
    return result


def instance_list():
    return [ROOT/'Instances 2'/f'U_10_{beta}_Num_{i}.txt' for beta in ('0.5','1.0') for i in range(1,6)]


def prepare(args):
    args.output.mkdir(parents=True,exist_ok=True)
    archive=Path('/tmp/pina_pardo_benchmark_audit/Instances.zip')
    if not archive.exists(): archive=args.output/'author_Instances.zip'
    if not archive.exists(): raise RuntimeError('Original author Instances.zip is required for independent data matching.')
    if archive.resolve() != (args.output/'author_Instances.zip').resolve():
        shutil.copy2(archive,args.output/'author_Instances.zip')
    z=zipfile.ZipFile(archive);checked=[]
    for path in instance_list():
        inst=read_instance(path,'border')
        original=[line.split() for line in z.read('Instances/'+path.name).decode().splitlines()[1:] if line.strip()]
        projection=[(Decimal(str(p.x)),Decimal(str(p.y)),Decimal(str(p.release))) for p in inst.nodes[1:]]
        expected=[(Decimal(p[1]),Decimal(p[2]),Decimal(p[0])) for p in original]
        if projection!=expected or any(p.typ!='D' for p in inst.nodes[1:]):raise RuntimeError('Author data mismatch: '+str(path))
        if inst.drone_capacity!=2 or inst.num_drones!=2 or inst.num_trucks!=2:raise RuntimeError('Fleet mismatch')
        checked.append({'instance':path.name,'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'customers':10,
                        'depot':'border','trucks':2,'drones':2,'drone_capacity':2})
    dump(args.output/'input_manifest.json',checked)


def case_metadata(path,method,args):
    parts=path.stem.split('_');baseline=references()[(str(Decimal(parts[2]).normalize()),int(parts[4]))]
    return {'instance':path.name,'method':method,'seed':1,'customers':10,'depot':'border','drone_capacity':2,
            'threads':args.threads,'time_limit_seconds':args.time,'paper_milp':baseline['milp'],
            'paper_matheuristic':baseline['mh'],'objective':None,'checkpoint_150':None,'status':'not_started'}


def milp_case(path,args):
    directory=args.output/path.stem/'milp';record_file=directory/'result.json'
    if record_file.exists():return json.loads(record_file.read_text())
    directory.mkdir(parents=True,exist_ok=True);record=case_metadata(path,'milp',args)
    inst=read_instance(path,'border');begin=time.monotonic()
    model=build_paper_milp(inst,require_docplex(),args.time,False)
    model.export_as_lp(str(directory/'model.lp'))
    record.update(variables=model.number_of_variables,constraints=model.number_of_constraints,build_seconds=time.monotonic()-begin)
    model.end()
    cmd=[str(args.cplex_runner),str(directory/'model.lp'),str(directory),str(args.time),str(args.threads)]
    dump(directory/'command.json',cmd)
    begin=time.monotonic()
    with (directory/'milp.log').open('w') as log:
        process=subprocess.run(cmd,cwd=directory,stdout=log,stderr=subprocess.STDOUT,timeout=args.time+120)
    record.update(exit_code=process.returncode,process_seconds=time.monotonic()-begin)
    metrics_path=directory/'solver_metrics.json'
    if metrics_path.exists():record.update(json.loads(metrics_path.read_text()))
    solution=directory/'milp_solution.sol'
    if solution.exists():
        routes,header,_=read_paper_solution(solution,inst)
        audit=export_audit(directory/'audit',routes,inst,header['objectiveValue'])
        if not audit['raw_replay']['valid'] or not audit['earliest_fixed_decisions']['valid']:
            record.update(status='physical_validation_failed',errors=audit['raw_replay']['errors']+audit['earliest_fixed_decisions']['errors'])
        else:
            record['raw_cplex_objective']=header['objectiveValue']
            record['objective']=audit['earliest_fixed_decisions']['objective']
            record['status']='verified_feasible'
            if record.get('best_bound',-1)>record['objective']+1e-4:raise RuntimeError('MILP bound exceeds independently feasible UB')
            record['certified_optimal']=abs(record['objective']-record.get('best_bound',-1))<=1e-5
    checkpoint=directory/'checkpoint_150.sol'
    if checkpoint.exists():
        routes,header,_=read_paper_solution(checkpoint,inst)
        audit=export_audit(directory/'audit_150',routes,inst,header['objectiveValue'])
        if audit['earliest_fixed_decisions']['valid']:
            record['checkpoint_150']=audit['earliest_fixed_decisions']['objective']
    if record.get('certified_optimal') and record.get('elapsed_seconds',args.time)>0 and record['elapsed_seconds']<150:
        record['checkpoint_150']=record['objective']
    dump(record_file,record);return record


def heuristic_case(path,method,args):
    directory=args.output/path.stem/method;record_file=directory/'result.json'
    if record_file.exists():return json.loads(record_file.read_text())
    directory.mkdir(parents=True,exist_ok=True);record=case_metadata(path,method,args)
    oracle_path=args.output/path.stem/'milp'/'result.json'
    bound=-1
    if oracle_path.exists():
        oracle=json.loads(oracle_path.read_text())
        if oracle.get('status')=='verified_feasible':bound=oracle.get('best_bound',-1)
    cmd=[str(args.solver),str(path),'--depot','1','--algorithm',method,'--threads',str(args.threads),
         '--time',str(args.time),'--run','1','--encoding-output',str(directory/'encoding.txt'),
         '--history-output',str(directory/'history.json'),'--certificate-bound',str(bound)]
    dump(directory/'command.json',cmd)
    begin=time.monotonic()
    with (directory/'search.log').open('w') as log:
        process=subprocess.run(cmd,cwd=directory,stdout=log,stderr=subprocess.STDOUT,timeout=args.time+120)
    record.update(exit_code=process.returncode,process_seconds=time.monotonic()-begin)
    if (directory/'history.json').exists():record.update(json.loads((directory/'history.json').read_text()))
    encoding=directory/'encoding.txt'
    if process.returncode==0 and encoding.exists():
        inst=read_instance(path,'border');routes,reported=read_encoding(encoding)
        audit=export_audit(directory/'audit',routes,inst,reported)
        if audit['raw_replay']['valid'] and audit['earliest_fixed_decisions']['valid']:
            record['search_objective']=reported
            record['objective']=audit['earliest_fixed_decisions']['objective'];record['status']='verified_feasible'
            record['checkpoint_150']=record.get('checkpoint_objective')
        else:record.update(status='physical_validation_failed',errors=audit['raw_replay']['errors'])
    checkpoint=directory/'checkpoint_150_encoding.txt'
    if checkpoint.exists():
        inst=read_instance(path,'border');routes,reported=read_encoding(checkpoint)
        audit=export_audit(directory/'audit_150',routes,inst,reported)
        if audit['raw_replay']['valid'] and audit['earliest_fixed_decisions']['valid']:
            record['checkpoint_150']=audit['earliest_fixed_decisions']['objective']
        else:
            record['checkpoint_validation_errors']=audit['raw_replay']['errors']
            record['checkpoint_150']=None
    dump(record_file,record);return record


def write_report(args):
    rows=[json.loads(p.read_text()) for p in sorted(args.output.glob('*/*/result.json'))]
    for r in rows:
        obj=r.get('objective')
        if r.get('status')=='verified_feasible' and obj is not None:
            r['difference_vs_paper_pct']=100*(obj-r['paper_milp'])/r['paper_milp']
    fields=['instance','method','status','objective','checkpoint_150','paper_milp','paper_matheuristic',
            'difference_vs_paper_pct','raw_cplex_objective','best_bound','relative_gap','certified_optimal',
            'elapsed_seconds','process_seconds','threads','time_limit_seconds','evaluations','cp_calls','cache_hits','fallbacks']
    with (args.output/'comparison.csv').open('w',encoding='utf-8-sig',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields,extrasaction='ignore');writer.writeheader();writer.writerows(rows)
    lines=['# Benchmark experiment report','',f'Completed records: {len(rows)}',
           '10 author-matched delivery-only instances; border depot; 2 trucks; 2 drones; capacity 2; seed 1.',
           f'Time limit: {args.time} seconds per method and instance; {args.threads} solver threads.',
           'All methods use the same corrected continuous-time physical semantics.',
           'A proven MILP lower bound is used only for certified early stopping, not to guide search.',
           'Differences below a published integer objective are not claimed as improvements of a proven optimum; original implementation precision is unverified.',
           '', '| Instance | Method | Verified objective | At 150 s | Paper MILP | Difference % | Status |',
           '|---|---|---:|---:|---:|---:|---|']
    for r in rows:
        fmt=lambda x:'' if x is None else f'{x:.6f}'
        lines.append(f'| {r["instance"]} | {r["method"]} | {fmt(r.get("objective"))} | {fmt(r.get("checkpoint_150"))} | {r["paper_milp"]} | {fmt(r.get("difference_vs_paper_pct"))} | {r["status"]} |')
    (args.output/'REPORT.md').write_text('\n'.join(lines)+'\n')
    return rows


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--phase',choices=['all','milp','heuristic','report'],default='all')
    parser.add_argument('--output',type=Path,default=ROOT/'report'/'benchmark_20261028')
    parser.add_argument('--time',type=float,default=3600)
    parser.add_argument('--threads',type=int,default=2)
    parser.add_argument('--jobs',type=int,default=8)
    parser.add_argument('--solver',type=Path,default=ROOT/'pdp_solver')
    parser.add_argument('--cplex-runner',type=Path,default=ROOT/'build-benchmark'/'cplex_benchmark_driver')
    parser.add_argument('--methods',nargs='+',default=['corrected','adaptive','memetic','full'])
    args=parser.parse_args();args.output=args.output.resolve()
    if args.jobs*args.threads>len(os.sched_getaffinity(0)):raise SystemExit('Requested CPU threads exceed available affinity')
    prepare(args)
    dump(args.output/'environment.json',{'python':sys.version,'available_cpus':len(os.sched_getaffinity(0)),
         'threads_per_solver':args.threads,'jobs':args.jobs,'time_limit_seconds':args.time,'seed':1,
         'sources':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
                    for p in list((ROOT/'src').glob('*'))+[ROOT/'MILP.py',ROOT/'paper_milp.py'] if p.is_file()}})
    phases=['milp','heuristic'] if args.phase=='all' else [args.phase]
    for phase in phases:
        if phase=='report':break
        tasks=[(p,None) for p in instance_list()] if phase=='milp' else [(p,m) for m in args.methods for p in instance_list()]
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.jobs) as pool:
            futures={pool.submit(milp_case,p,args) if phase=='milp' else pool.submit(heuristic_case,p,m,args):(p,m) for p,m in tasks}
            pending=set(futures)
            while pending:
                done,pending=concurrent.futures.wait(pending,timeout=30,return_when=concurrent.futures.FIRST_COMPLETED)
                if not done:print(f'RUNNING {phase}: {len(pending)} outstanding',flush=True)
                for future in done:
                    p,m=futures[future]
                    try:
                        r=future.result();print(f'DONE {r["instance"]} {r["method"]}: objective={r.get("objective")} status={r["status"]}',flush=True)
                    except Exception as error:
                        r=case_metadata(p,'milp' if phase=='milp' else m,args);r.update(status='execution_failed',error=repr(error))
                        dump(args.output/p.stem/r['method']/'result.json',r)
                        print('FAILED',p.name,m,repr(error),flush=True)
                    write_report(args)
    write_report(args)
if __name__=='__main__':main()
