#!/usr/bin/env python3
"""Replay every final/150 s solution and verify benchmark metadata, without solving."""
from __future__ import annotations
import argparse,collections,hashlib,json,math,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from MILP import read_instance
from benchmark_audit import read_encoding,read_milp_solution,replay,retime


def main():
    parser=argparse.ArgumentParser();parser.add_argument('directory',type=Path);args=parser.parse_args()
    root=args.directory.resolve();records=[(p,json.loads(p.read_text())) for p in sorted(root.glob('*/*/result.json'))]
    methods=collections.Counter(r['method'] for _,r in records)
    if methods!={'milp':10,'corrected':10,'adaptive':10,'memetic':10,'full':10}:
        raise SystemExit(f'Expected the complete 50-run batch, got {dict(methods)}')
    checked=[]
    oracle={r['instance']:r for _,r in records if r['method']=='milp'}
    for path,record in records:
        assert record['status']=='verified_feasible' and record['exit_code']==0,path
        assert record['time_limit_seconds']==3600 and record['threads']==2,path
        assert record['elapsed_seconds']<=3600.5,path
        inst=read_instance(ROOT/'Instances 2'/record['instance'],'border')
        directory=path.parent
        for checkpoint,field in [(False,'objective'),(True,'checkpoint_150')]:
            if record['method']=='milp':
                file=directory/('checkpoint_150.sol' if checkpoint else 'milp_solution.sol')
                routes,header,_=read_milp_solution(file,inst);reported=header['objectiveValue']
            else:
                file=directory/('checkpoint_150_encoding.txt' if checkpoint else 'encoding.txt')
                routes,reported=read_encoding(file)
            raw=replay(routes,inst,reported);earliest=replay(retime(routes,inst),inst,reported)
            assert raw['valid'] and earliest['valid'],(file,raw['errors'],earliest['errors'])
            assert math.isfinite(record[field]) and abs(earliest['objective']-record[field])<1e-5,(file,record[field],earliest['objective'])
            assert record[field]+1e-5>=oracle[record['instance']]['best_bound'],file
            checked.append({'instance':record['instance'],'method':record['method'],
                            'checkpoint':150 if checkpoint else 'final','objective':earliest['objective']})
        assert record['checkpoint_150']+1e-5>=record['objective'],path
    for row in json.loads((root/'input_manifest.json').read_text()):
        actual=hashlib.sha256((ROOT/'Instances 2'/row['instance']).read_bytes()).hexdigest()
        assert actual==row['sha256'],row['instance']
    report={'valid':True,'runs':len(records),'replayed_solutions':len(checked),
            'inputs_unchanged':10,'methods':dict(methods),'details':checked}
    directory=root/'validation';directory.mkdir(exist_ok=True)
    (directory/'final_output_audit.json').write_text(json.dumps(report,indent=2))
    print(f'All {len(records)} runs / {len(checked)} final and checkpoint solutions pass independent replay; 10 original inputs unchanged.')


if __name__=='__main__':main()
