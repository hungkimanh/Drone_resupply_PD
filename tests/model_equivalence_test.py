"""Compare compact and positional MILPs on analytically solvable instances.

Run with the licensed CPLEX C-API driver built as documented in the report.
"""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from MILP import Instance,Node,build_milp,manhattan,euclidean
RUNNER=Path(os.environ.get('CPLEX_BENCHMARK_RUNNER',str(ROOT/'build-benchmark'/'cplex_benchmark_driver')))

@unittest.skipUnless(RUNNER.exists(),'licensed CPLEX benchmark driver is unavailable')
class FormulationTests(unittest.TestCase):
    def test_handover_peak_capacity(self):
        nodes=[Node(0,0,0,'DEPOT',0,0),Node(1,1,0,'D',0,0),Node(2,2,0,'D',0,0)]
        inst=Instance(nodes,[[2*manhattan(a,b) for b in nodes] for a in nodes],
                      [[euclidean(a,b) for b in nodes] for a in nodes],
                      num_trucks=1,num_drones=1,truck_capacity=1)
        model=build_milp(inst,5,False,1)
        # At node 1 both parcels arrive together. Delivering parcel 1 would
        # reduce the final load to one, but the handover peak is two.
        model.add_constraint(model._pdp_vars['b'][0,1,1,1]==1)
        model.add_constraint(model._pdp_vars['b'][0,1,1,2]==1)
        model.add_constraint(model._pdp_vars['y'][0,1,1,1]==1)
        with tempfile.TemporaryDirectory() as temp:
            directory=Path(temp);model.export_as_lp(str(directory/'model.lp'));model.end()
            with (directory/'log.txt').open('w') as log:
                result=subprocess.run([str(RUNNER),str(directory/'model.lp'),str(directory),'5','1'],stdout=log,stderr=subprocess.STDOUT,timeout=15)
            metrics=json.loads((directory/'solver_metrics.json').read_text())
            self.assertEqual(result.returncode,6,(directory/'log.txt').read_text())
            self.assertIsNone(metrics['objective'])
            self.assertEqual(metrics['cplex_status'],103)

    def test_known_optima(self):
        cases=[([(1,0,50)],57), ([(1,0,0),(0,1,0)],7), ([(1,0,20),(0,1,0)],27)]
        with tempfile.TemporaryDirectory() as temp:
            for number,(points,expected) in enumerate(cases):
                nodes=[Node(0,0,0,'DEPOT',0,0)]+[Node(i,x,y,'D',release,0) for i,(x,y,release) in enumerate(points,1)]
                inst=Instance(nodes,[[2*manhattan(a,b) for b in nodes] for a in nodes],
                              [[euclidean(a,b) for b in nodes] for a in nodes])
                for formulation in ['compact','position']:
                    with self.subTest(case=number,formulation=formulation):
                        directory=Path(temp)/f'{number}_{formulation}';directory.mkdir()
                        model=build_milp(inst,5,False,None if formulation=='compact' else len(points))
                        model.export_as_lp(str(directory/'model.lp'));model.end()
                        with (directory/'log.txt').open('w') as log:
                            result=subprocess.run([str(RUNNER),str(directory/'model.lp'),str(directory),'5','1'],stdout=log,stderr=subprocess.STDOUT,timeout=15)
                        self.assertEqual(result.returncode,0,(directory/'log.txt').read_text())
                        metrics=json.loads((directory/'solver_metrics.json').read_text())
                        self.assertAlmostEqual(metrics['objective'],expected,places=5)
                        self.assertAlmostEqual(metrics['best_bound'],expected,places=5)

if __name__=='__main__':unittest.main()
