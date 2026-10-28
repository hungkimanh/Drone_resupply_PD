"""Analytical timeline and adversarial validation checks (no solver needed)."""
import copy
import sys
import unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
from MILP import Instance,Node
from benchmark_audit import replay,retime


class PhysicalReplayTests(unittest.TestCase):
    def setUp(self):
        self.inst=Instance([Node(0,0,0,'DEPOT',0,0),Node(1,1,0,'D',50,0)],
                           [[0,10],[10,0]],[[0,5],[5,0]],num_trucks=1,num_drones=1)
        self.routes=[[{'node':0,'packages':[]},
                      {'node':1,'packages':[1],'drone':{'id':0,'trip':1,'depart':50,'arrive':55,'return':65}},
                      {'node':0,'packages':[]}]]
    def test_release_and_component_sum(self):
        result=replay(self.routes,self.inst,73)
        self.assertTrue(result['valid'],result['errors'])
        self.assertAlmostEqual(result['objective'],73)
        truck=result['trucks'][0]
        self.assertAlmostEqual(truck['travel']+truck['service']+truck['handling']+truck['waiting'],73)
    def test_early_departure_rejected(self):
        routes=copy.deepcopy(self.routes)
        routes[0][1]['drone'].update(depart=49,arrive=54,**{'return':64})
        self.assertFalse(replay(routes,self.inst)['valid'])
    def test_underreported_objective_rejected(self):
        self.assertFalse(replay(self.routes,self.inst,72.9)['valid'])
    def test_fixed_decisions_retiming(self):
        routes=copy.deepcopy(self.routes)
        routes[0][1]['drone'].update(depart=60,arrive=65,**{'return':75})
        earliest=retime(routes,self.inst)
        self.assertAlmostEqual(replay(earliest,self.inst)['objective'],73)
    def test_duplicate_package_rejected(self):
        routes=copy.deepcopy(self.routes);routes[0][1]['packages']=[1,1]
        self.assertFalse(replay(routes,self.inst)['valid'])

    def test_capacity_before_delivering_own_package(self):
        # The own package is unloaded during service, after the drone handover.
        inst=copy.deepcopy(self.inst)
        inst.nodes.append(Node(2,1,0,'D',0,0))
        inst.truck_time=[[0,10,10],[10,0,0],[10,0,0]]
        inst.drone_time=[[0,5,5],[5,0,0],[5,0,0]]
        inst.truck_capacity=1
        routes=copy.deepcopy(self.routes)
        routes[0][0]['packages']=[2]
        routes[0].insert(-1,{'node':2,'packages':[]})
        self.assertFalse(replay(routes,inst)['valid'])

    def test_shared_drone_turnaround(self):
        inst=copy.deepcopy(self.inst)
        inst.num_trucks=2
        inst.nodes.append(Node(2,1,0,'D',0,0))
        inst.truck_time=[[0,10,10],[10,0,0],[10,0,0]]
        inst.drone_time=[[0,5,5],[5,0,0],[5,0,0]]
        routes=copy.deepcopy(self.routes)
        routes.append([{'node':0,'packages':[]},
          {'node':2,'packages':[2],'drone':{'id':0,'trip':2,'depart':66,'arrive':71,'return':81}},
          {'node':0,'packages':[]}])
        self.assertFalse(replay(routes,inst)['valid'])
        routes[1][1]['drone'].update(depart=70,arrive=75,**{'return':85})
        self.assertTrue(replay(routes,inst)['valid'])

    def test_service_frees_capacity_before_handover(self):
        inst=copy.deepcopy(self.inst)
        inst.nodes.append(Node(2,1,0,'D',0,0))
        inst.truck_time=[[0,10,10],[10,0,0],[10,0,0]]
        inst.drone_time=[[0,5,5],[5,0,0],[5,0,0]]
        inst.truck_capacity=1
        routes=[[{'node':0,'packages':[1]},
          {'node':1,'packages':[2],'drone':{'id':0,'trip':1,'depart':58,'arrive':63,'return':73}},
          {'node':2,'packages':[]},{'node':0,'packages':[]}]]
        self.assertTrue(replay(routes,inst)['valid'])

if __name__=='__main__':unittest.main()
