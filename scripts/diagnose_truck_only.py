#!/usr/bin/env python3
"""Independent exact subset DP for small truck-only release-date instances.

Alternative conventions below are diagnostics, never benchmark parameters.
No MILP, heuristic evaluator or drone scheduler is used in the DP.
"""
from __future__ import annotations
import argparse,csv,json,math,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from MILP import read_instance


def completion_time(inst,travel=None,service=None):
    travel=inst.truck_time if travel is None else travel
    service=inst.truck_service if service is None else service
    n=len(inst.nodes)-1
    if n>12:raise ValueError('Exact diagnostic DP is limited to 12 customers.')
    if any(node.typ!='D' for node in inst.nodes[1:]):raise ValueError('Delivery-only DP.')
    if inst.num_trucks!=2 or inst.truck_capacity<n:raise ValueError('Two nonbinding-capacity trucks required.')
    full=(1<<n)-1
    path={};duration=[0.]*(full+1);release=[0.]*(full+1)
    # Held-Karp: shortest depot tour for each possible set of orders.
    for mask in range(1,full+1):
        release[mask]=max(inst.nodes[i+1].release for i in range(n) if mask>>i&1)
        for j in range(n):
            if not mask>>j&1:continue
            rest=mask^(1<<j)
            path[mask,j]=(min(path[rest,k]+travel[k+1][j+1]
                for k in range(n) if rest>>k&1) if rest else travel[0][j+1])+service
        duration[mask]=min(path[mask,j]+travel[j+1][0] for j in range(n) if mask>>j&1)
    # For each set, partition it into chronologically ordered depot trips.
    finish=[0.]*(full+1)
    for mask in range(1,full+1):
        best=release[mask]+duration[mask]
        trip=(mask-1)&mask
        while trip:
            best=min(best,max(finish[mask^trip]+inst.depot_receive,release[trip])+duration[trip])
            trip=(trip-1)&mask
        finish[mask]=best
    return min(max(finish[s],finish[full^s]) for s in range(full+1))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,default=ROOT/'report/benchmark_20261028/diagnostics/truck_only')
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    with (ROOT/'Results/Truck-Only MILP Model Results.csv').open(encoding='utf-8-sig',newline='') as f:
        references={(r['Beta'],r['Instance']):float(r['Objective value']) for r in csv.DictReader(f)
          if r['Customers']=='10' and r['Depot location']=='Border' and r['Trucks']=='2'}
    rows=[]
    for beta in ['0.5','1.0']:
        for number in range(1,6):
            name=f'U_10_{beta}_Num_{number}'
            inst=read_instance(ROOT/'Instances 2'/(name+'.txt'),'border')
            values={'physical_definition':completion_time(inst),
              'service_zero_DIAGNOSTIC':completion_time(inst,service=0),
              'floor_each_travel_DIAGNOSTIC':completion_time(inst,travel=[[math.floor(t) for t in row] for row in inst.truck_time]),
              'nearest_each_travel_DIAGNOSTIC':completion_time(inst,travel=[[round(t) for t in row] for row in inst.truck_time])}
            for label,quantize in [('floor_coordinates_DIAGNOSTIC',math.floor),('round_coordinates_DIAGNOSTIC',round)]:
                coords=[(quantize(node.x),quantize(node.y)) for node in inst.nodes]
                travel=[[2*(abs(a-c)+abs(b-d)) for c,d in coords] for a,b in coords]
                values[label]=completion_time(inst,travel=travel)
            values['published_truck_only']=references[beta,str(number)]
            rows.append({'instance':name,**values})
            if beta=='0.5' and number==1:
                (args.output/'independent_subset_dp.json').write_text(json.dumps(values,indent=2))
    with (args.output/'diagnostic_matrix.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    print(json.dumps(rows,indent=2))


if __name__=='__main__':main()
