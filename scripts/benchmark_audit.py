"""Independent physical replay of exported truck/drone decisions; no optimizer."""
from __future__ import annotations
import collections
import copy
import csv
import json
import math
import re
from pathlib import Path
import xml.etree.ElementTree as ET

TOL=1e-5


def read_encoding(path):
    routes=[]; current=None; reported=None
    for line in Path(path).read_text().splitlines():
        p=line.split()
        if not p: continue
        if p[0] in {'HEURISTIC_CMAX','MILP_CMAX'}: reported=float(p[1])
        elif p[0]=='ROUTE': current=[]; routes.append(current)
        elif p[0]=='STOP':
            marker=p.index('DRONE') if 'DRONE' in p else len(p)
            stop={'node':int(p[1]),'packages':list(map(int,p[2:marker]))}
            if marker<len(p):
                stop['drone']={'id':int(p[marker+1]),'trip':int(p[marker+2]),
                               'depart':float(p[marker+3]),'arrive':float(p[marker+4]),'return':float(p[marker+5])}
            current.append(stop)
        elif p[0]=='END_ROUTE': current=None
    return routes,reported


def read_paper_solution(path,inst):
    root=ET.parse(path).getroot(); header=root.find('header').attrib.copy()
    values={v.attrib['name']:float(v.attrib['value']) for v in root.findall('./variables/variable')}
    val=lambda prefix,*key:values.get(prefix+'_'+ '_'.join(map(str,key)),0.0)
    n=len(inst.nodes)-1; end=n+1
    order={}
    for d in range(inst.num_drones):
        if val('drone_used',d)<.5: continue
        successor={i:j for i in [0]+list(range(1,n+1)) for j in list(range(1,n+1))+[end]
                   if i!=j and val('w',d,i,j)>.5}
        h=successor[0]; r=1; visited=set()
        while h!=end:
            if h in visited: raise ValueError('drone subtour')
            visited.add(h); order[d,h]=r; r+=1; h=successor[h]
    routes=[]
    for v in range(inst.num_trucks):
        successor={i:j for i in [0]+list(range(1,n+1)) for j in list(range(1,n+1))+[end]
                   if i!=j and val('x',v,i,j)>.5}
        route=[{'node':0,'packages':[i for i in range(1,n+1) if val('gamma',i)>.5 and val('z',v,i)>.5]}]
        h=successor[0]; visited=set()
        while h!=end:
            if h in visited: raise ValueError('truck subtour')
            visited.add(h)
            if val('theta',h)>.5:
                route.append({'node':0,'packages':[i for i in range(1,n+1) if val('load',h,i)>.5]})
            active=[d for d in range(inst.num_drones) if val('u',d,h)>.5]
            if len(active)>1: raise ValueError('multiple drone rendezvous')
            packages=[] if not active else [i for i in range(1,n+1) if val('supply',active[0],h,i)>.5]
            stop={'node':h,'packages':packages}
            if active:
                d=active[0]
                stop['drone']={'id':d,'trip':order[d,h],'depart':val('dr_dep',d,h),
                               'arrive':val('dr_arr',d,h),'return':val('dr_ret',d,h)}
            route.append(stop); h=successor[h]
        route.append({'node':0,'packages':[]}); routes.append(route)
    header['objectiveValue']=float(header['objectiveValue'])
    for name in ['bestObjective','MIPRelativeGap','solutionTime']:
        if name in header: header[name]=float(header[name])
    return routes,header,values


def read_milp_solution(path,inst):
    """Read either the compact benchmark model or the positional PDP model."""
    root=ET.parse(path).getroot()
    values={v.attrib['name']:float(v.attrib['value']) for v in root.findall('./variables/variable')}
    if not any(re.fullmatch(r'y_\d+_\d+_\d+_\d+',name) for name in values):
        return read_paper_solution(path,inst)
    header=root.find('header').attrib.copy();header['objectiveValue']=float(header['objectiveValue'])
    val=lambda prefix,*key:values.get(prefix+'_'+'_'.join(map(str,key)),0.0)
    visits=collections.defaultdict(list)
    targets={}
    for name,value in values.items():
        if value<=.5:continue
        match=re.fullmatch(r'y_(\d+)_(\d+)_(\d+)_(\d+)',name)
        if match:
            v,m,p,node=map(int,match.groups());visits[v,m].append((p,node))
        match=re.fullmatch(r'w_(\d+)_(\d+)_(\d+)_(\d+)_(\d+)_(\d+)',name)
        if match:
            d,r,v,m,p,node=map(int,match.groups());targets[v,m,p]=(d,r,node)
    routes=[]
    c1=[n.idx for n in inst.nodes[1:] if n.typ=='D']
    for v in range(inst.num_trucks):
        route=[]
        for vv,m in sorted(visits):
            if vv!=v:continue
            route.append({'node':0,'packages':[i for i in c1 if val('load0',v,m,i)>.5]})
            for p,node in sorted(visits[v,m]):
                stop={'node':node,'packages':[i for i in c1 if val('drone_pkg',v,m,p,i)>.5]}
                if (v,m,p) in targets:
                    d,r,h=targets[v,m,p]
                    if h!=node:raise ValueError('rendezvous differs from truck node')
                    stop['drone']={'id':d,'trip':r,'depart':val('dr_dep',d,r),
                                   'arrive':val('dr_arr',d,r),'return':val('dr_ret',d,r)}
                route.append(stop)
        if not route:route.append({'node':0,'packages':[]})
        route.append({'node':0,'packages':[]});routes.append(route)
    return routes,header,values


def retime(routes,inst):
    """Longest paths in the fixed decision precedence graph, using exact travel."""
    routes=copy.deepcopy(routes); edges=[]; times=[0.0]; events=[]; drones=collections.defaultdict(list)
    def variable(): times.append(0.0); return len(times)-1
    for v,route in enumerate(routes):
        prev=0; node=0
        for k,stop in enumerate(route):
            h=stop['node']; travel=inst.truck_time[node][h]; depart=variable()
            if h==0:
                handling=inst.depot_receive if 0<k<len(route)-1 else 0
                edges.append((prev,depart,travel+handling))
                for p in stop['packages']:edges.append((0,depart,inst.nodes[p].release))
            elif not stop['packages']:
                edges.append((prev,depart,travel+inst.truck_service))
            else:
                dr=stop['drone']; launch=variable(); own=h in stop['packages']; fly=inst.drone_time[0][h]
                edges.append((prev,launch,travel+(0 if own else inst.truck_service)-fly))
                for p in stop['packages']: edges.append((0,launch,inst.nodes[p].release))
                edges.append((launch,depart,fly+inst.resupply_time+(inst.truck_service if own else 0)))
                event=(launch,v,k,h); events.append(event); drones[dr['id']].append((dr['trip'],event))
            prev=depart;node=h
    for sorties in drones.values():
        sorties.sort()
        if [r for r,_ in sorties]!=list(range(1,len(sorties)+1)): raise ValueError('invalid sortie numbering')
        for (_,p),(_,q) in zip(sorties,sorties[1:]):
            edges.append((p[0],q[0],2*inst.drone_time[0][p[3]]+inst.resupply_time+inst.drone_load))
    for _ in range(len(times)):
        changed=False
        for a,b,lag in edges:
            if times[b]+1e-9<times[a]+lag: times[b]=times[a]+lag; changed=True
        if not changed: break
    else: raise ValueError('positive truck/drone precedence cycle')
    for launch,v,k,h in events:
        dr=routes[v][k]['drone']; dr['depart']=times[launch]
        dr['arrive']=dr['depart']+inst.drone_time[0][h]
        dr['return']=dr['arrive']+inst.resupply_time+inst.drone_time[h][0]
    return routes


def replay(routes,inst,reported=None):
    errors=[]; timeline=[]; sorties=[]; summaries=[]; served=collections.Counter(); sources=collections.Counter()
    def check(ok,message):
        if not ok: errors.append(message)
    check(len(routes)==inst.num_trucks,'wrong number of truck routes')
    delivery_only=inst.truck_capacity>=len(inst.nodes)-1 and all(n.typ=='D' for n in inst.nodes[1:])
    for v,route in enumerate(routes):
        check(route and route[0]['node']==0 and route[-1]['node']==0,f'truck {v}: missing depot endpoint')
        t=0.0;previous=0;stock=set(); picked=set()
        totals=collections.Counter()
        for k,stop in enumerate(route):
            h=stop['node']; packages=stop['packages']; travel=inst.truck_time[previous][h]; arrival=t+travel
            handling=service=wait=0.0
            if h==0:
                handling=inst.depot_receive if 0<k<len(route)-1 else 0.0
                if handling and not delivery_only:
                    check(not any(inst.nodes[p].typ=='D' for p in stock),f'C1 freight crosses a depot return on truck {v}')
                available=max([inst.nodes[p].release for p in packages]+[0.0])
                depart=max(arrival+handling,available);wait=depart-arrival-handling
                for p in packages:
                    check(inst.nodes[p].typ=='D',f'C2 package {p} loaded at depot')
                    check(p not in stock,f'package {p} loaded twice')
                    sources[p]+=1;stock.add(p)
            else:
                served[h]+=1; service=inst.truck_service; own=h in packages
                if not own: check(inst.nodes[h].typ!='D' or h in stock,f'package {h} not on truck')
                def serve_stock():
                    typ=inst.nodes[h].typ
                    if typ=='P': stock.add(h);picked.add(inst.nodes[h].pair_id)
                    elif typ=='DL':
                        partner=next((n.idx for n in inst.nodes[1:] if n.typ=='P' and n.pair_id==inst.nodes[h].pair_id),None)
                        check(partner in stock and inst.nodes[h].pair_id in picked,f'pair precedence at {h}')
                        stock.discard(partner)
                    else:
                        check(h in stock,f'package missing at service {h}');stock.discard(h)
                    check(len(stock)<=inst.truck_capacity,f'truck capacity after service at {h}')
                # When the own parcel is already onboard, service precedes the
                # handover. Capacity must follow this same physical chronology.
                if packages and not own: serve_stock()
                if packages:
                    dr=stop.get('drone');check(dr is not None,f'missing drone at {h}')
                    if dr is None: continue
                    check(0<=dr['id']<inst.num_drones,f'invalid drone ID at {h}')
                    check(len(packages)<=inst.drone_capacity,f'drone overload at {h}')
                    check(len(packages)==len(set(packages)),f'duplicate package in sortie at {h}')
                    ready=arrival+(0 if own else service); fly=inst.drone_time[0][h]
                    check(dr['depart']+TOL>=max(inst.nodes[p].release for p in packages),f'drone release at {h}')
                    check(abs(dr['arrive']-dr['depart']-fly)<TOL,f'drone flight at {h}')
                    check(dr['arrive']+TOL>=ready,f'drone arrives before truck ready at {h}')
                    check(abs(dr['return']-dr['arrive']-inst.resupply_time-fly)<TOL,f'drone return at {h}')
                    check(dr['return']-dr['depart']<=inst.drone_endurance+TOL,f'drone endurance at {h}')
                    handling=inst.resupply_time; wait=max(0.0,dr['arrive']-ready)
                    depart=arrival+service+handling+wait
                    for p in packages:
                        check(inst.nodes[p].typ=='D',f'C2 package {p} supplied by drone')
                        check(p not in stock,f'package {p} supplied twice')
                        sources[p]+=1;stock.add(p)
                    check(len(stock)<=inst.truck_capacity,f'truck capacity before service at {h}')
                    sorties.append(dict(dr,truck=v,node=h,packages=packages))
                else: depart=arrival+service
                if not packages or own: serve_stock()
            check(len(stock)<=inst.truck_capacity,f'truck capacity at {h}')
            totals.update(travel=travel,service=service,handling=handling,waiting=wait)
            timeline.append({'truck':v,'stop':k,'node':h,'packages':json.dumps(packages),'arrival':arrival,
                             'travel':travel,'service':service,'handling':handling,'waiting':wait,'departure':depart})
            t=depart;previous=h
        check(not stock,f'truck {v} has undelivered freight')
        check(abs(sum(totals.values())-t)<TOL,f'truck {v} component sum differs')
        summaries.append(dict(truck=v,return_time=t,**totals))
    for d in range(inst.num_drones):
        events=sorted((s for s in sorties if s['id']==d),key=lambda s:s['trip'])
        check([e['trip'] for e in events]==list(range(1,len(events)+1)),f'drone {d} sortie numbering')
        for a,b in zip(events,events[1:]): check(b['depart']+TOL>=a['return']+inst.drone_load,f'drone {d} overlaps')
    for n in inst.nodes[1:]:
        check(served[n.idx]==1,f'customer {n.idx} visited {served[n.idx]} times')
        if n.typ=='D': check(sources[n.idx]==1,f'package {n.idx} sourced {sources[n.idx]} times')
    cmax=max((s['return_time'] for s in summaries),default=0)
    if reported is not None: check(reported+TOL>=cmax,'reported objective is below physical makespan')
    return {'valid':not errors,'errors':errors,'objective':cmax,'reported_objective':reported,
            'reported_minus_replayed':None if reported is None else reported-cmax,
            'trucks':summaries,'sorties':sorties,'timeline':timeline}


def export_audit(directory,routes,inst,reported=None):
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    raw=replay(routes,inst,reported)
    earliest_routes=retime(routes,inst)
    canonical=replay(earliest_routes,inst,reported)
    result={'raw_replay':raw,'earliest_fixed_decisions':canonical}
    (directory/'audit.json').write_text(json.dumps(result,indent=2,ensure_ascii=False))
    for label,report in [('raw',raw),('earliest',canonical)]:
        with (directory/f'{label}_timeline.csv').open('w',encoding='utf-8-sig',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(report['timeline'][0]))
            writer.writeheader();writer.writerows(report['timeline'])
    lines=['# Explicit objective calculation','',f'Reported objective: {reported}',
           f'Raw fixed-drone-time replay: {raw["objective"]:.9f}',
           f'Earliest schedule of the same discrete decisions: {canonical["objective"]:.9f}',
           f'Physical validation: {canonical["valid"]}','','Per-truck identity: travel + service + handling + waiting = return time.','']
    for tr in canonical['trucks']:
        lines.append(f'Truck {tr["truck"]}: {tr["travel"]:.9f} + {tr["service"]:.9f} + {tr["handling"]:.9f} + {tr["waiting"]:.9f} = {tr["return_time"]:.9f}')
    lines+=['',f'C_max = max(truck return times) = {canonical["objective"]:.9f}']
    (directory/'objective_calculation.md').write_text('\n'.join(lines)+'\n')
    return result
