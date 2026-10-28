"""Compact delivery-only VRPRD-DR formulation (Pina-Pardo et al., 2024).

Truck arcs define master tours; intermediate depot returns are node decisions.
This formulation is selected only for delivery-only, nonbinding truck capacity.
The positional pickup/delivery formulation remains available in MILP.py.
"""
from __future__ import annotations


def build_paper_milp(inst, Model, time_limit=3600, log_output=True):
    n = len(inst.nodes) - 1
    customers = list(range(1, n + 1))
    end = n + 1
    trucks, drones = list(range(inst.num_trucks)), list(range(inst.num_drones))
    tt = lambda i, j: inst.truck_time[0 if i == end else i][0 if j == end else j]
    dd = lambda i: inst.drone_time[0][i]
    eligible = [i for i in customers if 2*dd(i) + inst.resupply_time <= inst.drone_endurance + 1e-9]
    # A constructive all-at-depot tour is a safe feasible upper bound.
    horizon = max(p.release for p in inst.nodes) + sum(tt(a,b) for a,b in zip([0]+customers,customers+[end])) + n*inst.truck_service + 1
    big_m = 2*horizon + max(map(max,inst.truck_time)) + 2*max(map(max,inst.drone_time)) + 20
    mdl = Model(name="paper_vrprd_dr")
    mdl.set_time_limit(time_limit)
    mdl.context.solver.log_output = log_output
    arcs = [(i,j) for i in [0]+customers for j in customers+[end] if i != j]
    x = mdl.binary_var_dict(((v,i,j) for v in trucks for i,j in arcs),name="x")
    z = mdl.binary_var_dict(((v,i) for v in trucks for i in customers),name="z")
    theta = mdl.binary_var_dict(customers,name="theta")
    initial = mdl.binary_var_dict(customers,name="gamma")
    load = mdl.binary_var_dict(((i,j) for i in customers for j in customers),name="load")
    supply = mdl.binary_var_dict(((d,h,j) for d in drones for h in eligible for j in customers),name="supply")
    u = mdl.binary_var_dict(((d,h) for d in drones for h in eligible),name="u")
    drone_used = mdl.binary_var_dict(drones,name="drone_used")
    drone_arcs = [(i,j) for i in [0]+eligible for j in eligible+[end] if i != j and (i,j)!=(0,end)]
    w = mdl.binary_var_dict(((d,i,j) for d in drones for i,j in drone_arcs),name="w")
    start = mdl.continuous_var_dict(trucks,lb=0,ub=horizon,name="start")
    returned = mdl.continuous_var_dict(trucks,lb=0,ub=horizon,name="return")
    arrive = mdl.continuous_var_dict(customers,lb=0,ub=horizon,name="arr")
    depart = mdl.continuous_var_dict(customers,lb=0,ub=horizon,name="dep")
    dr_dep = mdl.continuous_var_dict(((d,h) for d in drones for h in eligible),lb=0,ub=horizon,name="dr_dep")
    dr_arr = mdl.continuous_var_dict(((d,h) for d in drones for h in eligible),lb=0,ub=horizon,name="dr_arr")
    dr_ret = mdl.continuous_var_dict(((d,h) for d in drones for h in eligible),lb=0,ub=horizon,name="dr_ret")
    cmax = mdl.continuous_var(lb=0,ub=horizon,name="C_max")
    for i in customers:
        mdl.add_constraint(mdl.sum(z[v,i] for v in trucks)==1)
    for v in trucks:
        mdl.add_constraint(mdl.sum(x[v,0,j] for j in customers+[end])==1)
        mdl.add_constraint(mdl.sum(x[v,i,end] for i in [0]+customers)==1)
        for i in customers:
            mdl.add_constraint(mdl.sum(x[v,i,j] for j in customers+[end] if j!=i)==z[v,i])
            mdl.add_constraint(mdl.sum(x[v,j,i] for j in [0]+customers if j!=i)==z[v,i])
            mdl.add_constraint(start[v]>=inst.nodes[i].release-big_m*(2-initial[i]-z[v,i]))
            mdl.add_constraint(arrive[i]>=start[v]+tt(0,i)-big_m*(1-x[v,0,i]))
            mdl.add_constraint(returned[v]>=depart[i]+tt(i,end)-big_m*(1-x[v,i,end]))
            for j in customers:
                if i==j: continue
                detour=tt(i,0)+inst.depot_receive+tt(0,j)-tt(i,j)
                mdl.add_constraint(arrive[j]>=depart[i]+tt(i,j)+detour*theta[j]-big_m*(1-x[v,i,j]))
        mdl.add_constraint(returned[v]>=start[v]+mdl.sum(tt(i,j)*x[v,i,j] for i,j in arcs)
                           +inst.truck_service*mdl.sum(z[v,i] for i in customers))
        mdl.add_constraint(cmax>=returned[v])
    for v in trucks[1:]:
        mdl.add_constraint(mdl.sum(i*z[v-1,i] for i in customers)<=mdl.sum(i*z[v,i] for i in customers))
    for i in customers:
        rendezvous=mdl.sum(u[d,i] for d in drones) if i in eligible else 0
        mdl.add_constraint(rendezvous<=1)
        mdl.add_constraint(depart[i]>=arrive[i]+inst.truck_service+inst.resupply_time*rendezvous)
        mdl.add_constraint(theta[i]<=1-mdl.sum(x[v,0,i] for v in trucks))
        mdl.add_constraint(mdl.sum(load[i,j] for j in customers)>=theta[i])
        for j in customers:
            mdl.add_constraint(load[i,j]<=theta[i])
            mdl.add_constraint(arrive[i]>=(inst.nodes[j].release+tt(0,i))*load[i,j])
            if i!=j:
                mdl.add_constraint(depart[j]>=depart[i]+tt(i,j)+inst.truck_service-big_m*(1-load[i,j]))
            for v in trucks:
                mdl.add_constraint(z[v,i]+load[i,j]-1<=z[v,j])
        mdl.add_constraint(initial[i]+mdl.sum(load[h,i] for h in customers)
                           +mdl.sum(supply[d,h,i] for d in drones for h in eligible)==1)
        node_lb=inst.nodes[i].release+min(dd(i)+inst.resupply_time,tt(0,i))+inst.truck_service
        mdl.add_constraint(depart[i]>=node_lb)
        mdl.add_constraint(cmax>=depart[i]+tt(i,end))
    for d in drones:
        mdl.add_constraint(mdl.sum(w[d,0,j] for j in eligible)==drone_used[d])
        mdl.add_constraint(mdl.sum(w[d,i,end] for i in eligible)==drone_used[d])
        for h in eligible:
            mdl.add_constraint(mdl.sum(w[d,h,j] for j in eligible+[end] if j!=h)==u[d,h])
            mdl.add_constraint(mdl.sum(w[d,j,h] for j in [0]+eligible if j!=h)==u[d,h])
            mdl.add_constraint(mdl.sum(supply[d,h,j] for j in customers)>=u[d,h])
            mdl.add_constraint(mdl.sum(supply[d,h,j] for j in customers)<=inst.drone_capacity*u[d,h])
            mdl.add_constraint(dr_dep[d,h]<=horizon*u[d,h])
            mdl.add_constraint(dr_arr[d,h]==dr_dep[d,h]+dd(h)*u[d,h])
            mdl.add_constraint(dr_ret[d,h]==dr_arr[d,h]+(inst.resupply_time+dd(h))*u[d,h])
            mdl.add_constraint(dr_arr[d,h]>=arrive[h]+inst.truck_service*(1-supply[d,h,h])-big_m*(1-u[d,h]))
            mdl.add_constraint(depart[h]>=dr_arr[d,h]+inst.resupply_time+inst.truck_service*supply[d,h,h]-big_m*(1-u[d,h]))
            for j in customers:
                mdl.add_constraint(dr_dep[d,h]>=inst.nodes[j].release*supply[d,h,j])
                if h!=j:
                    mdl.add_constraint(depart[j]>=depart[h]+tt(h,j)+inst.truck_service-big_m*(1-supply[d,h,j]))
                for v in trucks:
                    mdl.add_constraint(z[v,h]+supply[d,h,j]-1<=z[v,j])
            for j in eligible:
                if j!=h:
                    mdl.add_constraint(dr_dep[d,j]>=dr_ret[d,h]+inst.drone_load-big_m*(1-w[d,h,j]))
        if d>0: mdl.add_constraint(drone_used[d]<=drone_used[d-1])
    mdl.minimize(cmax)
    mdl._paper_vars=locals().copy()
    mdl._pdp_vars={"sets":{"truck_trips":list(range(1,n+1))}}
    return mdl


def print_paper_solution(mdl, inst, solution=None):
    variables=mdl._paper_vars
    solution=solution if solution is not None else mdl.solution
    get=lambda v: solution.get_value(v)
    print(f"\nC_max = {solution.objective_value:.9f}")
    for v in variables['trucks']:
        successor={i:j for (vv,i,j),var in variables['x'].items() if vv==v and get(var)>.5}
        route=[]; node=successor.get(0,variables['end'])
        while node!=variables['end'] and len(route)<=len(inst.nodes):
            route.append(node); node=successor[node]
        if not route:
            print(f"Truck {v}: 0 -> 0 (unused)")
            continue
        print(f"Truck {v}: 0 -> "+' -> '.join(map(str,route))+f" -> 0; return={get(variables['returned'][v]):.9f}")
        for i in route:
            print(f"  node {i}: arr={get(variables['arrive'][i]):.9f}, dep={get(variables['depart'][i]):.9f}")
