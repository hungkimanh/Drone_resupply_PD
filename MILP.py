#!/usr/bin/env python3
"""
MILP model for the Truck-Drone Resupply PDP instances.

Solver backend: IBM CPLEX through docplex.mp.

Delivery-only benchmarks with nonbinding truck capacity use the compact
arc formulation in paper_milp.py. Pickup/delivery instances use trip positions:
  - truck routing over trip-position pairs,
  - multi-trip trucks with pickup load carried across depot returns,
  - pickup-delivery same-truck and global pickup-before-delivery constraints,
  - C1 delivery packages either loaded at the depot before a truck trip or supplied by drone,
  - drone trips with payload capacity, release dates, endurance and sequencing,
  - synchronization at the truck rendezvous position,
  - objective: minimize C_max over truck return times (the paper's objective).

Notes:
  - Original 3-column and six-column instance formats are accepted.
  - An optional depot row is ignored; --depot selects its physical coordinates.
  - This is intended as an exact MILP baseline for small instances such as n=10.
"""

import argparse
import math
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


def add_cplex_python_paths() -> None:
    """Add local CPLEX Studio Python packages when they are not installed globally."""
    roots = [
        Path(r"C:\Program Files\IBM\ILOG\CPLEX_Studio221"),
        Path(r"C:\Program Files\IBM\ILOG\CPLEX_Studio2211"),
        Path(r"C:\Program Files\IBM\ILOG\CPLEX_Studio2212"),
    ]
    version = f"{sys.version_info.major}.{sys.version_info.minor}"
    for root in roots:
        docplex_path = root / "python" / "docplex"
        cplex_path = root / "cplex" / "python" / version / "x64_win64"
        for candidate in (docplex_path, cplex_path):
            if candidate.exists():
                text = str(candidate)
                if text not in sys.path:
                    sys.path.insert(0, text)


def require_docplex():
    add_cplex_python_paths()
    try:
        from docplex.mp.model import Model
    except ImportError as exc:  # pragma: no cover - user environment dependent
        raise SystemExit(
            "Cannot import docplex. Please install/enable the CPLEX Python modeling API, e.g.\n"
            "  python -m pip install docplex\n"
            "and make sure IBM CPLEX is installed/licensed."
        ) from exc
    return Model


DEPOTS = {
    "center": (10.0, 10.0),
    "border": (0.0, 10.0),
    "outside": (-10.0, 10.0),
}


class Node:
    def __init__(self, idx: int, x: float, y: float, typ: str, release: float, pair_id: int) -> None:
        self.idx = idx
        self.x = x
        self.y = y
        self.typ = typ
        self.release = release
        self.pair_id = pair_id


class Instance:
    def __init__(
        self,
        nodes: List[Node],
        truck_time: List[List[float]],
        drone_time: List[List[float]],
        num_trucks: int = 2,
        num_drones: int = 2,
        truck_capacity: int = 50,
        drone_capacity: int = 2,
        truck_service: float = 3.0,
        depot_receive: float = 5.0,
        drone_load: float = 5.0,
        resupply_time: float = 5.0,
        drone_endurance: float = 90.0,
    ) -> None:
        self.nodes = nodes
        self.truck_time = truck_time
        self.drone_time = drone_time
        self.num_trucks = num_trucks
        self.num_drones = num_drones
        self.truck_capacity = truck_capacity
        self.drone_capacity = drone_capacity
        self.truck_service = truck_service
        self.depot_receive = depot_receive
        self.drone_load = drone_load
        self.resupply_time = resupply_time
        self.drone_endurance = drone_endurance


def manhattan(a: Node, b: Node) -> float:
    return abs(a.x - b.x) + abs(a.y - b.y)


def euclidean(a: Node, b: Node) -> float:
    return math.hypot(a.x - b.x, a.y - b.y)


def read_instance(path: Path, depot: str) -> Instance:
    if depot not in DEPOTS:
        raise ValueError(f"unknown depot mode: {depot}")

    dx, dy = DEPOTS[depot]
    nodes = [Node(0, dx, dy, "DEPOT", 0.0, 0)]

    with path.open("r", encoding="utf-8-sig") as f:
        for line_number, raw in enumerate(f, 1):
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if parts[0] in {"ri", "id"}:
                continue
            if len(parts) == 3:
                release, x, y = map(float, parts)
                file_id, typ, pair_id = len(nodes), "D", 0
            elif len(parts) == 6:
                file_id, x, y = int(parts[0]), float(parts[1]), float(parts[2])
                typ, release, pair_id = parts[3], float(parts[4]), int(parts[5])
                if file_id == 0:
                    if len(nodes) != 1 or typ != "D" or release != 0 or pair_id != 0:
                        raise ValueError(f"invalid depot row at {path}:{line_number}")
                    continue
            else:
                raise ValueError(f"expected 3 or 6 columns at {path}:{line_number}")
            if file_id != len(nodes) or typ not in {"D", "P", "DL"} or release < 0 or not all(map(math.isfinite, (x, y, release))):
                raise ValueError(f"invalid customer at {path}:{line_number}")
            nodes.append(Node(file_id, x, y, typ, release, pair_id))

    if len(nodes) <= 1:
        raise ValueError(f"no customers read from {path}")

    n = len(nodes)
    truck_time = [[0.0 for _ in range(n)] for _ in range(n)]
    drone_time = [[0.0 for _ in range(n)] for _ in range(n)]
    for i in range(n):
        for j in range(n):
            truck_time[i][j] = manhattan(nodes[i], nodes[j]) / 30.0 * 60.0
            drone_time[i][j] = euclidean(nodes[i], nodes[j]) / 60.0 * 60.0

    inst = Instance(nodes=nodes, truck_time=truck_time, drone_time=drone_time)
    if len(nodes) - 1 > 20:
        inst.drone_capacity = 10
    return inst


def pair_map(inst: Instance) -> Dict[int, int]:
    pickups: Dict[int, int] = {}
    deliveries: Dict[int, int] = {}
    for node in inst.nodes[1:]:
        if node.typ == "P":
            pickups[node.pair_id] = node.idx
        elif node.typ == "DL":
            deliveries[node.pair_id] = node.idx
    return {pickups[pair_id]: deliveries[pair_id] for pair_id in pickups.keys() & deliveries.keys()}


def build_milp(
    inst: Instance,
    time_limit: Optional[float] = None,
    log_output: bool = True,
    max_truck_trips: Optional[int] = None,
):
    Model = require_docplex()
    if all(node.typ == "D" for node in inst.nodes[1:]) and inst.truck_capacity >= len(inst.nodes)-1 and max_truck_trips is None:
        from paper_milp import build_paper_milp
        return build_paper_milp(inst, Model, time_limit or 3600, log_output)
    mdl = Model(name="truck_drone_resupply_pdp")
    mdl.context.cplex_parameters.threads = 8
    if time_limit is not None and time_limit > 0:
        mdl.set_time_limit(time_limit)
    mdl.context.solver.log_output = log_output

    customers = [node.idx for node in inst.nodes[1:]]
    c1_nodes = [node.idx for node in inst.nodes[1:] if node.typ == "D"]
    pickup_nodes = [node.idx for node in inst.nodes[1:] if node.typ == "P"]
    delivery_nodes = [node.idx for node in inst.nodes[1:] if node.typ == "DL"]
    positions = list(range(1, len(customers) + 1))
    truck_trips = list(range(1, (max_truck_trips or len(customers)) + 1))
    trucks = list(range(inst.num_trucks))
    drones = list(range(inst.num_drones))
    trip_slots = list(range(1, max(1, len(c1_nodes)) + 1))
    big_m = 100000.0

    y = mdl.binary_var_dict(
        ((v, m, p, i) for v in trucks for m in truck_trips for p in positions for i in customers),
        name="y",
    )
    used = mdl.binary_var_dict(((v, m, p) for v in trucks for m in truck_trips for p in positions), name="used")
    trip_on = mdl.binary_var_dict(((v, m) for v in trucks for m in truck_trips), name="truck_trip")
    z = mdl.binary_var_dict(
        (
            (v, m, p, i, j)
            for v in trucks
            for m in truck_trips
            for p in positions[:-1]
            for i in customers
            for j in customers
        ),
        name="z",
    )
    last = mdl.binary_var_dict(
        ((v, m, p, i) for v in trucks for m in truck_trips for p in positions for i in customers),
        name="last",
    )

    t0 = mdl.continuous_var_dict(((v, m) for v in trucks for m in truck_trips), lb=0, name="t0")
    arr = mdl.continuous_var_dict(((v, m, p) for v in trucks for m in truck_trips for p in positions), lb=0, name="arr")
    dep = mdl.continuous_var_dict(((v, m, p) for v in trucks for m in truck_trips for p in positions), lb=0, name="dep")
    truck_ret = mdl.continuous_var_dict(((v, m) for v in trucks for m in truck_trips), lb=0, name="truck_ret")

    load0 = mdl.binary_var_dict(((v, m, i) for v in trucks for m in truck_trips for i in c1_nodes), name="load0")
    b = mdl.binary_var_dict(
        ((v, m, p, i) for v in trucks for m in truck_trips for p in positions for i in c1_nodes),
        name="drone_pkg",
    )
    rs = mdl.binary_var_dict(((v, m, p) for v in trucks for m in truck_trips for p in positions), name="rs")
    own_from_drone = mdl.binary_var_dict(
        ((v, m, p, i) for v in trucks for m in truck_trips for p in positions for i in c1_nodes),
        name="own_drone",
    )
    load_start = mdl.continuous_var_dict(((v, m) for v in trucks for m in truck_trips), lb=0, name="load_start")
    load_after = mdl.continuous_var_dict(
        ((v, m, p) for v in trucks for m in truck_trips for p in positions),
        lb=0,
        name="load_after",
    )
    load_end = mdl.continuous_var_dict(((v, m) for v in trucks for m in truck_trips), lb=0, name="load_end")

    w = mdl.binary_var_dict(
        (
            (d, r, v, m, p, h)
            for d in drones
            for r in trip_slots
            for v in trucks
            for m in truck_trips
            for p in positions
            for h in customers
        ),
        name="w",
    )
    trip_used = mdl.binary_var_dict(((d, r) for d in drones for r in trip_slots), name="trip_used")
    dr_dep = mdl.continuous_var_dict(((d, r) for d in drones for r in trip_slots), lb=0, name="dr_dep")
    dr_arr = mdl.continuous_var_dict(((d, r) for d in drones for r in trip_slots), lb=0, name="dr_arr")
    dr_ret = mdl.continuous_var_dict(((d, r) for d in drones for r in trip_slots), lb=0, name="dr_ret")

    cmax = mdl.continuous_var(lb=0, name="C_max")

    for v in trucks:
        for m in truck_trips:
            mdl.add_constraint(used[v, m, 1] == trip_on[v, m], ctname=f"trip_first_{v}_{m}")
            if m > 1:
                mdl.add_constraint(trip_on[v, m] <= trip_on[v, m - 1], ctname=f"truck_trip_no_gap_{v}_{m}")
            for p in positions:
                mdl.add_constraint(mdl.sum(y[v, m, p, i] for i in customers) == used[v, m, p], ctname=f"one_node_{v}_{m}_{p}")
                if p > 1:
                    mdl.add_constraint(used[v, m, p] <= used[v, m, p - 1], ctname=f"no_gap_{v}_{m}_{p}")

    for i in customers:
        mdl.add_constraint(
            mdl.sum(y[v, m, p, i] for v in trucks for m in truck_trips for p in positions) == 1,
            ctname=f"visit_{i}",
        )

    assigned = {
        (v, i): mdl.sum(y[v, m, p, i] for m in truck_trips for p in positions)
        for v in trucks
        for i in customers
    }
    order_of = {
        (v, i): mdl.sum(((m - 1) * len(positions) + p) * y[v, m, p, i] for m in truck_trips for p in positions)
        for v in trucks
        for i in customers
    }

    for p_node, dl_node in pair_map(inst).items():
        for v in trucks:
            mdl.add_constraint(assigned[v, p_node] == assigned[v, dl_node], ctname=f"same_truck_{v}_{p_node}_{dl_node}")
            mdl.add_constraint(
                order_of[v, p_node] + 1 <= order_of[v, dl_node] + big_m * (1 - assigned[v, p_node]),
                ctname=f"precedence_{v}_{p_node}_{dl_node}",
            )

    for v in trucks:
        for m in truck_trips:
            for i in c1_nodes:
                served_here = mdl.sum(y[v, m, p, i] for p in positions)
                # Source and delivery share truck-trip index m, so C1 freight
                # loaded at the depot or supplied by drone cannot cross a depot.
                mdl.add_constraint(load0[v, m, i] + mdl.sum(b[v, m, p, i] for p in positions) == served_here, ctname=f"source_{v}_{m}_{i}")
                mdl.add_indicator(load0[v, m, i], t0[v, m] >= inst.nodes[i].release, name=f"release_load0_{v}_{m}_{i}")
                for p in positions:
                    mdl.add_constraint(b[v, m, p, i] <= mdl.sum(y[v, m, q, i] for q in positions if q >= p), ctname=f"before_service_{v}_{m}_{p}_{i}")
                    mdl.add_constraint(b[v, m, p, i] <= used[v, m, p], ctname=f"rs_on_used_{v}_{m}_{p}_{i}")

    for v in trucks:
        for m in truck_trips:
            total_drone_pkgs = mdl.sum(b[v, m, p, i] for p in positions for i in c1_nodes)
            for p in positions:
                pkgs_at_p = mdl.sum(b[v, m, p, i] for i in c1_nodes)
                mdl.add_constraint(pkgs_at_p <= inst.drone_capacity * rs[v, m, p], ctname=f"drone_cap_{v}_{m}_{p}")
                mdl.add_constraint(pkgs_at_p >= rs[v, m, p], ctname=f"rs_has_pkg_{v}_{m}_{p}")
            mdl.add_constraint(total_drone_pkgs <= len(c1_nodes) * trip_on[v, m], ctname=f"drone_only_active_trip_{v}_{m}")

    for v in trucks:
        for m in truck_trips:
            for p in positions:
                for i in c1_nodes:
                    mdl.add_constraint(own_from_drone[v, m, p, i] <= y[v, m, p, i], ctname=f"own_y_ub_{v}_{m}_{p}_{i}")
                    mdl.add_constraint(own_from_drone[v, m, p, i] <= b[v, m, p, i], ctname=f"own_b_ub_{v}_{m}_{p}_{i}")
                    mdl.add_constraint(own_from_drone[v, m, p, i] >= y[v, m, p, i] + b[v, m, p, i] - 1, ctname=f"own_lb_{v}_{m}_{p}_{i}")

    for v in trucks:
        for m in truck_trips:
            for p in positions[:-1]:
                for i in customers:
                    for j in customers:
                        mdl.add_constraint(z[v, m, p, i, j] <= y[v, m, p, i], ctname=f"z1_{v}_{m}_{p}_{i}_{j}")
                        mdl.add_constraint(z[v, m, p, i, j] <= y[v, m, p + 1, j], ctname=f"z2_{v}_{m}_{p}_{i}_{j}")
                        mdl.add_constraint(z[v, m, p, i, j] >= y[v, m, p, i] + y[v, m, p + 1, j] - 1, ctname=f"z3_{v}_{m}_{p}_{i}_{j}")

    for v in trucks:
        for m in truck_trips:
            for p in positions:
                for i in customers:
                    next_used = used[v, m, p + 1] if p < positions[-1] else 0
                    mdl.add_constraint(last[v, m, p, i] <= y[v, m, p, i], ctname=f"last_y_{v}_{m}_{p}_{i}")
                    mdl.add_constraint(last[v, m, p, i] <= 1 - next_used, ctname=f"last_next_{v}_{m}_{p}_{i}")
                    mdl.add_constraint(last[v, m, p, i] >= y[v, m, p, i] - next_used, ctname=f"last_lb_{v}_{m}_{p}_{i}")

    for v in trucks:
        for m in truck_trips:
            if m == 1:
                mdl.add_constraint(load_start[v, m] == mdl.sum(load0[v, m, i] for i in c1_nodes), ctname=f"load_start_first_{v}")
            else:
                mdl.add_constraint(load_start[v, m] == load_end[v, m - 1] + mdl.sum(load0[v, m, i] for i in c1_nodes), ctname=f"load_start_carry_{v}_{m}")
            mdl.add_constraint(load_start[v, m] <= inst.truck_capacity * trip_on[v, m], ctname=f"truck_cap_start_{v}_{m}")

            for p in positions:
                prev_load = load_start[v, m] if p == 1 else load_after[v, m, p - 1]
                delta = (
                    mdl.sum(y[v, m, p, i] for i in pickup_nodes)
                    - mdl.sum(y[v, m, p, i] for i in delivery_nodes)
                    - mdl.sum(y[v, m, p, i] for i in c1_nodes)
                    + mdl.sum(b[v, m, p, i] for i in c1_nodes)
                )
                mdl.add_constraint(load_after[v, m, p] >= prev_load + delta - big_m * (1 - used[v, m, p]), ctname=f"load_lb_{v}_{m}_{p}")
                mdl.add_constraint(load_after[v, m, p] <= prev_load + delta + big_m * (1 - used[v, m, p]), ctname=f"load_ub_{v}_{m}_{p}")
                mdl.add_constraint(load_after[v, m, p] <= inst.truck_capacity * used[v, m, p], ctname=f"truck_cap_after_{v}_{m}_{p}")
                # If the drone brings the parcel served at this very stop,
                # handover precedes service: that parcel is still onboard at
                # the peak load, before delivery removes it.
                mdl.add_constraint(load_after[v, m, p] + mdl.sum(own_from_drone[v, m, p, i] for i in c1_nodes)
                                   <= inst.truck_capacity * used[v, m, p], ctname=f"truck_cap_handover_{v}_{m}_{p}")

            mdl.add_constraint(load_end[v, m] <= inst.truck_capacity * trip_on[v, m], ctname=f"load_end_cap_{v}_{m}")
            for p in positions:
                for i in customers:
                    mdl.add_constraint(load_end[v, m] >= load_after[v, m, p] - big_m * (1 - last[v, m, p, i]), ctname=f"load_end_lb_{v}_{m}_{p}_{i}")
                    mdl.add_constraint(load_end[v, m] <= load_after[v, m, p] + big_m * (1 - last[v, m, p, i]), ctname=f"load_end_ub_{v}_{m}_{p}_{i}")

            # Only picked-up C2 freight may remain onboard across the depot.
            c2_carry = mdl.sum(
                mdl.sum(y[v, r, p, p_node] for r in truck_trips if r <= m for p in positions)
                - mdl.sum(y[v, r, p, dl_node] for r in truck_trips if r <= m for p in positions)
                for p_node, dl_node in pair_map(inst).items()
            )
            mdl.add_constraint(load_end[v, m] == c2_carry, ctname=f"load_end_c2_only_{v}_{m}")

    for v in trucks:
        for m in truck_trips:
            if m > 1:
                mdl.add_indicator(trip_on[v, m], t0[v, m] >= truck_ret[v, m - 1] + inst.depot_receive, name=f"truck_trip_seq_{v}_{m}")
            for i in c1_nodes:
                mdl.add_indicator(load0[v, m, i], t0[v, m] >= inst.nodes[i].release, name=f"load_release_{v}_{m}_{i}")
            for i in customers:
                mdl.add_indicator(y[v, m, 1, i], arr[v, m, 1] == t0[v, m] + inst.truck_time[0][i], name=f"start_time_{v}_{m}_{i}")
            for p in positions:
                mdl.add_indicator(used[v, m, p], dep[v, m, p] >= arr[v, m, p] + inst.truck_service + inst.resupply_time * rs[v, m, p], name=f"service_rs_{v}_{m}_{p}")
                mdl.add_constraint(
                    dep[v, m, p]
                    <= arr[v, m, p] + inst.truck_service
                    + big_m * rs[v, m, p] + big_m * (1 - used[v, m, p]),
                    ctname=f"plain_service_ub_{v}_{m}_{p}",
                )
            for p in positions[:-1]:
                for i in customers:
                    for j in customers:
                        mdl.add_indicator(z[v, m, p, i, j], arr[v, m, p + 1] == dep[v, m, p] + inst.truck_time[i][j], name=f"travel_{v}_{m}_{p}_{i}_{j}")
            for p in positions:
                for i in customers:
                    mdl.add_indicator(last[v, m, p, i], truck_ret[v, m] == dep[v, m, p] + inst.truck_time[i][0], name=f"return_{v}_{m}_{p}_{i}")
            mdl.add_indicator(trip_on[v, m], cmax >= truck_ret[v, m], name=f"truck_cmax_{v}_{m}")

    for v in trucks:
        for m in truck_trips:
            for p in positions:
                mdl.add_constraint(
                    mdl.sum(w[d, r, v, m, p, h] for d in drones for r in trip_slots for h in customers) == rs[v, m, p],
                    ctname=f"rs_trip_assign_{v}_{m}_{p}",
                )

    for d in drones:
        for r in trip_slots:
            mdl.add_constraint(
                mdl.sum(w[d, r, v, m, p, h] for v in trucks for m in truck_trips for p in positions for h in customers) == trip_used[d, r],
                ctname=f"one_trip_target_{d}_{r}",
            )
            if r > 1:
                mdl.add_constraint(trip_used[d, r] <= trip_used[d, r - 1], ctname=f"drone_no_gap_{d}_{r}")
                mdl.add_indicator(trip_used[d, r], dr_dep[d, r] >= dr_ret[d, r - 1] + inst.drone_load, name=f"drone_seq_{d}_{r}")
            else:
                mdl.add_indicator(trip_used[d, r], dr_dep[d, r] >= 0, name=f"first_drone_load_{d}_{r}")
            mdl.add_indicator(trip_used[d, r], dr_ret[d, r] - dr_dep[d, r] <= inst.drone_endurance, name=f"endurance_{d}_{r}")

    for d in drones:
        for r in trip_slots:
            for v in trucks:
                for m in truck_trips:
                    for p in positions:
                        for h in customers:
                            mdl.add_constraint(w[d, r, v, m, p, h] <= y[v, m, p, h], ctname=f"target_node_{d}_{r}_{v}_{m}_{p}_{h}")

    for v in trucks:
        for m in truck_trips:
            for p in positions:
                for i in c1_nodes:
                    for d in drones:
                        for r in trip_slots:
                            mdl.add_constraint(
                                dr_dep[d, r] >= inst.nodes[i].release
                                - big_m * (2 - b[v, m, p, i] - mdl.sum(w[d, r, v, m, p, h] for h in customers)),
                                ctname=f"dr_release_{d}_{r}_{v}_{m}_{p}_{i}",
                            )

    for d in drones:
        for r in trip_slots:
            for v in trucks:
                for m in truck_trips:
                    for p in positions:
                        for h in customers:
                            ww = w[d, r, v, m, p, h]
                            fly_out = inst.drone_time[0][h]
                            fly_back = inst.drone_time[h][0]
                            mdl.add_indicator(ww, dr_arr[d, r] == dr_dep[d, r] + fly_out, name=f"dr_arr_{d}_{r}_{v}_{m}_{p}_{h}")
                            mdl.add_indicator(ww, dr_arr[d, r] >= arr[v, m, p], name=f"dr_no_wait_truck_{d}_{r}_{v}_{m}_{p}_{h}")
                            # If h can be served with freight already onboard,
                            # service precedes handover (the decoder's MD rule).
                            own_h_from_drone = own_from_drone[v, m, p, h] if h in c1_nodes else 0
                            mdl.add_constraint(
                                dr_arr[d, r] >= arr[v, m, p] + inst.truck_service
                                - big_m * (1 - ww + own_h_from_drone),
                                ctname=f"serve_before_receive_{d}_{r}_{v}_{m}_{p}_{h}",
                            )
                            mdl.add_indicator(ww, dr_ret[d, r] == dr_arr[d, r] + inst.resupply_time + fly_back, name=f"dr_ret_flight_{d}_{r}_{v}_{m}_{p}_{h}")
                            mdl.add_indicator(ww, dep[v, m, p] >= dr_arr[d, r] + inst.resupply_time, name=f"sync_after_drone_{d}_{r}_{v}_{m}_{p}_{h}")
                            for i in c1_nodes:
                                mdl.add_constraint(
                                    dep[v, m, p] >= dr_arr[d, r] + inst.resupply_time + inst.truck_service
                                    - big_m * (2 - ww - own_from_drone[v, m, p, i]),
                                    ctname=f"own_receive_before_service_{d}_{r}_{v}_{m}_{p}_{h}_{i}",
                                )

    mdl.minimize(cmax)

    mdl._pdp_vars = {
        "y": y,
        "used": used,
        "trip_on": trip_on,
        "load0": load0,
        "b": b,
        "rs": rs,
        "w": w,
        "trip_used": trip_used,
        "arr": arr,
        "dep": dep,
        "truck_ret": truck_ret,
        "load_start": load_start,
        "load_after": load_after,
        "load_end": load_end,
        "dr_dep": dr_dep,
        "dr_arr": dr_arr,
        "dr_ret": dr_ret,
        "cmax": cmax,
        "sets": {
            "customers": customers,
            "c1_nodes": c1_nodes,
            "positions": positions,
            "truck_trips": truck_trips,
            "trucks": trucks,
            "drones": drones,
            "trip_slots": trip_slots,
        },
    }
    return mdl


def polish_fixed_incumbent(mdl, time_limit: float = 30.0, log_output: bool = False):
    """Re-optimize continuous timing after fixing the incumbent's binaries.

    A time-limited CPLEX MIP incumbent may contain arbitrary waiting slack because
    most scheduling constraints are lower bounds. Route encodings do not retain
    that slack, so the heuristic decoder reconstructs an earlier schedule. Fixing
    all discrete decisions and resolving produces the canonical MILP objective
    for the same route/loading/resupply decisions.
    """
    incumbent = mdl.solution
    if incumbent is None:
        return None
    for var in mdl.iter_binary_vars():
        fixed = 1 if incumbent.get_value(var) > 0.5 else 0
        var.lb = fixed
        var.ub = fixed
    mdl.set_time_limit(max(0.1, time_limit))
    return mdl.solve(log_output=log_output)


def _val(x) -> float:
    return 0.0 if x is None else float(x.solution_value)


def print_solution(mdl, inst: Instance, solution=None) -> None:
    solution = solution if solution is not None else mdl.solution
    _val = lambda var: 0.0 if var is None else float(solution.get_value(var))
    if hasattr(mdl, "_paper_vars"):
        from paper_milp import print_paper_solution
        return print_paper_solution(mdl, inst, solution)
    data = mdl._pdp_vars
    y = data["y"]
    trip_on = data["trip_on"]
    load0 = data["load0"]
    b = data["b"]
    w = data["w"]
    arr = data["arr"]
    dep = data["dep"]
    truck_ret = data["truck_ret"]
    load_start = data["load_start"]
    load_end = data["load_end"]
    dr_dep = data["dr_dep"]
    dr_arr = data["dr_arr"]
    dr_ret = data["dr_ret"]
    cmax = data["cmax"]
    sets = data["sets"]

    print(f"\nC_max = {_val(cmax):.2f}")

    for v in sets["trucks"]:
        any_truck_trip = False
        for m in sets["truck_trips"]:
            if _val(trip_on[v, m]) <= 0.5:
                continue
            any_truck_trip = True
            route: List[Tuple[int, int]] = []
            for p in sets["positions"]:
                chosen = [i for i in sets["customers"] if _val(y[v, m, p, i]) > 0.5]
                if chosen:
                    route.append((p, chosen[0]))
            if not route:
                continue

            initial = [i for i in sets["c1_nodes"] if _val(load0[v, m, i]) > 0.5]
            route_nodes = [node for _, node in route]
            print(f"\nTruck {v} trip {m}: 0 -> " + " -> ".join(map(str, route_nodes)) + " -> 0")
            print(f"  initial load: {initial}, load_start={_val(load_start[v, m]):.2f}, load_end={_val(load_end[v, m]):.2f}, return={_val(truck_ret[v, m]):.2f}")
            for p, node in route:
                pkgs = [i for i in sets["c1_nodes"] if _val(b[v, m, p, i]) > 0.5]
                tag = f" drone_pkgs={pkgs}" if pkgs else ""
                print(f"  pos {p:02d} node {node:3d}: arr={_val(arr[v, m, p]):8.2f} dep={_val(dep[v, m, p]):8.2f}{tag}")
        if not any_truck_trip:
            print(f"Truck {v}: 0 -> 0")

    print("\nDrone trips:")
    any_trip = False
    for d in sets["drones"]:
        for r in sets["trip_slots"]:
            targets = [
                (v, m, p, h)
                for v in sets["trucks"]
                for m in sets["truck_trips"]
                for p in sets["positions"]
                for h in sets["customers"]
                if _val(w[d, r, v, m, p, h]) > 0.5
            ]
            if not targets:
                continue
            any_trip = True
            v, m, p, h = targets[0]
            pkgs = [i for i in sets["c1_nodes"] if _val(b[v, m, p, i]) > 0.5]
            print(
                f"  drone {d} trip {r}: h={h}, truck={v}, truck_trip={m}, pos={p}, "
                f"pkgs={pkgs}, dep={_val(dr_dep[d, r]):.2f}, "
                f"arr={_val(dr_arr[d, r]):.2f}, ret={_val(dr_ret[d, r]):.2f}"
            )
    if not any_trip:
        print("  none")


def find_cplex_exe(user_path: Optional[Path] = None) -> Optional[Path]:
    if user_path is not None:
        return user_path if user_path.exists() else None
    candidates = [
        Path.home() / "CPLEX_Studio2211" / "cplex" / "bin" / "x86-64_linux" / "cplex",
        Path(r"C:\Program Files\IBM\ILOG\CPLEX_Studio221\cplex\bin\x64_win64\cplex.exe"),
        Path(r"C:\Program Files\IBM\ILOG\CPLEX_Studio2211\cplex\bin\x64_win64\cplex.exe"),
        Path(r"C:\Program Files\IBM\ILOG\CPLEX_Studio2212\cplex\bin\x64_win64\cplex.exe"),
    ]
    for exe in candidates:
        if exe.exists():
            return exe
    return None


def solve_with_cplex_exe(
    mdl,
    cplex_exe: Path,
    lp_path: Path,
    sol_path: Path,
    time_limit: Optional[float],
    mipgap: Optional[float],
    mipstart_path: Optional[Path] = None,
    inst: Optional[Instance] = None,
) -> int:
    lp_path.parent.mkdir(parents=True, exist_ok=True)
    sol_path.parent.mkdir(parents=True, exist_ok=True)
    mdl.export_as_lp(str(lp_path))

    commands = [f'read "{lp_path}"']
    if mipstart_path is not None:
        commands.append(f'read "{mipstart_path}"')
    if time_limit is not None and time_limit > 0:
        commands.append(f"set timelimit {time_limit}")
    if mipgap is not None:
        commands.append(f"set mip tolerances mipgap {mipgap}")
    commands.extend(
        [
            "optimize",
            f'write "{sol_path}"',
            "display solution objective",
            "quit",
        ]
    )
    proc = subprocess.run(
        [str(cplex_exe)],
        input="\n".join(commands) + "\n",
        universal_newlines=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    print(proc.stdout)
    if sol_path.exists():
        print(f"CPLEX solution file: {sol_path}")
        if inst is not None:
            audit_exported_solution(inst, sol_path)
    return proc.returncode


def audit_exported_solution(inst: Instance, sol_path: Path) -> None:
    """Print and save the arithmetic of a returned solution, without optimizing."""
    from scripts.benchmark_audit import read_milp_solution, export_audit
    routes, header, _ = read_milp_solution(sol_path, inst)
    directory = sol_path.parent / (sol_path.stem + "_audit")
    audit = export_audit(directory, routes, inst, header["objectiveValue"])
    if not audit["raw_replay"]["valid"] or not audit["earliest_fixed_decisions"]["valid"]:
        raise ValueError(f"Physical replay failed: {audit['raw_replay']['errors']}")
    print((directory / "objective_calculation.md").read_text())
    print(f"Detailed timeline CSVs and audit: {directory}")


def write_mst(path: Path, values: Dict[str, float], problem_name: str, obj_value: Optional[float] = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    objective = "" if obj_value is None else f' objectiveValue="{obj_value:.12g}"'
    with path.open("w", encoding="utf-8") as f:
        f.write('<?xml version = "1.0" encoding="UTF-8" standalone="yes"?>\n')
        f.write('<CPLEXSolution version="1.2">\n')
        f.write(f' <header problemName="{problem_name}" solutionName="heuristic_start"{objective}/>\n')
        f.write(" <variables>\n")
        for idx, (name, value) in enumerate(sorted(values.items())):
            f.write(f'  <variable name="{name}" index="{idx}" value="{value:.12g}"/>\n')
        f.write(" </variables>\n")
        f.write("</CPLEXSolution>\n")


def heuristic_u10_num1_start() -> Dict[str, float]:
    """MIP start from the heuristic solution observed for U_10_0.5_Num_1_pd."""
    values: Dict[str, float] = {"C_max": 158.40}

    routes = {
        0: [3, 1, 2],
        1: [4, 9, 5, 6, 10, 7, 8],
    }
    drone_pkgs = {
        (0, 1, 1): [3],
        (0, 1, 2): [1, 2],
        (1, 1, 3): [5, 6],
        (1, 1, 6): [7, 8],
    }
    initial_load = {
        0: [],
        1: [9],
    }
    trip_assignments = [
        # drone, trip, truck, truck trip, position, rendezvous
        (0, 1, 0, 1, 1, 3),
        (0, 2, 0, 1, 2, 1),
        (1, 1, 1, 1, 3, 5),
        (1, 2, 1, 1, 6, 7),
    ]

    c1_nodes = [1, 2, 3, 5, 6, 7, 8, 9]

    for v, route in routes.items():
        m = 1
        values[f"truck_trip_{v}_{m}"] = 1
        for p, node in enumerate(route, start=1):
            values[f"used_{v}_{m}_{p}"] = 1
            values[f"y_{v}_{m}_{p}_{node}"] = 1
        if route:
            last_p = len(route)
            values[f"last_{v}_{m}_{last_p}_{route[-1]}"] = 1
        for p in range(1, len(route)):
            values[f"z_{v}_{m}_{p}_{route[p - 1]}_{route[p]}"] = 1

        for pkg in initial_load.get(v, []):
            values[f"load0_{v}_{m}_{pkg}"] = 1

        for (vv, mm, p), pkgs in drone_pkgs.items():
            if vv != v:
                continue
            values[f"rs_{v}_{mm}_{p}"] = 1
            node_at_p = route[p - 1]
            for pkg in pkgs:
                values[f"drone_pkg_{v}_{mm}_{p}_{pkg}"] = 1
                if pkg == node_at_p and pkg in c1_nodes:
                    values[f"own_drone_{v}_{mm}_{p}_{pkg}"] = 1

    for d, r, v, m, p, h in trip_assignments:
        values[f"trip_used_{d}_{r}"] = 1
        values[f"w_{d}_{r}_{v}_{m}_{p}_{h}"] = 1

    return values


def diverse_u10_start() -> Dict[str, float]:
    """Multi-trip fixture shared with ``pdp_solver --check-diverse``."""
    # Leave continuous timing variables (including C_max) unspecified so CPLEX
    # can complete the combinatorial start with the MILP's own scheduler.
    values: Dict[str, float] = {}
    routes = {
        (0, 1): [4, 3],
        (0, 2): [10, 5, 1, 2],
        (1, 1): [9],
        (1, 2): [6, 7, 8],
    }
    initial_load = {
        (0, 1): [],
        (0, 2): [5, 1],
        (1, 1): [9],
        (1, 2): [6],
    }
    drone_pkgs = {
        (0, 1, 1): [3],       # pickup 4 is served before rendezvous
        (0, 2, 3): [2],       # delivery 1 is served before rendezvous
        (1, 2, 2): [7, 8],    # delivery 7 is supplied before service
    }
    trip_assignments = [
        (0, 1, 0, 1, 1, 4),
        (0, 2, 0, 2, 3, 1),
        (1, 1, 1, 2, 2, 7),
    ]
    c1_nodes = {1, 2, 3, 5, 6, 7, 8, 9}

    for (v, m), route in routes.items():
        values[f"truck_trip_{v}_{m}"] = 1
        for p, node in enumerate(route, start=1):
            values[f"used_{v}_{m}_{p}"] = 1
            values[f"y_{v}_{m}_{p}_{node}"] = 1
        values[f"last_{v}_{m}_{len(route)}_{route[-1]}"] = 1
        for p in range(1, len(route)):
            values[f"z_{v}_{m}_{p}_{route[p - 1]}_{route[p]}"] = 1

        for pkg in initial_load[(v, m)]:
            values[f"load0_{v}_{m}_{pkg}"] = 1

        for (vv, mm, p), pkgs in drone_pkgs.items():
            if (vv, mm) != (v, m):
                continue
            values[f"rs_{v}_{m}_{p}"] = 1
            node_at_p = route[p - 1]
            for pkg in pkgs:
                values[f"drone_pkg_{v}_{m}_{p}_{pkg}"] = 1
                if pkg == node_at_p and pkg in c1_nodes:
                    values[f"own_drone_{v}_{m}_{p}_{pkg}"] = 1

    for d, r, v, m, p, h in trip_assignments:
        values[f"trip_used_{d}_{r}"] = 1
        values[f"w_{d}_{r}_{v}_{m}_{p}_{h}"] = 1
    return values


def main(argv: Optional[Iterable[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Build/solve MILP for Truck-Drone Resupply PDP with CPLEX.")
    parser.add_argument("instance", type=Path, help="Original 3-column benchmark or six-column PDP file")
    parser.add_argument("--depot", choices=sorted(DEPOTS), default="border", help="Depot position mode")
    parser.add_argument("--time", type=float, default=300.0, help="CPLEX time limit in seconds")
    parser.add_argument("--mipgap", type=float, default=None, help="Relative MIP gap, e.g. 0.01")
    parser.add_argument("--max-truck-trips", type=int, default=None, help="Maximum trips allowed for each truck; default is number of customers")
    parser.add_argument("--export-lp", type=Path, default=None, help="Export model to .lp and continue")
    parser.add_argument("--cplex-exe", type=Path, default=None, help="Path to cplex.exe for command-line solve fallback")
    parser.add_argument("--sol-file", type=Path, default=Path("results/milp_solution.sol"), help="Output .sol file when using cplex.exe")
    parser.add_argument("--heuristic-u10-start", action="store_true", help="Use the known heuristic solution for U_10_0.5_Num_1_pd as a MIP start")
    parser.add_argument("--diverse-u10-start", action="store_true", help="Use the multi-trip U_10 validation fixture as a MIP start")
    parser.add_argument("--mst-file", type=Path, default=Path("results/heuristic_start.mst"), help="MIP start file written/read by CPLEX")
    parser.add_argument("--no-solve", action="store_true", help="Only build/export the model")
    parser.add_argument("--quiet", action="store_true", help="Disable CPLEX log output")
    parser.add_argument("--polish-time", type=float, default=30.0, help="Seconds to normalize timing after fixing incumbent binaries; 0 disables")
    args = parser.parse_args(list(argv) if argv is not None else None)

    inst = read_instance(args.instance, args.depot)
    mdl = build_milp(inst, time_limit=args.time, log_output=not args.quiet, max_truck_trips=args.max_truck_trips)
    if args.mipgap is not None:
        mdl.parameters.mip.tolerances.mipgap = args.mipgap

    print(
        f"Built MILP: {len(inst.nodes) - 1} customers, "
        f"{mdl._pdp_vars['sets']['truck_trips'][-1]} max truck trips, "
        f"{mdl.number_of_variables} vars, {mdl.number_of_constraints} constraints"
    )

    if args.export_lp:
        mdl.export_as_lp(str(args.export_lp))
        print(f"Exported LP: {args.export_lp}")

    mipstart_path = None
    if args.heuristic_u10_start or args.diverse_u10_start:
        values = diverse_u10_start() if args.diverse_u10_start else heuristic_u10_num1_start()
        start_obj = 270.556095 if args.diverse_u10_start else 158.40
        mipstart_path = args.mst_file
        write_mst(mipstart_path, values, str(args.export_lp or "milp_model.lp"), obj_value=start_obj)
        print(f"Wrote heuristic MIP start: {mipstart_path} ({len(values)} nonzero values)")

    if args.no_solve:
        return 0

    search_started = time.monotonic()
    try:
        sol = mdl.solve(log_output=not args.quiet)
    except Exception as exc:
        cplex_exe = find_cplex_exe(args.cplex_exe)
        if cplex_exe is None:
            print(f"Docplex solve failed: {exc}")
            print("No cplex.exe found for fallback.")
            return 3
        lp_path = args.export_lp or Path("results/milp_model.lp")
        print(f"Docplex solve failed: {exc}")
        print(f"Fallback: solving exported LP with {cplex_exe}")
        remaining = max(0.0, args.time - (time.monotonic() - search_started)) if args.time > 0 else None
        if remaining is not None and remaining <= 0:
            print("No solver budget remains for CLI fallback.")
            return 2
        return solve_with_cplex_exe(mdl, cplex_exe, lp_path, args.sol_file, remaining, args.mipgap, mipstart_path, inst)

    if sol is None:
        print("No feasible solution found by CPLEX within the limit.")
        return 2

    search_status = mdl.solve_details.status
    search_bound = mdl.solve_details.best_bound
    raw_objective = sol.objective_value
    remaining = max(0.0, args.time - (time.monotonic() - search_started)) if args.time > 0 else args.polish_time
    polish_budget = min(args.polish_time, remaining)
    # Docplex rounds sub-second model limits up to one second.
    if polish_budget >= 1.0:
        try:
            polished = polish_fixed_incumbent(mdl, polish_budget, log_output=not args.quiet)
        except Exception as exc:
            print(f"Timing polishing failed; retaining the feasible incumbent: {exc}")
            polished = None
        if polished is not None:
            sol = polished
            print(f"Raw incumbent C_max: {raw_objective:.9f}")
            print(f"Polished fixed-decision C_max: {sol.objective_value:.9f}")
    print(f"Search status: {search_status}")
    print(f"Search best bound: {search_bound}")
    print_solution(mdl, inst, sol)
    args.sol_file.parent.mkdir(parents=True, exist_ok=True)
    exported = Path(sol.export_as_sol(path=str(args.sol_file.parent), basename=args.sol_file.name))
    if exported != args.sol_file:
        exported.replace(args.sol_file)
    audit_exported_solution(inst, args.sol_file)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
