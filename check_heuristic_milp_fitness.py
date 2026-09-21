#!/usr/bin/env python3
"""Run heuristic solutions and recompute their fitness with fixed MILP decisions."""

from __future__ import annotations

import argparse
import csv
import re
import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from milp import read_instance, require_docplex


FIELDS = [
    "instance", "heuristic_cmax", "milp_cmax", "delta", "milp_status",
    "heuristic_exit", "note",
]


def parse_encoding(path: Path):
    routes = []
    route = None
    objective = None
    for line in path.read_text(encoding="ascii").splitlines():
        parts = line.split()
        if not parts:
            continue
        if parts[0] == "HEURISTIC_CMAX":
            objective = float(parts[1])
        elif parts[0] == "ROUTE":
            route = []
            routes.append(route)
        elif parts[0] == "STOP":
            marker = parts.index("DRONE") if "DRONE" in parts else len(parts)
            packages = [int(x) for x in parts[2:marker]]
            drone = None
            if marker < len(parts):
                drone = (int(parts[marker + 1]), int(parts[marker + 2]))
            route.append((int(parts[1]), packages, drone))
    if objective is None or not routes:
        raise ValueError(f"Malformed encoding: {path}")
    return objective, routes


def route_decisions(routes):
    trips = {}
    drone_targets = set()
    max_trips = 1
    for v, route in enumerate(routes):
        m = 0
        p = 0
        for idx, (node, packages, drone) in enumerate(route):
            if node == 0:
                if idx == len(route) - 1:
                    continue
                m += 1
                p = 0
                trips[(v, m)] = {"load": set(packages), "stops": []}
                max_trips = max(max_trips, m)
                continue
            p += 1
            trips[(v, m)]["stops"].append((p, node, set(packages)))
            if drone is not None:
                d, r = drone
                drone_targets.add((d, r, v, m, p, node))
    return trips, drone_targets, max_trips


def fix_decisions(model, routes):
    data = model._pdp_vars
    sets = data["sets"]
    trips, drone_targets, _ = route_decisions(routes)

    for v in sets["trucks"]:
        for m in sets["truck_trips"]:
            trip = trips.get((v, m))
            data["trip_on"][v, m].lb = data["trip_on"][v, m].ub = int(trip is not None)
            stop_map = {} if trip is None else {p: (node, pkgs) for p, node, pkgs in trip["stops"]}
            load = set() if trip is None else trip["load"]
            for i in sets["c1_nodes"]:
                value = int(i in load)
                data["load0"][v, m, i].lb = data["load0"][v, m, i].ub = value
            for p in sets["positions"]:
                chosen = stop_map.get(p)
                for i in sets["customers"]:
                    value = int(chosen is not None and chosen[0] == i)
                    data["y"][v, m, p, i].lb = data["y"][v, m, p, i].ub = value
                packages = set() if chosen is None else chosen[1]
                for i in sets["c1_nodes"]:
                    value = int(i in packages)
                    data["b"][v, m, p, i].lb = data["b"][v, m, p, i].ub = value
                value = int(bool(packages))
                data["rs"][v, m, p].lb = data["rs"][v, m, p].ub = value

    used_sorties = {(d, r) for d, r, *_ in drone_targets}
    for d in sets["drones"]:
        for r in sets["trip_slots"]:
            value = int((d, r) in used_sorties)
            data["trip_used"][d, r].lb = data["trip_used"][d, r].ub = value
            for v in sets["trucks"]:
                for m in sets["truck_trips"]:
                    for p in sets["positions"]:
                        for h in sets["customers"]:
                            value = int((d, r, v, m, p, h) in drone_targets)
                            var = data["w"][d, r, v, m, p, h]
                            var.lb = var.ub = value


def build_fixed_timing_model(inst, routes, time_limit):
    """Build the continuous MILP submodel induced by one fixed encoding."""
    Model = require_docplex()
    model = Model(name="fixed_heuristic_timing")
    model.set_time_limit(time_limit)
    trips, drone_targets, _ = route_decisions(routes)
    cmax = model.continuous_var(lb=0, name="C_max")
    truck_return = {}
    stop_vars = {}

    for (v, m), trip in sorted(trips.items()):
        t0 = model.continuous_var(lb=0, name=f"t0_{v}_{m}")
        if m > 1:
            model.add_constraint(t0 >= truck_return[v, m - 1] + inst.depot_receive)
        for pkg in trip["load"]:
            model.add_constraint(t0 >= inst.nodes[pkg].release)

        previous_depart = None
        previous_node = 0
        for p, node, packages in trip["stops"]:
            arrive = model.continuous_var(lb=0, name=f"arr_{v}_{m}_{p}")
            depart = model.continuous_var(lb=0, name=f"dep_{v}_{m}_{p}")
            if previous_depart is None:
                model.add_constraint(arrive == t0 + inst.truck_time[0][node])
            else:
                model.add_constraint(arrive == previous_depart + inst.truck_time[previous_node][node])
            service_and_handover = inst.truck_service + (inst.resupply_time if packages else 0.0)
            model.add_constraint(depart >= arrive + service_and_handover)
            if not packages:
                model.add_constraint(depart <= arrive + inst.truck_service)
            stop_vars[v, m, p] = (node, packages, arrive, depart)
            previous_depart = depart
            previous_node = node

        returned = model.continuous_var(lb=0, name=f"ret_{v}_{m}")
        model.add_constraint(returned == previous_depart + inst.truck_time[previous_node][0])
        model.add_constraint(cmax >= returned)
        truck_return[v, m] = returned

    event_by_sortie = {}
    for d, r, v, m, p, node in sorted(drone_targets):
        packages = stop_vars[v, m, p][1]
        depart = model.continuous_var(lb=0, name=f"dr_dep_{d}_{r}")
        arrive = model.continuous_var(lb=0, name=f"dr_arr_{d}_{r}")
        returned = model.continuous_var(lb=0, name=f"dr_ret_{d}_{r}")
        event_by_sortie[d, r] = (depart, arrive, returned)
        max_release = max(inst.nodes[pkg].release for pkg in packages)
        model.add_constraint(depart >= max_release + inst.drone_load)
        if r > 1:
            model.add_constraint(depart >= event_by_sortie[d, r - 1][2] + inst.drone_load)
        else:
            model.add_constraint(depart >= inst.drone_load)
        model.add_constraint(arrive == depart + inst.drone_time[0][node])
        model.add_constraint(returned == arrive + inst.resupply_time + inst.drone_time[node][0])
        model.add_constraint(returned - depart <= inst.drone_endurance)
        model.add_constraint(cmax >= returned)

        _, _, truck_arrive, truck_depart = stop_vars[v, m, p]
        own_from_drone = node in packages
        if own_from_drone:
            model.add_constraint(arrive >= truck_arrive)
            model.add_constraint(truck_depart >= arrive + inst.resupply_time + inst.truck_service)
        else:
            model.add_constraint(arrive >= truck_arrive + inst.truck_service)
            model.add_constraint(truck_depart >= arrive + inst.resupply_time)

    model.minimize(cmax)
    return model


def check_one(args):
    instance, solver, out_dir, heuristic_time, milp_time = args
    encoding = out_dir / "encodings" / f"{instance.stem}.txt"
    log = out_dir / "logs" / f"{instance.stem}.log"
    encoding.parent.mkdir(parents=True, exist_ok=True)
    log.parent.mkdir(parents=True, exist_ok=True)
    cmd = [str(solver), str(instance), "--depot", "1", "--time", str(heuristic_time),
           "--run", "1", "--encoding-output", str(encoding)]
    proc = subprocess.run(cmd, text=True, stdout=subprocess.PIPE,
                          stderr=subprocess.STDOUT, check=False)
    log.write_text(proc.stdout, encoding="utf-8", errors="replace")
    row = {"instance": instance.name, "heuristic_exit": proc.returncode, "note": ""}
    if proc.returncode != 0 or not encoding.exists():
        row["note"] = "heuristic failed or produced no encoding"
        return row

    model = None
    try:
        heuristic_cmax, routes = parse_encoding(encoding)
        inst = read_instance(instance, "border")
        model = build_fixed_timing_model(inst, routes, milp_time)
        model.context.cplex_parameters.threads = 1
        solution = model.solve(log_output=False)
        row["heuristic_cmax"] = f"{heuristic_cmax:.9f}"
        row["milp_status"] = str(model.solve_details.status)
        if solution is None:
            row["note"] = "fixed heuristic decisions infeasible in MILP"
        else:
            milp_cmax = float(solution.objective_value)
            row["milp_cmax"] = f"{milp_cmax:.9f}"
            row["delta"] = f"{milp_cmax - heuristic_cmax:.9f}"
            if abs(milp_cmax - heuristic_cmax) > 1e-6:
                row["note"] = "objective mismatch"
    except Exception as exc:
        row["note"] = f"ERROR: {type(exc).__name__}: {exc}"
    finally:
        if model is not None:
            model.end()
    return row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instances", type=Path, default=Path("Instance/Instance"))
    parser.add_argument("--solver", type=Path, default=Path("build-msvc-nmake-release/pdp_solver.exe"))
    parser.add_argument("--output", type=Path, default=Path("results/heuristic_milp_fitness"))
    parser.add_argument("--heuristic-time", type=float, default=150.0)
    parser.add_argument("--milp-time", type=float, default=60.0)
    parser.add_argument("--jobs", type=int, default=5)
    parser.add_argument("--sizes", nargs="+", type=int, default=[10, 15])
    args = parser.parse_args()
    instances = sorted(
        path
        for size in args.sizes
        for path in args.instances.glob(f"U_{size}_*_pd.txt")
    )
    args.output.mkdir(parents=True, exist_ok=True)
    csv_path = args.output / "summary.csv"
    jobs = [(p.resolve(), args.solver.resolve(), args.output.resolve(),
             args.heuristic_time, args.milp_time) for p in instances]
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        with ProcessPoolExecutor(max_workers=args.jobs) as pool:
            futures = [pool.submit(check_one, job) for job in jobs]
            for index, future in enumerate(as_completed(futures), 1):
                row = future.result()
                writer.writerow(row)
                stream.flush()
                print(f"[{index}/{len(jobs)}] {row['instance']}: "
                      f"H={row.get('heuristic_cmax', 'NA')} "
                      f"M={row.get('milp_cmax', 'NA')} {row.get('note', '')}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
