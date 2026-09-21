#!/usr/bin/env python3
"""Solve every instance with MILP, then validate its incumbent in the C++ decoder."""

from __future__ import annotations

import argparse
import csv
import gc
import re
import subprocess
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

from milp import build_milp, polish_fixed_incumbent, read_instance


RESULT_FIELDS = [
    "instance",
    "customers",
    "milp_status",
    "raw_milp_cmax",
    "milp_cmax",
    "best_bound",
    "solve_seconds",
    "polish_seconds",
    "decode_valid",
    "decode_feasible",
    "decode_cmax",
    "decode_penalty",
    "objective_delta",
    "note",
]


def value(var) -> float:
    return 0.0 if var is None else float(var.solution_value)


def extract_encoding(model):
    data = model._pdp_vars
    sets = data["sets"]
    y = data["y"]
    trip_on = data["trip_on"]
    load0 = data["load0"]
    drone_pkg = data["b"]
    w = data["w"]
    dr_dep = data["dr_dep"]
    dr_arr = data["dr_arr"]
    dr_ret = data["dr_ret"]
    routes = []

    for truck in sets["trucks"]:
        route = []
        for trip in sets["truck_trips"]:
            if value(trip_on[truck, trip]) <= 0.5:
                continue
            initial = [
                node
                for node in sets["c1_nodes"]
                if value(load0[truck, trip, node]) > 0.5
            ]
            route.append((0, initial, -1, -1, -1.0, -1.0, -1.0))
            for pos in sets["positions"]:
                chosen = [
                    node
                    for node in sets["customers"]
                    if value(y[truck, trip, pos, node]) > 0.5
                ]
                if not chosen:
                    break
                packages = [
                    node
                    for node in sets["c1_nodes"]
                    if value(drone_pkg[truck, trip, pos, node]) > 0.5
                ]
                assignment = [
                    (drone, sortie)
                    for drone in sets["drones"]
                    for sortie in sets["trip_slots"]
                    for rendezvous in sets["customers"]
                    if value(w[drone, sortie, truck, trip, pos, rendezvous]) > 0.5
                ]
                drone, sortie = assignment[0] if assignment else (-1, -1)
                timing = (-1.0, -1.0, -1.0) if drone < 0 else (
                    value(dr_dep[drone, sortie]),
                    value(dr_arr[drone, sortie]),
                    value(dr_ret[drone, sortie]),
                )
                route.append((chosen[0], packages, drone, sortie, *timing))
        route.append((0, [], -1, -1, -1.0, -1.0, -1.0))
        routes.append(route)
    return routes


def write_encoding(path: Path, milp_cmax: float,
                   routes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="ascii", newline="\n") as output:
        output.write(f"MILP_CMAX {milp_cmax:.12f}\n")
        for truck, route in enumerate(routes):
            output.write(f"ROUTE {truck}\n")
            for node, packages, drone, sortie, depart, arrive, returned in route:
                suffix = "" if not packages else " " + " ".join(map(str, packages))
                if drone >= 0:
                    suffix += (
                        f" DRONE {drone} {sortie}"
                        f" {depart:.12f} {arrive:.12f} {returned:.12f}"
                    )
                output.write(f"STOP {node}{suffix}\n")
            output.write("END_ROUTE\n")


def decode(decoder: Path, instance: Path, encoding: Path) -> dict:
    proc = subprocess.run(
        [str(decoder), str(instance), str(encoding), "1"],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    match = re.search(
        r"DECODE_RESULT valid=(\d+) feasible=(\d+) "
        r"cmax=([0-9.eE+-]+) penalty=([0-9.eE+-]+)",
        proc.stdout,
    )
    if match is None:
        raise RuntimeError(f"decoder exit={proc.returncode}: {proc.stdout[-1000:]}")
    return {
        "decode_valid": int(match.group(1)),
        "decode_feasible": int(match.group(2)),
        "decode_cmax": float(match.group(3)),
        "decode_penalty": float(match.group(4)),
    }


def append_result(csv_path: Path, row: dict) -> None:
    exists = csv_path.exists() and csv_path.stat().st_size > 0
    with csv_path.open("a", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=RESULT_FIELDS)
        if not exists:
            writer.writeheader()
        writer.writerow({key: row.get(key, "") for key in RESULT_FIELDS})


def completed_instances(csv_path: Path) -> set[str]:
    if not csv_path.exists():
        return set()
    with csv_path.open("r", encoding="utf-8", newline="") as source:
        return {row["instance"] for row in csv.DictReader(source)}


def solve_instance(path: Path, time_limit: float, output: Path, decoder: Path,
                   max_truck_trips: int | None, threads: int,
                   polish_time: float) -> dict:
    row = {"instance": path.name, "note": ""}
    model = None
    try:
        instance = read_instance(path, "border")
        row["customers"] = len(instance.nodes) - 1
        model = build_milp(
            instance,
            time_limit=time_limit,
            log_output=False,
            max_truck_trips=max_truck_trips,
        )
        model.context.cplex_parameters.threads = threads
        started = time.monotonic()
        solution = model.solve(log_output=False)
        row["solve_seconds"] = f"{time.monotonic() - started:.3f}"
        row["milp_status"] = model.solve_details.status
        row["best_bound"] = model.solve_details.best_bound
        if solution is None:
            row["note"] = "no incumbent"
        else:
            row["raw_milp_cmax"] = f"{float(solution.objective_value):.9f}"
            polish_started = time.monotonic()
            polished = polish_fixed_incumbent(model, polish_time, log_output=False)
            row["polish_seconds"] = f"{time.monotonic() - polish_started:.3f}"
            if polished is not None:
                solution = polished
            milp_cmax = float(solution.objective_value)
            row["milp_cmax"] = f"{milp_cmax:.9f}"
            encoding_path = output / "encodings" / f"{path.stem}.txt"
            write_encoding(encoding_path, milp_cmax, extract_encoding(model))
            decoded = decode(decoder, path.resolve(), encoding_path.resolve())
            row.update(decoded)
            row["objective_delta"] = f"{decoded['decode_cmax'] - milp_cmax:.9f}"
            if not decoded["decode_valid"]:
                row["note"] = "decoder invalid"
            elif abs(decoded["decode_cmax"] - milp_cmax) > 1e-6:
                row["note"] = "objective mismatch"
    except Exception as exc:
        row["note"] = f"ERROR: {type(exc).__name__}: {exc}"
    finally:
        if model is not None:
            model.end()
        del model
        gc.collect()
    return row


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--instances", type=Path, default=Path("Instances 3"))
    parser.add_argument("--pattern", default="*.txt")
    parser.add_argument("--time", type=float, default=300.0)
    parser.add_argument("--output", type=Path, default=Path("results/milp_decoder_batch"))
    parser.add_argument("--decoder", type=Path, default=Path("build-msvc-nmake-release/pdp_decode_encoding.exe"))
    parser.add_argument("--max-truck-trips", type=int, default=None)
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--threads-per-job", type=int, default=1)
    parser.add_argument("--polish-time", type=float, default=30.0)
    parser.add_argument("--fresh", action="store_true")
    args = parser.parse_args(list(argv) if argv is not None else None)

    args.output.mkdir(parents=True, exist_ok=True)
    summary = args.output / "summary.csv"
    if args.fresh and summary.exists():
        summary.unlink()
    completed = completed_instances(summary)
    instances = sorted(
        path for path in args.instances.glob(args.pattern)
        if path.name != "mô tả dữ liệu.txt"
    )

    pending = [path for path in instances if path.name not in completed]
    print(
        f"Batch: {len(pending)} pending, {len(completed)} completed, "
        f"jobs={args.jobs}, threads/job={args.threads_per_job}",
        flush=True,
    )
    with ProcessPoolExecutor(max_workers=args.jobs) as pool:
        futures = {
            pool.submit(
                solve_instance,
                path.resolve(),
                args.time,
                args.output.resolve(),
                args.decoder.resolve(),
                args.max_truck_trips,
                args.threads_per_job,
                args.polish_time,
            ): path
            for path in pending
        }
        done_count = len(completed)
        for future in as_completed(futures):
            path = futures[future]
            try:
                row = future.result()
            except Exception as exc:
                row = {
                    "instance": path.name,
                    "note": f"WORKER ERROR: {type(exc).__name__}: {exc}",
                }
            append_result(summary, row)
            done_count += 1
            print(
                f"[{done_count}/{len(instances)}] DONE {path.name}: "
                f"MILP={row.get('milp_cmax', 'NA')} "
                f"decode={row.get('decode_cmax', 'NA')} "
                f"valid={row.get('decode_valid', 'NA')} {row.get('note', '')}",
                flush=True,
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
