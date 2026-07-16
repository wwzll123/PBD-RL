#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
from collections import defaultdict
from pathlib import Path

import gemmi


def _load_structure(path: Path) -> gemmi.Structure:
    if path.suffix.lower() == ".cif":
        return gemmi.make_structure_from_block(gemmi.cif.read_file(str(path)).sole_block())
    return gemmi.read_structure(str(path))


def _dist(a, b) -> float:
    return math.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2 + (a.z - b.z) ** 2)


def _atom_map(residue) -> dict[str, gemmi.Position]:
    return {atom.name.strip(): atom.pos for atom in residue}


def _quantile(values: list[float], q: float) -> float:
    values = sorted(values)
    return values[min(len(values) - 1, int(q * (len(values) - 1)))]


def _summarize(values: list[float]) -> dict[str, float]:
    values = sorted(v for v in values if math.isfinite(v))
    if not values:
        return {"n": 0.0, "p50": float("nan"), "p95": float("nan"), "min": float("nan"), "max": float("nan")}
    return {
        "n": float(len(values)),
        "min": values[0],
        "p50": _quantile(values, 0.5),
        "p95": _quantile(values, 0.95),
        "max": values[-1],
    }


def check_file(path: Path) -> tuple[bool, list[str]]:
    structure = _load_structure(path)
    messages: list[str] = []
    ok = True
    for mi, model in enumerate(structure):
        if mi > 0:
            break
        for chain in model:
            residues = list(chain)
            metrics: dict[str, list[float]] = defaultdict(list)
            for i, residue in enumerate(residues):
                atoms = _atom_map(residue)
                for key, a, b in [("N-CA", "N", "CA"), ("CA-C", "CA", "C"), ("C-O", "C", "O")]:
                    if a in atoms and b in atoms:
                        metrics[key].append(_dist(atoms[a], atoms[b]))
                if i + 1 < len(residues):
                    next_atoms = _atom_map(residues[i + 1])
                    if "CA" in atoms and "CA" in next_atoms:
                        metrics["CA-CA_next"].append(_dist(atoms["CA"], next_atoms["CA"]))
                    if "C" in atoms and "N" in next_atoms:
                        metrics["C-N_next"].append(_dist(atoms["C"], next_atoms["N"]))

            limits = {
                "N-CA": (1.0, 2.0),
                "CA-C": (1.0, 2.1),
                "C-O": (0.8, 1.7),
                "CA-CA_next": (2.8, 4.8),
                "C-N_next": (0.8, 2.0),
            }
            for key, (lo, hi) in limits.items():
                stat = _summarize(metrics[key])
                messages.append(
                    f"{path.name} chain={chain.name} {key} "
                    f"n={int(stat['n'])} p50={stat['p50']:.3f} p95={stat['p95']:.3f}"
                )
                if stat["n"] <= 0 or not (lo <= stat["p50"] <= hi):
                    ok = False
    return ok, messages


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="+")
    args = parser.parse_args()
    all_ok = True
    for raw in args.paths:
        ok, messages = check_file(Path(raw))
        all_ok = all_ok and ok
        for msg in messages:
            print(msg)
    if not all_ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
