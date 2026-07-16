import argparse
import gzip
import json
import os
import pickle
import shutil
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

from pxdesign.utils.infer import convert_to_bioassembly_dict
from pxdesign.utils.inputs import parse_yaml_to_json


def _str2bool(v: str) -> bool:
    return str(v).lower() in {"1", "true", "yes", "y", "on"}


@dataclass
class RebuildJob:
    yaml_path: str
    source_structure_path: str
    output_pkl_path: str


@dataclass
class RebuildResult:
    yaml_path: str
    source_structure_path: str
    output_pkl_path: str
    status: str
    message: str
    backup_path: str | None = None


def _resolve_yaml_files(
    yaml_dir: str | None,
    yaml_list: list[str] | None,
    yaml_manifest: str | None,
    yaml_glob: list[str],
) -> list[Path]:
    files: list[Path] = []

    if yaml_list:
        files.extend(Path(x).expanduser().resolve() for x in yaml_list)

    if yaml_manifest:
        manifest_path = Path(yaml_manifest).expanduser().resolve()
        if not manifest_path.exists():
            raise FileNotFoundError(f"yaml_manifest not found: {manifest_path}")
        if not manifest_path.is_file():
            raise FileNotFoundError(f"yaml_manifest is not a regular file: {manifest_path}")
        for raw_line in manifest_path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            p = Path(line).expanduser()
            if not p.is_absolute():
                p = (manifest_path.parent / p).resolve()
            else:
                p = p.resolve()
            files.append(p)

    if yaml_dir:
        base = Path(yaml_dir).expanduser().resolve()
        if not base.is_dir():
            raise FileNotFoundError(f"yaml_dir not found or not a directory: {base}")
        for pattern in yaml_glob:
            files.extend(base.rglob(pattern))

    uniq: list[Path] = []
    seen: set[Path] = set()
    for p in sorted(files):
        if p.suffix.lower() not in {".yaml", ".yml"}:
            continue
        if p in seen:
            continue
        seen.add(p)
        uniq.append(p)

    if not uniq:
        raise ValueError("No YAML files found. Check --yaml_dir / --yaml_list / --yaml_glob.")

    for p in uniq:
        if not p.exists():
            raise FileNotFoundError(f"YAML file not found: {p}")
        if not p.is_file():
            raise FileNotFoundError(f"Not a regular file: {p}")
    return uniq


def _parse_yaml_task(yaml_path: Path) -> dict:
    cwd = os.getcwd()
    try:
        os.chdir(str(yaml_path.parent))
        parsed = parse_yaml_to_json(str(yaml_path), json_path=None)
    finally:
        os.chdir(cwd)
    if not isinstance(parsed, list) or len(parsed) != 1:
        raise ValueError(f"Expected one task from YAML, got {type(parsed)} for {yaml_path}")
    return parsed[0]


def _resolve_target_file(yaml_path: Path, structure_file: str) -> Path:
    p = Path(structure_file).expanduser()
    if p.is_absolute():
        return p.resolve()
    return (yaml_path.parent / p).resolve()


def _derive_source_structure_path(
    yaml_path: Path,
    structure_file: str,
    structure_dir: Path | None,
) -> Path:
    target_path = _resolve_target_file(yaml_path, structure_file)
    target_name = target_path.name

    if target_name.endswith(".cif") or target_name.endswith(".pdb"):
        if target_path.exists():
            return target_path
        if structure_dir is not None:
            alt = (structure_dir / target_name).resolve()
            if alt.exists():
                return alt
        raise FileNotFoundError(f"Source structure file not found: {target_path}")

    if target_name.endswith(".pkl.gz"):
        stem = target_name[: -len(".pkl.gz")]
        candidates: list[Path] = [
            (yaml_path.parent / f"{stem}.cif").resolve(),
            (yaml_path.parent / f"{stem}.pdb").resolve(),
        ]
        if structure_dir is not None:
            candidates.extend(
                [
                    (structure_dir / f"{stem}.cif").resolve(),
                    (structure_dir / f"{stem}.pdb").resolve(),
                ]
            )
        for candidate in candidates:
            if candidate.exists():
                return candidate
        raise FileNotFoundError(
            f"Cannot infer source CIF/PDB for {yaml_path.name} from target.file={structure_file}"
        )

    raise ValueError(
        f"Unsupported target.file for {yaml_path.name}: {structure_file}. "
        "Expected .cif, .pdb, or .pkl.gz"
    )


def _derive_output_pkl_path(
    yaml_path: Path,
    structure_file: str,
    source_structure_path: Path,
    output_dir: Path,
) -> Path:
    target_path = _resolve_target_file(yaml_path, structure_file)
    if target_path.name.endswith(".pkl.gz"):
        return (output_dir / target_path.name).resolve()
    return (output_dir / f"{source_structure_path.stem}.pkl.gz").resolve()


def _validate_rebuilt_pkl(path: Path) -> None:
    with gzip.open(path, "rb") as f:
        obj = pickle.load(f)
    if not isinstance(obj, dict):
        raise ValueError(f"Rebuilt pkl is not a dict: {path}")
    if "atom_array" not in obj:
        raise ValueError(f"Rebuilt pkl missing atom_array: {path}")
    atom_array = obj["atom_array"]
    try:
        atom_count = len(atom_array)
    except Exception as e:
        raise ValueError(f"Invalid atom_array in rebuilt pkl: {path}") from e
    if atom_count <= 0:
        raise ValueError(f"Rebuilt pkl has empty atom_array: {path}")


def _copy_backup_if_needed(src: Path, backup_path: Path) -> None:
    backup_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, backup_path)


def _build_job(
    yaml_path: Path,
    structure_dir: Path | None,
    output_dir: Path,
) -> RebuildJob:
    task = _parse_yaml_task(yaml_path)
    cond = task.get("condition", {})
    structure_file = cond.get("structure_file")
    if not isinstance(structure_file, str):
        raise ValueError(f"Missing condition.structure_file in {yaml_path}")
    source_structure_path = _derive_source_structure_path(
        yaml_path, structure_file, structure_dir
    )
    output_pkl_path = _derive_output_pkl_path(
        yaml_path, structure_file, source_structure_path, output_dir
    )
    return RebuildJob(
        yaml_path=str(yaml_path),
        source_structure_path=str(source_structure_path),
        output_pkl_path=str(output_pkl_path),
    )


def _run_one_job(
    job: RebuildJob,
    backup_dir: str | None,
    force: bool,
    dry_run: bool,
) -> RebuildResult:
    yaml_path = Path(job.yaml_path)
    source_structure_path = Path(job.source_structure_path)
    output_pkl_path = Path(job.output_pkl_path)

    if output_pkl_path.exists() and not force:
        return RebuildResult(
            yaml_path=job.yaml_path,
            source_structure_path=job.source_structure_path,
            output_pkl_path=job.output_pkl_path,
            status="skipped",
            message="output exists and --force is false",
        )

    if dry_run:
        return RebuildResult(
            yaml_path=job.yaml_path,
            source_structure_path=job.source_structure_path,
            output_pkl_path=job.output_pkl_path,
            status="dry_run",
            message="would rebuild and replace",
        )

    task = _parse_yaml_task(yaml_path)
    task["condition"]["structure_file"] = str(source_structure_path)

    output_pkl_path.parent.mkdir(parents=True, exist_ok=True)
    backup_path_str: str | None = None
    with tempfile.TemporaryDirectory(
        prefix="pxdesign_rebuild_pkl_", dir=str(output_pkl_path.parent)
    ) as tmp_dir_str:
        tmp_dir = Path(tmp_dir_str)
        convert_to_bioassembly_dict(task, str(tmp_dir))
        tmp_pkl_path = tmp_dir / f"{source_structure_path.stem}.pkl.gz"
        if not tmp_pkl_path.exists():
            raise FileNotFoundError(f"Expected rebuilt pkl not found: {tmp_pkl_path}")
        _validate_rebuilt_pkl(tmp_pkl_path)

        if output_pkl_path.exists() and backup_dir is not None:
            backup_root = Path(backup_dir)
            backup_path = (backup_root / output_pkl_path.name).resolve()
            _copy_backup_if_needed(output_pkl_path, backup_path)
            backup_path_str = str(backup_path)

        os.replace(tmp_pkl_path, output_pkl_path)

    return RebuildResult(
        yaml_path=job.yaml_path,
        source_structure_path=job.source_structure_path,
        output_pkl_path=job.output_pkl_path,
        status="rebuilt",
        message="success",
        backup_path=backup_path_str,
    )


def _write_failures_log(path: Path, failures: list[RebuildResult]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for item in failures:
            f.write(json.dumps(asdict(item), ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Rebuild local PXDesign-compatible pkl.gz caches from the source "
            "CIF/PDB files referenced by YAMLs, then atomically replace the "
            "existing pkl.gz files."
        )
    )
    parser.add_argument("--yaml_dir", type=str, default=None)
    parser.add_argument("--yaml_list", nargs="+", default=None)
    parser.add_argument("--yaml_manifest", type=str, default=None)
    parser.add_argument(
        "--yaml_glob",
        nargs="+",
        default=["*.yaml", "*.yml"],
        help="Glob pattern(s) used under --yaml_dir.",
    )
    parser.add_argument(
        "--structure_dir",
        type=str,
        default=None,
        help=(
            "Optional directory to search for source CIF/PDB files when target.file "
            "already points to a pkl.gz."
        ),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory where rebuilt pkl.gz files will be written/replaced.",
    )
    parser.add_argument(
        "--backup_dir",
        type=str,
        default=None,
        help=(
            "Backup directory for overwritten pkl.gz files. "
            "Default: <output_dir>/_pkl_backups/<timestamp>"
        ),
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=max(1, min((os.cpu_count() or 1), 8)),
        help="Number of worker processes.",
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--dry_run", type=_str2bool, default=False)
    parser.add_argument("--continue_on_error", type=_str2bool, default=True)
    parser.add_argument("--force", type=_str2bool, default=True)
    parser.add_argument(
        "--failures_log",
        type=str,
        default=None,
        help="Optional jsonl path to store failed rebuild records.",
    )
    args = parser.parse_args()

    if not args.yaml_dir and not args.yaml_list and not args.yaml_manifest:
        raise ValueError(
            "You must provide at least one of --yaml_dir, --yaml_list or --yaml_manifest."
        )

    yaml_files = _resolve_yaml_files(
        args.yaml_dir, args.yaml_list, args.yaml_manifest, args.yaml_glob
    )
    if args.limit is not None:
        yaml_files = yaml_files[: max(0, args.limit)]

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    structure_dir = (
        Path(args.structure_dir).expanduser().resolve()
        if args.structure_dir is not None
        else None
    )

    backup_dir: str | None
    if args.dry_run:
        backup_dir = None
    elif args.backup_dir:
        backup_dir = str(Path(args.backup_dir).expanduser().resolve())
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_dir = str((output_dir / "_pkl_backups" / timestamp).resolve())

    jobs_by_output: dict[str, RebuildJob] = {}
    duplicate_yaml_count = 0
    for yaml_path in yaml_files:
        job = _build_job(yaml_path, structure_dir, output_dir)
        if job.output_pkl_path in jobs_by_output:
            duplicate_yaml_count += 1
            continue
        jobs_by_output[job.output_pkl_path] = job

    jobs = list(jobs_by_output.values())
    print(
        f"[RebuildPKL] YAML files={len(yaml_files)} unique_outputs={len(jobs)} "
        f"duplicate_yaml_refs={duplicate_yaml_count}"
    )

    results: list[RebuildResult] = []
    failures: list[RebuildResult] = []
    with ProcessPoolExecutor(max_workers=max(1, args.workers)) as executor:
        future_to_job = {
            executor.submit(_run_one_job, job, backup_dir, args.force, args.dry_run): job
            for job in jobs
        }
        for future in as_completed(future_to_job):
            job = future_to_job[future]
            try:
                result = future.result()
            except Exception as e:  # noqa: BLE001
                result = RebuildResult(
                    yaml_path=job.yaml_path,
                    source_structure_path=job.source_structure_path,
                    output_pkl_path=job.output_pkl_path,
                    status="failed",
                    message=str(e),
                )
                if not args.continue_on_error:
                    results.append(result)
                    failures.append(result)
                    print(
                        f"[RebuildPKL] failed yaml={job.yaml_path} output={job.output_pkl_path} "
                        f"error={result.message}"
                    )
                    break
            results.append(result)
            if result.status == "failed":
                failures.append(result)
            print(
                f"[RebuildPKL] {result.status} yaml={result.yaml_path} "
                f"source={result.source_structure_path} output={result.output_pkl_path} "
                f"message={result.message}"
            )

    if args.failures_log and failures:
        _write_failures_log(Path(args.failures_log).expanduser().resolve(), failures)

    rebuilt = sum(r.status == "rebuilt" for r in results)
    skipped = sum(r.status == "skipped" for r in results)
    dry_run = sum(r.status == "dry_run" for r in results)
    failed = sum(r.status == "failed" for r in results)
    print(
        f"[RebuildPKL] done rebuilt={rebuilt} skipped={skipped} dry_run={dry_run} "
        f"failed={failed} backup_dir={backup_dir}"
    )

    if failed > 0 and not args.continue_on_error:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
