import argparse
import subprocess
import sys
from pathlib import Path
import yaml
import itertools

PARAMS_FILE = Path("params.yaml")

EXCLUDED_RECEIVERS    = {"Ublox6"}
EXCLUDED_JAMMING_TYPES = {"none"}

REPORTS_ROOT   = "PARSED_REPORTS"
PROCESSED_ROOT = "PARSED_PROCESSED"


def _split(comma_separated: str | None) -> set | None:
    return set(comma_separated.split(",")) if comma_separated else None


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="JaGuard: Automated Experimental Sweep",
        allow_abbrev=False
    )
    p.add_argument(
        "--data-root",
        default="gnss/data/parsed",
        help="Path to root data directory (default: gnss/data/parsed)"
    )
    p.add_argument(
        "-r", "--receivers",
        help="Comma-separated list of receivers (e.g., Ublox10,GP01)"
    )
    p.add_argument(
        "--seeds",
        help="Comma-separated list of seeds (default: 42,789,1011,1263,2024)"
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print experiment matrix without executing"
    )
    return p.parse_args()


def discover_experiments(root: Path, want_receivers: set) -> list[dict]:
    """
    Locates all receiver, jamming_type, and jamming_power combinations.
    Assumes structure: DATA_ROOT / receiver / jamming_type / jamming_power
    """
    experiments = []
    if not root.exists():
        print(f"ERROR: DATA_ROOT {root} not found!")
        return experiments

    for receiver_dir in root.iterdir():
        if not receiver_dir.is_dir() or receiver_dir.name in EXCLUDED_RECEIVERS:
            continue
        if want_receivers and receiver_dir.name not in want_receivers:
            continue

        for jamming_dir in receiver_dir.iterdir():
            if not jamming_dir.is_dir() or jamming_dir.name in EXCLUDED_JAMMING_TYPES:
                continue

            for power_dir in jamming_dir.iterdir():
                if not power_dir.is_dir():
                    continue

                experiments.append({
                    "receiver":     receiver_dir.name,
                    "jamming_type": jamming_dir.name,
                    "jamming_power": power_dir.name
                })

    return experiments


def run_single_experiment(exp_params: dict, seed: int, dry_run: bool) -> bool:
    """
    Updates params.yaml and executes the DVC pipeline for a single configuration.
    """
    rec = exp_params['receiver']
    jt  = exp_params['jamming_type']
    jp  = exp_params['jamming_power']
    exp_id = f"{rec}_{jt}_{jp}_seed{seed}"

    print(f"\n>>> INITIATING EXPERIMENT: {exp_id}")

    if dry_run:
        return True

    try:
        with open(PARAMS_FILE, "r") as f:
            params = yaml.safe_load(f)
    except FileNotFoundError:
        print(f"ERROR: {PARAMS_FILE} missing!", file=sys.stderr)
        return False

    params["receiver_name"] = rec
    params["jamming_type"]  = jt
    params["jamming_power"] = jp
    params["seed"]          = seed

    if "train" not in params:
        params["train"] = {}

    rel_path = f"{rec}/{jt}/{jp}/seed_{seed}"
    params["output_dir"]             = f"{PROCESSED_ROOT}/{rel_path}"
    params["train"]["output_dir"]    = f"{REPORTS_ROOT}/{rel_path}"

    with open(PARAMS_FILE, "w") as f:
        yaml.dump(params, f)

    res = subprocess.run(["dvc", "repro"], check=False)

    if res.returncode != 0:
        print(f"! CRITICAL: Experiment {exp_id} FAILED during DVC execution", file=sys.stderr)
        return False

    return True


def main() -> None:
    ns = parse_args()

    DATA_ROOT      = Path(ns.data_root).expanduser()
    want_receivers = _split(ns.receivers)
    seeds_str      = ns.seeds if ns.seeds else "42,789,1011,1263,2024"
    seeds          = [int(s.strip()) for s in seeds_str.split(",")]

    print("--- Discovering target configurations ---")
    experiments_to_run = discover_experiments(DATA_ROOT, want_receivers)

    if not experiments_to_run:
        print("No valid receiver/jamming configurations found.")
        return

    all_combinations = list(itertools.product(experiments_to_run, seeds))
    print(f"Configurations discovered:    {len(experiments_to_run)}")
    print(f"Seeds to test:                {seeds}")
    print(f"Total pipeline runs scheduled: {len(all_combinations)}")

    all_failures = []

    for exp_params, seed in all_combinations:
        success = run_single_experiment(exp_params, seed, ns.dry_run)
        if not ns.dry_run and not success:
            rec = exp_params['receiver']
            jt  = exp_params['jamming_type']
            jp  = exp_params['jamming_power']
            all_failures.append(f"{rec}_{jt}_{jp}_seed{seed}")

    if ns.dry_run:
        print("\n(Dry-run complete. No changes made.)")
        return

    if all_failures:
        print("\n--- EXPERIMENT FAILURES ---")
        for name in all_failures:
            print(f" - {name}")
    else:
        print("\n--- ALL EXPERIMENTS COMPLETED SUCCESSFULLY! ---")


if __name__ == "__main__":
    main()