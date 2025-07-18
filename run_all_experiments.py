import argparse
import subprocess
import sys
from pathlib import Path
import yaml
import itertools 

DATA_ROOT = Path("~/shared/Ivana_GNN/Sateliti/GNSSGraphDetect/data/parsed").expanduser()
PARAMS_FILE = Path("params.yaml")
EXCLUDED_JAMMING_TYPES = {"none", ".ipynb_checkpoints"}
EXCLUDED_RECEIVERS = {"Ublox6", ".ipynb_checkpoints"}

def _split(comma_separated: str | None) -> set | None:
    return set(comma_separated.split(",")) if comma_separated else None

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run GNSS-Graph-Detect experiments.", allow_abbrev=False)
    p.add_argument("-r", "--receivers", help="Comma-separated list of receivers to INCLUDE")
    p.add_argument("-j", "--jams", help="Comma-separated list of jamming types to INCLUDE")
    p.add_argument("-p", "--powers", help="Comma-separated list of jamming power folders to INCLUDE")
    p.add_argument("--seeds", help="Comma-separated list of seeds (default: 42,145,156,5,85)")
    p.add_argument("--dry-run", action="store_true", help="Print planned experiments and exit")
    return p.parse_args()

def discover_experiments(root: Path, want_receivers: set, want_jams: set, want_powers: set) -> list[dict]:
    '''
    Prolazi kroz direktorije i pronalazi sve validne kombinacije
    receiver/jamming_type/power.
    '''
    experiments = []
    for receiver_dir in root.iterdir():
        if not receiver_dir.is_dir() or receiver_dir.name in EXCLUDED_RECEIVERS:
            continue
        if want_receivers and receiver_dir.name not in want_receivers:
            continue

        for jam_dir in receiver_dir.iterdir():
            if not jam_dir.is_dir() or jam_dir.name in EXCLUDED_JAMMING_TYPES:
                continue
            if want_jams and jam_dir.name not in want_jams:
                continue

            for power_dir in jam_dir.iterdir():
                if not power_dir.is_dir():
                    continue
                if want_powers and power_dir.name not in want_powers:
                    continue
                
                experiments.append({
                    "receiver": receiver_dir.name,
                    "jamming_type": jam_dir.name,
                    "power": power_dir.name
                })
    return experiments

def run_single_experiment(exp_params: dict, seed: int, dry_run: bool) -> bool:
    '''
    Ažurira params.yaml i pokreće 'dvc repro' za jedan eksperiment
    Vraća True ako je uspješno, False ako nije.
    '''
    exp_name = f"exp_{exp_params['receiver']}_{exp_params['jamming_type']}_{exp_params['power']}_seed{seed}".replace("-", "m")
    print(f"> Running: {exp_name}")
    
    if dry_run:
        return True

    #  Update params.yaml 
    try:
        with open(PARAMS_FILE, "r") as f:
            params = yaml.safe_load(f)
    except FileNotFoundError:
        print(f"ERROR: {PARAMS_FILE} not found!", file=sys.stderr)
        return False

    params["receiver_name"] = exp_params["receiver"]
    params["jamming_type"] = exp_params["jamming_type"]
    params["jamming_power"] = exp_params["power"]
    params["seed"] = seed

    if "train" in params:
        params["train"]["output_dir"] = f"reports/{exp_params['receiver']}/{exp_params['jamming_type']}/{exp_params['power']}/seed_{seed}"
    params["output_dir"] = f"data/processed/{exp_params['receiver']}/{exp_params['jamming_type']}/{exp_params['power']}/seed_{seed}"

    with open(PARAMS_FILE, "w") as f:
        yaml.dump(params, f)

    #  Run DVC pipeline : 
    res = subprocess.run(["dvc", "repro"], check=False)
    #res = subprocess.run(["dvc", "repro"], check=False, capture_output=True, text=True)
    if res.returncode != 0:
        print(f"! Experiment {exp_name} FAILED", file=sys.stderr)
        print("--- DVC STDOUT ---", file=sys.stderr)
        print(res.stdout, file=sys.stderr)
        print("--- DVC STDERR ---", file=sys.stderr)
        print(res.stderr, file=sys.stderr)
        return False
    
    return True

def main() -> None:
    ns = parse_args()
    want_receivers = _split(ns.receivers)
    want_jams      = _split(ns.jams)
    want_powers    = _split(ns.powers)
    seeds = [int(s.strip()) for s in ns.seeds.split(",")] if ns.seeds else [42, 145, 156, 5, 85]

    print("--- Discovering experiments ---")
    experiments_to_run = discover_experiments(DATA_ROOT, want_receivers, want_jams, want_powers)
    
    if not experiments_to_run:
        print("No experiment combinations found for the selected filters.")
        return

    # Spajamo eksperimente i seedove u jednu listu
    all_combinations = list(itertools.product(experiments_to_run, seeds))
    print(f"Found {len(experiments_to_run)} experiment configurations, running for {len(seeds)} seeds.")
    print(f"Total experiments to run: {len(all_combinations)}")

    all_failures = []
    for exp_params, seed in all_combinations:
        success = run_single_experiment(exp_params, seed, ns.dry_run)
        if not success:
            exp_name = f"{exp_params['receiver']}_{exp_params['jamming_type']}_{exp_params['power']}_seed{seed}"
            all_failures.append(exp_name)

    #  Ispis rezultata 
    if ns.dry_run:
        print("\n(Dry-run complete - no experiments executed.)")
        return
    
    if all_failures:
        print("\n--- Some experiments failed: ---")
        for name in all_failures:
            print(f" - {name}")
    else:
        print("\n--- All selected experiments completed successfully!--- 

if __name__ == "__main__":
    main()
    