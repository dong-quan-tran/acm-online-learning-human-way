"""Compare the shared loader with Nguyen's original saved split."""
import argparse
import json
from pathlib import Path
from src.data import get_online_loader

def verify(membership, original, seed):
    saved = json.loads(Path(original).read_text(encoding="utf-8"))
    expected_train = saved["stream_train_split_rows"]
    expected_val = saved["val_train_split_rows"]
    train = get_online_loader(membership=membership, seed=seed)
    val = get_online_loader("validation", membership=membership, seed=seed)
    actual_train = train.dataset.indices
    actual_val = val.dataset.indices
    checks = {
        "training_count": len(actual_train) == len(expected_train) == 9000,
        "validation_count": len(actual_val) == len(expected_val) == 1000,
        "training_ids_and_order": actual_train == expected_train,
        "validation_ids_and_order": actual_val == expected_val,
        "training_unique": len(set(actual_train)) == 9000,
        "validation_unique": len(set(actual_val)) == 1000,
        "no_overlap": not set(actual_train).intersection(actual_val),
    }
    print(json.dumps(checks, indent=2))
    if not all(checks.values()):
        raise SystemExit("Shared split verification failed")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--membership", default="data/shared_c20_membership.json")
    parser.add_argument("--original", required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    verify(args.membership, args.original, args.seed)
