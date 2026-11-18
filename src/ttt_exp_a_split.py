import json
from pathlib import Path

def load_exp_a_split(path):
    """Return ordered CIFAR-10 test row IDs for adaptation and held-out scoring."""
    with Path(path).open(encoding="utf-8") as handle:
        manifest = json.load(handle)

    def test_rows(key):
        ids = manifest[key]
        if not ids:
            raise ValueError(f"{key} is empty")
        rows = []
        for source_id in ids:
            split, row = source_id.split(":", 1)
            if split != "test" or not row.isdecimal() or int(row) >= 10000:
                raise ValueError(f"Invalid CIFAR-10 test ID: {source_id}")
            rows.append(int(row))
        if len(rows) != len(set(rows)):
            raise ValueError(f"{key} contains duplicate rows")
        return rows

    adaptation = test_rows("adaptation_ids")
    held_out = test_rows("held_out_ids")
    development = test_rows("development_ids")
    if set(adaptation) & set(held_out) or set(development) & (set(adaptation) | set(held_out)):
        raise ValueError("Development, adaptation and held-out rows must be disjoint")
    return adaptation, held_out
