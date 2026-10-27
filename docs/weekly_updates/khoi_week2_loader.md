# Khoi Week 2 shared loader evidence

Date: 2026-10-27. Branch: `khoi/online-data-loader`.

## Repository and dependency status

Started clean at `752de0e`. Updated main with `git pull --ff-only origin main`;
main remained `c2ac92d` (Revise assignment instructions and team roles).
`git merge --no-edit main` on Khoi's branch reported already up to date.
No conflicts or local work requiring preservation.

After `git fetch origin`, inspected `origin/nguyen/small-resnet` at `8060729`
using `git ls-tree -r --name-only` and the notebook cells. Its only training
notebook is the old class-incremental notebook. Neither that branch nor main
contains Nguyen's current online trainer, saved 9000/1000 row membership,
or stream-order metadata. The local data/config inventory also has no such
membership. `data/task_split.json` defines classes, not validation membership.
Baseline/EWC configurations belong to the older experiment.

**Pending:** Nguyen must publish his original train/validation source-row IDs
and current online training code, including its preprocessing settings. The
actual experiment membership and trainer integration cannot be verified yet.
The old notebook was not changed. No replacement experiment split was saved.

## Loader contract

`get_online_loader(split="train", batch_size=32, seed=42, membership=...)`
accepts a mapping or JSON path with these fields:

- `source_split`: `"train"` (original Hugging Face CIFAR-100 training rows).
- `train`: exactly 9000 original integer row IDs.
- `validation`: exactly 1000 original integer row IDs.

Supply Nguyen's original membership, converting its representation to these
fields if needed without changing row membership. No membership is generated
by the loader; omission raises an explicit dependency error. Validation uses
`split="validation"` and retains the supplied row order. The union must be
exactly the 10000 training-source rows with fine labels 0-19, with no duplicates
or overlap. Stream seed shuffles training order only; DataLoader does not
shuffle again. Items return `(image, label, "train:<original row>")`.

Existing `ToTensor()` is unchanged: float CHW images scaled to [0, 1], with
no added normalization. Matching unpublished trainer preprocessing remains
pending; no transform agreement is claimed.

Once the membership file exists, the shared interface is:

```python
from src.data import get_online_loader

train = get_online_loader(membership="path/to/published_membership.json")
validation = get_online_loader("validation", membership="path/to/published_membership.json")
for images, labels, sample_ids in train:
    # Pass images and labels to the published trainer; retain IDs for auditing.
    pass
```

## Actual checks and results

Tests use real cached CIFAR-100 images/labels. Their **test-only** membership
uses the first 9000 selected source rows for training and remaining 1000 for
validation. This exercises the contract and is NOT Nguyen's membership or an
approved experimental split. The source dataset is loaded once per suite and
reused by test fixtures; no test images or membership files are committed.

PowerShell commands (project virtual environment; cached dataset, no downloads):

```powershell
$env:HF_HUB_OFFLINE='1'
$env:HF_DATASETS_OFFLINE='1'
$env:PYTEST_DEBUG_TEMPROOT=(Resolve-Path .pytest_cache).Path
.\.venv\Scripts\python.exe -m pytest tests/test_online_loader.py -v
.\.venv\Scripts\python.exe -m pytest -v
```

Actual final output:

```text
13 passed in 29.30s
14 passed in 27.79s
```

An earlier attempt hit a sandbox temporary-directory error; placing pytest's
temporary root in the existing repository cache fixed it. Tests also check
that missing or invalid membership fails rather than silently creating a split.

Verified on the test-only membership:

- Train 9000; validation 1000; overlap 0; union 10000; labels 0-19.
- Same seed (42) gives identical order; seed 43 changes order but not membership.
- Validation membership and order are unchanged across stream seeds.
- All 9000 training IDs appear exactly once, in dataset order.
- Batch size 32; `drop_last=False`; 282 batches; final batch 8.
- Train and validation image batches `(32, 3, 32, 32)`, labels `(32,)`, 32 IDs.
- Example training IDs: `train:23112`, `train:16572`, `train:17820`.
- Example validation IDs: `train:44801`, `train:44810`, `train:44811`.
- Source rows, labels, pixel scaling, and JSON membership loading checked.

Lightweight integration: `test_batch_consumer_integration` loads both splits,
unpacks images/labels/IDs, runs cross-entropy and backward through a small
100-output PyTorch consumer, and checks a validation forward pass. This is
an interface smoke check, not Nguyen's unpublished online pipeline. The
separate local inspection of both batch types produced the shapes and IDs
above. No training accuracy or completed experiment is claimed.
