from pathlib import Path
import sys

REQUIRED = [
    'README.md',
    'REPORT_TEMPLATE.md',
    'requirements.txt',
    'src/dataset.py',
    'src/model.py',
    'src/train.py',
    'src/utils.py',
    'docs/LAB_GUIDE_LAB2.md',
    'docs/WANDB_GUIDE.md',
    'configs/baseline_scratch.json',
    'configs/baseline_transfer.json',
    'ci/smoke_train.py',
    '.github/workflows/validate-lab2.yml',
]

root = Path(__file__).resolve().parents[1]
missing = [p for p in REQUIRED if not (root / p).exists()]
if missing:
    print('Missing files:')
    for item in missing:
        print('-', item)
    sys.exit(1)
print('Structure OK')
