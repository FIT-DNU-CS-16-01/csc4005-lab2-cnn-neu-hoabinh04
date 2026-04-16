from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
TMP = ROOT / '.tmp_ci_data'
if TMP.exists():
    shutil.rmtree(TMP)
TMP.mkdir(parents=True, exist_ok=True)

classes = ['Crazing', 'Inclusion', 'Patches', 'Pitted_Surface', 'Rolled-in_Scale', 'Scratches']
rng = np.random.default_rng(42)
for class_name in classes:
    class_dir = TMP / class_name
    class_dir.mkdir(parents=True, exist_ok=True)
    prefix = class_name.lower().replace('rolled-in_scale', 'rolled-in_scale')
    for idx in range(10):
        arr = rng.integers(0, 255, size=(64, 64), dtype=np.uint8)
        Image.fromarray(arr, mode='L').save(class_dir / f'{prefix}_{idx:02d}.png')

cmd = [
    sys.executable,
    '-m',
    'src.train',
    '--data_dir', str(TMP),
    '--model_name', 'cnn_small',
    '--train_mode', 'scratch',
    '--run_name', 'ci_smoke',
    '--epochs', '1',
    '--batch_size', '8',
    '--img_size', '64',
    '--patience', '1',
]
subprocess.run(cmd, cwd=ROOT, check=True)
print('Smoke train OK')
