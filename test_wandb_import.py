import sys
sys.path.insert(0, '.')
try:
    from src import train
    print(f"train module imported OK")
    print(f"wandb in train: {train.wandb}")
except Exception as e:
    print(f"Error: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()
