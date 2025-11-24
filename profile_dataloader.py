import time
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from training.data.datasets.sope_dataset import SopeCentersnapDataset
from training.data.datasets.sope_dataloader import pad_collate

# ---------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------
DATA_ROOT = "/home/mirshad7/Downloads/data/Omni6DPose/SOPE"
BATCH_SIZE = 16
NUM_WORKERS = 6
N_BATCHES = 20  # number of batches to profile

# ---------------------------------------------------------------------
# 1️⃣ Baseline: measure pure dataset __getitem__
# ---------------------------------------------------------------------
print("Measuring raw dataset (no DataLoader)...")
dataset = SopeCentersnapDataset(DATA_ROOT)

times = []
for i in tqdm(range(50)):
    t0 = time.time()
    sample = dataset[i]
    times.append(time.time() - t0)

print(f"\n[Dataset only] mean={sum(times)/len(times):.4f}s "
      f"std={torch.std(torch.tensor(times)):.4f}s "
      f"max={max(times):.4f}s")

# ---------------------------------------------------------------------
# 2️⃣ Measure full DataLoader iteration
# ---------------------------------------------------------------------
print("\nMeasuring DataLoader (with workers & collate)...")
loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=True,
    persistent_workers=True,
    collate_fn=pad_collate,
)

batch_times = []
collate_times = []

start_all = time.time()
for i, batch in enumerate(loader):
    t_batch = time.time() - start_all
    batch_times.append(t_batch)
    start_all = time.time()
    if i >= N_BATCHES:
        break

print(f"\n[DataLoader total] mean batch={sum(batch_times)/len(batch_times):.3f}s "
      f"(≈ {sum(batch_times)/(len(batch_times)*BATCH_SIZE):.4f}s/sample)")

# ---------------------------------------------------------------------
# 3️⃣ Optional: profile pad_collate separately
# ---------------------------------------------------------------------
print("\nProfiling pad_collate() overhead...")
samples = [dataset[i] for i in range(BATCH_SIZE)]
t0 = time.time()
batch = pad_collate(samples)
collate_time = time.time() - t0
print(f"pad_collate() for {BATCH_SIZE} samples: {collate_time:.4f}s "
      f"(≈ {collate_time/BATCH_SIZE:.4f}s/sample)")

# ---------------------------------------------------------------------
print("\n✅ Profiling complete.")
