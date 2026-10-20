"""Train one fold of the HGNetV2-B0 SED branch on the pilot cache (Kaggle GPU kernel).

Training data is every pseudo-labelled window, every focal clip (a random 5 s crop
per visit) and the labelled soundscape windows outside fold K, drawn uniformly
from one concatenated dataset. 13th place found that this simple mix beat
ratio-tuned sampling once the pseudo-labels were strong (0.934 vs 0.894). Labelled
fold K is held out as a sanity check only; the leaderboard decides.

Outputs, in /kaggle/working or $SED_OUT:
    sed_fold{K}_ema.pt   EMA weights (state_dict of the SED module)
    train_log.csv        epoch, train_loss, val_auc, sec, samples_per_sec, loader_wait_frac
    val_pred.npy         EMA probabilities on labelled fold K, in labelled_meta.csv order
    env.txt              torch / timm / CUDA versions, GPU, peak memory

    python3 sed_train.py --selftest
    SED_CACHE=... SED_OUT=... python3 sed_train.py --smoke
"""
import os
import subprocess
import sys
import time

import numpy as np
import pandas as pd
import torch
from torch.utils.data import ConcatDataset, DataLoader, Dataset

WIN = 160_000
EPOCHS, WARMUP, BATCH = 20, 5, 64
LR, WD, EMA_DECAY = 5e-4, 1e-4, 0.999
TIMM = "timm==1.0.30"


def labelled_folds(meta, k=4, seed=0):
    """Fold per labelled window, grouped by file, the same on every platform."""
    from sklearn.model_selection import GroupKFold

    fold = np.zeros(len(meta), int)
    split = GroupKFold(k, shuffle=True, random_state=seed).split(meta, groups=meta.filename)
    for f, (_, va) in enumerate(split):
        fold[va] = f
    return fold


class WindowSet(Dataset):
    """Rows of an [N, 160000] int16 array. The memmap is opened lazily, inside each
    worker: pickling an open np.memmap copies the whole array (macOS spawns workers)."""

    def __init__(self, path, targets, rows=None):
        self.path, self.targets = path, targets
        self.rows = np.arange(len(targets)) if rows is None else np.asarray(rows)
        self._a = None

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        if self._a is None:
            self._a = np.load(self.path, mmap_mode="r")
        r = self.rows[i]
        return self._a[r].astype(np.float32) / 32767, self.targets[r].astype(np.float32)


class FocalSet(Dataset):
    """A random 5 s crop of each focal clip, zero-padded when the clip is shorter."""

    def __init__(self, path, index, targets, seed=0):
        self.path, self.targets, self.seed = path, targets, seed
        self.offset, self.length = index.offset.to_numpy(), index.length.to_numpy()
        self._a = self.rng = None

    def __len__(self):
        return len(self.offset)

    def __getitem__(self, i):
        if self._a is None:
            self._a = np.load(self.path, mmap_mode="r")
            # Each worker holds a copy of this object; seed from torch's per-worker
            # seed so the workers do not all draw the same sequence of crops.
            info = torch.utils.data.get_worker_info()
            self.rng = np.random.default_rng(self.seed if info is None else info.seed)
        off, n = self.offset[i], self.length[i]
        start = off + (self.rng.integers(0, n - WIN + 1) if n > WIN else 0)
        x = np.zeros(WIN, np.float32)
        seg = self._a[start:start + min(n, WIN)]
        x[:len(seg)] = seg.astype(np.float32) / 32767
        return x, self.targets[i].astype(np.float32)


def mixup(x, y, lam, perm):
    return lam * x + (1 - lam) * x[perm], lam * y + (1 - lam) * y[perm]


def find(marker, root="/kaggle/input"):
    for dirpath, _, files in os.walk(root):
        if marker in files:
            return dirpath
    raise FileNotFoundError(f"{marker} not found under {root}")


def pseudo_targets(cache, labels, root="/kaggle/input"):
    """The targets file for a teacher: in the cache if it is there, else wherever
    it was shipped under root (Phase B's targets are a dataset of their own)."""
    name = f"pseudo_targets_{labels}.npy"
    path = f"{cache}/{name}"
    return path if os.path.exists(path) else f"{find(name, root)}/{name}"


@torch.no_grad()
def predict(model, fe, ds, device, batch=64):
    model.eval()
    out = []
    for x, _ in DataLoader(ds, batch_size=batch):
        out.append(torch.sigmoid(model(fe(x.to(device)))[0]).float().cpu().numpy())
    return np.concatenate(out)


def main():
    args = sys.argv[1:]
    get = lambda flag, default: args[args.index(flag) + 1] if flag in args else default
    fold, labels = int(get("--fold", 0)), get("--labels", "A")
    smoke = "--smoke" in args
    epochs, batch = (1, 8) if smoke else (int(get("--epochs", EPOCHS)), BATCH)

    local = "SED_CACHE" in os.environ
    cache = os.environ["SED_CACHE"] if local else find("pseudo_audio.npy")
    code = os.path.dirname(os.path.abspath(__file__)) if local else find("sed_model.py")
    out = os.environ.get("SED_OUT", "/kaggle/working")
    os.makedirs(out, exist_ok=True)
    sys.path.insert(0, code)
    try:
        import timm
    except ModuleNotFoundError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", TIMM])
        import timm
    from cv import macro_auc
    from sed_model import SED, LogMel, sed_loss

    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    cuda = device == "cuda"
    torch.manual_seed(fold)
    lm = pd.read_csv(f"{cache}/labelled_meta.csv")
    lt = np.load(f"{cache}/labelled_targets.npy")
    folds = labelled_folds(lm)
    targets = pseudo_targets(cache, labels)
    print(f"pseudo targets: {targets}", flush=True)
    train_ds = ConcatDataset([
        WindowSet(f"{cache}/pseudo_audio.npy", np.load(targets)),
        FocalSet(f"{cache}/focal_audio.npy", pd.read_csv(f"{cache}/focal_index.csv"),
                 np.load(f"{cache}/focal_targets.npy"), seed=fold),
        WindowSet(f"{cache}/labelled_audio.npy", lt, rows=np.where(folds != fold)[0]),
    ])
    val_rows = np.where(folds == fold)[0]
    val_ds = WindowSet(f"{cache}/labelled_audio.npy", lt, rows=val_rows)
    workers = 0 if smoke else os.cpu_count()
    loader = DataLoader(train_ds, batch_size=batch, shuffle=True, drop_last=True,
                        num_workers=workers, persistent_workers=workers > 0, pin_memory=cuda)
    steps = 2 if smoke else len(loader)
    print(f"device {device}, train {len(train_ds):,} (pseudo {len(train_ds.datasets[0]):,}, "
          f"focal {len(train_ds.datasets[1]):,}, labelled {len(train_ds.datasets[2])}), "
          f"val {len(val_ds)}, {steps} steps x {epochs} epochs", flush=True)

    fe = LogMel().to(device)
    model = SED(pretrained=not smoke).to(device)
    ema = timm.utils.ModelEmaV3(model, decay=EMA_DECAY, use_warmup=True)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=LR, total_steps=epochs * steps,
                                                pct_start=min(WARMUP / epochs, 0.3))
    scaler = torch.amp.GradScaler(enabled=cuda)
    beta = torch.distributions.Beta(1.0, 1.0)

    log = []
    for epoch in range(epochs):
        model.train()
        t0 = tick = time.time()
        wait, total, seen = 0.0, 0.0, 0
        for step, (x, y) in enumerate(loader):
            wait += time.time() - tick
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            with torch.no_grad():
                mel = fe(x)
            mel, y = mixup(mel, y, float(beta.sample()), torch.randperm(len(y), device=device))
            with torch.autocast("cuda", enabled=cuda):
                clip, frame = model(mel)
            loss = sed_loss(clip.float(), frame.float(), y)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            ema.update(model)
            total += loss.item()
            seen += len(y)
            tick = time.time()
            if step + 1 >= steps:
                break
        sec = time.time() - t0
        val = predict(ema.module, fe, val_ds, device)
        auc = macro_auc(lt[val_rows].astype(np.float32), val)
        log.append(dict(epoch=epoch, train_loss=total / steps, val_auc=auc, sec=sec,
                        samples_per_sec=seen / sec, loader_wait_frac=wait / sec))
        print(f"epoch {epoch:2d}  loss {total / steps:.4f}  val_auc {auc:.4f}  {sec:6.0f}s  "
              f"{seen / sec:6.0f}/s  wait {wait / sec:.0%}", flush=True)

    torch.save(ema.module.state_dict(), f"{out}/sed_fold{fold}_ema.pt")
    np.save(f"{out}/val_pred.npy", val)
    pd.DataFrame(log).to_csv(f"{out}/train_log.csv", index=False)
    with open(f"{out}/env.txt", "w") as fh:
        fh.write(f"torch {torch.__version__}\ntimm {timm.__version__}\n"
                 f"cuda {torch.version.cuda}\ndevice {device}\n")
        if cuda:
            fh.write(f"gpu {torch.cuda.get_device_name()}\n"
                     f"peak_mem_gb {torch.cuda.max_memory_allocated() / 1e9:.2f}\n")
    print(f"wrote {out}", flush=True)


def selftest():
    meta = pd.DataFrame({"filename": [f"f{i}.ogg" for i in range(6) for _ in range(3)],
                         "start_s": [0, 5, 10] * 6})
    f = labelled_folds(meta)
    assert (f == labelled_folds(meta)).all(), "folds must be deterministic"
    assert meta.assign(fold=f).groupby("filename").fold.nunique().max() == 1
    assert set(f) == {0, 1, 2, 3}, set(f)

    short, long_ = 3 * 32_000, 10 * 32_000
    audio = np.concatenate([np.full(short, 1000, np.int16), np.full(long_, -2000, np.int16)])
    index = pd.DataFrame({"filename": ["a.ogg", "b.ogg"], "offset": [0, short],
                          "length": [short, long_]})
    targets = np.zeros((2, 234), np.float16)
    targets[0, 3] = targets[1, 7] = 1
    np.save(path := f"/tmp/sed_train_selftest_{os.getpid()}.npy", audio)
    try:
        fs = FocalSet(path, index, targets, seed=0)
        for _ in range(5):
            x, y = fs[0]
            assert x.shape == (160_000,) and y[3] == 1 and y.sum() == 1
            assert np.allclose(x[:short], 1000 / 32767) and (x[short:] == 0).all()
            x, _ = fs[1]
            assert x.shape == (160_000,) and np.allclose(x, -2000 / 32767), "crop left clip b"

        # Every DataLoader worker gets a copy of the dataset; if they also share one
        # RNG state, worker 0's k-th crop equals worker 1's k-th crop.
        ramp = np.arange(60 * 32_000, dtype=np.int64) % 30_000
        np.save(path, ramp.astype(np.int16))
        same = pd.DataFrame({"filename": ["c.ogg"] * 4, "offset": [0] * 4, "length": [len(ramp)] * 4})
        ld = DataLoader(FocalSet(path, same, np.zeros((4, 234), np.float16)), batch_size=2,
                        num_workers=2)
        a, b = [x for x, _ in ld]
        assert not torch.equal(a[0], b[0]), "workers drew identical crops"
    finally:
        os.remove(path)

    # Phase B's targets ship in their own dataset, next to a cache that holds A's.
    import tempfile

    with tempfile.TemporaryDirectory() as d:
        os.makedirs(f"{d}/cache")
        os.makedirs(f"{d}/input/targets/v1")
        np.save(f"{d}/cache/pseudo_targets_A.npy", np.zeros(1))
        np.save(f"{d}/input/targets/v1/pseudo_targets_B.npy", np.zeros(1))
        assert pseudo_targets(f"{d}/cache", "A", f"{d}/input") == f"{d}/cache/pseudo_targets_A.npy"
        assert pseudo_targets(f"{d}/cache", "B", f"{d}/input") == \
            f"{d}/input/targets/v1/pseudo_targets_B.npy"

    torch.manual_seed(0)
    xb, yb = torch.randn(4, 1, 8, 8), torch.rand(4, 234)
    perm = torch.tensor([1, 0, 3, 2])
    xm, ym = mixup(xb, yb, 0.3, perm)
    assert torch.allclose(ym, 0.3 * yb + 0.7 * yb[perm])
    assert torch.allclose(xm, 0.3 * xb + 0.7 * xb[perm])
    print("sed_train self-checks passed")


if __name__ == "__main__":
    selftest() if "--selftest" in sys.argv else main()
