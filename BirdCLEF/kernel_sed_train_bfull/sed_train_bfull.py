"""Full-scale B: sed_train.py unchanged, on all 127,104 unlabelled windows (--labels Bfull).

Their audio (~41 GB as int16) is too big to upload, so it is decoded here from the
competition's train_soundscapes onto scratch disk, by the pilot cache's own code.
The pilot cache's focal and labelled files and the Bfull targets are linked beside it.
KAGGLE_INPUT, SED_SCRATCH and SED_SMOKE exist for the local test only.
"""
import os
import runpy
import shutil
import subprocess
import sys
import time

ROOT = os.environ.get("KAGGLE_INPUT", "/kaggle/input")
SCRATCH = os.environ.get("SED_SCRATCH", "/tmp/sedfull")


def find(marker):
    for dirpath, _, files in os.walk(ROOT):
        if marker in files:
            return dirpath
    raise FileNotFoundError(f"{marker} not found under {ROOT}")


def main():
    try:
        import soundfile  # noqa: F401
    except ModuleNotFoundError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "soundfile"])

    code, comp = find("sed_train.py"), find("taxonomy.csv")
    cache, targets = find("focal_audio.npy"), find("pseudo_targets_Bfull.npy")
    sys.path.insert(0, code)

    import pandas as pd

    from sed_cache import stage_full

    os.makedirs(SCRATCH, exist_ok=True)
    mem = "n/a"
    if os.path.exists("/proc/meminfo"):
        with open("/proc/meminfo") as fh:
            mem = next(line.split(":")[1].strip() for line in fh if line.startswith("MemTotal"))
    print(f"code={code} comp={comp} cache={cache} targets={targets}", flush=True)
    print(f"scratch {SCRATCH}: {shutil.disk_usage(SCRATCH).free / 1e9:.1f} GB free; "
          f"MemTotal {mem}; {os.cpu_count()} CPUs", flush=True)

    meta = pd.read_csv(f"{targets}/full_meta.csv")
    info = stage_full(SCRATCH, meta, f"{comp}/train_soundscapes", cache,
                      f"{targets}/pseudo_targets_Bfull.npy", os.cpu_count())
    print(f"decoded {info['rows']:,} windows in {info['sec'] / 60:.1f} min "
          f"({info['rows'] / info['sec']:.0f} windows/s)", flush=True)

    os.environ["SED_CACHE"] = SCRATCH
    os.environ.setdefault("SED_OUT", "/kaggle/working")
    sys.argv = ["sed_train.py", "--labels", "Bfull"] + (["--smoke"] if os.environ.get("SED_SMOKE") else [])
    t0 = time.time()
    runpy.run_path(f"{code}/sed_train.py", run_name="__main__")
    print(f"training finished in {(time.time() - t0) / 60:.1f} min", flush=True)


# The decode pool's workers re-import this file under spawn/forkserver; only the
# parent may stage and train.
if __name__ == "__main__":
    main()
