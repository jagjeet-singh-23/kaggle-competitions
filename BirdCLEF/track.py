"""BirdCLEF's experiment history in MLflow: every leaderboard submission and every SED
training run, in one place (notebooks/experiments.ipynb reads them back).

Two experiments:
    birdclef-lb         one run per Kaggle submission: what it was, LB public and private,
                        and local CV where RESULTS.md has a platform-independent one
    birdclef-sed-train  one run per SED training kernel: val_auc, loss, speed and loader
                        wait per epoch; train_log.csv, env.txt and the kernel log attached

Scores come from the Kaggle CLI. What each submission was is SUBMISSIONS below, written
from RESULTS.md. Every run carries a `key` tag, so a rerun adds only what is missing: a
submission still PENDING gets its run now and its scores on the next run.

    MLFLOW_TRACKING_URI=http://127.0.0.1:5050 python3 track.py
    python3 track.py --selftest
"""
import csv
import io
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime, timezone

JOBS = os.environ.get("BIRDCLEF_JOBS", "/Volumes/Extreme SSD/kaggle/BirdCLEF/jobs")
COMPETITION = "birdclef-2026"
BLEND = "birdclef26-perch-sed-blend"

# Every SED run shares this recipe (sed_model.py, sed_train.py).
SED_RECIPE = {"backbone": "hgnetv2_b0.ssld_stage2_ft_in1k", "n_mels": 256, "epochs": 20,
              "warmup_epochs": 5, "batch": 64, "optimizer": "AdamW", "lr": 5e-4,
              "weight_decay": 1e-4, "schedule": "OneCycle", "mixup": True, "ema_decay": 0.999,
              "fold": 0}

TRAINING = {
    "sed-A": {"dir": "sed-out", "kernel": "birdclef26-sed-train", "labels": "A",
              "teacher": "public pseudo-labels (pseudo_labels_v2s_oof_raw.csv)"},
    "sed-B": {"dir": "sed-out-b", "kernel": "birdclef26-sed-train-b", "labels": "B",
              "teacher": "Perch kernel, quantile-mapped onto A"},
    "sed-Bfull": {"dir": "sed-out-bfull", "kernel": "birdclef26-sed-train-bfull", "labels": "Bfull",
                  "teacher": "Perch kernel, quantile-mapped onto A"},
}

BIRDNET = {"encoder": "BirdNET V2.4", "head": "one-vs-rest logistic, C=0.01, balanced",
           "zero_shot_blend": "mapped columns", "cold_start": "none", "smoothing": "none"}
COLD = {"cold_start": "focal heads, 30 per class"}
SMOOTH = {"smoothing": "k=6 max, alpha=0.2"}
PERCH = {**BIRDNET, **COLD, **SMOOTH, "encoder": "Perch v2"}

# local_cv: 5-fold grouped CV over the 739 labelled windows, 10 seeds paired (RESULTS
# section 14). Earlier local numbers moved by up to 0.02 between machines (section 13),
# and cold-start heads are invisible to local CV, so the other runs have none.
SUBMISSIONS = {
    56580377: {"name": "BirdNET v1", "family": "birdnet", "kernel": "birdclef26-birdnet-head",
               "params": BIRDNET},
    56583907: {"name": "BirdNET v3: + cold-start heads", "family": "birdnet",
               "kernel": "birdclef26-birdnet-head", "params": {**BIRDNET, **COLD}},
    56602966: {"name": "BirdNET v4: + smoothing", "family": "birdnet",
               "kernel": "birdclef26-birdnet-head", "local_cv": 0.8880,
               "params": {**BIRDNET, **COLD, **SMOOTH}},
    57422504: {"name": "Perch v1", "family": "perch", "kernel": "birdclef26-perch-head",
               "local_cv": 0.9144, "params": PERCH},
    57434345: {"name": "SED pilot A + Perch, w=0.5", "family": "perch+sed", "kernel": BLEND,
               "kernel_version": 1, "train": "sed-A",
               "params": {**PERCH, "mode": "blend", "w_sed": 0.5, "sed_onnx": "sed_fold0.onnx"}},
    57434639: {"name": "SED pilot A alone", "family": "sed", "kernel": BLEND, "kernel_version": 2,
               "train": "sed-A", "params": {"mode": "sed", "sed_onnx": "sed_fold0.onnx"}},
    57441058: {"name": "SED pilot B + Perch, w=0.5", "family": "perch+sed", "kernel": BLEND,
               "kernel_version": 3, "train": "sed-B",
               "params": {**PERCH, "mode": "blend", "w_sed": 0.5, "sed_onnx": "sed_fold0_B.onnx"}},
    57441268: {"name": "SED pilot B alone", "family": "sed", "kernel": BLEND, "kernel_version": 4,
               "train": "sed-B", "params": {"mode": "sed", "sed_onnx": "sed_fold0_B.onnx"}},
    57592821: {"name": "SED full B + Perch, w=0.5", "family": "perch+sed", "kernel": BLEND,
               "kernel_version": 6, "train": "sed-Bfull",
               "params": {**PERCH, "mode": "blend", "w_sed": 0.5, "sed_onnx": "sed_fold0_Bfull.onnx"}},
    57592972: {"name": "SED full B alone", "family": "sed", "kernel": BLEND, "kernel_version": 7,
               "train": "sed-Bfull", "params": {"mode": "sed", "sed_onnx": "sed_fold0_Bfull.onnx"}},
    57594049: {"name": "SED full B + Perch, w=0.25 (probe)", "family": "perch+sed", "kernel": BLEND,
               "kernel_version": 8, "train": "sed-Bfull",
               "params": {**PERCH, "mode": "blend", "w_sed": 0.25, "sed_onnx": "sed_fold0_Bfull.onnx"}},
}


def to_ms(date):
    """Kaggle's UTC timestamp, e.g. '2026-10-24 11:33:09.777000', in epoch milliseconds."""
    dt = datetime.strptime(date, "%Y-%m-%d %H:%M:%S.%f").replace(tzinfo=timezone.utc)
    return round(dt.timestamp() * 1000)


def kaggle_submissions():
    exe = os.path.join(os.path.dirname(sys.executable), "kaggle")
    cmd = [exe if os.path.exists(exe) else "kaggle", "competitions", "submissions", COMPETITION,
           "--csv", "--page-size", "200"]
    return subprocess.run(cmd, capture_output=True, text=True, check=True).stdout


def parse_submissions(text):
    """The CLI's CSV (after any warning lines) -> {ref: row}; a blank score is None."""
    lines = text.splitlines()
    start = next(i for i, line in enumerate(lines) if line.startswith("ref,"))
    score = lambda v: float(v) if v else None
    return {int(r["ref"]): {"date": r["date"], "description": r["description"],
                            "status": r["status"].removeprefix("SubmissionStatus."),
                            "public": score(r["publicScore"]), "private": score(r["privateScore"])}
            for r in csv.DictReader(io.StringIO("\n".join(lines[start:])))}


def log_lines(text):
    """A Kaggle kernel log, a JSON list of stream chunks, as lines."""
    return "".join(e.get("data", "") for e in json.loads(text)).splitlines()


def train_counts(lines):
    pat = (r"device \S+, train ([\d,]+) \(pseudo ([\d,]+), focal ([\d,]+), labelled ([\d,]+)\), "
           r"val ([\d,]+), (\d+) steps")
    m = next(m for line in lines if (m := re.match(pat, line)))
    keys = ("n_train", "n_pseudo", "n_focal", "n_labelled", "n_val", "steps_per_epoch")
    return {k: int(v.replace(",", "")) for k, v in zip(keys, m.groups())}


def decode_min(lines):
    """Minutes the in-kernel audio decode took; only full-scale B decodes, others None."""
    for line in lines:
        if m := re.match(r"decoded [\d,]+ windows in ([\d.]+) min", line):
            return float(m.group(1))
    return None


def read_env(text):
    return dict(line.split(" ", 1) for line in text.splitlines() if line.strip())


def experiment(client, name, note):
    exp = client.get_experiment_by_name(name)
    eid = exp.experiment_id if exp else client.create_experiment(name)
    client.set_experiment_tag(eid, "mlflow.note.content", note)
    return eid


def find_run(client, eid, key):
    runs = client.search_runs([eid], filter_string=f"tags.key = '{key}'")
    return runs[0] if runs else None


def log_training(client, eid, key):
    """One run per SED training kernel. Its outputs never change, so a logged run is left alone."""
    from mlflow.entities import Metric, Param

    if run := find_run(client, eid, key):
        return run.info.run_id, False
    t = TRAINING[key]
    d = f"{JOBS}/{t['dir']}/fold0"
    files = [f"{d}/{t['kernel']}.log", f"{d}/env.txt", f"{d}/train_log.csv"]
    with open(files[0]) as fh:
        lines = log_lines(fh.read())
    with open(files[1]) as fh:
        env = read_env(fh.read())
    with open(files[2]) as fh:
        epochs = list(csv.DictReader(fh))

    params = {**SED_RECIPE, **train_counts(lines), "kernel": t["kernel"], "labels": t["labels"],
              "teacher": t["teacher"], **{k: v for k, v in env.items() if k != "peak_mem_gb"}}
    now = round(time.time() * 1000)
    metrics = [Metric(name, float(row[name]), now, int(row["epoch"])) for row in epochs
               for name in ("val_auc", "train_loss", "sec", "samples_per_sec", "loader_wait_frac")]
    auc = [float(row["val_auc"]) for row in epochs]
    summary = {"val_auc_final": auc[-1], "val_auc_max": max(auc), "peak_mem_gb": float(env["peak_mem_gb"]),
               "gpu_min": sum(float(row["sec"]) for row in epochs) / 60}
    if (dm := decode_min(lines)) is not None:
        summary["decode_min"] = dm

    rid = client.create_run(eid, run_name=key, tags={"key": key, "kernel": t["kernel"]}).info.run_id
    client.log_batch(rid, metrics=metrics + [Metric(k, v, now, 0) for k, v in summary.items()],
                     params=[Param(k, str(v)) for k, v in params.items()])
    for f in files:
        client.log_artifact(rid, f)
    client.set_terminated(rid)
    return rid, True


def log_submission(client, eid, sid, sub, score, train_runs):
    """One run per submission. A logged run gets only the metrics it lacks, which is how a
    submission first logged while PENDING picks up its scores."""
    from mlflow.entities import Metric, Param

    when = to_ms(score["date"]) if score else round(time.time() * 1000)
    metrics = {"local_cv": sub.get("local_cv")}
    if score:
        metrics |= {"lb_public": score["public"], "lb_private": score["private"]}
    metrics = {k: v for k, v in metrics.items() if v is not None}

    key = f"lb:{sid}"
    run = find_run(client, eid, key)
    new = run is None
    if new:
        tags = {"key": key, "submission_id": str(sid), "family": sub["family"], "kernel": sub["kernel"]}
        params = {"kernel": sub["kernel"], **sub["params"]}
        if "kernel_version" in sub:
            params["kernel_version"] = sub["kernel_version"]
        if "train" in sub:
            tags["sed_train_run_id"] = train_runs[sub["train"]]
            params["sed_train"] = sub["train"]
        if score:
            params |= {"date": score["date"], "description": score["description"]}
        run = client.create_run(eid, start_time=when, run_name=sub["name"], tags=tags)
        client.log_batch(run.info.run_id, params=[Param(k, str(v)) for k, v in params.items()])
        client.set_terminated(run.info.run_id, end_time=when)
    rid = run.info.run_id
    added = sorted(k for k in metrics if new or k not in run.data.metrics)
    if added:
        client.log_batch(rid, metrics=[Metric(k, metrics[k], when, 0) for k in added])
    if score:
        client.set_tag(rid, "status", score["status"])
    return new, added


def selftest():
    text = ("Warning: Looks like you're using an outdated `kaggle` version\n"
            "ref,fileName,date,description,status,publicScore,privateScore\n"
            '2,submission.csv,2026-10-24 12:20:48.733000,"a, b",SubmissionStatus.PENDING,,\n'
            "1,submission.csv,2026-09-26 13:26:29.450000,v1,SubmissionStatus.COMPLETE,0.83593,0.83575\n")
    s = parse_submissions(text)
    assert set(s) == {1, 2}, s
    assert s[1]["public"] == 0.83593 and s[1]["private"] == 0.83575 and s[1]["status"] == "COMPLETE"
    assert s[2]["public"] is None and s[2]["private"] is None and s[2]["status"] == "PENDING"
    assert s[2]["description"] == "a, b"
    assert to_ms("1970-01-01 00:00:01.500000") == 1500

    lines = ["pseudo targets: /tmp/sedfull/pseudo_targets_Bfull.npy",
             "device cuda, train 132,952 (pseudo 127,104, focal 5,313, labelled 535), val 204, "
             "2077 steps x 20 epochs",
             "decoded 127,104 windows in 6.3 min (337 windows/s)"]
    assert train_counts(lines) == {"n_train": 132952, "n_pseudo": 127104, "n_focal": 5313,
                                   "n_labelled": 535, "n_val": 204, "steps_per_epoch": 2077}
    assert decode_min(lines) == 6.3 and decode_min(lines[:2]) is None
    log = json.dumps([{"stream_name": "stdout", "data": "a\nb\n"}, {"data": "c\n"}])
    assert log_lines(log) == ["a", "b", "c"]
    assert read_env("torch 2.11.0+cu128\ngpu Tesla T4\npeak_mem_gb 3.22\n") == {
        "torch": "2.11.0+cu128", "gpu": "Tesla T4", "peak_mem_gb": "3.22"}

    for sid, sub in SUBMISSIONS.items():
        assert sub.get("train") in (None, *TRAINING), sid
        assert sub["family"] in ("birdnet", "perch", "perch+sed", "sed"), sid
        assert (sub["family"] in ("perch+sed", "sed")) == ("train" in sub), sid
    print("track self-checks passed")


def main():
    if not os.environ.get("MLFLOW_TRACKING_URI"):
        sys.exit("set MLFLOW_TRACKING_URI (the local server is http://127.0.0.1:5050)")
    os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
    from mlflow.tracking import MlflowClient

    client = MlflowClient()
    scores = parse_submissions(kaggle_submissions())
    if unknown := sorted(set(scores) - set(SUBMISSIONS)):
        print(f"not described in SUBMISSIONS, skipped: {unknown}")

    train_eid = experiment(client, "birdclef-sed-train",
                           "BirdCLEF+ 2026 SED training kernels (Kaggle T4), one fold each. "
                           "val_auc is fold 0 of the labelled windows and is inflated: "
                           "every teacher saw those labels (RESULTS.md section 16).")
    lb_eid = experiment(client, "birdclef-lb",
                        "BirdCLEF+ 2026 leaderboard submissions, public and private. "
                        "Competition closed 2026-06-03; later submissions are scored, not ranked.")
    train_runs = {}
    for key in TRAINING:
        train_runs[key], new = log_training(client, train_eid, key)
        print(f"{key}: {'logged' if new else 'already logged'}")
    for sid, sub in SUBMISSIONS.items():
        new, added = log_submission(client, lb_eid, sid, sub, scores.get(sid), train_runs)
        status = scores[sid]["status"] if sid in scores else "not on Kaggle"
        print(f"{sid} {sub['name']}: {'new run' if new else 'exists'}, {status}"
              + (f", logged {', '.join(added)}" if added else ""))


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        selftest()
    else:
        main()
