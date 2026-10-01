import argparse
import json
import subprocess
import sys
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Dict
import numpy as np
import yaml

from yolo import YoloTrainer
from deeplabv3 import DeeplabTrainer
from segformer import SegformerTrainer
from rfdetr_trainer import RFDETRTrainer

TRAINERS = {
    "yolo": YoloTrainer,
    "deeplabv3": DeeplabTrainer,
    "segformer": SegformerTrainer,
    "rfdetr": RFDETRTrainer,
}


def ensure_split_files(dataset_dir: Path) -> None:
    """Ensure train/val/test files exist; duplicate valid.txt to val.txt if needed."""
    train_file = dataset_dir / "train.txt"
    val_file = dataset_dir / "val.txt"
    valid_file = dataset_dir / "valid.txt"
    test_file = dataset_dir / "test.txt"

    if not val_file.exists() and valid_file.exists():
        val_file.write_text(valid_file.read_text(encoding="utf-8"), encoding="utf-8")

    for required in [train_file, val_file, test_file]:
        if not required.exists():
            raise FileNotFoundError(f"Missing split file: {required}")


IMG_EXTS = {".jpg", ".jpeg", ".png"}


def collect_incident_images(data_root: Path, incidents) -> list:
    """Collect images (that have a YOLO label) from crops/<incident>/<date>/images."""
    images = []
    for incident in incidents:
        incident_dir = data_root / incident
        if not incident_dir.is_dir():
            raise FileNotFoundError(f"Incident folder not found: {incident_dir}")
        for path in sorted(incident_dir.rglob("*")):
            if path.suffix.lower() not in IMG_EXTS or path.parent.name != "images":
                continue
            if path.parent.parent.joinpath("labels", path.stem + ".txt").exists():
                images.append(path.resolve())
    if not images:
        raise ValueError(f"No labeled images found for incidents: {incidents}")
    return images


def write_fold_splits(data_root: Path, fold: Dict[str, Any], split_dir: Path) -> None:
    split_dir.mkdir(parents=True, exist_ok=True)
    for split in ["train", "val", "test"]:
        images = collect_incident_images(data_root, fold[split])
        (split_dir / f"{split}.txt").write_text("\n".join(map(str, images)) + "\n", encoding="utf-8")
        print(f"  {split}: {len(images)} images from {len(fold[split])} incidents")


COUNT_KEYS = ["TP", "FP", "FN", "TN"]

METRICS_DEFINITION = {
    "Acc": "pixel accuracy per image, averaged over test images",
    "F1": "oil-class F1 (2TP/(2TP+FP+FN)) per image, averaged over images where defined",
    "Oil IoU": "oil-class IoU per image, averaged over images where defined",
    "BG IoU": "background-class IoU per image, averaged over images where defined",
    "mIoU": "(Oil IoU + BG IoU) / 2, from per-image IoUs",
    "Oil IoU (agg)": "TP / (TP + FP + FN) from pixel counts summed over all test images",
    "BG IoU (agg)": "TN / (TN + FP + FN) from pixel counts summed over all test images",
    "mIoU (agg)": "(Oil IoU (agg) + BG IoU (agg)) / 2",
    "TP/FP/FN/TN": "pixel counts summed over all test images, oil = positive class",
}


def _to_float(metrics: Dict[str, Any]) -> Dict[str, float]:
    return {k: float(v) for k, v in metrics.items() if v is not None}


def log_test_metrics(run, metrics: Dict[str, float], prefix: str = "test") -> None:
    """Log scalar metrics, TP/FP/FN/TN counts and a confusion matrix to wandb."""
    if run is None:
        return
    import wandb

    run.log({f"{prefix}/{k}": v for k, v in metrics.items()})
    run.summary.update({f"{prefix}/{k}": v for k, v in metrics.items()})

    tp, fp, fn, tn = (int(metrics[k]) for k in COUNT_KEYS)
    table = wandb.Table(
        columns=["Actual", "Predicted", "nPredictions"],
        data=[["oil", "oil", tp], ["oil", "background", fn], ["background", "oil", fp],
              ["background", "background", tn]],
    )
    run.log({f"{prefix}/confusion_matrix": wandb.plot_table(
        "wandb/confusion_matrix/v1", table,
        {"Actual": "Actual", "Predicted": "Predicted", "nPredictions": "nPredictions"},
        {"title": f"Confusion matrix (pixels, {prefix})"},
    )})


def ensure_active_run(run, run_dir: Path):
    """Ultralytics' wandb callback finishes the run at the end of training; reopen the same run to log test results."""
    if run is None:
        return None
    import wandb

    if wandb.run is not None:
        return wandb.run
    return wandb.init(id=run.id, project=run.project, entity=run.entity, resume="allow", dir=str(run_dir))


def find_best_weights(train_dir: Path):
    weights_dir = train_dir / "weights"
    if (weights_dir / "best.pt").exists():
        return weights_dir / "best.pt"
    candidates = sorted(weights_dir.glob("*.pt*")) if weights_dir.is_dir() else []
    return candidates[-1] if candidates else None


def log_provenance(run, run_dir: Path, train_dir: Path, metrics, repo_root: Path) -> None:
    """Link the wandb run to the local run dir / weights and attach the weights as a model artifact."""
    if run is None:
        return
    import wandb

    weights = find_best_weights(train_dir)
    run.config.update({
        "run_dir": str(run_dir.relative_to(repo_root)),
        "weights_path": str(weights.relative_to(repo_root)) if weights else None,
    }, allow_val_change=True)
    if weights is None:
        print(f"No weights found under {train_dir / 'weights'}; skipping model artifact")
        return

    artifact = wandb.Artifact(
        name=f"model-{run.name}".replace(" ", "_"),
        type="model",
        metadata={"git": git_info(repo_root), "run_dir": str(run_dir.relative_to(repo_root)),
                  **{f"test/{k}": v for k, v in (metrics or {}).items()}},
    )
    artifact.add_file(str(weights), name=weights.name)
    run.log_artifact(artifact, aliases=["best", "latest"])


def _git(repo_root: Path, *args: str) -> str:
    try:
        out = subprocess.run(["git", *args], cwd=repo_root, capture_output=True, text=True, check=True)
        return out.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return ""


def git_info(repo_root: Path) -> Dict[str, Any]:
    return {
        "commit": _git(repo_root, "rev-parse", "HEAD"),
        "branch": _git(repo_root, "rev-parse", "--abbrev-ref", "HEAD"),
        "dirty": bool(_git(repo_root, "status", "--porcelain", "--untracked-files=no")),
    }


def wandb_settings(exp: Dict[str, Any], defaults: Dict[str, Any]) -> Dict[str, Any]:
    return {**defaults.get("wandb", {}), **exp.get("wandb", {})}


def init_wandb(cfg: Dict[str, Any], *, exp, model_key, group, name, job_type, run_dir, extra_config,
               split_strategy, repo_root, config_path):
    """Start a wandb run carrying config file, git info and metric definitions. Returns the run or None."""
    if not cfg.get("enabled", False):
        return None
    import wandb

    tags = [model_key, exp["name"], f"split:{split_strategy}", *cfg.get("tags", [])]
    run = wandb.init(
        project=cfg.get("project", "oil-spill-seg"),
        entity=cfg.get("entity"),
        group=group,
        name=name,
        job_type=job_type,
        tags=tags,
        dir=str(run_dir),
        config={
            "experiment": exp["name"],
            "model": model_key,
            "split_strategy": split_strategy,
            "git": git_info(repo_root),
            "config_file": str(config_path.relative_to(repo_root)),
            "experiment_config": exp,  # full experiment block from the YAML
            "metrics_definition": METRICS_DEFINITION,
            **extra_config,
        },
    )
    # the YAML itself, downloadable from the run's Files tab
    run.save(str(config_path), base_path=str(config_path.parent), policy="now")
    return run


class _Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for st in self.streams:
            st.write(data)
            st.flush()
        return len(data)

    def flush(self):
        for st in self.streams:
            st.flush()

    def isatty(self):
        return False


@contextmanager
def tee_log(log_path: Path):
    """Mirror stdout/stderr to log_path while still printing to the terminal."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    out, err = sys.stdout, sys.stderr
    with log_path.open("w", encoding="utf-8") as f:
        sys.stdout, sys.stderr = _Tee(out, f), _Tee(err, f)
        try:
            yield
        finally:
            sys.stdout, sys.stderr = out, err


def run_cross_validation(exp, defaults, repo_root, config_path) -> None:
    """Incident-split k-fold: each fold is trained, validated and tested on its own incident lists."""
    name = exp["name"]
    model_key = exp["model"].lower()
    trainer_cls = TRAINERS.get(model_key)
    if trainer_cls is None:
        raise ValueError(f"Unsupported model '{model_key}'. Supported: {list(TRAINERS.keys())}")

    cv = exp["cross_validation"]
    data_root = repo_root / cv["data_root"]
    folds = cv["folds"]
    split_strategy = cv.get("split_strategy", "incident")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_root = Path(defaults.get("output_root", "runs"))
    group = f"{timestamp}_{name}_{model_key}_incident-split"
    exp_dir = repo_root / output_root / group
    exp_dir.mkdir(parents=True, exist_ok=True)

    wb_cfg = wandb_settings(exp, defaults)
    if wb_cfg.get("enabled", False) and model_key == "yolo":
        from ultralytics import settings
        settings.update({"wandb": True})
    wb_kwargs = dict(exp=exp, model_key=model_key, group=group, split_strategy=split_strategy,
                     repo_root=repo_root, config_path=config_path)

    test_cfg = exp.get("test", {})
    save_flag = exp.get("save", defaults.get("save", True))
    all_metrics = []

    for i, fold in enumerate(folds, start=1):
        fold_name = f"fold{i}"
        print(f"\n--- {name} / {fold_name} ---")
        fold_dir = exp_dir / fold_name
        split_dir = fold_dir / "splits"
        write_fold_splits(data_root, fold, split_dir)

        run = init_wandb(wb_cfg, name=f"{name}_{model_key}_{fold_name}", job_type="train", run_dir=fold_dir,
                         extra_config={"fold": fold_name, "incidents": fold, **exp.get("load", {}),
                                       **exp.get("params", {})}, **wb_kwargs)

        trainer = trainer_cls()
        trainer.load_model(**exp.get("load", {}))
        trainer.train(str(split_dir), dst=str(fold_dir / "train"), save=save_flag, **exp.get("params", {}))

        run = ensure_active_run(run, fold_dir)
        metrics = None
        if test_cfg.get("enabled", True):
            metrics = _to_float(trainer.test(
                str(split_dir),
                dst=str(fold_dir / "test"),
                file_name=test_cfg.get("file_name", "test.txt"),
                color_coded=test_cfg.get("color_coded", False),
                log=test_cfg.get("log", True),
                save=test_cfg.get("save", True),
            ))
            all_metrics.append(metrics)
            log_test_metrics(run, metrics)
        log_provenance(run, fold_dir, fold_dir / "train", metrics, repo_root)

        trainer.remove_model()
        if run is not None:
            run.finish()

    if all_metrics:
        keys = [k for k in all_metrics[0] if k not in COUNT_KEYS]
        summary = {k: {"mean": float(np.mean([m[k] for m in all_metrics])),
                       "std": float(np.std([m[k] for m in all_metrics])),
                       "folds": [m[k] for m in all_metrics]} for k in keys}
        pooled = {k: sum(int(m[k]) for m in all_metrics) for k in COUNT_KEYS}  # counts summed over folds
        (exp_dir / "cv_summary.json").write_text(
            json.dumps({"metrics": summary, "pooled_counts": pooled}, indent=4), encoding="utf-8")
        print("\n----- mean ± std over folds -----")
        for k, v in summary.items():
            print(f"{k:>14} : {v['mean']:.3f} ± {v['std']:.3f}")
        run = init_wandb(wb_cfg, name=f"{name}_{model_key}_cv-summary", job_type="summary", run_dir=exp_dir,
                         extra_config={"folds": folds}, **wb_kwargs)
        if run is not None:
            run.summary.update({f"cv/{k}_{s}": v[s] for k, v in summary.items() for s in ("mean", "std")})
            log_test_metrics(run, {k: float(v) for k, v in pooled.items()}, prefix="cv_pooled")
            run.finish()


def run_experiment(exp: Dict[str, Any], defaults: Dict[str, Any], repo_root: Path, config_path: Path) -> None:
    split_strategy = exp.get("cross_validation", {}).get("split_strategy", "incident") \
        if "cross_validation" in exp else exp.get("split_strategy", "unspecified")
    log_name = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{exp['name']}_{exp['model'].lower()}_{split_strategy}.log"
    with tee_log(repo_root / defaults.get("log_dir", "logs") / log_name):
        _run_experiment(exp, defaults, repo_root, config_path)


def _run_experiment(exp: Dict[str, Any], defaults: Dict[str, Any], repo_root: Path, config_path: Path) -> None:
    if "cross_validation" in exp:
        run_cross_validation(exp, defaults, repo_root, config_path)
        return

    name = exp["name"]
    model_key = exp["model"].lower()
    trainer_cls = TRAINERS.get(model_key)
    if trainer_cls is None:
        raise ValueError(f"Unsupported model '{model_key}'. Supported: {list(TRAINERS.keys())}")

    dataset_dir = repo_root / exp["dataset"]
    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset dir not found: {dataset_dir}")
    ensure_split_files(dataset_dir)

    output_root = Path(defaults.get("output_root", "runs/experiments"))
    dst = repo_root / output_root / name
    dst.mkdir(parents=True, exist_ok=True)

    wb_cfg = wandb_settings(exp, defaults)
    if wb_cfg.get("enabled", False) and model_key == "yolo":
        from ultralytics import settings
        settings.update({"wandb": True})
    train_params = exp.get("params", {})
    run = init_wandb(wb_cfg, exp=exp, model_key=model_key, group=name, name=name, job_type="train", run_dir=dst,
                     extra_config={"dataset": exp["dataset"], **exp.get("load", {}), **train_params},
                     split_strategy=exp.get("split_strategy", "unspecified"), repo_root=repo_root,
                     config_path=config_path)

    trainer = trainer_cls()
    trainer.load_model(**exp.get("load", {}))

    save_flag = exp.get("save", defaults.get("save", True))
    trainer.train(str(dataset_dir), dst=str(dst), save=save_flag, **train_params)

    run = ensure_active_run(run, dst)
    metrics = None
    test_cfg = exp.get("test", {})
    if test_cfg.get("enabled", False):
        metrics = _to_float(trainer.test(
            str(dataset_dir),
            dst=str(dst / "test"),
            file_name=test_cfg.get("file_name", "test.txt"),
            color_coded=test_cfg.get("color_coded", False),
            log=test_cfg.get("log", True),
            save=test_cfg.get("save", True),
        ))
        log_test_metrics(run, metrics)
    log_provenance(run, dst, dst, metrics, repo_root)

    trainer.remove_model()
    if run is not None:
        run.finish()


def load_config(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run multiple segmentation experiments")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/experiments/exp2_1_inc.yaml",
        help="Path to experiment config YAML",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent
    config_path = (repo_root / args.config).resolve()
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    cfg = load_config(config_path)
    defaults = cfg.get("defaults", {})
    experiments = cfg.get("experiments", [])
    if not experiments:
        raise ValueError("No experiments defined in config")

    for exp in experiments:
        print(f"\n=== Running experiment: {exp['name']} ===")
        run_experiment(exp, defaults, repo_root, config_path)


if __name__ == "__main__":
    main()
