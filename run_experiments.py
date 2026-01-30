import argparse
from pathlib import Path
from typing import Any, Dict
import yaml

from yolo import YoloTrainer
from deeplabv3 import DeeplabTrainer
from segformer import SegformerTrainer

TRAINERS = {
    "yolo": YoloTrainer,
    "deeplabv3": DeeplabTrainer,
    "segformer": SegformerTrainer,
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


def run_experiment(exp: Dict[str, Any], defaults: Dict[str, Any], repo_root: Path) -> None:
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

    trainer = trainer_cls()
    trainer.load_model(**exp.get("load", {}))

    save_flag = exp.get("save", defaults.get("save", True))
    train_params = exp.get("params", {})
    trainer.train(str(dataset_dir), dst=str(dst), save=save_flag, **train_params)

    test_cfg = exp.get("test", {})
    if test_cfg.get("enabled", False):
        test_dst = dst / "test"
        trainer.test(
            str(dataset_dir),
            dst=str(test_dst),
            file_name=test_cfg.get("file_name", "test.txt"),
            color_coded=test_cfg.get("color_coded", False),
            log=test_cfg.get("log", True),
            save=test_cfg.get("save", True),
        )

    trainer.remove_model()


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
        run_experiment(exp, defaults, repo_root)


if __name__ == "__main__":
    main()
