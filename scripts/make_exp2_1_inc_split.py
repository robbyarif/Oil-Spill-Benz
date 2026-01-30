from pathlib import Path
from typing import Dict, Iterable, List

ALLOWED_EXT = {".jpg", ".jpeg", ".png"}
LABEL_EXT = {".png", ".txt"}
SPLIT_MAP: Dict[str, List[str]] = {
    "train": [
        "20240725_喀麥隆籍「多芬DOLPHIN」貨輪擱淺漏油案",
    ],
    "val": [
        "20240725_蒙古籍「凱塔(KETA)」貨輪發電機故障漏油案",
    ],
    "test": [
        "20240725_多哥籍「阿諾(ALANO)」貨輪疑似流錨協處案",
        "20241208_巴拿馬籍「液態寶石」油輪擱淺案",
    ],
}


def collect_images(images_root: Path, labels_root: Path, incident_folders: Iterable[str]) -> List[Path]:
    missing = [name for name in incident_folders if not (images_root / name).exists()]
    if missing:
        raise FileNotFoundError(f"Missing incident folders under {images_root}: {', '.join(missing)}")

    images: List[Path] = []
    for incident in incident_folders:
        incident_dir = images_root / incident
        for path in incident_dir.rglob("*"):
            if path.is_file() and path.suffix.lower() in ALLOWED_EXT:
                rel = path.relative_to(images_root)
                label_stem = (labels_root / rel).with_suffix("")
                has_label = False
                for ext in LABEL_EXT:
                    if (label_stem.with_suffix(ext)).exists():
                        has_label = True
                        break
                if has_label:
                    images.append(path)
                else:
                    print(f"skip (no label): {path}")
    return sorted({p for p in images})


def write_split(paths: List[Path], dst: Path, repo_root: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    with dst.open("w", encoding="utf-8") as f:
        for path in paths:
            rel = path.relative_to(repo_root).as_posix()
            f.write(f"{rel}\n")


def main() -> None:
    repo_root = Path(__file__).resolve().parent.parent
    images_root = repo_root / "DV4" / "images"
    labels_root = repo_root / "DV4" / "labels"
    output_root = repo_root / "datasets" / "processed" / "exp2.1_inc-split"

    for split_name, folders in SPLIT_MAP.items():
        paths = collect_images(images_root, labels_root, folders)
        write_split(paths, output_root / f"{split_name}.txt", repo_root)
        print(f"{split_name}: {len(paths)} images")


if __name__ == "__main__":
    main()
