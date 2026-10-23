"""Tek görüntü veya klasörde pipeline çalıştırıp sonuç CSV'si üretir."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.pipeline import PipelineConfig, process_file


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="SmartDoc klasik görüntü işleme pipeline'ı")
    parser.add_argument("input", nargs="?", default="data/test", help="Görüntü dosyası veya klasör (varsayılan: data/test)")
    parser.add_argument("--out-dir", default="results/pipeline", help="Çıktı klasörü")
    parser.add_argument("--strategy", choices=("canny", "threshold"), default="canny")
    parser.add_argument("--canny-mode", choices=("fixed", "auto", "otsu"), default="fixed")
    parser.add_argument("--binarize", choices=("none", "otsu", "adaptive_mean", "adaptive_gaussian"), default="otsu")
    parser.add_argument("--allow-unrectified", action="store_true")
    return parser


def run() -> None:
    args = _parser().parse_args()
    source = Path(args.input)
    out_dir = Path(args.out_dir)
    config = PipelineConfig(
        detection_strategy=args.strategy,
        canny_mode=args.canny_mode,
        binarize=args.binarize,
        allow_unrectified=args.allow_unrectified,
    )
    if source.is_file():
        files = [source]
    elif source.is_dir():
        files = sorted(path for path in source.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES)
    else:
        raise FileNotFoundError(f"Girdi yolu bulunamadı: {source}")
    if not files:
        print(f"İşlenecek fotoğraf yok: {source}")
        return

    rows: list[dict[str, object]] = []
    for path in files:
        try:
            result = process_file(path, out_dir / path.stem, config)
            before, after = result.info_before, result.info_after
            rows.append({
                "Dosya": str(path),
                "Durum": result.detection.status,
                "Uyarılar": " | ".join(result.warnings),
                "Adım_süreleri_ms": ";".join(f"{s['ad']}={float(s['sure_ms']):.2f}" for s in result.steps),
                "Parlaklık_önce": before.get("mean_brightness", ""),
                "Kontrast_önce": before.get("std_contrast", ""),
                "Keskinlik_önce": before.get("sharpness", ""),
                "Parlaklık_sonra": after.get("mean_brightness", ""),
                "Kontrast_sonra": after.get("std_contrast", ""),
                "Keskinlik_sonra": after.get("sharpness", ""),
            })
        except (OSError, ValueError, TypeError) as exc:
            rows.append({"Dosya": str(path), "Durum": "error", "Uyarılar": str(exc)})
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("Dosya", "Durum", "Uyarılar", "Adım_süreleri_ms", "Parlaklık_önce", "Kontrast_önce", "Keskinlik_önce", "Parlaklık_sonra", "Kontrast_sonra", "Keskinlik_sonra"))
        writer.writeheader()
        writer.writerows(rows)
    print(f"{len(rows)} dosya işlendi; özet: {out_dir / 'summary.csv'}")


if __name__ == "__main__":
    run()
