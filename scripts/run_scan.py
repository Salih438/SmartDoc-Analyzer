"""Tek görüntüyü veya klasördeki görüntüleri tarayıp PDF üretir."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import cv2

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.analysis.image_info import load_image
from src.export.pdf import images_to_pdf
from src.pipeline import PipelineConfig, process_image
from src.scan import SCAN_MODES, ScanMode, make_scan


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg"}


def _image_files(source: Path) -> list[Path]:
    """Dosya veya klasördeki görüntüleri Unicode uyumlu ve sıralı döndürür."""
    if source.is_file():
        if source.suffix.lower() not in IMAGE_SUFFIXES:
            raise ValueError(f"Desteklenmeyen görüntü uzantısı: {source.suffix}")
        return [source]
    if source.is_dir():
        return sorted(
            (path for path in source.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES),
            key=lambda path: str(path).casefold(),
        )
    raise FileNotFoundError(f"Girdi yolu bulunamadı: {source}")


def run_scan(
    source: str | Path,
    pdf_path: str | Path,
    mode: ScanMode = "color",
    page_size: str = "a4",
    dpi: int = 300,
    allow_unrectified: bool = False,
) -> int:
    """Girdi dosyası/klasörünü işler ve tarama sayfalarını tek PDF'te birleştirir."""
    input_path = Path(source)
    files = _image_files(input_path)
    if not files:
        raise ValueError(f"İşlenecek fotoğraf bulunamadı: {input_path}")

    scans: list[np.ndarray] = []
    config = PipelineConfig(allow_unrectified=allow_unrectified)
    for path in files:
        image = load_image(path)
        result = process_image(image, config)
        scan_source = result.rectified_image
        if scan_source is None:
            if result.detection.status != "ok" and not allow_unrectified:
                raise ValueError(
                    f"{path}: belge sınırı tespit edilemedi "
                    f"({result.detection.status}: {result.detection.reason}); "
                    "tam görüntüyle devam etmek için --allow-unrectified kullanın."
                )
            scan_source = result.original
        scans.append(make_scan(scan_source, mode))

    images_to_pdf(scans, Path(pdf_path), page_size=page_size, dpi=dpi)  # type: ignore[arg-type]
    return len(scans)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Belge görüntülerini tarama PDF'ine dönüştür")
    parser.add_argument("input", help="Tek görüntü veya görüntü klasörü")
    parser.add_argument("--mode", choices=SCAN_MODES, default="color", help="Tarama çıktı modu")
    parser.add_argument("--pdf", required=True, help="Oluşturulacak PDF yolu")
    parser.add_argument("--page-size", choices=("a4", "fit"), default="a4")
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument(
        "--allow-unrectified", action="store_true",
        help="Belge tespiti başarısızsa tam görüntünün kullanılmasını açıkça onayla",
    )
    return parser


def main() -> None:
    parser = _parser()
    args = parser.parse_args()
    try:
        count = run_scan(
            args.input, args.pdf, args.mode, args.page_size, args.dpi, args.allow_unrectified
        )
    except (OSError, TypeError, ValueError, cv2.error) as exc:
        parser.error(f"Tarama/PDF oluşturulamadı: {exc}")
    print(f"{count} görüntü PDF'e aktarıldı: {args.pdf}")


if __name__ == "__main__":
    main()
