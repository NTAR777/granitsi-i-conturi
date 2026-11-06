#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""CLI лабораторной работы «Границы и контуры объектов».

Примеры:
    python cli.py --source coins
    python cli.py --source coins -m canny --canny-sigma 3 --min-area 200
    python cli.py --source ~/Pictures/scan --out results/scan
    python cli.py --list
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # CLI рисует только в файлы

import pipeline as P  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    par = argparse.ArgumentParser(
        prog="cli.py",
        description="Выделение границ (Sobel/Scharr/Canny), морфологическая очистка "
                    "масок, поиск контуров и подсчёт объектов по площади.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    par.add_argument("--source", "-s", default="coins",
                     help=f"демо ({'/'.join(P.DEMO_SOURCES)}) или путь к файлу/папке")
    par.add_argument("--methods", "-m", nargs="+", choices=P.METHOD_ORDER, default=None,
                     help="какие детекторы границ запускать")
    par.add_argument("--sobel-thresh", type=float, default=None, dest="sobel_thresh",
                     help="порог Sobel: доля от max магнитуды (0..1)")
    par.add_argument("--scharr-thresh", type=float, default=None, dest="scharr_thresh",
                     help="порог Scharr: доля от max магнитуды (0..1)")
    par.add_argument("--canny-sigma", type=float, default=None, dest="canny_sigma",
                     help="Canny: сигма гауссова сглаживания, px")
    par.add_argument("--canny-low", type=float, default=None, dest="canny_low",
                     help="Canny: нижний порог гистерезиса (0..1)")
    par.add_argument("--canny-high", type=float, default=None, dest="canny_high",
                     help="Canny: верхний порог гистерезиса (0..1)")
    par.add_argument("--morph-op", choices=("none", "opening", "closing", "both"),
                     default=None, dest="morph_op",
                     help="морфологическая операция очистки масок")
    par.add_argument("--morph-size", type=int, default=None, dest="morph_size",
                     help="радиус структурного элемента disk, px")
    par.add_argument("--morph-iters", type=int, default=None, dest="morph_iters",
                     help="число итераций морфологии")
    par.add_argument("--no-fill-holes", action="store_true", dest="no_fill_holes",
                     help="не заливать дыры в масках перед подсчётом")
    par.add_argument("--min-area", type=float, default=None, dest="min_area",
                     help="минимальная площадь объекта, px")
    par.add_argument("--max-area", type=float, default=None, dest="max_area",
                     help="максимальная площадь объекта, px (0 — без ограничения)")
    par.add_argument("--out", "-o", type=Path, default=None,
                     help="папка результата (по умолчанию results/<имя источника>)")
    par.add_argument("--no-panel", action="store_true", dest="no_panel",
                     help="не сохранять сводный panel.png")
    par.add_argument("--list", action="store_true",
                     help="показать доступные демо-изображения и выйти")
    return par


def params_from_args(args: argparse.Namespace) -> P.Params:
    p = P.Params(methods=tuple(args.methods or P.METHOD_ORDER))
    if args.sobel_thresh is not None:
        p.sobel_thresh = args.sobel_thresh
    if args.scharr_thresh is not None:
        p.scharr_thresh = args.scharr_thresh
    if args.canny_sigma is not None:
        p.canny_sigma = args.canny_sigma
    if args.canny_low is not None:
        p.canny_low = args.canny_low
    if args.canny_high is not None:
        p.canny_high = args.canny_high
    if args.morph_op is not None:
        p.morph_op = args.morph_op
    if args.morph_size is not None:
        p.morph_size = args.morph_size
    if args.morph_iters is not None:
        p.morph_iters = args.morph_iters
    if args.no_fill_holes:
        p.fill_holes = False
    if args.min_area is not None:
        p.min_area = args.min_area
    if args.max_area is not None:
        p.max_area = args.max_area
    return P.validate_params(p)


def run_frame(img, name: str, p: P.Params, out_dir: Path, with_panel: bool) -> dict:
    """Обработать один кадр, сохранить «до/после», вернуть счётчики."""
    result = P.process(img, p, name=name)
    P.save_result(result, p, out_dir, with_panel=with_panel)
    return {m: info["count"] for m, info in result["methods"].items()}


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if args.list:
        print("Демо-изображения scikit-image:", ", ".join(P.DEMO_SOURCES))
        return 0

    try:
        p = params_from_args(args)
    except ValueError as e:
        print(f"Ошибка параметров: {e}", file=sys.stderr)
        return 2

    src = args.source
    path = Path(src).expanduser()
    out_root = args.out
    summary = []

    try:
        if src in P.DEMO_SOURCES:
            out_root = out_root or Path("results") / src
            counts = run_frame(P.load_demo(src), src, p, out_root, not args.no_panel)
            summary.append((src, counts))
        elif path.is_dir():
            files = P.list_image_files(path)
            if not files:
                print(f"В папке {path} нет изображений", file=sys.stderr)
                return 2
            out_root = out_root or Path("results") / path.name
            for f in files:
                img = P.load_image_file(f)
                counts = run_frame(img, f.stem, p, out_root / f.stem, not args.no_panel)
                summary.append((f.name, counts))
        elif path.is_file():
            img = P.load_image_file(path)
            out_root = out_root or Path("results") / path.stem
            counts = run_frame(img, path.stem, p, out_root, not args.no_panel)
            summary.append((path.name, counts))
        else:
            print(f"Источник «{src}» не найден. Демо: {', '.join(P.DEMO_SOURCES)}",
                  file=sys.stderr)
            return 2
    except (OSError, ValueError) as e:
        print(f"Ошибка: {e}", file=sys.stderr)
        return 1

    print("\nКоличество объектов на кадр (после фильтра по площади):")
    for name, counts in summary:
        row = "  ".join(f"{m}={cnt}" for m, cnt in counts.items())
        print(f"  {name}: {row}")
    print(f"\nРезультат сохранён в: {out_root.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
