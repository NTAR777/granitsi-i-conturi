# -*- coding: utf-8 -*-
"""Ядро лабораторной работы «Границы и контуры объектов».

Пайплайн:
    1. Загрузка изображения (демо scikit-image или файл пользователя) -> полутон.
    2. Границы: Sobel, Scharr (градиентные фильтры) и Canny (гистерезис).
    3. Очистка масок морфологией: открытие/закрытие (disk), заливка дыр.
    4. Контуры и подсчёт объектов (OpenCV), фильтрация по площади в пикселях.
    5. Сохранение «до/после» (PNG), сводный panel.png и results.json.

Модуль не зависит от интерфейса: его используют и CLI (cli.py), и GUI (gui.py).
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import cv2
import numpy as np
from PIL import Image
from scipy import ndimage
from skimage import color, data, feature, filters, morphology, util

DEMO_SOURCES: Tuple[str, ...] = ("coins", "page", "camera", "astronaut")
METHOD_ORDER: Tuple[str, ...] = ("sobel", "scharr", "canny")
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


@dataclass
class Params:
    """Параметры пайплайна. Значения по умолчанию подобраны под data.coins."""

    methods: Tuple[str, ...] = ("sobel", "scharr", "canny")
    # Порог Sobel/Scharr — доля от максимума магнитуды градиента (0..1):
    # относительный порог устойчив к общей яркости кадра.
    sobel_thresh: float = 0.10
    scharr_thresh: float = 0.10
    # Canny: sigma — радиус гауссова сглаживания (px), low/high — пороги гистерезиса.
    canny_sigma: float = 2.0
    canny_low: float = 0.10
    canny_high: float = 0.25
    # Морфология: none | opening | closing | both (closing, затем opening).
    morph_op: str = "closing"
    morph_size: int = 3     # радиус структурного элемента (disk), px
    morph_iters: int = 1
    fill_holes: bool = True  # залить внутренние области контуров перед подсчётом
    # Фильтр объектов по площади (px). max_area <= 0 — без верхней границы.
    min_area: float = 100.0
    max_area: float = 0.0


# ----------------------------------------------------------------------------
# Загрузка данных
# ----------------------------------------------------------------------------

def load_demo(name: str) -> np.ndarray:
    """Демо-изображение scikit-image (coins/page/camera/astronaut), uint8."""
    if name not in DEMO_SOURCES:
        raise ValueError(f"Нет демо «{name}». Доступны: {', '.join(DEMO_SOURCES)}")
    return np.asarray(getattr(data, name)())


def load_image_file(path: str | Path) -> np.ndarray:
    """Файл пользователя -> RGB uint8 (через Pillow)."""
    with Image.open(path) as im:
        im.load()
        if im.mode in ("RGBA", "LA", "P"):
            im = im.convert("RGB")
        return np.asarray(im)


def list_image_files(folder: str | Path) -> List[Path]:
    """Все изображения в папке (без рекурсии), отсортированные по имени."""
    folder = Path(folder)
    return sorted(p for p in folder.iterdir() if p.suffix.lower() in IMAGE_EXTS)


# ----------------------------------------------------------------------------
# Базовые преобразования
# ----------------------------------------------------------------------------

def to_gray(img: np.ndarray) -> np.ndarray:
    """RGB или полутон uint8 -> float [0..1] формы (H, W)."""
    img = np.asarray(img)
    if img.ndim == 3:
        return color.rgb2gray(img[..., :3])
    return util.img_as_float64(img)


def to_rgb_u8(img: np.ndarray) -> np.ndarray:
    """Любое входное изображение -> RGB uint8 (подложка для контуров)."""
    img = np.asarray(img)
    if img.ndim == 2:
        return np.stack([img] * 3, axis=-1)
    if img.dtype != np.uint8:
        img = util.img_as_ubyte(img)
    return np.ascontiguousarray(img[..., :3])


def validate_params(p: Params) -> Params:
    """Проверка параметров. Бросает ValueError с понятным сообщением."""
    if not p.methods:
        raise ValueError("Выберите хотя бы один метод (sobel/scharr/canny).")
    unknown = [m for m in p.methods if m not in METHOD_ORDER]
    if unknown:
        raise ValueError(f"Неизвестный метод: {', '.join(unknown)}")
    for attr in ("sobel_thresh", "scharr_thresh"):
        v = float(getattr(p, attr))
        if not 0.0 <= v <= 1.0:
            raise ValueError(f"{attr} должен быть в диапазоне [0..1], получено {v}")
    if not 0.0 <= float(p.canny_low) < float(p.canny_high):
        raise ValueError("Требуется 0 <= canny_low < canny_high "
                         f"(сейчас low={p.canny_low}, high={p.canny_high})")
    if p.canny_sigma <= 0:
        raise ValueError("canny_sigma должен быть > 0")
    if p.morph_op not in ("none", "opening", "closing", "both"):
        raise ValueError(f"Неизвестная морфологическая операция: «{p.morph_op}»")
    if int(p.morph_size) < 1 or int(p.morph_iters) < 1:
        raise ValueError("morph_size и morph_iters должны быть >= 1")
    if p.min_area < 0:
        raise ValueError("min_area должен быть >= 0")
    return p


# ----------------------------------------------------------------------------
# Шаги пайплайна
# ----------------------------------------------------------------------------

def compute_edges(gray: np.ndarray, p: Params) -> Dict[str, np.ndarray]:
    """Бинарные маски границ для каждого выбранного метода."""
    out: Dict[str, np.ndarray] = {}
    for m in METHOD_ORDER:
        if m not in p.methods:
            continue
        if m in ("sobel", "scharr"):
            mag = filters.sobel(gray) if m == "sobel" else filters.scharr(gray)
            peak = float(mag.max())
            norm = mag / peak if peak > 0 else mag
            thresh = p.sobel_thresh if m == "sobel" else p.scharr_thresh
            out[m] = norm >= thresh
        else:  # canny
            out[m] = feature.canny(
                gray,
                sigma=p.canny_sigma,
                low_threshold=p.canny_low,
                high_threshold=p.canny_high,
            )
    return out


def clean_mask(mask: np.ndarray, p: Params) -> np.ndarray:
    """Морфологическая очистка маски границ."""
    out = np.asarray(mask, dtype=bool)
    if int(p.morph_size) > 0:
        fp = morphology.disk(int(p.morph_size))
        if p.morph_op in ("closing", "both"):
            for _ in range(int(p.morph_iters)):
                out = morphology.closing(out, fp)
        if p.morph_op in ("opening", "both"):
            for _ in range(int(p.morph_iters)):
                out = morphology.opening(out, fp)
    if p.fill_holes:
        # binary_fill_holes переехал в scipy.ndimage (из skimage.morphology в 0.26)
        out = ndimage.binary_fill_holes(out)
    return out


def find_objects(mask: np.ndarray, p: Params) -> Tuple[int, List[float], List[np.ndarray]]:
    """Внешние контуры (OpenCV) + фильтр по площади.

    Возвращает (количество объектов, площади по возрастанию, отобранные контуры).
    """
    mask_u8 = np.asarray(mask, dtype=np.uint8) * 255
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    objs: List[np.ndarray] = []
    for c in contours:
        area = float(cv2.contourArea(c))
        if area < p.min_area:
            continue
        if p.max_area > 0 and area > p.max_area:
            continue
        objs.append(c)
    areas = sorted(float(cv2.contourArea(c)) for c in objs)
    return len(objs), areas, objs


def make_overlay(rgb_u8: np.ndarray, contours: Sequence[np.ndarray]) -> np.ndarray:
    """RGB-подложка с красными контурами отобранных объектов."""
    bgr = cv2.cvtColor(np.ascontiguousarray(rgb_u8), cv2.COLOR_RGB2BGR)
    if len(contours):
        cv2.drawContours(bgr, list(contours), -1, (0, 0, 255), 2)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def process(img: np.ndarray, p: Params, name: str = "image") -> dict:
    """Полный пайплайн одного кадра: границы -> морфология -> контуры -> подсчёт."""
    p = validate_params(p)
    rgb_u8 = to_rgb_u8(img)
    gray = to_gray(img)
    edges = compute_edges(gray, p)
    methods: Dict[str, dict] = {}
    for m, raw in edges.items():
        mask = clean_mask(raw, p)
        count, areas, objs = find_objects(mask, p)
        methods[m] = {
            "edges": raw,
            "mask": mask,
            "count": count,
            "areas": areas,
            "overlay": make_overlay(rgb_u8, objs),
        }
    return {
        "name": name,
        "is_color": bool(np.asarray(img).ndim == 3),
        "rgb_u8": rgb_u8,
        "gray_u8": util.img_as_ubyte(gray),
        "methods": methods,
    }


# ----------------------------------------------------------------------------
# Визуализация и сохранение
# ----------------------------------------------------------------------------

def build_panel(result: dict, width: float = 11.0, row_height: float = 2.4):
    """Сводная картинка «до/после» (matplotlib Figure без pyplot — годится и для GUI)."""
    import matplotlib.figure

    methods = [m for m in METHOD_ORDER if m in result["methods"]]
    rows = 1 + len(methods)
    fig = matplotlib.figure.Figure(layout="constrained",
                                   figsize=(width, row_height * rows), dpi=100)
    gs = fig.add_gridspec(rows, 3)

    ax = fig.add_subplot(gs[0, :])
    ax.imshow(result["gray_u8"], cmap="gray")
    ax.set_title(f"Оригинал (полутон): {result['name']}")
    ax.axis("off")

    for i, m in enumerate(methods, start=1):
        info = result["methods"][m]
        cells = (
            (info["edges"], "gray", f"{m}: границы"),
            (info["mask"], "gray", f"{m}: маска после морфологии"),
            (info["overlay"], None, f"{m}: контуры · объектов: {info['count']}"),
        )
        for col, (img, cmap, title) in enumerate(cells):
            ax_ = fig.add_subplot(gs[i, col])
            ax_.imshow(img, cmap=cmap)
            ax_.set_title(title, fontsize=10)
            ax_.axis("off")
    return fig


def save_result(result: dict, p: Params, out_dir: str | Path,
                with_panel: bool = True) -> List[Path]:
    """Папка «до/после»: before/original.png, after/*, panel.png, results.json."""
    d = Path(out_dir)
    (d / "before").mkdir(parents=True, exist_ok=True)
    after = d / "after"
    after.mkdir(exist_ok=True)
    saved: List[Path] = []

    base = d / "before" / "original.png"
    if result["is_color"]:
        Image.fromarray(result["rgb_u8"]).save(base)
    else:
        Image.fromarray(result["gray_u8"]).save(base)
    saved.append(base)

    for m, info in result["methods"].items():
        targets = {
            f"edges_{m}.png": info["edges"].astype(np.uint8) * 255,
            f"mask_{m}.png": info["mask"].astype(np.uint8) * 255,
            f"overlay_{m}.png": info["overlay"],
        }
        for fname, arr in targets.items():
            fpath = after / fname
            Image.fromarray(arr).save(fpath)
            saved.append(fpath)

    if with_panel:
        ppath = d / "panel.png"
        build_panel(result).savefig(ppath, dpi=150, bbox_inches="tight")
        saved.append(ppath)

    summary = {
        "image": result["name"],
        "counts": {m: info["count"] for m, info in result["methods"].items()},
        "areas_px": {m: [round(a, 1) for a in info["areas"]]
                     for m, info in result["methods"].items()},
        "params": asdict(p),
    }
    jpath = d / "results.json"
    jpath.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    saved.append(jpath)
    return saved
