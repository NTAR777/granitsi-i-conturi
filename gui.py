#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Графический интерфейс (tkinter + matplotlib) для лабораторной работы.

Запуск:
    python gui.py

Слева — все параметры пайплайна (те же, что и в CLI), справа — картинка
«до/после»: оригинал, границы каждого метода, маски после морфологии,
контуры с подсчитанным числом объектов.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("TkAgg")

import tkinter as tk  # noqa: E402
from tkinter import filedialog, messagebox, ttk  # noqa: E402

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg  # noqa: E402

import pipeline as P  # noqa: E402


class App(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("Границы и контуры объектов — лабораторная работа")
        self.geometry("1280x880")
        self.minsize(1000, 620)

        self.source: str = "coins"   # имя демо scikit-image или путь к файлу
        self.params = P.Params()
        self.result: dict | None = None
        self._canvas: FigureCanvasTkAgg | None = None

        self._build_toolbar()
        self._build_sidebar()
        self._build_canvas_area()

        self.status = tk.StringVar(value="Готово. Выберите источник и нажмите «Запустить».")
        ttk.Label(self, textvariable=self.status, relief="sunken", anchor="w",
                  padding=(8, 3)).pack(side="bottom", fill="x")

        self.after(250, self.run)  # дать окну отрисоваться, затем показать coins

    # ------------------------------------------------------------------ UI

    def _build_toolbar(self) -> None:
        bar = ttk.Frame(self, padding=(8, 6))
        bar.pack(side="top", fill="x")

        ttk.Label(bar, text="Источник:").pack(side="left")
        for name in P.DEMO_SOURCES:
            ttk.Button(bar, text=name, width=9,
                       command=lambda n=name: self.use_demo(n)).pack(side="left", padx=2)
        ttk.Button(bar, text="Открыть файл…", command=self.open_file)\
            .pack(side="left", padx=(10, 2))

        ttk.Separator(bar, orient="vertical").pack(side="left", fill="y", padx=10)
        ttk.Button(bar, text="Запустить", command=self.run).pack(side="left", padx=2)
        ttk.Button(bar, text="Сохранить результаты", command=self.save_results)\
            .pack(side="left", padx=2)

        self.source_var = tk.StringVar(value="coins (демо scikit-image)")
        ttk.Label(bar, textvariable=self.source_var).pack(side="right", padx=6)

    def _build_sidebar(self) -> None:
        side = ttk.Frame(self, padding=(8, 2))
        side.pack(side="left", fill="y")

        box = ttk.Labelframe(side, text="Методы границ", padding=6)
        box.pack(fill="x", pady=4)
        self.var_methods = {m: tk.BooleanVar(value=True) for m in P.METHOD_ORDER}
        for m in P.METHOD_ORDER:
            ttk.Checkbutton(box, text=m, variable=self.var_methods[m]).pack(anchor="w")

        box = ttk.Labelframe(side, text="Пороги детекторов", padding=6)
        box.pack(fill="x", pady=4)
        self.ent_sobel = self._entry(box, "Sobel: доля от max (0..1)", "0.10")
        self.ent_scharr = self._entry(box, "Scharr: доля от max (0..1)", "0.10")
        self.ent_sigma = self._entry(box, "Canny: sigma", "2.0")
        self.ent_low = self._entry(box, "Canny: low", "0.10")
        self.ent_high = self._entry(box, "Canny: high", "0.25")

        box = ttk.Labelframe(side, text="Морфология", padding=6)
        box.pack(fill="x", pady=4)
        self.cmb_op = ttk.Combobox(box, values=("closing", "opening", "both", "none"),
                                   state="readonly", width=10)
        self.cmb_op.set("closing")
        self.cmb_op.pack(anchor="w", pady=(0, 3))
        self.ent_morph_size = self._entry(box, "радиус disk (px)", "3")
        self.ent_morph_iters = self._entry(box, "итераций", "1")
        self.var_fill = tk.BooleanVar(value=True)
        ttk.Checkbutton(box, text="заливать дыры (fill holes)",
                        variable=self.var_fill).pack(anchor="w", pady=(3, 0))

        box = ttk.Labelframe(side, text="Фильтр по площади (px)", padding=6)
        box.pack(fill="x", pady=4)
        self.ent_min_area = self._entry(box, "минимальная", "100")
        self.ent_max_area = self._entry(box, "максимальная (0 = нет)", "0")

        ttk.Label(side, foreground="#666666", justify="left", wraplength=300, text=(
            "Пайплайн: полутон → границы (Sobel/Scharr/Canny) → морфологическая "
            "очистка → контуры (OpenCV) → фильтр по площади → количество объектов.\n\n"
            "«Сохранить результаты» пишет results/<имя>/: before/, after/, "
            "panel.png, results.json."
        )).pack(fill="x", pady=(12, 0))

    def _build_canvas_area(self) -> None:
        self.frame_canvas = ttk.Frame(self, padding=(4, 4))
        self.frame_canvas.pack(side="left", fill="both", expand=True)

    def _entry(self, parent, label: str, value: str) -> tk.StringVar:
        row = ttk.Frame(parent)
        row.pack(fill="x", pady=1)
        ttk.Label(row, text=label, width=28).pack(side="left")
        var = tk.StringVar(value=value)
        ttk.Entry(row, textvariable=var, width=8).pack(side="left")
        return var

    # ------------------------------------------------------------- события

    def use_demo(self, name: str) -> None:
        self.source = name
        self.source_var.set(f"{name} (демо scikit-image)")

    def open_file(self) -> None:
        path = filedialog.askopenfilename(
            title="Выберите изображение",
            filetypes=[("Изображения", "*.png *.jpg *.jpeg *.bmp *.tif *.tiff *.webp"),
                       ("Все файлы", "*.*")])
        if path:
            self.source = path
            self.source_var.set(f"{Path(path).name} (файл)")

    def _collect_params(self) -> P.Params:
        def f(var: tk.StringVar, label: str, lo: float | None = None,
              hi: float | None = None) -> float:
            raw = var.get().strip().replace(",", ".")
            try:
                v = float(raw)
            except ValueError:
                raise ValueError(f"Поле «{label}»: введите число (получено «{raw}»).")
            if lo is not None and v < lo or hi is not None and v > hi:
                span = f"[{lo}..{hi}]" if hi is not None else f">= {lo}"
                raise ValueError(f"Поле «{label}»: значение вне диапазона {span}.")
            return v

        def i(var: tk.StringVar, label: str, lo: int = 1) -> int:
            v = int(f(var, label, lo))
            return v

        methods = tuple(m for m in P.METHOD_ORDER if self.var_methods[m].get())
        return P.validate_params(P.Params(
            methods=methods,
            sobel_thresh=f(self.ent_sobel, "порог Sobel", 0.0, 1.0),
            scharr_thresh=f(self.ent_scharr, "порог Scharr", 0.0, 1.0),
            canny_sigma=f(self.ent_sigma, "sigma", 0.0),
            canny_low=f(self.ent_low, "canny low", 0.0, 1.0),
            canny_high=f(self.ent_high, "canny high", 0.0, 1.0),
            morph_op=self.cmb_op.get(),
            morph_size=i(self.ent_morph_size, "радиус disk"),
            morph_iters=i(self.ent_morph_iters, "итераций"),
            fill_holes=self.var_fill.get(),
            min_area=f(self.ent_min_area, "мин. площадь", 0.0),
            max_area=f(self.ent_max_area, "макс. площадь", 0.0),
        ))

    def run(self) -> None:
        try:
            p = self._collect_params()
        except ValueError as e:
            messagebox.showwarning("Параметры", str(e), parent=self)
            return

        try:
            if self.source in P.DEMO_SOURCES:
                img, name = P.load_demo(self.source), self.source
            else:
                img, name = P.load_image_file(self.source), Path(self.source).stem
            result = P.process(img, p, name=name)
        except Exception as e:  # noqa: BLE001 — показываем пользователю любые ошибки
            messagebox.showerror("Ошибка", str(e), parent=self)
            return

        self.result = result
        self.params = p
        self._show(result)
        counts = "  ".join(f"{m}={info['count']}" for m, info in result["methods"].items())
        self.status.set(f"{name}: объектов — {counts}")

    def _show(self, result: dict) -> None:
        # Подгоняем размер рисунка под текущую площадь окна (dpi = 100).
        w_px = max(self.frame_canvas.winfo_width(), 900)
        h_px = max(self.frame_canvas.winfo_height(), 640)
        n_rows = 1 + len(result["methods"])
        fig = P.build_panel(
            result,
            width=(w_px - 30) / 100.0,
            row_height=max((h_px - 50) / 100.0 / n_rows, 1.7),
        )
        if self._canvas is not None:
            self._canvas.get_tk_widget().destroy()
        self._canvas = FigureCanvasTkAgg(fig, master=self.frame_canvas)
        self._canvas.draw()
        self._canvas.get_tk_widget().pack(fill="both", expand=True)

    def save_results(self) -> None:
        if self.result is None:
            messagebox.showinfo("Нет результата", "Сначала нажмите «Запустить».", parent=self)
            return
        out_dir = Path("results") / self.result["name"]
        try:
            saved = P.save_result(self.result, self.params, out_dir)
        except Exception as e:  # noqa: BLE001
            messagebox.showerror("Ошибка сохранения", str(e), parent=self)
            return
        self.status.set(f"Сохранено: {len(saved)} файлов в {out_dir.resolve()}")
        messagebox.showinfo("Сохранено",
                            f"Файлов: {len(saved)}\n{out_dir.resolve()}", parent=self)


if __name__ == "__main__":
    App().mainloop()
