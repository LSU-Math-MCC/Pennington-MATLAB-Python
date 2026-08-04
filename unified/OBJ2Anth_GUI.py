import sys
import os
import queue
import shlex
import threading
import subprocess
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox
import customtkinter as ctk

# Ensure repo root is on sys.path when running this file directly inside the unified folder
_HERE = Path(__file__).resolve().parent  # .../unified
_ROOT = _HERE.parent  # .../Pennington-MATLAB-Python
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from unified.obj2anthro.backends.slice.slice_updated_A import run_pipeline as slice_run_pipeline
from unified.cli import main as cli_main

if getattr(sys, 'frozen', False):
    BASE_RESULTS = Path(sys.executable).parent / "results"
else:
    BASE_RESULTS = Path(__file__).parent.parent / "results"

# Tell Windows we are DPI-aware so tk.Text is not silently downscaled.
try:
    import ctypes
    ctypes.windll.shcore.SetProcessDpiAwareness(2)
except Exception:
    pass

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
OBJ_EXTS = {".obj"}
ANTHRO_METHODS = ["auto", "caesar", "ansur"]
UNIT_OPTIONS = ["auto", "mm", "cm", "m", "in"]
FORCE_STAGE = ["auto", "img2obj"]
STAGE_KEYS = ["img2obj", "obj2anthro"]
STAGE_LABELS = ["📷  img → OBJ", "📐  OBJ → Anthro"]
STAGE_TIPS = [
    "Converts image files\ninto 3-D OBJ meshes",
    "Extracts anthropometric\nmeasurements from OBJ",
]
LOG_COLORS = {
    "CMD": "#ce93d8", "STAGE": "#4fc3f7", "SUCCESS": "#a5d6a7",
    "WARNING": "#ffe082", "ERROR": "#ef9a9a", "DIM": "#777777", "INFO": "#cccccc",
}


def detect_input_type(path):
    p = Path(path)
    if p.is_dir():
        return "folder"
    ext = p.suffix.lower()
    if ext in IMG_EXTS:
        return "image"
    if ext in OBJ_EXTS:
        return "obj"
    return "unknown"


def classify_log_line(line):
    lo = line.lower()
    if line.startswith("$"):
        return "CMD"
    if any(k in lo for k in ("stage", "pipeline", "img2obj", "obj2anthro", "running")):
        return "STAGE"
    if any(k in lo for k in ("complete", "success", "done", "finish", "saved")):
        return "SUCCESS"
    if any(k in lo for k in ("warning", "warn")):
        return "WARNING"
    if any(k in lo for k in ("error", "traceback", "exception", "failed", "critical")):
        return "ERROR"
    if line.strip() == "":
        return "DIM"
    return "INFO"


class _GUIStream:
    """Redirects print() from pipeline functions into the GUI log console."""
    def __init__(self, log_queue):
        self._queue = log_queue

    def write(self, text):
        for line in text.splitlines():
            if line.strip():
                self._queue.put(("line", line))

    def flush(self):
        pass


class StageIndicator(ctk.CTkFrame):

    STATE_TEXT = {
        "idle": "● idle", "active": "⟳ running",
        "done": "✔ done", "error": "✖ error", "skipped": "— skipped",
    }
    STATE_COLOR = {
        "idle": "#888888", "active": "#4fc3f7",
        "done": "#a5d6a7", "error": "#ef9a9a", "skipped": "#555555",
    }
    BORDER_CLR = {
        "idle": "#444444", "active": "#1f538d",
        "done": "#1b5e20", "error": "#7f1d1d", "skipped": "#333333",
    }

    def __init__(self, master, **kwargs):
        super().__init__(master, fg_color="transparent", **kwargs)
        self._labels = {}
        self._boxes = {}
        n = len(STAGE_KEYS)
        for c in range(2 * n - 1):
            self.grid_columnconfigure(c, weight=1 if c % 2 == 0 else 0)
        for i, (key, label, tip) in enumerate(zip(STAGE_KEYS, STAGE_LABELS, STAGE_TIPS)):
            box = ctk.CTkFrame(self, corner_radius=10, border_width=2,
                               border_color="#444444", fg_color="#1e1e1e")
            box.grid(row=0, column=i * 2, sticky="nsew", padx=6, pady=6)
            box.grid_columnconfigure(0, weight=1)
            ctk.CTkLabel(box, text=label, font=ctk.CTkFont(size=13, weight="bold"),
                         wraplength=160).grid(row=0, column=0, pady=(10, 2), padx=10)
            ctk.CTkLabel(box, text=tip, font=ctk.CTkFont(size=10), text_color="#aaaaaa",
                         wraplength=160, justify="center").grid(row=1, column=0, pady=(0, 4), padx=10)
            status = ctk.CTkLabel(box, text="● idle", font=ctk.CTkFont(size=11),
                                  text_color="#888888")
            status.grid(row=2, column=0, pady=(0, 10))
            self._boxes[key] = box
            self._labels[key] = status
            if i < n - 1:
                ctk.CTkLabel(self, text="  ›  ", font=ctk.CTkFont(size=26),
                             text_color="#555555").grid(row=0, column=i * 2 + 1)

    def set_state(self, stage, state):
        if stage not in self._boxes:
            return
        self._labels[stage].configure(
            text=self.STATE_TEXT[state],
            text_color=self.STATE_COLOR[state],
        )
        self._boxes[stage].configure(border_color=self.BORDER_CLR[state])

    def reset_all(self):
        for key in self._boxes:
            self.set_state(key, "idle")


class LogConsole(ctk.CTkFrame):

    def __init__(self, master, **kwargs):
        super().__init__(master, fg_color="#111111", corner_radius=8, **kwargs)
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=1)
        self._text = tk.Text(
            self, bg="#111111", fg="#cccccc", font=("Consolas", 12),
            relief="flat", bd=8, wrap="word", state="disabled",
            selectbackground="#2a2a2a", cursor="arrow",
        )
        self._text.grid(row=0, column=0, sticky="nsew")
        sb = ctk.CTkScrollbar(self, command=self._text.yview)
        sb.grid(row=0, column=1, sticky="ns")
        self._text.configure(yscrollcommand=sb.set)
        for tag, color in LOG_COLORS.items():
            self._text.tag_configure(tag, foreground=color)

    def append(self, line, tag=None):
        if tag is None:
            tag = classify_log_line(line)
        self._text.configure(state="normal")
        self._text.insert("end", line + "\n", tag)
        self._text.configure(state="disabled")
        self._text.see("end")

    def clear(self):
        self._text.configure(state="normal")
        self._text.delete("1.0", "end")
        self._text.configure(state="disabled")


class App(ctk.CTk):

    def __init__(self):
        super().__init__()
        self.title("Unified Anthropometry Pipeline")
        self.geometry("1020x860")
        self.minsize(820, 680)
        self._process = None
        self._log_queue = queue.Queue()
        self._running = False
        self._n_slices_var       = tk.StringVar(value="200")
        self._height_scale_var   = tk.StringVar(value="1.0")
        self._no_images_var      = tk.BooleanVar(value=False)
        self._no_aligned_obj_var = tk.BooleanVar(value=False)
        self._build_ui()
        self._poll_log_queue()

    def _build_ui(self):
        self.grid_rowconfigure(3, weight=1)
        self.grid_columnconfigure(0, weight=1)
        self._build_header()
        self._build_stage_indicator()
        self._build_config_panel()
        self._build_log_panel()
        self._build_bottom_bar()

    def _build_header(self):
        hdr = ctk.CTkFrame(self, fg_color="#1a1a2e", corner_radius=0)
        hdr.grid(row=0, column=0, sticky="ew")
        hdr.grid_columnconfigure(0, weight=1)
        ctk.CTkLabel(hdr, text="⚡  Unified Anthropometry Pipeline",
                     font=ctk.CTkFont(size=20, weight="bold"),
                     text_color="#4fc3f7").grid(row=0, column=0, pady=(14, 2), padx=20, sticky="w")
        ctk.CTkLabel(hdr, text="Image → OBJ → Anthropometric Measurements",
                     font=ctk.CTkFont(size=12),
                     text_color="#888888").grid(row=1, column=0, pady=(0, 12), padx=20, sticky="w")

    def _build_stage_indicator(self):
        wrap = ctk.CTkFrame(self, fg_color="#161616", corner_radius=0)
        wrap.grid(row=1, column=0, sticky="ew")
        wrap.grid_columnconfigure(0, weight=1)
        ctk.CTkLabel(wrap, text="PIPELINE STAGES",
                     font=ctk.CTkFont(size=10, weight="bold"),
                     text_color="#555555").grid(row=0, column=0, pady=(8, 0), padx=16, sticky="w")
        self.stage_indicator = StageIndicator(wrap)
        self.stage_indicator.grid(row=1, column=0, sticky="ew", padx=10, pady=(2, 10))

    def _build_config_panel(self):
        outer = ctk.CTkFrame(self, fg_color="#1a1a1a", corner_radius=0)
        outer.grid(row=2, column=0, sticky="ew")
        outer.grid_columnconfigure(0, weight=1)

        # --- Input row ---
        iw = ctk.CTkFrame(outer, fg_color="#222222", corner_radius=8)
        iw.grid(row=0, column=0, sticky="ew", padx=14, pady=(12, 6))
        iw.grid_columnconfigure(1, weight=1)
        ctk.CTkLabel(iw, text="Input", font=ctk.CTkFont(size=12, weight="bold"),
                     text_color="#aaaaaa").grid(row=0, column=0, padx=(12, 6), pady=(10, 2), sticky="w")
        self._input_badge = ctk.CTkLabel(iw, text="— not set —",
                                         font=ctk.CTkFont(size=11), text_color="#888888")
        self._input_badge.grid(row=0, column=1, pady=(10, 2), sticky="w")
        self._input_var = tk.StringVar()
        ctk.CTkEntry(iw, textvariable=self._input_var,
                     placeholder_text="Select a file or folder…",
                     font=ctk.CTkFont(size=12), height=34,
                     ).grid(row=1, column=0, columnspan=2, sticky="ew", padx=(12, 6), pady=(0, 10))
        self._input_var.trace_add("write", self._on_input_changed)
        br = ctk.CTkFrame(iw, fg_color="transparent")
        br.grid(row=1, column=2, padx=(0, 10), pady=(0, 10))
        ctk.CTkButton(br, text="File",   width=64, height=34,
                      command=self._browse_file).grid(row=0, column=0, padx=(0, 4))
        ctk.CTkButton(br, text="Folder", width=74, height=34,
                      command=self._browse_folder).grid(row=0, column=1)

        # --- Options row ---
        opt = ctk.CTkFrame(outer, fg_color="transparent")
        opt.grid(row=1, column=0, sticky="ew", padx=14, pady=4)
        opt.grid_columnconfigure((0, 1, 2, 3), weight=1)
        self._anthro_var = tk.StringVar(value="auto")
        self._units_var  = tk.StringVar(value="auto")
        self._stage_var  = tk.StringVar(value="auto")
        self._make_option_box(opt, "Anthro Method", ANTHRO_METHODS, self._anthro_var, col=0)
        self._make_option_box(opt, "Units",         UNIT_OPTIONS,   self._units_var,  col=1)
        self._make_option_box(opt, "Force Stage",   FORCE_STAGE,    self._stage_var,  col=2)

        ob = ctk.CTkFrame(opt, fg_color="#222222", corner_radius=8)
        ob.grid(row=0, column=3, sticky="nsew", padx=4, pady=2)
        ob.grid_columnconfigure(0, weight=1)
        ctk.CTkLabel(ob, text="Output Dir (optional)",
                     font=ctk.CTkFont(size=11, weight="bold"),
                     text_color="#aaaaaa").grid(row=0, column=0, columnspan=2, padx=10, pady=(8, 2), sticky="w")
        self._output_var = tk.StringVar()
        ctk.CTkEntry(ob, textvariable=self._output_var, placeholder_text="Same as input",
                     font=ctk.CTkFont(size=11), height=30,
                     ).grid(row=1, column=0, sticky="ew", padx=(10, 4), pady=(0, 8))
        ctk.CTkButton(ob, text="…", width=30, height=30,
                      command=self._browse_output).grid(row=1, column=1, padx=(0, 8), pady=(0, 8))

        # --- Slice options row ---
        slc = ctk.CTkFrame(outer, fg_color="#222222", corner_radius=8)
        slc.grid(row=2, column=0, sticky="ew", padx=14, pady=(4, 6))
        slc.grid_columnconfigure((1, 3), weight=1)
        ctk.CTkLabel(slc, text="Slice Options",
                     font=ctk.CTkFont(size=11, weight="bold"),
                     text_color="#aaaaaa").grid(row=0, column=0, padx=(12, 8), pady=8, sticky="w")
        ctk.CTkLabel(slc, text="N Slices",
                     font=ctk.CTkFont(size=11), text_color="#888888"
                     ).grid(row=0, column=1, padx=(0, 4), pady=8, sticky="e")
        ctk.CTkEntry(slc, textvariable=self._n_slices_var, width=70, height=30,
                     font=ctk.CTkFont(size=11)
                     ).grid(row=0, column=2, padx=(0, 16), pady=8, sticky="w")
        ctk.CTkLabel(slc, text="Height Scale → cm",
                     font=ctk.CTkFont(size=11), text_color="#888888"
                     ).grid(row=0, column=3, padx=(0, 4), pady=8, sticky="e")
        ctk.CTkEntry(slc, textvariable=self._height_scale_var, width=70, height=30,
                     font=ctk.CTkFont(size=11)
                     ).grid(row=0, column=4, padx=(0, 16), pady=8, sticky="w")
        ctk.CTkCheckBox(slc, text="No Images", variable=self._no_images_var,
                        font=ctk.CTkFont(size=11)
                        ).grid(row=0, column=5, padx=(0, 12), pady=8)
        ctk.CTkCheckBox(slc, text="No Aligned OBJ", variable=self._no_aligned_obj_var,
                        font=ctk.CTkFont(size=11)
                        ).grid(row=0, column=6, padx=(0, 12), pady=8)

        # --- Extra CLI flags (img2obj stage only) ---
        adv = ctk.CTkFrame(outer, fg_color="#222222", corner_radius=8)
        adv.grid(row=3, column=0, sticky="ew", padx=14, pady=(0, 12))
        adv.grid_columnconfigure(1, weight=1)
        ctk.CTkLabel(adv, text="Extra img2obj flags",
                     font=ctk.CTkFont(size=11, weight="bold"),
                     text_color="#aaaaaa").grid(row=0, column=0, padx=(12, 8), pady=8, sticky="w")
        self._extra_var = tk.StringVar()
        ctk.CTkEntry(adv, textvariable=self._extra_var,
                     placeholder_text="e.g.  --verbose --no-cache  (only applied to img→OBJ stage)",
                     font=ctk.CTkFont(size=11), height=30,
                     ).grid(row=0, column=1, sticky="ew", padx=(0, 12), pady=8)

    def _make_option_box(self, parent, label, values, var, col):
        box = ctk.CTkFrame(parent, fg_color="#222222", corner_radius=8)
        box.grid(row=0, column=col, sticky="nsew", padx=4, pady=2)
        box.grid_columnconfigure(0, weight=1)
        ctk.CTkLabel(box, text=label, font=ctk.CTkFont(size=11, weight="bold"),
                     text_color="#aaaaaa").grid(row=0, column=0, padx=10, pady=(8, 2), sticky="w")
        ctk.CTkOptionMenu(box, values=values, variable=var,
                          font=ctk.CTkFont(size=12), height=30,
                          ).grid(row=1, column=0, sticky="ew", padx=10, pady=(0, 8))

    def _build_log_panel(self):
        wrap = ctk.CTkFrame(self, fg_color="#111111", corner_radius=0)
        wrap.grid(row=3, column=0, sticky="nsew")
        wrap.grid_rowconfigure(1, weight=1)
        wrap.grid_columnconfigure(0, weight=1)
        hdr = ctk.CTkFrame(wrap, fg_color="#161616", corner_radius=0)
        hdr.grid(row=0, column=0, sticky="ew")
        hdr.grid_columnconfigure(0, weight=1)
        ctk.CTkLabel(hdr, text="OUTPUT LOG",
                     font=ctk.CTkFont(size=10, weight="bold"),
                     text_color="#555555").grid(row=0, column=0, padx=14, pady=6, sticky="w")
        ctk.CTkButton(hdr, text="Clear", width=60, height=26,
                      fg_color="#2a2a2a", hover_color="#333333",
                      command=self._clear_log).grid(row=0, column=1, padx=10, pady=4, sticky="e")
        self.log_console = LogConsole(wrap)
        self.log_console.grid(row=1, column=0, sticky="nsew")

    def _build_bottom_bar(self):
        bar = ctk.CTkFrame(self, fg_color="#1a1a1a", corner_radius=0, height=56)
        bar.grid(row=4, column=0, sticky="ew")
        bar.grid_columnconfigure(0, weight=1)
        bar.grid_propagate(False)
        self._status_var = tk.StringVar(value="Ready")
        ctk.CTkLabel(bar, textvariable=self._status_var,
                     font=ctk.CTkFont(size=11),
                     text_color="#888888").grid(row=0, column=0, padx=16, pady=14, sticky="w")
        bg = ctk.CTkFrame(bar, fg_color="transparent")
        bg.grid(row=0, column=1, padx=14, pady=8, sticky="e")
        self._copy_btn = ctk.CTkButton(bg, text="Copy Command", width=120, height=36,
                                       fg_color="#2a2a2a", hover_color="#333333",
                                       command=self._copy_command)
        self._copy_btn.grid(row=0, column=0, padx=(0, 8))
        self._stop_btn = ctk.CTkButton(bg, text="■  Stop", width=90, height=36,
                                       fg_color="#7f1d1d", hover_color="#991b1b",
                                       command=self._stop_pipeline, state="disabled")
        self._stop_btn.grid(row=0, column=1, padx=(0, 8))
        self._run_btn = ctk.CTkButton(bg, text="▶  Run Pipeline", width=140, height=36,
                                      font=ctk.CTkFont(size=13, weight="bold"),
                                      fg_color="#1f538d", hover_color="#2563a8",
                                      command=self._run_pipeline)
        self._run_btn.grid(row=0, column=2)

    def _browse_file(self):
        path = filedialog.askopenfilename(
            title="Select input file",
            filetypes=[
                ("Supported files", "*.obj *.jpg *.jpeg *.png *.bmp *.tif *.tiff *.webp"),
                ("OBJ meshes", "*.obj"),
                ("Images", "*.jpg *.jpeg *.png *.bmp *.tif *.tiff *.webp"),
                ("All files", "*.*"),
            ],
        )
        if path:
            self._input_var.set(path)

    def _browse_folder(self):
        path = filedialog.askdirectory(title="Select input folder")
        if path:
            self._input_var.set(path)

    def _browse_output(self):
        path = filedialog.askdirectory(title="Select output folder")
        if path:
            self._output_var.set(path)

    def _on_input_changed(self, *_):
        path = self._input_var.get().strip()
        if not path:
            self._input_badge.configure(text="— not set —", text_color="#888888")
            self._status_var.set("Ready")
            return
        kind = detect_input_type(path)
        badge_map = {
            "image":   ("🖼  Image file",   "#ffe082"),
            "obj":     ("📦  OBJ mesh",     "#4fc3f7"),
            "folder":  ("📁  Folder",       "#a5d6a7"),
            "unknown": ("❓  Unknown type", "#ef9a9a"),
        }
        text, color = badge_map.get(kind, ("❓  Unknown", "#ef9a9a"))
        self._input_badge.configure(text=text, text_color=color)
        if kind in ("image", "folder"):
            self._status_var.set("Will run: img2obj  →  obj2anthro")
        elif kind == "obj":
            self._status_var.set("Will run: obj2anthro  (img2obj skipped)")
        else:
            self._status_var.set("Unknown input type — check path")

    def _build_commands(self):
        path = self._input_var.get().strip()
        kind = detect_input_type(path)
        user_out = self._output_var.get().strip()

        input_name = Path(path).stem
        base_out = user_out if user_out else str(BASE_RESULTS / input_name)
        Path(base_out).mkdir(parents=True, exist_ok=True)

        if kind == "obj":
            return [{
                "type": "slice",
                "input": path,
                "output_dir": base_out,
                "n_slices": int(self._n_slices_var.get().strip() or 200),
                "height_scale_to_cm": float(self._height_scale_var.get().strip() or 1.0),
                "save_images": not self._no_images_var.get(),
                "save_aligned_obj": not self._no_aligned_obj_var.get(),
                "is_folder": False,
            }]

        # Image or folder: img2obj first, then slice
        img2obj_out = str(Path(base_out) / "img2obj_output")
        slice_out   = str(Path(base_out) / "slice_output")

        extra = self._extra_var.get().strip()
        extra_tokens = []
        if extra:
            try:
                extra_tokens = shlex.split(extra)
            except ValueError as exc:
                messagebox.showerror("Invalid extra flags", str(exc))
                return []

        img2obj_argv = [
            "--input", path,
            "--method", "auto",
            "--out", img2obj_out,
        ] + extra_tokens

        return [
            {"type": "img2obj", "argv": img2obj_argv},
            {
                "type": "slice",
                "input": img2obj_out,
                "output_dir": slice_out,
                "n_slices": int(self._n_slices_var.get().strip() or 200),
                "height_scale_to_cm": float(self._height_scale_var.get().strip() or 1.0),
                "save_images": not self._no_images_var.get(),
                "save_aligned_obj": not self._no_aligned_obj_var.get(),
                "is_folder": True,
            },
        ]

    def _copy_command(self):
        stages = self._build_commands()
        if not stages:
            return
        parts = []
        for s in stages:
            if s["type"] == "img2obj":
                parts.append("python -m unified " + " ".join(shlex.quote(a) for a in s["argv"]))
            elif s["type"] == "slice":
                parts.append(
                    f"slice_pipeline(input={s['input']}, "
                    f"output_dir={s['output_dir']}, "
                    f"n_slices={s['n_slices']})"
                )
        self.clipboard_clear()
        self.clipboard_append("  &&  ".join(parts))
        self._status_var.set("Command(s) copied to clipboard ✔")

    def _run_pipeline(self):
        if self._running:
            return
        path = self._input_var.get().strip()
        if not path:
            messagebox.showwarning("No input", "Please select an input file or folder.")
            return
        if not Path(path).exists():
            messagebox.showerror("Path not found", f"Input path does not exist:\n{path}")
            return
        stages = self._build_commands()
        if not stages:
            return
        self._running = True
        self._run_btn.configure(state="disabled")
        self._stop_btn.configure(state="normal")
        self.stage_indicator.reset_all()
        self.log_console.clear()
        for s in stages:
            if s["type"] == "img2obj":
                self.log_console.append(
                    "$ python -m unified " + " ".join(shlex.quote(a) for a in s["argv"]), "CMD"
                )
            elif s["type"] == "slice":
                self.log_console.append(
                    f"$ slice_pipeline(input={s['input']}, n_slices={s['n_slices']})", "CMD"
                )
        self.log_console.append("", "DIM")
        self._status_var.set("Running…")
        kind = detect_input_type(path)
        if kind in ("image", "folder"):
            self.stage_indicator.set_state("img2obj",    "active")
            self.stage_indicator.set_state("obj2anthro", "active")
        elif kind == "obj":
            self.stage_indicator.set_state("img2obj",    "skipped")
            self.stage_indicator.set_state("obj2anthro", "active")
        threading.Thread(target=self._run_stages, args=(stages,), daemon=True).start()

    def _run_stages(self, stages):
        stream = _GUIStream(self._log_queue)
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        try:
            for stage in stages:
                sys.stdout = stream
                sys.stderr = stream

                if stage["type"] == "img2obj":
                    try:
                        rc = cli_main(stage["argv"])
                        if rc != 0:
                            self._log_queue.put(("done", rc))
                            return
                    except SystemExit as e:
                        if e.code != 0:
                            self._log_queue.put(("done", e.code or 1))
                            return

                elif stage["type"] == "slice":
                    try:
                        slice_run_pipeline(
                            input_path=Path(stage["input"]),
                            output_dir=Path(stage["output_dir"]),
                            n_slices=stage["n_slices"],
                            recursive=stage["is_folder"],
                            height_scale_to_cm=stage["height_scale_to_cm"],
                            save_images=stage["save_images"],
                            save_aligned_obj=stage["save_aligned_obj"],
                        )
                    except SystemExit as e:
                        if e.code != 0:
                            self._log_queue.put(("done", e.code or 1))
                            return

            self._log_queue.put(("done", 0))

        except Exception as exc:
            self._log_queue.put(("line", f"[GUI ERROR] {exc}"))
            self._log_queue.put(("done", 1))
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr

    def _stop_pipeline(self):
        # For direct calls we can't kill a thread, but we flag it
        self.log_console.append("", "DIM")
        self.log_console.append("⚠  Stop requested — current stage will finish.", "WARNING")
        self._status_var.set("Stopping…")
        self._set_idle()

    def _set_idle(self):
        self._running = False
        self._run_btn.configure(state="normal")
        self._stop_btn.configure(state="disabled")

    def _clear_log(self):
        self.log_console.clear()

    def _poll_log_queue(self):
        try:
            while True:
                kind, value = self._log_queue.get_nowait()
                if kind == "line":
                    self.log_console.append(value)
                    self._update_stages_from_line(value)
                elif kind == "done":
                    self._on_pipeline_done(value)
        except queue.Empty:
            pass
        self.after(80, self._poll_log_queue)

    def _update_stages_from_line(self, line):
        lo = line.lower()
        if "img2obj" in lo:
            if any(k in lo for k in ("start", "running", "begin", "processing")):
                self.stage_indicator.set_state("img2obj", "active")
            elif any(k in lo for k in ("done", "complete", "finish")):
                self.stage_indicator.set_state("img2obj", "done")
            elif any(k in lo for k in ("error", "failed")):
                self.stage_indicator.set_state("img2obj", "error")
        if "obj2anthro" in lo or "slice" in lo or "pipeline" in lo:
            if any(k in lo for k in ("start", "running", "begin", "processing")):
                self.stage_indicator.set_state("obj2anthro", "active")
            elif any(k in lo for k in ("done", "complete", "finish")):
                self.stage_indicator.set_state("obj2anthro", "done")
            elif any(k in lo for k in ("error", "failed")):
                self.stage_indicator.set_state("obj2anthro", "error")

    def _on_pipeline_done(self, returncode):
        self.log_console.append("", "DIM")
        if returncode == 0:
            self.log_console.append("✔  Pipeline completed successfully.", "SUCCESS")
            self._status_var.set("Done ✔")
            self.stage_indicator.set_state("obj2anthro", "done")
        else:
            self.log_console.append(f"✖  Pipeline exited with code {returncode}.", "ERROR")
            self._status_var.set(f"Failed (exit code {returncode})")
        self._set_idle()


if __name__ == "__main__":
    app = App()
    app.mainloop()