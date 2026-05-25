import os # os 模块提供了丰富的方法来处理文件和目录
import threading # threading 模块提供了创建和管理线程的功能
import subprocess 
import tkinter as tk
from tkinter import filedialog, ttk, messagebox
from PIL import Image, ImageTk

PREVIEW_SIZE = (360, 260)  # 预览尺寸（缩略图）


class ImageViewer(tk.Toplevel):
    """
    原图查看器（支持拖拽平移 + 滚轮缩放）
    - 左键拖拽：平移
    - 滚轮：缩放（以鼠标位置为中心）
    - Ctrl+0：还原到初始适配屏幕比例
    """
    MIN_SCALE = 0.10
    MAX_SCALE = 8.00
    ZOOM_STEP = 1.12  # 每次滚轮缩放倍率（可调：1.08 更细，1.2 更快）

    def __init__(self, master, path: str, title: str = "Image Viewer"):
        super().__init__(master)
        self.title(title)
        self.geometry("900x650")
        self.minsize(520, 420)

        self.path = path
        self._tk_img = None
        self._pil_img = None

        self.base_scale = 1.0   # 初始“适配屏幕”比例
        self.scale = 1.0        # 当前显示比例

        # Top info bar
        top = ttk.Frame(self, padding=(10, 8)) 
        top.pack(fill="x")
        self.lbl_info = ttk.Label(top, text=path)
        self.lbl_info.pack(side="left", fill="x", expand=True)

        tip = ttk.Label(top, text="滚轮缩放 | 左键拖动平移 ", foreground="#555")
        tip.pack(side="right")

        # Canvas area
        body = ttk.Frame(self, padding=(10, 0, 10, 10))
        body.pack(fill="both", expand=True)
        body.rowconfigure(0, weight=1)
        body.columnconfigure(0, weight=1)

        self.canvas = tk.Canvas(body, bg="#111111", highlightthickness=0)
        self.canvas.grid(row=0, column=0, sticky="nsew")

        ybar = ttk.Scrollbar(body, orient="vertical", command=self.canvas.yview)
        ybar.grid(row=0, column=1, sticky="ns")
        xbar = ttk.Scrollbar(body, orient="horizontal", command=self.canvas.xview)
        xbar.grid(row=1, column=0, sticky="we")

        self.canvas.configure(yscrollcommand=ybar.set, xscrollcommand=xbar.set)

        # Mouse drag to pan
        self.canvas.bind("<ButtonPress-1>", self._start_pan)
        self.canvas.bind("<B1-Motion>", self._do_pan)

        # Mouse wheel zoom (Windows)
        self.canvas.bind("<MouseWheel>", self._on_mousewheel)

        # Reset zoom
        self.bind("<Control-0>", lambda e: self.reset_zoom())
        self.bind("<Control-Key-0>", lambda e: self.reset_zoom())

        self._load_image()

    def _start_pan(self, event):
        self.canvas.scan_mark(event.x, event.y)

    def _do_pan(self, event):
        self.canvas.scan_dragto(event.x, event.y, gain=1)

    def _load_image(self):
        try:
            img = Image.open(self.path).convert("RGB")
            self._pil_img = img

            # Compute base_scale to fit screen
            screen_w = self.winfo_screenwidth()
            screen_h = self.winfo_screenheight()
            max_w = int(screen_w * 0.85)
            max_h = int(screen_h * 0.80)

            w, h = img.size
            self.base_scale = min(max_w / w, max_h / h, 1.0)
            self.scale = self.base_scale

            self._render()

        except Exception as e:
            messagebox.showerror("错误", f"打开图片失败：{e}")
            self.destroy()

    def _render(self, keep_anchor=None):
        """
        Render image at current self.scale.
        keep_anchor: (canvas_x, canvas_y) - keep this canvas coord at same screen spot by centering zoom.
        """
        img = self._pil_img
        if img is None:
            return

        w, h = img.size
        new_w = max(1, int(w * self.scale))
        new_h = max(1, int(h * self.scale))

        show = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
        self._tk_img = ImageTk.PhotoImage(show)

        # If we want to zoom around a point, record pre-render scroll fractions
        if keep_anchor is not None:
            ax, ay = keep_anchor
            # Convert anchor to relative fraction in scrollregion
            # We approximate using current scrollregion size before redraw
            old_region = self.canvas.cget("scrollregion")
            try:
                x0, y0, x1, y1 = map(float, old_region.split())
                old_w = max(1.0, x1 - x0)
                old_h = max(1.0, y1 - y0)
                fx = (self.canvas.canvasx(ax) - x0) / old_w
                fy = (self.canvas.canvasy(ay) - y0) / old_h
            except Exception:
                fx = fy = None
        else:
            fx = fy = None

        self.canvas.delete("all")
        self.canvas.create_image(0, 0, image=self._tk_img, anchor="nw")
        self.canvas.configure(scrollregion=(0, 0, new_w, new_h))

        # Restore view so the anchor stays near the cursor position (best-effort)
        if fx is not None and fy is not None:
            self.canvas.xview_moveto(min(max(fx, 0.0), 1.0))
            self.canvas.yview_moveto(min(max(fy, 0.0), 1.0))

        pct = int(self.scale * 100)
        self.title(f"{os.path.basename(self.path)}")
        self.lbl_info.configure(text=f"缩放: {pct}%")

    def _on_mousewheel(self, event):
        """
        Windows: event.delta >0 scroll up, <0 scroll down.
        Zoom around mouse position.
        """
        if self._pil_img is None:
            return

        # direction
        if event.delta > 0:
            new_scale = self.scale * self.ZOOM_STEP
        else:
            new_scale = self.scale / self.ZOOM_STEP

        new_scale = max(self.MIN_SCALE, min(self.MAX_SCALE, new_scale))
        if abs(new_scale - self.scale) < 1e-6:
            return

        self.scale = new_scale
        self._render(keep_anchor=(event.x, event.y))

    def reset_zoom(self):
        self.scale = self.base_scale
        self._render()


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("高斯模糊")
        self.geometry("1040x720")
        self.minsize(980, 650)

        self.input_path = tk.StringVar()
        self.output_path = tk.StringVar(value=os.path.join(os.path.expanduser("~"), "Desktop", "out.jpg"))
        self.nproc = tk.IntVar(value=8)

        self.ksize = tk.IntVar(value=21)       # odd
        self.sigma = tk.DoubleVar(value=5.0)   # >0

        self._img_in_tk = None
        self._img_out_tk = None

        self._build_style()
        self._build_ui()

    def _build_style(self):
        style = ttk.Style(self)
        try:
            style.theme_use("vista")
        except tk.TclError:
            style.theme_use(style.theme_names()[0])

        style.configure(".", font=("Microsoft YaHei UI", 10))
        style.configure("Title.TLabel", font=("Microsoft YaHei UI", 14, "bold"))
        style.configure("Run.TButton", font=("Microsoft YaHei UI", 10, "bold"))

    def _build_ui(self):
        root = ttk.Frame(self, padding=12)
        root.pack(fill="both", expand=True)

        ttk.Label(root, text="MPI 高斯模糊处理", style="Title.TLabel").pack(anchor="w", pady=(0, 10))

        settings = ttk.Frame(root)
        settings.pack(fill="x", pady=(0, 10))

        path_card = ttk.LabelFrame(settings, text="输入/输出", padding=10)
        path_card.pack(side="left", fill="x", expand=True, padx=(0, 10))
        path_card.columnconfigure(1, weight=1)

        ttk.Label(path_card, text="输入图片：").grid(row=0, column=0, sticky="w", pady=4)
        ttk.Entry(path_card, textvariable=self.input_path).grid(row=0, column=1, sticky="we", padx=6, pady=4)
        ttk.Button(path_card, text="浏览", command=self.pick_input, width=10).grid(row=0, column=2, sticky="e", pady=4)

        ttk.Label(path_card, text="输出图片：").grid(row=1, column=0, sticky="w", pady=4)
        ttk.Entry(path_card, textvariable=self.output_path).grid(row=1, column=1, sticky="we", padx=6, pady=4)
        ttk.Button(path_card, text="浏览", command=self.pick_output, width=10).grid(row=1, column=2, sticky="e", pady=4)

        param_card = ttk.LabelFrame(settings, text="参数", padding=10)
        param_card.pack(side="left", fill="x")

        ttk.Label(param_card, text="进程数 P：").grid(row=0, column=0, sticky="w", pady=4)
        ttk.Spinbox(param_card, from_=1, to=64, textvariable=self.nproc, width=8).grid(row=0, column=1, sticky="w", pady=4)

        ttk.Label(param_card, text="卷积核尺寸（奇数）：").grid(row=1, column=0, sticky="w", pady=4)
        ttk.Spinbox(param_card, from_=3, to=31, increment=2, textvariable=self.ksize, width=8).grid(row=1, column=1, sticky="w", pady=4)

        ttk.Label(param_card, text="标准差：").grid(row=2, column=0, sticky="w", pady=4)
        ttk.Entry(param_card, textvariable=self.sigma, width=10).grid(row=2, column=1, sticky="w", pady=4)

        run_card = ttk.Frame(settings)
        run_card.pack(side="left", fill="y", padx=(10, 0))

        self.btn_run = ttk.Button(run_card, text="运行~", style="Run.TButton", command=self.run_job, width=12)
        self.btn_run.pack(fill="x")

        self.btn_clear = ttk.Button(run_card, text="清空日志", command=self.clear_log, width=12)
        self.btn_clear.pack(fill="x", pady=(8, 0))

        self.progress = ttk.Progressbar(run_card, mode="indeterminate", length=120)
        self.progress.pack(fill="x", pady=(12, 0))

        preview = ttk.Frame(root)
        preview.pack(fill="both", expand=True)

        lf_in = ttk.LabelFrame(preview, text="输入图片预览", padding=10)
        lf_in.pack(side="left", fill="both", expand=True, padx=(0, 10))

        lf_out = ttk.LabelFrame(preview, text="输出图片预览", padding=10)
        lf_out.pack(side="left", fill="both", expand=True)

        self.preview_in = ttk.Label(lf_in, cursor="hand2")
        self.preview_in.pack(fill="both", expand=True)

        self.preview_out = ttk.Label(lf_out, cursor="hand2")
        self.preview_out.pack(fill="both", expand=True)

        self.preview_in.bind("<Button-1>", lambda e: self.open_viewer("in"))
        self.preview_out.bind("<Button-1>", lambda e: self.open_viewer("out"))

        log_card = ttk.LabelFrame(root, text="日志", padding=10)
        log_card.pack(fill="both", expand=False, pady=(10, 0))
        log_card.columnconfigure(0, weight=1)

        self.txt = tk.Text(log_card, height=8, wrap="word", font=("Consolas", 9))
        self.txt.grid(row=0, column=0, sticky="nsew")

        sb = ttk.Scrollbar(log_card, orient="vertical", command=self.txt.yview)
        sb.grid(row=0, column=1, sticky="ns")
        self.txt.configure(yscrollcommand=sb.set)

    def _log(self, s):
        self.txt.insert("end", s)
        self.txt.see("end")

    def clear_log(self):
        self.txt.delete("1.0", "end")

    def pick_input(self):
        p = filedialog.askopenfilename(filetypes=[("Images", "*.png;*.jpg;*.jpeg;*.bmp"), ("All", "*.*")])
        if not p:
            return
        self.input_path.set(p)
        self._load_preview(p, which="in")

        if not self.output_path.get().strip():
            base, ext = os.path.splitext(p)
            self.output_path.set(base + "_blur" + ext)

    def pick_output(self):
        p = filedialog.asksaveasfilename(
            defaultextension=".jpg",
            filetypes=[("JPEG", "*.jpg"), ("PNG", "*.png"), ("All", "*.*")],
        )
        if not p:
            return
        self.output_path.set(p)

    def _load_preview(self, path, which):
        try:
            img = Image.open(path).convert("RGB")
            img.thumbnail(PREVIEW_SIZE)
            tkimg = ImageTk.PhotoImage(img)
            if which == "in":
                self._img_in_tk = tkimg
                self.preview_in.configure(image=tkimg)
            else:
                self._img_out_tk = tkimg
                self.preview_out.configure(image=tkimg)
        except Exception as e:
            self._log(f"[Preview] 加载失败: {e}\n")

    def open_viewer(self, which):
        if which == "in":
            path = self.input_path.get().strip()
            if not path or not os.path.exists(path):
                messagebox.showinfo("提示", "请先选择输入图片。")
                return
            ImageViewer(self, path, title="输入图片")
        else:
            path = self.output_path.get().strip()
            if not path or not os.path.exists(path):
                messagebox.showinfo("提示", "请先运行生成输出图片。")
                return
            ImageViewer(self, path, title="输出图片")

    def _set_running(self, running: bool):
        if running:
            self.btn_run.configure(state="disabled")
            self.progress.start(12)
        else:
            self.btn_run.configure(state="normal")
            self.progress.stop()

    def run_job(self):
        inp = self.input_path.get().strip()
        outp = self.output_path.get().strip()

        if not inp or not os.path.exists(inp):
            messagebox.showerror("错误", "请选择有效的输入图片。")
            return
        if not outp:
            messagebox.showerror("错误", "请选择输出路径。")
            return

        n = int(self.nproc.get())
        ksize = int(self.ksize.get())
        sigma = float(self.sigma.get())

        if ksize < 3 or ksize % 2 == 0:
            messagebox.showerror("错误", "卷积核尺寸必须为奇数且 >= 3。")
            return
        if sigma <= 0:
            messagebox.showerror("错误", "标准差必须 > 0。")
            return

        script = os.path.join(os.path.dirname(__file__), "mpi_blurV3_gaussian.py")
        cmd = [
            "mpiexec", "-n", str(n), "python", script,
            "--input", inp, "--output", outp,
            "--ksize", str(ksize),
            "--sigma", str(sigma),
        ]

        self._log("\n" + "=" * 72 + "\n")
        self._log(f"运行参数：P={n}, ksize={ksize}, sigma={sigma}\n")
        self._log("$ " + " ".join(cmd) + "\n\n")

        self._set_running(True)

        def worker():
            try:
                p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
                for line in p.stdout:
                    self.after(0, self._log, line)
                rc = p.wait()

                def finish():
                    self._set_running(False)
                    self._log(f"\n[GUI] 结束，exit code={rc}\n")
                    if rc == 0 and os.path.exists(outp):
                        self._load_preview(outp, "out")

                self.after(0, finish)

            except Exception as e:
                def fail():
                    self._set_running(False)
                    self._log(f"[GUI] 启动失败: {e}\n")
                self.after(0, fail)

        threading.Thread(target=worker, daemon=True).start()

# 第一次启动会闪退，第二次启动才正常
if __name__ == "__main__":
    App().mainloop()