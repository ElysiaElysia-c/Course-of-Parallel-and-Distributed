import os
import threading
import subprocess
import tkinter as tk
from tkinter import filedialog, ttk, messagebox
from PIL import Image, ImageTk


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("MPI Image Filter (Windows)")
        self.geometry("980x600")

        self.input_path = tk.StringVar()
        self.output_path = tk.StringVar(value=os.path.join(os.path.expanduser("~"), "Desktop", "out.jpg"))
        self.nproc = tk.IntVar(value=8)

        self._img_in_tk = None
        self._img_out_tk = None

        self._build_ui()

    def _build_ui(self):
        frm = ttk.Frame(self, padding=10)
        frm.pack(fill="both", expand=True)

        r = 0
        ttk.Label(frm, text="Input image:").grid(row=r, column=0, sticky="w")
        ttk.Entry(frm, textvariable=self.input_path, width=80).grid(row=r, column=1, sticky="we", padx=6)
        ttk.Button(frm, text="Browse...", command=self.pick_input).grid(row=r, column=2, sticky="e")
        r += 1

        ttk.Label(frm, text="Output image:").grid(row=r, column=0, sticky="w")
        ttk.Entry(frm, textvariable=self.output_path, width=80).grid(row=r, column=1, sticky="we", padx=6)
        ttk.Button(frm, text="Browse...", command=self.pick_output).grid(row=r, column=2, sticky="e")
        r += 1

        ttk.Label(frm, text="MPI processes (-n):").grid(row=r, column=0, sticky="w")
        ttk.Spinbox(frm, from_=1, to=64, textvariable=self.nproc, width=10).grid(row=r, column=1, sticky="w", padx=6)
        ttk.Button(frm, text="Run", command=self.run_job).grid(row=r, column=2, sticky="e")
        r += 1

        imgfrm = ttk.Frame(frm)
        imgfrm.grid(row=r, column=0, columnspan=3, sticky="nsew", pady=10)
        imgfrm.columnconfigure(0, weight=1)
        imgfrm.columnconfigure(1, weight=1)

        ttk.Label(imgfrm, text="Input").grid(row=0, column=0)
        ttk.Label(imgfrm, text="Output").grid(row=0, column=1)

        self.preview_in = ttk.Label(imgfrm)
        self.preview_in.grid(row=1, column=0, padx=10)
        self.preview_out = ttk.Label(imgfrm)
        self.preview_out.grid(row=1, column=1, padx=10)

        r += 1
        ttk.Label(frm, text="Logs:").grid(row=r, column=0, sticky="w")
        r += 1

        self.txt = tk.Text(frm, height=10)
        self.txt.grid(row=r, column=0, columnspan=3, sticky="nsew")
        frm.rowconfigure(r, weight=1)
        frm.columnconfigure(1, weight=1)

    def _log(self, s):
        self.txt.insert("end", s)
        self.txt.see("end")

    def pick_input(self):
        p = filedialog.askopenfilename(filetypes=[("Images", "*.png;*.jpg;*.jpeg;*.bmp"), ("All", "*.*")])
        if not p:
            return
        self.input_path.set(p)
        self._load_preview(p, which="in")

    def pick_output(self):
        p = filedialog.asksaveasfilename(defaultextension=".jpg",
                                        filetypes=[("JPEG", "*.jpg"), ("PNG", "*.png"), ("All", "*.*")])
        if not p:
            return
        self.output_path.set(p)

    def _load_preview(self, path, which):
        try:
            img = Image.open(path).convert("RGB")
            img.thumbnail((420, 320))
            tkimg = ImageTk.PhotoImage(img)
            if which == "in":
                self._img_in_tk = tkimg
                self.preview_in.configure(image=tkimg)
            else:
                self._img_out_tk = tkimg
                self.preview_out.configure(image=tkimg)
        except Exception as e:
            self._log(f"Preview load failed: {e}\n")

    def run_job(self):
        inp = self.input_path.get().strip()
        outp = self.output_path.get().strip()
        if not inp or not os.path.exists(inp):
            messagebox.showerror("Error", "Choose a valid input image.")
            return

        n = int(self.nproc.get())

        # 计算脚本路径（同目录下的 mpi_blur.py）
        script = os.path.join(os.path.dirname(__file__), "mpi_blur.py")

        cmd = ["mpiexec", "-n", str(n), "python", script, "--input", inp, "--output", outp]
        self._log("\n$ " + " ".join(cmd) + "\n")

        def worker():
            try:
                p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
                for line in p.stdout:
                    self.after(0, self._log, line)
                rc = p.wait()
                if rc == 0 and os.path.exists(outp):
                    self.after(0, self._load_preview, outp, "out")
                else:
                    self.after(0, self._log, f"[GUI] exit code {rc}\n")
            except Exception as e:
                self.after(0, self._log, f"[GUI] failed: {e}\n")

        threading.Thread(target=worker, daemon=True).start()


if __name__ == "__main__":
    # pip install pillow
    App().mainloop()