import os
import numpy as np
import matplotlib.pyplot as plt

# 固定 ksize=21, sigma=5，不同进程数的数据（来自你的测试结果）
P = np.array([1, 2, 4, 6, 8], dtype=int)
T_total = np.array([8.629952, 6.280245, 4.925080, 4.475618, 4.284471], dtype=float)

# 加速比 S(P)=T(1)/T(P)
T1 = T_total[P == 1][0]
speedup = T1 / T_total

outdir = "figures_P_only"
os.makedirs(outdir, exist_ok=True)

# 图1：总时间 vs 进程数
plt.figure(figsize=(7.2, 4.6))
plt.plot(P, T_total, marker="o", linewidth=2)
plt.xticks(P)  # 只显示你的 P 取值
plt.xlabel("P (MPI processes) ")
plt.ylabel("Total time (s) ")
plt.title("ksize=21, sigma=5")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(outdir, "time.png"), dpi=180)
plt.close()

# 图2：加速比 vs 进程数（含理想线）
plt.figure(figsize=(7.2, 4.6))
plt.plot(P, speedup, marker="o", linewidth=2, label="Measured speedup")
plt.plot(P, P, linestyle="--", linewidth=1.6, label="Ideal speedup (S=P)")
plt.xticks(P)
plt.xlabel("P (MPI processes) ")
plt.ylabel("Speedup S(P)=T(1)/T(P)")
plt.title("ksize=21, sigma=5")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(outdir, "speedup.png"), dpi=180)
plt.close()

print("已生成图像：")
print(os.path.join(outdir, "time.png"))
print(os.path.join(outdir, "speedup.png"))

# 可选：打印加速比数值，方便写报告
print("\nSpeedup:")
for p, s in zip(P, speedup):
    print(f"P={p:2d}  S={s:.3f}")