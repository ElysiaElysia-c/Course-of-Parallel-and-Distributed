import matplotlib.pyplot as plt
    # 存储测试结果：进程数、串行时间、三种并行时间、加速比
results = {
        "proc_nums": [],
        "serial_t": [],
        "bcast_t": [],
        "scatter_t": [],
        "async_t": [],
        "speedup_bcast": [],
        "speedup_scatter": [],
        "speedup_async": []
    }

def plot_performance(res):
    """可视化：运行时间对比 + 加速比曲线"""
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # 子图1：运行时间对比 
    ax1.plot(res["proc_nums"], res["serial_t"], color='black', marker='o', label='串行', linewidth=2)
    ax1.plot(res["proc_nums"], res["bcast_t"], color='red', marker='s', label='广播(Bcast)', linewidth=2)
    ax1.plot(res["proc_nums"], res["scatter_t"], color='blue', marker='^', label='分散(Scatter)', linewidth=2)
    ax1.plot(res["proc_nums"], res["async_t"], color='green', marker='*', label='异步(Async)', linewidth=2)
    ax1.set_xlabel('Number of processes')
    ax1.set_ylabel('Runtime (s)')
    ax1.set_title('Comparison of running time by different methods')
    ax1.legend()
    ax1.grid(True)

    # 子图2：加速比曲线
    # ideal_speedup = res["proc_nums"]  # 理想线性加速比
    # ax2.plot(res["proc_nums"], ideal_speedup, 'k--', label='理想加速比', linewidth=2)
    ax2.plot(res["proc_nums"], res["speedup_bcast"], color='red', marker='s', label='广播(Bcast)', linewidth=2)
    ax2.plot(res["proc_nums"], res["speedup_scatter"], color='blue', marker='^', label='分散(Scatter)', linewidth=2)
    ax2.plot(res["proc_nums"], res["speedup_async"], color='green', marker='*', label='异步(Async)', linewidth=2)
    ax2.set_xlabel('Number of processes')
    ax2.set_ylabel('Speedup')
    ax2.set_title('Speedup Curve') # 加速比曲线
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig('performance_analysis.png', dpi=300)
    plt.show()
    print("\n 性能图表已保存为 performance_analysis.png")

if __name__ == "__main__":
    results["proc_nums"].append(1)
    results["proc_nums"].append(3)
    results["proc_nums"].append(4)
    results["proc_nums"].append(6)
    results["proc_nums"].append(8)

    results["serial_t"].append(0.0124)
    results["serial_t"].append(0.0124)
    results["serial_t"].append(0.0124)
    results["serial_t"].append(0.0124) 
    results["serial_t"].append(0.0124)

    results["bcast_t"].append(0.0138)
    results["bcast_t"].append(0.0060)
    results["bcast_t"].append(0.0053)
    results["bcast_t"].append(0.0043)
    results["bcast_t"].append(0.0042)

    results["scatter_t"].append(0.0140)
    results["scatter_t"].append(0.0055)
    results["scatter_t"].append(0.0051)
    results["scatter_t"].append(0.0036)
    results["scatter_t"].append(0.0034)

    results["async_t"].append(0.0128)
    results["async_t"].append(0.0052)
    results["async_t"].append(0.0050)
    results["async_t"].append(0.0026)
    results["async_t"].append(0.0030)

    # 计算加速比：加速比 = 串行时间 / 并行时间
    results["speedup_bcast"].append(0.8969879213739055)
    results["speedup_bcast"].append(2.048111450823396)
    results["speedup_bcast"].append(2.4949378561143236)
    results["speedup_bcast"].append(3.071606038254193)
    results["speedup_bcast"].append(3.0270760201891074) # Bcast加速比

    results["speedup_scatter"].append(0.8876775826139494)
    results["speedup_scatter"].append(2.249082643997827)
    results["speedup_scatter"].append(2.5931953836185873)
    results["speedup_scatter"].append(3.6850289659101394)
    results["speedup_scatter"].append(3.788203128390898) # Scatter加速比
    
    results["speedup_async"].append(0.9699082397202218)
    results["speedup_async"].append(2.3817883046690507)
    results["speedup_async"].append(2.662775930287634)
    results["speedup_async"].append(4.972754526087114)
    results["speedup_async"].append(4.38184631759493) # Async加速比

    plot_performance(results)


