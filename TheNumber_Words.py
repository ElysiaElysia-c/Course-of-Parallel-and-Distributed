from mpi4py import MPI
import numpy as np
import random
import string
import time
import matplotlib.pyplot as plt

# ===================== 1. 全局配置 =====================
# 随机字符串总数量
TOTAL_STR_NUM = 5000
# 每个字符串长度（随机10-20字符）
STR_LEN_RANGE = (10, 20)
# 26个大写英文字母
ALPHABET = list(string.ascii_uppercase)
ALPHABET_SIZE = len(ALPHABET)

# MPI初始化
comm = MPI.COMM_WORLD
rank = comm.Get_rank()  # 当前进程编号
size = comm.Get_size()  # 总进程数


# ===================== 2. 数据生成（仅主进程执行） =====================
def generate_random_strings(total_num):
    """生成指定数量的随机大写字母字符串"""
    random.seed(42)  # 固定随机种子，保证数据一致
    strings = []
    for _ in range(total_num):
        length = random.randint(*STR_LEN_RANGE)
        s = ''.join(random.choice(ALPHABET) for _ in range(length))
        strings.append(s)
    return strings


# ===================== 3. 串行基准实现 =====================
def serial_count(strings):
    """串行统计字母频次（用于对比性能）"""
    count = np.zeros(ALPHABET_SIZE, dtype=int)
    for s in strings:
        for char in s:
            idx = ALPHABET.index(char)
            count[idx] += 1
    return count


# ===================== 4. 并行实现1：广播(Bcast) =====================
def parallel_bcast(strings):
    """
    广播方式：主进程将全部数据广播给所有进程
    每个进程处理自己分片的数据
    """
    # 1. 主进程广播完整数据
    strings = comm.bcast(strings, root=0)

    # 2. 数据划分（负载均衡：均分）
    local_strings = split_data_balanced(strings, rank, size)

    # 3. 本地统计
    local_count = np.zeros(ALPHABET_SIZE, dtype=int)
    for s in local_strings:
        for char in s:
            idx = ALPHABET.index(char)
            local_count[idx] += 1

    # 4. 全局规约求和
    global_count = comm.reduce(local_count, op=MPI.SUM, root=0)
    return global_count


# ===================== 5. 并行实现2：分散(Scatter) =====================
def parallel_scatter(strings):
    """
    分散方式：主进程将数据切分后分发给各进程
    无需全量数据传输，通信量更小
    """
    local_data = None

    if rank == 0:
        # 主进程：数据切分（负载均衡）
        chunks = split_chunks_balanced(strings, size)
        # 转换为numpy数组适配Scatter
        chunks = np.array(chunks, dtype=object)

    # 2. Scatter分发数据
    local_data = comm.scatter(chunks if rank == 0 else None, root=0)

    # 3. 本地统计
    local_count = np.zeros(ALPHABET_SIZE, dtype=int)
    for s in local_data:
        for char in s:
            idx = ALPHABET.index(char)
            local_count[idx] += 1

    # 4. 规约求和
    global_count = comm.reduce(local_count, op=MPI.SUM, root=0)
    return global_count


# ===================== 6. 并行实现3：异步(Isend/Irecv) =====================
def parallel_async(strings):
    """
    异步通信方式：非阻塞发送/接收，计算与通信重叠
    """
    local_count = np.zeros(ALPHABET_SIZE, dtype=int)

    if rank == 0:
        # 主进程：异步发送分片数据
        chunks = split_chunks_balanced(strings, size)
        reqs = []
        for i in range(1, size):
            req = comm.isend(chunks[i], dest=i, tag=i)
            reqs.append(req)
        # 主进程处理自己的分片
        local_strings = chunks[0]
        # 等待所有发送完成
        MPI.Request.Waitall(reqs)
    else:
        # 从进程：异步接收数据
        local_strings = comm.recv(source=0, tag=rank)

    # 本地统计
    for s in local_strings:
        for char in s:
            idx = ALPHABET.index(char)
            local_count[idx] += 1

    # 异步规约
    global_count = None
    if rank == 0:
        global_count = np.zeros(ALPHABET_SIZE, dtype=int)
    comm.Reduce(local_count, global_count, op=MPI.SUM, root=0)
    return global_count


# ===================== 工具函数：负载均衡划分 =====================
def split_data_balanced(data, rank, size):
    """均匀划分数据，保证负载均衡"""
    n = len(data)
    avg = n // size
    remainder = n % size

    # 前remainder个进程多分配1个，实现完美负载均衡
    if rank < remainder:
        start = rank * (avg + 1)
        end = start + avg + 1
    else:
        start = remainder * (avg + 1) + (rank - remainder) * avg
        end = start + avg
    return data[start:end]


def split_chunks_balanced(data, size):
    """将数据切分为size个均衡分片"""
    chunks = []
    for i in range(size):
        chunk = split_data_balanced(data, i, size)
        chunks.append(chunk)
    return chunks


# ===================== 7. 性能测试与可视化 =====================
def test_performance():
    """测试三种并行方式+串行的性能，绘制加速比曲线"""
    # 仅主进程生成数据
    strings = generate_random_strings(TOTAL_STR_NUM) if rank == 0 else None

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

    # 测试：1~最大进程数
    for p in range(1, size + 1):
        if rank == 0:
            print(f"\n========== 测试进程数：{p} ==========")

        # 仅使用前p个进程测试
        if rank >= p:
            continue

        comm.Barrier()  # 同步所有进程

        # ---------------- 串行测试 ----------------
        if rank == 0:
            t_start = time.time()
            serial_res = serial_count(strings)
            t_serial = time.time() - t_start
            print(f"串行耗时：{t_serial:.4f}s")
        else:
            t_serial = 0
            serial_res = None

        comm.Barrier()

        # ---------------- 广播方式测试 ----------------
        t_start = time.time()
        bcast_res = parallel_bcast(strings)
        comm.Barrier()
        t_bcast = time.time() - t_start
        if rank == 0:
            print(f"广播(Bcast)耗时：{t_bcast:.4f}s")

        # ---------------- 分散方式测试 ----------------
        t_start = time.time()
        scatter_res = parallel_scatter(strings)
        comm.Barrier()
        t_scatter = time.time() - t_start
        if rank == 0:
            print(f"分散(Scatter)耗时：{t_scatter:.4f}s")

        # ---------------- 异步方式测试 ----------------
        t_start = time.time()
        async_res = parallel_async(strings)
        comm.Barrier()
        t_async = time.time() - t_start
        if rank == 0:
            print(f"异步(Async)耗时：{t_async:.4f}s")

        # 主进程记录结果
        if rank == 0:
            results["proc_nums"].append(p)
            results["serial_t"].append(t_serial)
            results["bcast_t"].append(t_bcast)
            results["scatter_t"].append(t_scatter)
            results["async_t"].append(t_async)

            # 计算加速比：加速比 = 串行时间 / 并行时间
            results["speedup_bcast"].append(t_serial / t_bcast)
            results["speedup_scatter"].append(t_serial / t_scatter)
            results["speedup_async"].append(t_serial / t_async)

            # 验证结果正确性
            assert np.array_equal(serial_res, bcast_res)
            assert np.array_equal(serial_res, scatter_res)
            assert np.array_equal(serial_res, async_res)
            print("所有方式统计结果一致！")

    # 主进程可视化
    if rank == 0:
        # plot_performance(results)
        print_letter_count(serial_res, strings)


def plot_performance(res):
    """可视化：运行时间对比 + 加速比曲线"""
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # 子图1：运行时间对比
    ax1.plot(res["proc_nums"], res["serial_t"], 'o-', label='串行', linewidth=2)
    ax1.plot(res["proc_nums"], res["bcast_t"], 's-', label='广播(Bcast)', linewidth=2)
    ax1.plot(res["proc_nums"], res["scatter_t"], '^-', label='分散(Scatter)', linewidth=2)
    ax1.plot(res["proc_nums"], res["async_t"], '*-', label='异步(Async)', linewidth=2)
    ax1.set_xlabel('Number of processes')
    ax1.set_ylabel('Runtime (s)')
    ax1.set_title('Comparison of running time by different methods')
    ax1.legend()
    ax1.grid(True)

    # 子图2：加速比曲线
    ideal_speedup = res["proc_nums"]  # 理想线性加速比
    ax2.plot(res["proc_nums"], ideal_speedup, 'k--', label='理想加速比', linewidth=2)
    ax2.plot(res["proc_nums"], res["speedup_bcast"], 's-', label='广播(Bcast)', linewidth=2)
    ax2.plot(res["proc_nums"], res["speedup_scatter"], '^-', label='分散(Scatter)', linewidth=2)
    ax2.plot(res["proc_nums"], res["speedup_async"], '*-', label='异步(Async)', linewidth=2)
    ax2.set_xlabel('Number of processes')
    ax2.set_ylabel('Speedup')
    ax2.set_title('Speedup Curve') # 加速比曲线
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig('performance_analysis.png', dpi=300)
    plt.show()
    print("\n 性能图表已保存为 performance_analysis.png")


def print_letter_count(count_arr, strings):
    """打印统计结果"""
    total_chars = sum(len(s) for s in strings)
    print("\n========== 字母统计结果 ==========")
    print(f"总字符串数量：{TOTAL_STR_NUM}")
    print(f"总字符数：{total_chars}")
    for i, char in enumerate(ALPHABET):
        print(f"{char}：{count_arr[i]} 次")


# ===================== 主函数 =====================
if __name__ == "__main__":
    test_performance()