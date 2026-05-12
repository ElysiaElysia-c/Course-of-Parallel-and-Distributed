from mpi4py import MPI
import numpy as np
import random
import string
import matplotlib.pyplot as plt

# =====================================================
# 1. 全局参数设置
# =====================================================

# 总字符串数量（调大后更容易看到并行加速）
TOTAL_STR_NUM = 5000

# 每个字符串长度范围
STR_LEN_RANGE = (10, 20)

# 大写字母表
ALPHABET = list(string.ascii_uppercase)

# 字母数量
ALPHABET_SIZE = len(ALPHABET)

# 建立字典：字符 -> 下标
# 比 ALPHABET.index() 快很多
CHAR_TO_IDX = {c: i for i, c in enumerate(ALPHABET)}

# MPI初始化
comm = MPI.COMM_WORLD
rank = comm.Get_rank()   # 当前进程编号
size = comm.Get_size()   # 总进程数


# =====================================================
# 2. 数据生成（只有主进程执行）
# =====================================================
def generate_random_strings(total_num):
    """
    生成随机字符串
    """
    random.seed(42)

    strings = []

    for _ in range(total_num):
        length = random.randint(*STR_LEN_RANGE)

        s = ''.join(
            random.choice(ALPHABET)
            for _ in range(length)
        )

        strings.append(s)

    return strings


# =====================================================
# 3. 数据均匀划分（负载均衡）
# =====================================================
def split_data_balanced(data, rank, size):
    """
    把数据平均分给 size 个进程
    """

    n = len(data)

    avg = n // size
    remainder = n % size

    # 前 remainder 个进程多分一个
    if rank < remainder:
        start = rank * (avg + 1)
        end = start + avg + 1
    else:
        start = remainder * (avg + 1) + (rank - remainder) * avg
        end = start + avg

    return data[start:end]


def split_chunks_balanced(data, size):
    """
    切成 size 份
    """
    return [
        split_data_balanced(data, i, size)
        for i in range(size)
    ]


# =====================================================
# 4. 串行统计
# =====================================================
def serial_count(strings):
    """
    串行统计
    """
    count = np.zeros(ALPHABET_SIZE, dtype=int)

    for s in strings:
        for char in s:
            idx = CHAR_TO_IDX[char]
            count[idx] += 1

    return count


# =====================================================
# 5. 并行方式1：Bcast
# =====================================================
def parallel_bcast(strings):
    """
    广播方式：
    所有进程都收到完整数据，
    然后各自处理自己的那一部分
    """

    # 广播完整数据
    strings = comm.bcast(strings, root=0)

    # 本地分片
    local_strings = split_data_balanced(strings, rank, size)

    # 本地统计
    local_count = np.zeros(ALPHABET_SIZE, dtype=int)

    for s in local_strings:
        for char in s:
            local_count[CHAR_TO_IDX[char]] += 1

    # 汇总
    global_count = comm.reduce(
        local_count,
        op=MPI.SUM,
        root=0
    )

    return global_count


# =====================================================
# 6. 并行方式2：Scatter
# =====================================================
def parallel_scatter(strings):
    """
    Scatter：
    主进程切好后直接分发
    """

    if rank == 0:
        chunks = split_chunks_balanced(strings, size)
    else:
        chunks = None

    # 每个进程拿到自己的数据
    local_data = comm.scatter(chunks, root=0)

    local_count = np.zeros(ALPHABET_SIZE, dtype=int)

    for s in local_data:
        for char in s:
            local_count[CHAR_TO_IDX[char]] += 1

    global_count = comm.reduce(
        local_count,
        op=MPI.SUM,
        root=0
    )

    return global_count


# =====================================================
# 7. 并行方式3：异步通信
# =====================================================
def parallel_async(strings):
    """
    Isend + Irecv
    真异步
    """

    local_count = np.zeros(ALPHABET_SIZE, dtype=int)

    # ---------------- 主进程 ----------------
    if rank == 0:

        chunks = split_chunks_balanced(strings, size)

        reqs = []

        # 异步发送
        for i in range(1, size):
            comm.send(
                chunks[i],
                dest=i,
                tag=i
            )

        # 自己先计算
        local_strings = chunks[0]

        for s in local_strings:
            for char in s:
                local_count[CHAR_TO_IDX[char]] += 1

        # 等待发送完成
        MPI.Request.Waitall(reqs)

    # ---------------- 从进程 ----------------
    else:

        # 异步接收
        req = comm.irecv(
            source=0,
            tag=rank
        )

        local_strings = req.wait()

        for s in local_strings:
            for char in s:
                local_count[CHAR_TO_IDX[char]] += 1

    # Reduce
    global_count = None

    if rank == 0:
        global_count = np.zeros(ALPHABET_SIZE, dtype=int)

    comm.Reduce(
        local_count,
        global_count,
        op=MPI.SUM,
        root=0
    )

    return global_count


# =====================================================
# 8. 打印统计结果
# =====================================================
def print_result(count_arr, strings):
    """
    打印统计结果
    """
    total_chars = sum(len(s) for s in strings)

    print("\n========== 统计结果 ==========")
    print("总字符串数:", TOTAL_STR_NUM)
    print("总字符数:", total_chars)

    for i, c in enumerate(ALPHABET):
        print(f"{c}: {count_arr[i]}")


# =====================================================
# 9. 性能测试
# =====================================================
def test():
    """
    性能测试
    """

    # 只有主进程生成数据
    if rank == 0:
        strings = generate_random_strings(TOTAL_STR_NUM)
    else:
        strings = None

    # ---------------- 串行 ----------------
    if rank == 0:
        t1 = MPI.Wtime()

        serial_res = serial_count(strings)

        t_serial = MPI.Wtime() - t1

        print(f"\n串行耗时: {t_serial:.4f}s")
    else:
        serial_res = None

    comm.Barrier()

    # ---------------- Bcast ----------------
    t1 = MPI.Wtime()

    bcast_res = parallel_bcast(strings)

    comm.Barrier()

    t_bcast = MPI.Wtime() - t1

    if rank == 0:
        print(f"Bcast耗时: {t_bcast:.4f}s")

    # ---------------- Scatter ----------------
    t1 = MPI.Wtime()

    scatter_res = parallel_scatter(strings)

    comm.Barrier()

    t_scatter = MPI.Wtime() - t1

    if rank == 0:
        print(f"Scatter耗时: {t_scatter:.4f}s")

    # ---------------- Async ----------------
    t1 = MPI.Wtime()

    async_res = parallel_async(strings)

    comm.Barrier()

    t_async = MPI.Wtime() - t1

    if rank == 0:
        print(f"Async耗时: {t_async:.4f}s")

    # ---------------- 验证结果 ----------------
    if rank == 0:
        assert np.array_equal(serial_res, bcast_res)
        assert np.array_equal(serial_res, scatter_res)
        assert np.array_equal(serial_res, async_res)

        print("\n结果一致，验证通过")

        print("\n加速比：")
        print("Bcast :", t_serial / t_bcast)
        print("Scatter:", t_serial / t_scatter)
        print("Async :", t_serial / t_async)

        print_result(serial_res, strings)


# =====================================================
# 主函数
# =====================================================
if __name__ == "__main__":
    test()