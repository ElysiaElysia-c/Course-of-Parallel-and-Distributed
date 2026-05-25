from mpi4py import MPI

MAXSIZE = 1000

def main():
    comm = MPI.COMM_WORLD
    myid = comm.Get_rank()
    numprocs = comm.Get_size()

    data = None

    # 进程0直接生成数组，不需要文件！
    if myid == 0:
        data = list(range(MAXSIZE))  # 生成 0~999 的数组

    # 广播数据
    data = comm.bcast(data, root=0)

    # 计算每个进程的区间
    x = MAXSIZE // numprocs
    low = myid * x
    high = low + x

    if myid == numprocs - 1:
        high = MAXSIZE

    # 局部求和
    mysum = sum(data[i] for i in range(low, high))

    # 归约求和
    total = comm.reduce(mysum, op=MPI.SUM, root=0)

    # 只有进程0输出最终结果
    if myid == 0:
        print(f"数组长度：{MAXSIZE}")
        print(f"运行进程数：{numprocs}")
        print(f"数组 0~999 总和 = {total}")

if __name__ == "__main__":
    main()