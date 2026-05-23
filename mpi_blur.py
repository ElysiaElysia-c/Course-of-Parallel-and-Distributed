import argparse
import numpy as np
from mpi4py import MPI
from PIL import Image


def split_rows(h, size, rank):
    """Return (start, end) row indices for this rank; near-even split."""
    base = h // size
    rem = h % size
    start = rank * base + min(rank, rem)
    end = start + base + (1 if rank < rem else 0)
    return start, end


def scatter_rows(comm, full_img):
    """
    Scatter 2D uint8 image by contiguous row blocks using Scatterv.
    Returns local block (2D) and metadata for gather.
    """
    rank = comm.Get_rank()
    size = comm.Get_size()

    if rank == 0:
        h, w = full_img.shape
    else:
        h = w = None
    h = comm.bcast(h, root=0)
    w = comm.bcast(w, root=0)

    # counts/displs in elements (not bytes)
    counts = np.zeros(size, dtype=np.int64)
    displs = np.zeros(size, dtype=np.int64)
    for r in range(size):
        s, e = split_rows(h, size, r)
        counts[r] = (e - s) * w
        displs[r] = s * w

    local_n = int(counts[rank])
    local_buf = np.empty(local_n, dtype=np.uint8)

    if rank == 0:
        sendbuf = [full_img.ravel(), counts, displs, MPI.UNSIGNED_CHAR]
    else:
        sendbuf = None

    comm.Scatterv(sendbuf, local_buf, root=0)
    local = local_buf.reshape((-1, w))  # local_h x w
    return local, (h, w, counts, displs)


def exchange_halo(comm, local, radius=1):
    """
    Exchange halo rows with neighbors.
    local: local_h x w
    returns extended array with halos: (local_h + 2*radius) x w
    """
    rank = comm.Get_rank()
    size = comm.Get_size()
    local_h, w = local.shape

    top = rank - 1
    bot = rank + 1
    has_top = top >= 0
    has_bot = bot < size

    # Prepare halo buffers
    recv_top = np.empty((radius, w), dtype=np.uint8)
    recv_bot = np.empty((radius, w), dtype=np.uint8)

    send_top = local[:radius, :].copy()
    send_bot = local[-radius:, :].copy()

    t0 = MPI.Wtime()

    # Use Sendrecv for simplicity (blocking but deadlock-safe)
    if has_top:
        comm.Sendrecv(send_top, dest=top, sendtag=11,
                    recvbuf=recv_top, source=top, recvtag=22)
    else:
        # edge padding: replicate first row
        recv_top[:] = local[:1, :]

    if has_bot:
        comm.Sendrecv(send_bot, dest=bot, sendtag=22,
                    recvbuf=recv_bot, source=bot, recvtag=11)
    else:
        recv_bot[:] = local[-1:, :]

    t1 = MPI.Wtime()

    ext = np.vstack([recv_top, local, recv_bot])
    return ext, (t1 - t0)


def box_blur_3x3(ext):
    """
    ext: (local_h+2) x w, uint8, radius=1
    returns blurred local_h x w
    Edge handling for left/right uses 'edge' padding via np.pad.
    """
    # pad left/right by 1 (replicate)
    padded = np.pad(ext, ((0, 0), (1, 1)), mode="edge").astype(np.float32)

    # 3x3 sum via slicing (fast, vectorized)
    s = (
        padded[:-2, :-2] + padded[:-2, 1:-1] + padded[:-2, 2:] +
        padded[1:-1, :-2] + padded[1:-1, 1:-1] + padded[1:-1, 2:] +
        padded[2:, :-2] + padded[2:, 1:-1] + padded[2:, 2:]
    )
    out = (s / 9.0).clip(0, 255).astype(np.uint8)
    return out


def gather_rows(comm, local_out, meta):
    rank = comm.Get_rank()
    h, w, counts, displs = meta
    local_buf = local_out.ravel()

    if rank == 0:
        full_buf = np.empty(h * w, dtype=np.uint8)
        recvbuf = [full_buf, counts, displs, MPI.UNSIGNED_CHAR]
    else:
        recvbuf = None

    comm.Gatherv(local_buf, recvbuf, root=0)

    if rank == 0:
        return full_buf.reshape((h, w))
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Input image path")
    parser.add_argument("--output", required=True, help="Output image path")
    args = parser.parse_args()

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    if rank == 0:
        img = Image.open(args.input).convert("L")
        full = np.array(img, dtype=np.uint8)
    else:
        full = None

    comm.Barrier()
    t_start = MPI.Wtime()

    local, meta = scatter_rows(comm, full)
    ext, t_comm = exchange_halo(comm, local, radius=1)

    t0 = MPI.Wtime()
    local_out = box_blur_3x3(ext)
    t1 = MPI.Wtime()
    t_comp = t1 - t0

    out = gather_rows(comm, local_out, meta)

    comm.Barrier()
    t_end = MPI.Wtime()

    # reduce timings
    t_total = comm.reduce(t_end - t_start, op=MPI.MAX, root=0)
    t_comm_max = comm.reduce(t_comm, op=MPI.MAX, root=0)
    t_comp_max = comm.reduce(t_comp, op=MPI.MAX, root=0)

    if rank == 0:
        Image.fromarray(out, mode="L").save(args.output)
        print(f"[MPI] total_time_s={t_total:.6f} comm_time_s(max)={t_comm_max:.6f} comp_time_s(max)={t_comp_max:.6f}")


if __name__ == "__main__":
# 用多进程 / 多机集群对图片做 3x3 均值模糊 

# 执行方式：
# mpiexec -n 4 python mpi_blur.py --input cat.png --output cat1.png 
    main()