import argparse
import numpy as np
from mpi4py import MPI
from PIL import Image


def split_rows(h, size, rank):
    base = h // size
    rem = h % size
    start = rank * base + min(rank, rem)
    end = start + base + (1 if rank < rem else 0)
    return start, end


def scatter_rows(comm, full_img):
    rank = comm.Get_rank()
    size = comm.Get_size()

    if rank == 0:
        h, w = full_img.shape
    else:
        h = w = None
    h = comm.bcast(h, root=0)
    w = comm.bcast(w, root=0)

    counts = np.zeros(size, dtype=np.int64)
    displs = np.zeros(size, dtype=np.int64)
    for r in range(size):
        s, e = split_rows(h, size, r)
        counts[r] = (e - s) * w
        displs[r] = s * w

    local_n = int(counts[comm.Get_rank()])
    local_buf = np.empty(local_n, dtype=np.uint8)

    if comm.Get_rank() == 0:
        sendbuf = [full_img.ravel(), counts, displs, MPI.UNSIGNED_CHAR]
    else:
        sendbuf = None

    comm.Scatterv(sendbuf, local_buf, root=0)
    local = local_buf.reshape((-1, w))
    return local, (h, w, counts, displs)


def exchange_halo(comm, local, radius):
    rank = comm.Get_rank()
    size = comm.Get_size()
    local_h, w = local.shape

    top = rank - 1
    bot = rank + 1
    has_top = top >= 0
    has_bot = bot < size

    recv_top = np.empty((radius, w), dtype=np.uint8)
    recv_bot = np.empty((radius, w), dtype=np.uint8)
    send_top = local[:radius, :].copy()
    send_bot = local[-radius:, :].copy()

    t0 = MPI.Wtime()

    # top
    if has_top:
        comm.Sendrecv(send_top, dest=top, sendtag=11,
                    recvbuf=recv_top, source=top, recvtag=22)
    else:
        recv_top[:] = local[:1, :]

    # bottom
    if has_bot:
        comm.Sendrecv(send_bot, dest=bot, sendtag=22,
                    recvbuf=recv_bot, source=bot, recvtag=11)
    else:
        recv_bot[:] = local[-1:, :]

    t1 = MPI.Wtime()
    ext = np.vstack([recv_top, local, recv_bot])
    return ext, (t1 - t0)


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


def make_gaussian_kernel(ksize, sigma):
    r = ksize // 2
    ax = np.arange(-r, r + 1, dtype=np.float32)
    xx, yy = np.meshgrid(ax, ax)
    k = np.exp(-(xx * xx + yy * yy) / (2.0 * sigma * sigma)).astype(np.float32)
    k /= np.sum(k)
    return k


def conv_from_ext(ext, kernel):
    ksize = kernel.shape[0]
    r = ksize // 2

    padded = np.pad(ext, ((0, 0), (r, r)), mode="edge").astype(np.float32)
    local_h = ext.shape[0] - 2 * r
    w = ext.shape[1]
    out = np.zeros((local_h, w), dtype=np.float32)

    for dy in range(ksize):
        for dx in range(ksize):
            out += kernel[dy, dx] * padded[dy:dy + local_h, dx:dx + w]

    return out.clip(0, 255).astype(np.uint8)


def sobel_from_ext(ext):
    p = np.pad(ext, ((0, 0), (1, 1)), mode="edge").astype(np.float32)
    gx = (
        -1 * p[:-2, :-2] + 0 * p[:-2, 1:-1] + 1 * p[:-2, 2:] +
        -2 * p[1:-1, :-2] + 0 * p[1:-1, 1:-1] + 2 * p[1:-1, 2:] +
        -1 * p[2:, :-2] + 0 * p[2:, 1:-1] + 1 * p[2:, 2:]
    )
    gy = (
        -1 * p[:-2, :-2] + -2 * p[:-2, 1:-1] + -1 * p[:-2, 2:] +
         0 * p[1:-1, :-2] +  0 * p[1:-1, 1:-1] +  0 * p[1:-1, 2:] +
         1 * p[2:, :-2] +  2 * p[2:, 1:-1] +  1 * p[2:, 2:]
    )
    mag = np.sqrt(gx * gx + gy * gy)
    return mag.clip(0, 255).astype(np.uint8)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--op", choices=["box", "gaussian", "sobel"], default="box")
    parser.add_argument("--ksize", type=int, default=3, help="odd >=3 (box/gaussian)")
    parser.add_argument("--sigma", type=float, default=1.2, help="gaussian sigma")
    args = parser.parse_args()

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    if args.op == "sobel":
        ksize = 3
        radius = 1
    else:
        ksize = args.ksize
        if ksize < 3 or ksize % 2 == 0:
            raise ValueError("ksize must be odd and >=3")
        radius = ksize // 2

    if rank == 0:
        img = Image.open(args.input).convert("L")
        full = np.array(img, dtype=np.uint8)
    else:
        full = None

    comm.Barrier()
    t_start = MPI.Wtime()

    local, meta = scatter_rows(comm, full)
    ext, t_comm = exchange_halo(comm, local, radius=radius)

    t0 = MPI.Wtime()
    if args.op == "box":
        kernel = np.ones((ksize, ksize), dtype=np.float32) / (ksize * ksize)
        local_out = conv_from_ext(ext, kernel)
    elif args.op == "gaussian":
        kernel = make_gaussian_kernel(ksize, args.sigma)
        local_out = conv_from_ext(ext, kernel)
    else:  # sobel
        local_out = sobel_from_ext(ext)
    t1 = MPI.Wtime()
    t_comp = t1 - t0

    out = gather_rows(comm, local_out, meta)

    comm.Barrier()
    t_end = MPI.Wtime()

    t_total = comm.reduce(t_end - t_start, op=MPI.MAX, root=0)
    t_comm_max = comm.reduce(t_comm, op=MPI.MAX, root=0)
    t_comp_max = comm.reduce(t_comp, op=MPI.MAX, root=0)

    if rank == 0:
        Image.fromarray(out, mode="L").save(args.output)
        print(f"[MPI] op={args.op} ksize={ksize} sigma={args.sigma} "
            f"total_time_s={t_total:.6f} comm_time_s(max)={t_comm_max:.6f} comp_time_s(max)={t_comp_max:.6f}")


if __name__ == "__main__":
    main()