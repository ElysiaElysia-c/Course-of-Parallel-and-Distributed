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
    """
    Scatter RGB image by contiguous row blocks using Scatterv.
    full_img on rank0: uint8 array (H, W, 3)
    Returns local block (local_h, W, 3) and metadata.
    """
    rank = comm.Get_rank()
    size = comm.Get_size()

    if rank == 0:
        h, w, ch = full_img.shape
        assert ch == 3
    else:
        h = w = None

    h = comm.bcast(h, root=0)
    w = comm.bcast(w, root=0)

    w3 = w * 3  # elements per row in flattened RGB

    counts = np.zeros(size, dtype=np.int64)
    displs = np.zeros(size, dtype=np.int64)
    for r in range(size):
        s, e = split_rows(h, size, r)
        counts[r] = (e - s) * w3
        displs[r] = s * w3

    local_n = int(counts[rank])
    local_buf = np.empty(local_n, dtype=np.uint8)

    if rank == 0:
        sendbuf = [full_img.reshape(-1), counts, displs, MPI.UNSIGNED_CHAR]
    else:
        sendbuf = None

    comm.Scatterv(sendbuf, local_buf, root=0)

    local_h = local_n // w3
    local = local_buf.reshape((local_h, w, 3))
    return local, (h, w, counts, displs)


def exchange_halo(comm, local, radius):
    """
    local: (local_h, W, 3)
    returns ext: (local_h + 2*radius, W, 3)
    """
    rank = comm.Get_rank()
    size = comm.Get_size()
    local_h, w, ch = local.shape
    assert ch == 3

    top = rank - 1
    bot = rank + 1
    has_top = top >= 0
    has_bot = bot < size

    w3 = w * 3

    # Halo buffers as flat rows for Sendrecv convenience
    recv_top = np.empty((radius, w3), dtype=np.uint8)
    recv_bot = np.empty((radius, w3), dtype=np.uint8)

    send_top = local[:radius, :, :].reshape((radius, w3)).copy()
    send_bot = local[-radius:, :, :].reshape((radius, w3)).copy()

    t0 = MPI.Wtime()

    if has_top:
        comm.Sendrecv(send_top, dest=top, sendtag=11,
                    recvbuf=recv_top, source=top, recvtag=22)
    else:
        recv_top[:] = send_top[:1, :]  # replicate first row

    if has_bot:
        comm.Sendrecv(send_bot, dest=bot, sendtag=22,
                    recvbuf=recv_bot, source=bot, recvtag=11)
    else:
        recv_bot[:] = send_bot[-1:, :]  # replicate last row

    t1 = MPI.Wtime()

    ext_top = recv_top.reshape((radius, w, 3))
    ext_bot = recv_bot.reshape((radius, w, 3))
    ext = np.vstack([ext_top, local, ext_bot])
    return ext, (t1 - t0)


def gather_rows(comm, local_out, meta):
    """
    local_out: (local_h, W, 3)
    meta: (H, W, counts, displs) where counts/displs are in elements of uint8
    """
    rank = comm.Get_rank()
    h, w, counts, displs = meta

    local_buf = local_out.reshape(-1)

    if rank == 0:
        full_buf = np.empty(h * w * 3, dtype=np.uint8)
        recvbuf = [full_buf, counts, displs, MPI.UNSIGNED_CHAR]
    else:
        recvbuf = None

    comm.Gatherv(local_buf, recvbuf, root=0)

    if rank == 0:
        return full_buf.reshape((h, w, 3))
    return None


def make_gaussian_kernel(ksize, sigma):
    r = ksize // 2
    ax = np.arange(-r, r + 1, dtype=np.float32)
    xx, yy = np.meshgrid(ax, ax)
    k = np.exp(-(xx * xx + yy * yy) / (2.0 * sigma * sigma)).astype(np.float32)
    k /= np.sum(k)
    return k


def conv_from_ext(ext, kernel):
    """
    ext: (local_h + 2r, W, 3) uint8
    kernel: (ksize, ksize) float32
    returns: (local_h, W, 3) uint8
    """
    ksize = kernel.shape[0]
    r = ksize // 2

    local_h = ext.shape[0] - 2 * r
    w = ext.shape[1]

    out = np.empty((local_h, w, 3), dtype=np.uint8)

    for c in range(3):
        ext_c = ext[:, :, c]  # (local_h+2r, W)
        padded = np.pad(ext_c, ((0, 0), (r, r)), mode="edge").astype(np.float32)

        acc = np.zeros((local_h, w), dtype=np.float32)
        for dy in range(ksize):
            for dx in range(ksize):
                acc += kernel[dy, dx] * padded[dy:dy + local_h, dx:dx + w]

        out[:, :, c] = acc.clip(0, 255).astype(np.uint8)

    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--ksize", type=int, default=11, help="odd >=3")
    parser.add_argument("--sigma", type=float, default=2.0, help="gaussian sigma")
    args = parser.parse_args()

    if args.ksize < 3 or args.ksize % 2 == 0:
        raise ValueError("ksize must be odd and >=3")
    if args.sigma <= 0:
        raise ValueError("sigma must be > 0")

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    radius = args.ksize // 2

    if rank == 0:
        img = Image.open(args.input).convert("RGB")
        full = np.array(img, dtype=np.uint8)
    else:
        full = None

    comm.Barrier()
    t_start = MPI.Wtime()

    local, meta = scatter_rows(comm, full)
    ext, t_comm = exchange_halo(comm, local, radius=radius)

    t0 = MPI.Wtime()
    kernel = make_gaussian_kernel(args.ksize, args.sigma)
    local_out = conv_from_ext(ext, kernel)
    t1 = MPI.Wtime()
    t_comp = t1 - t0

    out = gather_rows(comm, local_out, meta)

    comm.Barrier()
    t_end = MPI.Wtime()

    t_total = comm.reduce(t_end - t_start, op=MPI.MAX, root=0)
    t_comm_max = comm.reduce(t_comm, op=MPI.MAX, root=0)
    t_comp_max = comm.reduce(t_comp, op=MPI.MAX, root=0)

    if rank == 0:
        Image.fromarray(out, mode="RGB").save(args.output)
        print(
            f"[MPI][GAUSSIAN] ksize={args.ksize} sigma={args.sigma} "
            f"total_time_s={t_total:.6f} comm_time_s(max)={t_comm_max:.6f} comp_time_s(max)={t_comp_max:.6f}"
        )


if __name__ == "__main__":
    main()