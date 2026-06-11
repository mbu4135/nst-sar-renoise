#!/usr/bin/env python3
"""
Incoherent temporal average of polarimetric covariance-matrix diagonal terms
and |C12| from a coregistered Sentinel-1 IW RSLC stack (VV + VH).

Per epoch t (linear scale):
    C11_t      = |VV_t|^2                 (VV intensity)
    C22_t      = |VH_t|^2                 (VH intensity)
    C12mag_t   = |VV_t * conj(VH_t)| = |VV_t| * |VH_t|

Output = nanmean over the 100 epochs, where a pixel value of 0 in an SLC
(i.e. complex 0, outside the valid burst footprint) is treated as NaN and
excluded from that pixel's mean.  Pixels with no valid epoch are written as 0.

Memory: the stack is ~1.4 TB, so we never hold a full image. We sweep over
azimuth row-blocks; for each block we loop the 100 dates, reading only that
block from each VV/VH file and accumulating per-pixel sum + count.

I/O note: GAMMA binary files are big-endian. Inputs are read as '>c8'
(big-endian complex64). Outputs (.img) are written as big-endian float32
('>f4') for consistency with the GAMMA ecosystem.
"""

import os
import sys
import glob
import time

import numpy as np

VV_DIR = "/media/sdb8TB/sentinel1/korea/rslc_prep_vv"
VH_DIR = "/media/sdb8TB/sentinel1/korea/rslc_prep_vh"
OUT_DIR = "/media/sdb8TB/sentinel1/korea/ave_dir"

NCOLS = 68647        # range_samples
NROWS = 13124        # azimuth_lines
IN_DTYPE = np.dtype(">c8")    # GAMMA FCOMPLEX, big-endian
OUT_DTYPE = np.dtype(">f4")   # big-endian float32

BLOCK_ROWS = 1024    # azimuth lines per block (~0.56 GB per complex block)


def get_dates():
    vv = sorted(os.path.basename(p)[:8]
                for p in glob.glob(os.path.join(VV_DIR, "*.rslc")))
    vh = sorted(os.path.basename(p)[:8]
                for p in glob.glob(os.path.join(VH_DIR, "*.rslc")))
    if vv != vh:
        sys.exit("ERROR: VV and VH date lists differ")
    return vv


def read_block(path, start_row, nrows):
    """Read `nrows` azimuth lines starting at `start_row` -> (nrows, NCOLS) complex64."""
    count = nrows * NCOLS
    offset = start_row * NCOLS * IN_DTYPE.itemsize
    with open(path, "rb") as f:
        f.seek(offset)
        buf = np.fromfile(f, dtype=IN_DTYPE, count=count)
    if buf.size != count:
        sys.exit(f"ERROR: short read on {path} "
                 f"(got {buf.size}, expected {count})")
    return buf.reshape(nrows, NCOLS)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    dates = get_dates()
    n = len(dates)
    print(f"{n} dates, image {NROWS} x {NCOLS}, block = {BLOCK_ROWS} rows",
          flush=True)

    out_c11 = os.path.join(OUT_DIR, "C11.img")
    out_c22 = os.path.join(OUT_DIR, "C22.img")
    out_c12 = os.path.join(OUT_DIR, "C12_mag.img")

    t0 = time.time()
    with open(out_c11, "wb") as fc11, \
         open(out_c22, "wb") as fc22, \
         open(out_c12, "wb") as fc12:

        for start in range(0, NROWS, BLOCK_ROWS):
            nb = min(BLOCK_ROWS, NROWS - start)
            shp = (nb, NCOLS)

            sum_c11 = np.zeros(shp, dtype=np.float64)
            sum_c22 = np.zeros(shp, dtype=np.float64)
            sum_c12 = np.zeros(shp, dtype=np.float64)
            cnt_c11 = np.zeros(shp, dtype=np.int32)
            cnt_c22 = np.zeros(shp, dtype=np.int32)
            cnt_c12 = np.zeros(shp, dtype=np.int32)

            for d in dates:
                vv = read_block(os.path.join(VV_DIR, f"{d}.rslc"), start, nb)
                vh = read_block(os.path.join(VH_DIR, f"{d}.rslc"), start, nb)

                amp_vv = np.abs(vv).astype(np.float64)   # |VV|
                amp_vh = np.abs(vh).astype(np.float64)   # |VH|

                m_vv = amp_vv > 0.0      # valid (non-zero) VV pixel
                m_vh = amp_vh > 0.0      # valid (non-zero) VH pixel
                m_12 = m_vv & m_vh       # both valid -> C12 defined

                sum_c11 += np.where(m_vv, amp_vv * amp_vv, 0.0)
                sum_c22 += np.where(m_vh, amp_vh * amp_vh, 0.0)
                sum_c12 += np.where(m_12, amp_vv * amp_vh, 0.0)
                cnt_c11 += m_vv
                cnt_c22 += m_vh
                cnt_c12 += m_12

            # nanmean: divide by count, 0 where no valid epoch
            ave_c11 = np.where(cnt_c11 > 0, sum_c11 / np.maximum(cnt_c11, 1), 0.0)
            ave_c22 = np.where(cnt_c22 > 0, sum_c22 / np.maximum(cnt_c22, 1), 0.0)
            ave_c12 = np.where(cnt_c12 > 0, sum_c12 / np.maximum(cnt_c12, 1), 0.0)

            ave_c11.astype(OUT_DTYPE).tofile(fc11)
            ave_c22.astype(OUT_DTYPE).tofile(fc22)
            ave_c12.astype(OUT_DTYPE).tofile(fc12)

            done = start + nb
            el = time.time() - t0
            print(f"  rows {start:5d}-{done:5d}/{NROWS}  "
                  f"({100*done/NROWS:5.1f}%)  elapsed {el/60:.1f} min",
                  flush=True)

    print(f"DONE in {(time.time()-t0)/60:.1f} min", flush=True)
    print(f"Wrote:\n  {out_c11}\n  {out_c22}\n  {out_c12}", flush=True)
    print("Format: big-endian float32, "
          f"{NROWS} lines x {NCOLS} samples, 0 = no-data", flush=True)


if __name__ == "__main__":
    main()
