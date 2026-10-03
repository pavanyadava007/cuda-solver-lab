#!/usr/bin/env python3
"""Builds results/RESULTS.md, the PNG figures and the README results block
from the raw files written by bench.sh / profile.sh / sanitize.sh.

Every number in the README comes from this script; nothing is typed by hand.
"""
from __future__ import annotations

import csv
import re
from collections import defaultdict
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "results" / "raw"
FIG = ROOT / "results" / "figures"
NCU = ROOT / "results" / "ncu"
SPEC_BW = 300.0  # GB/s, L4 datasheet (192-bit GDDR6 at 6251 MHz: 24 B x 2 x 6.251 GHz)

# Reference palette (light mode) from the dataviz guide, fixed order.
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]

GPU_ORDER = ["csr_scalar", "csr_vector32", "csr_vector4", "stencil", "stencil_graph", "fused",
             "fused_graph", "fused_rows", "fused_rows_graph", "cusparse"]
LABEL = {
    "cpu_serial/csr": "CPU serial (CSR)",
    "cpu_openmp16/csr": "CPU OpenMP 16 thr (CSR)",
    "cpu_openmp32/csr": "CPU OpenMP 32 thr (CSR)",
    "fortran_openacc_gfortran/stencil_3loops": "Fortran OpenACC gfortran, GPU",
    "fortran_serial_gfortran/stencil_3loops": "Fortran serial gfortran, CPU",
    "fortran_openacc_nvfortran/stencil_3loops": "Fortran OpenACC nvfortran, GPU",
    "fortran_stdpar_nvfortran/stencil_3loops": "Fortran do concurrent nvfortran, GPU",
    "cuda/csr_scalar": "CUDA CSR scalar",
    "cuda/csr_vector32": "CUDA CSR vector (32 lanes/row)",
    "cuda/csr_vector4": "CUDA CSR vector (4 lanes/row)",
    "cuda/stencil": "CUDA matrix-free stencil",
    "cuda/stencil_graph": "CUDA stencil + CUDA Graph",
    "cuda/fused": "CUDA fused (2 kernels/iter)",
    "cuda/fused_graph": "CUDA fused + CUDA Graph",
    "cuda/fused_rows": "CUDA fused, row-strip SpMV (2 kernels/iter)",
    "cuda/fused_rows_graph": "CUDA fused row-strip + CUDA Graph",
    "cuda/cusparse": "cuSPARSE SpMV + cuBLAS",
}


FORTRAN_GPU = ["fortran_openacc_gfortran/stencil_3loops", "fortran_openacc_nvfortran/stencil_3loops",
               "fortran_stdpar_nvfortran/stencil_3loops"]
NSYS_FORTRAN = [("cg_acc_gpu", "gfortran OpenACC"), ("cg_acc_nvf", "nvfortran OpenACC"),
                ("cg_dc_nvf", "nvfortran do concurrent")]


def read_csv(path: Path) -> list[dict]:
    with path.open() as f:
        return list(csv.DictReader(f))


def key(r: dict) -> str:
    impl = r["impl"]
    if impl == "cpu_openmp":
        impl = f"cpu_openmp{r['threads']}"
    return f"{impl}/{r['variant']}"


def fmt(x: float, nd: int = 2) -> str:
    return f"{x:,.{nd}f}"


def md_table(header: list[str], rows: list[list[str]]) -> str:
    out = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    out += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(out)


def style(ax, title: str, xlabel: str, ylabel: str) -> None:
    ax.set_facecolor(SURFACE)
    ax.figure.set_facecolor(SURFACE)
    ax.set_title(title, color=INK, fontsize=11, loc="left")
    ax.set_xlabel(xlabel, color=INK2)
    ax.set_ylabel(ylabel, color=INK2)
    ax.tick_params(colors=INK2, labelsize=8)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


# --------------------------------------------------------------- ncu parsing --
UNIT = {"byte": 1, "Kbyte": 1e3, "Mbyte": 1e6, "Gbyte": 1e9, "usecond": 1e-6, "msecond": 1e-3,
        "nsecond": 1e-9, "ms": 1e-3, "us": 1e-6, "ns": 1e-9, "Gbyte/s": 1.0, "Mbyte/s": 1e-3,
        "Tbyte/s": 1e3}


def ncu_metrics(path: Path) -> dict:
    rows = list(csv.reader(path.open()))
    head, units, vals = rows[0], rows[1], rows[2]

    def get(name: str, scale_unit: bool = True) -> float:
        i = head.index(name)
        v = float(vals[i].replace(",", ""))
        return v * UNIT.get(units[i], 1.0) if scale_unit else v

    m = {
        "kernel": vals[head.index("Kernel Name")].split("(")[0],
        "time_s": get("gpu__time_duration.sum"),
        "dram_pct": get("gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed", False),
        "dram_gbs": get("dram__bytes.sum.per_second"),
        "dram_read": get("dram__bytes_read.sum"),
        "dram_write": get("dram__bytes_write.sum"),
        "occ_achieved": get("sm__warps_active.avg.pct_of_peak_sustained_active", False),
        "occ_theory": get("sm__maximum_warps_per_active_cycle_pct", False),
        "sectors": get("l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum", False),
        "requests": get("l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum", False),
        "l2_hit": get("lts__t_sector_hit_rate.pct", False),
        "regs": get("launch__registers_per_thread", False),
    }
    m["sect_per_req"] = m["sectors"] / m["requests"] if m["requests"] else 0.0
    return m


# ------------------------------------------------------------------- main ----
def main() -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    env = (RAW / "env.txt").read_text()
    date = re.search(r"date: (\d{4}-\d{2}-\d{2})", env).group(1)
    cuda = re.search(r"release ([\d.]+)", env).group(1)
    tag = f"measured on NVIDIA L4, CUDA {cuda}, {date}"

    # ---- peaks
    peaks = {r["probe"]: float(r["value"]) for r in read_csv(RAW / "peaks.csv")}
    bw_probes = ["copy_f64", "copy_f64x2", "triad_f64", "read_f64", "write_f64", "memcpy_d2d"]
    best_bw = max(peaks[p] for p in bw_probes)
    fp64 = peaks["fma_f64"]
    ridge = fp64 / best_bw

    sec = [f"_All numbers: {tag}. Generated by `scripts/make_report.py` from `results/raw/`._\n"]
    sec.append("### Hardware ceilings (bw_probe)\n")
    rows = [[p, fmt(peaks[p], 1) + " GB/s", fmt(100 * peaks[p] / SPEC_BW, 1) + " %"] for p in bw_probes]
    rows += [["fma_f64", fmt(fp64, 0) + " GFLOP/s", ""], ["fma_f32", fmt(peaks["fma_f32"], 0) + " GFLOP/s", ""]]
    sec.append(md_table(["probe", "best of 20", "% of 300 GB/s spec"], rows))
    sec.append(
        f"\nBest measured bandwidth: **{fmt(best_bw, 1)} GB/s** "
        f"({fmt(100 * best_bw / SPEC_BW, 1)} % of the 300 GB/s datasheet value; ECC is enabled on this GPU). "
        f"FP64 FMA peak: **{fmt(fp64, 0)} GFLOP/s** (FP32/FP64 = {peaks['fma_f32'] / fp64:.0f}). "
        f"FP64 ridge point = {fmt(fp64, 0)} / {fmt(best_bw, 1)} = **{ridge:.2f} flop/byte**.\n")

    # ---- CG fixed-iteration runs
    cg = read_csv(RAW / "cg.csv")
    fixed = [r for r in cg if r["mode"] == "fixed"]
    by = defaultdict(dict)  # key -> n -> row
    for r in fixed:
        by[key(r)][int(r["n"])] = r
    sizes = sorted({int(r["n"]) for r in fixed})
    order = ["cpu_serial/csr", "cpu_openmp16/csr", "cpu_openmp32/csr",
             ] + FORTRAN_GPU + [f"cuda/{v}" for v in GPU_ORDER]
    order = [k for k in order if k in by]

    sec.append("### CG: time per iteration (ms), fixed iteration count\n")
    rows = []
    for k in order:
        rows.append([LABEL.get(k, k)] + [fmt(float(by[k][n]["ms_per_iter"]), 3) if n in by[k] else "-" for n in sizes])
    sec.append(md_table(["implementation"] + [f"{n}^2" for n in sizes], rows))

    big = max(sizes)
    serial = float(by["cpu_serial/csr"][big]["ms_per_iter"])
    omp = min(float(by[k][big]["ms_per_iter"]) for k in ("cpu_openmp16/csr", "cpu_openmp32/csr") if k in by)
    lib = float(by["cuda/cusparse"][big]["ms_per_iter"])
    sec.append(f"\n### CG at {big}^2 ({int(by['cuda/fused'][big]['unknowns']):,} unknowns): bandwidth, roofline, speedups\n")
    rows = []
    for k in order:
        r = by[k].get(big)
        if not r:
            continue
        ms = float(r["ms_per_iter"])
        gbs = float(r["eff_gbs"])
        flops = 2 * float(r["nnz"]) + 10 * float(r["unknowns"])
        ai = flops / float(r["model_bytes_per_iter"])
        rows.append([LABEL.get(k, k), fmt(ms, 3), fmt(float(r["model_bytes_per_iter"]) / 1e6, 0),
                     fmt(gbs, 1), fmt(100 * gbs / SPEC_BW, 1) if k.startswith(("cuda", "fortran_openacc", "fortran_stdpar")) else "-",
                     fmt(float(r["gflops"]), 1), f"{ai:.3f}", fmt(serial / ms, 1) + "x",
                     fmt(lib / ms, 2) + "x"])
    sec.append(md_table(["implementation", "ms/iter", "model MB/iter", "eff. GB/s", "% of 300 GB/s",
                         "GFLOP/s", "flop/byte", "vs CPU serial", "vs cuSPARSE"], rows))
    gpu_keys = [f"cuda/{v}" for v in GPU_ORDER if f"cuda/{v}" in by and big in by[f"cuda/{v}"]]
    best_gpu_key = min(gpu_keys, key=lambda k: float(by[k][big]["ms_per_iter"]))
    best_gpu = float(by[best_gpu_key][big]["ms_per_iter"])
    ai_fused = (2 * float(by["cuda/fused"][big]["nnz"]) + 10 * float(by["cuda/fused"][big]["unknowns"])) / float(
        by["cuda/fused"][big]["model_bytes_per_iter"])
    sec.append(
        f"\nRoofline: the fused CG iteration does {ai_fused:.2f} flop/byte against a ridge point of {ridge:.2f} "
        f"flop/byte, so the attainable rate is bandwidth x intensity = {fmt(best_bw * ai_fused, 0)} GFLOP/s, "
        f"{fmt(100 * best_bw * ai_fused / fp64, 1)} % of FP64 peak: CG is memory bound by a factor of "
        f"{ridge / ai_fused:.1f}, and the only lever is bytes moved per iteration.\n\n"
        f"Speedups at {big}^2 (best GPU variant = {LABEL[best_gpu_key]}): "
        f"**{serial / best_gpu:.1f}x** vs CPU serial, **{omp / best_gpu:.1f}x** vs best CPU OpenMP, "
        f"**{lib / best_gpu:.2f}x** vs cuSPARSE + cuBLAS; CPU OpenMP vs serial: {serial / omp:.1f}x.\n")

    # Fortran GPU builds against each other and the C++ / CUDA versions
    fk = [k for k in FORTRAN_GPU if k in by and big in by[k]]
    if len(fk) > 1:
        gf = float(by[fk[0]][big]["ms_per_iter"])
        parts = []
        for k in fk[1:]:
            ms = float(by[k][big]["ms_per_iter"])
            parts.append(f"{LABEL[k]} {fmt(ms, 3)} ms/iter ({gf / ms:.2f}x faster than gfortran, "
                         f"{ms / best_gpu:.2f}x the time of the best CUDA variant, {omp / ms:.1f}x faster than "
                         f"best CPU OpenMP)")
        sec.append(f"Fortran at {big}^2: gfortran OpenACC {fmt(gf, 3)} ms/iter; " + "; ".join(parts) + ".\n")

    # small-size launch overhead
    small = min(sizes)
    st, stg = float(by["cuda/stencil"][small]["ms_per_iter"]), float(by["cuda/stencil_graph"][small]["ms_per_iter"])
    fu, fug = float(by["cuda/fused"][small]["ms_per_iter"]), float(by["cuda/fused_graph"][small]["ms_per_iter"])
    sec.append(
        f"Launch-bound regime ({small}^2, everything L2 resident): stencil {fmt(st * 1e3, 1)} us/iter -> "
        f"{fmt(stg * 1e3, 1)} us with a CUDA Graph ({st / stg:.2f}x); fused {fmt(fu * 1e3, 1)} -> {fmt(fug * 1e3, 1)} us "
        f"({fu / fug:.2f}x). At {big}^2 the graph changes stencil by "
        f"{float(by['cuda/stencil'][big]['ms_per_iter']) / float(by['cuda/stencil_graph'][big]['ms_per_iter']):.3f}x "
        f"(launch cost is hidden behind ms-long kernels).\n")

    # ---- time to solution
    tol = [r for r in cg if r["mode"] == "tol"]
    if tol:
        n_t = int(tol[0]["n"])
        ser_t = next(float(r["time_s"]) for r in tol if key(r) == "cpu_serial/csr")
        sec.append(f"### CG time to solution, {n_t}^2, relative residual 1e-8\n")
        rows = []
        for r in tol:
            rows.append([LABEL.get(key(r), key(r)), r["iters"], fmt(float(r["time_s"]), 3),
                         f"{float(r['rel_res_true']):.2e}", f"{float(r['max_err_analytic']):.3e}",
                         fmt(ser_t / float(r["time_s"]), 1) + "x"])
        sec.append(md_table(["implementation", "iterations", "seconds", "true rel. residual",
                             "max error vs analytic", "vs CPU serial"], rows))
        sec.append("\nGPU solves check the residual every iteration (graph variants every 10), which adds a "
                   "device-to-host copy per check; it is included in these times.\n")

    # ---- Jacobi
    jac = read_csv(RAW / "jacobi.csv")
    jsizes = sorted({int(r["n"]) for r in jac})
    jk = ["cpu", "naive", "smem_tile", "regs_stream"]
    jby = {(r["kernel"], int(r["n"])): r for r in jac}
    sec.append("### Jacobi sweep: effective bandwidth (GB/s, 24 B per point update)\n")
    rows = []
    for k in jk:
        name = {"cpu": "CPU OpenMP 16 threads", "naive": "CUDA naive (global loads)",
                "smem_tile": "CUDA shared-memory tile 32x8", "regs_stream": "CUDA register streaming (16 rows/thread)"}[k]
        rows.append([name] + [fmt(float(jby[(k, n)]["eff_gbs"]), 1) + f" ({fmt(float(jby[(k, n)]['ms_per_sweep']), 3)} ms)"
                              if (k, n) in jby else "-" for n in jsizes])
    sec.append(md_table(["kernel"] + [f"{n}^2" for n in jsizes], rows))

    # ---- ncu
    reps = sorted(NCU.glob("*.raw.csv"))
    if reps:
        sec.append(f"\n### Nsight Compute, one launch per kernel at {big}^2 (`ncu --set full`)\n")
        rows = []
        prof = {}
        for p in reps:
            m = ncu_metrics(p)
            prof[p.name.replace(".raw.csv", "")] = m
        want = ["spmv_csr_scalar", "spmv_csr_vector32", "spmv_csr_vector4", "cusparse_spmv", "spmv_stencil",
                "dot_finalize", "fused_p_spmv_dot", "fused_rows_p_spmv_dot", "fused_xr_dot"]
        for name in [w for w in want if w in prof]:
            m = prof[name]
            rows.append([name, fmt(m["time_s"] * 1e3, 3), fmt(m["dram_pct"], 1), fmt(m["dram_gbs"], 1),
                         fmt(m["dram_read"] / 1e6, 0), fmt(m["dram_write"] / 1e6, 0), fmt(m["occ_achieved"], 1),
                         fmt(m["occ_theory"], 0), f"{m['sect_per_req']:.2f}", fmt(m["l2_hit"], 1), f"{m['regs']:.0f}"])
        sec.append(md_table(["profile", "duration ms", "DRAM % of peak", "DRAM GB/s", "DRAM read MB",
                             "DRAM write MB", "achieved occ. %", "theor. occ. %", "sectors/request (global ld)",
                             "L2 hit %", "regs/thread"], rows))
        n_unk = float(big) * big
        if "fused_p_spmv_dot" in prof and "fused_rows_p_spmv_dot" in prof:
            a, b = prof["fused_p_spmv_dot"], prof["fused_rows_p_spmv_dot"]
            model = 32.0 * n_unk / 1e6
            sec.append(
                f"\nFirst fused kernel, model traffic {fmt(model, 0)} MB (read r, p_old; write p_new, Ap): the 1D "
                f"grid-stride version moves {fmt((a['dram_read'] + a['dram_write']) / 1e6, 0)} MB "
                f"(reads {a['dram_read'] / (16.0 * n_unk):.2f}x the model), the row-strip version "
                f"{fmt((b['dram_read'] + b['dram_write']) / 1e6, 0)} MB (reads {b['dram_read'] / (16.0 * n_unk):.2f}x); "
                f"kernel time {fmt(a['time_s'] * 1e3, 2)} -> {fmt(b['time_s'] * 1e3, 2)} ms under ncu.\n")
        sec.append("\nncu locks clocks to base during profiling, so durations differ slightly from the timed runs. "
                   "Sectors/request is averaged over all global loads of the kernel: a fully coalesced warp load "
                   "touches 8 sectors (32 B each) for 8-byte values and 4 for 4-byte values; higher means "
                   "scattered accesses.\n")
        perm = (NCU / "permission_error.txt").read_text().strip()
        if perm:
            sec.append(f"Unprivileged `ncu` on this host fails with:\n\n```\n{perm}\n```\n\n"
                       "(RmProfilingAdminOnly=1), so the profiles were taken with passwordless `sudo`.\n")

    # ---- nsys census
    ks = ROOT / "results" / "nsys" / "cusparse_n1024_cuda_gpu_kern_sum.csv"
    if ks.exists():
        kr = read_csv(ks)
        lib_rows = [r for r in kr if "dot_finalize" not in r["Name"]]
        total = sum(int(r["Instances"]) for r in lib_rows)
        sec.append("### Nsight Systems kernel census, cuSPARSE + cuBLAS path (n = 1024, 10 warm-up + 100 timed iterations)\n")
        rows = [[re.sub(r"<(?!unnamed>).*", "", r["Name"]).replace("void ", ""), r["Instances"],
                 f"{int(r['Instances']) / 110:.0f}"] for r in lib_rows]
        sec.append(md_table(["kernel", "instances", "per iteration"], rows))
        sec.append(f"\n**{total / 110:.0f} GPU kernels per CG iteration** for the library path vs 2 for the fused variant.\n")

    # ---- nsys census of the Fortran GPU builds
    fr = [(b, name) for b, name in NSYS_FORTRAN
          if (ROOT / "results" / "nsys" / f"{b}_n1024_cuda_gpu_kern_sum.csv").exists()]
    if fr:
        sec.append("### Nsight Systems, Fortran GPU builds (n = 1024, 100 fixed iterations)\n")
        rows = []
        for b, name in fr:
            kr = read_csv(ROOT / "results" / "nsys" / f"{b}_n1024_cuda_gpu_kern_sum.csv")
            ar = read_csv(ROOT / "results" / "nsys" / f"{b}_n1024_cuda_api_sum.csv")
            for r in kr:
                rows.append([name, r["Name"], f"{int(r['Instances']) / 100:.0f}",
                             fmt(float(r["Avg (ns)"]) / 1e3, 1)])
            api = {r["Name"]: int(r["Num Calls"]) for r in ar}
            ktot = sum(float(r["Total Time (ns)"]) for r in kr) / 1e6 / 100
            calls = ", ".join(f"{c} {api[c] / 100:.0f}" for c in ("cuMemAlloc_v2", "cuMemFree_v2", "cuStreamSynchronize")
                              if c in api)
            rows.append([name, f"**sum of kernel time per iteration: {fmt(ktot, 3)} ms**",
                         f"{sum(int(r['Instances']) for r in kr) / 100:.0f}", f"API calls/iter: {calls}"])
        sec.append(md_table(["build", "kernel", "per iteration", "avg us"], rows))
        sec.append("\nKernel names are compiler generated: gfortran numbers the offloaded regions "
                   "(`MAIN__$_omp_fn$N`), nvfortran names them by source line, with a `__red` kernel finishing "
                   "each reduction. `-Minfo` compiler feedback for the nvfortran builds is in `results/minfo/`.\n")

    # ---- sanitizer
    sz = ROOT / "results" / "sanitizer" / "summary.csv"
    if sz.exists():
        srows = read_csv(sz)
        clean = [r for r in srows if r["exit_code"] == "0"]
        dirty = [r for r in srows if r["exit_code"] != "0"]
        sec.append(f"### compute-sanitizer\n\n{len(clean)} of {len(srows)} runs clean "
                   "(memcheck, racecheck, synccheck, initcheck x every CG variant + Jacobi).")
        for r in dirty:
            sec.append(f" Not clean: `{r['binary']} {r['variant']} {r['tool']}`: {r['summary']}.")
        sec.append("\n")

    text = "\n".join(sec)
    (ROOT / "results" / "RESULTS.md").write_text("# Results\n\n" + text)
    readme = ROOT / "README.md"
    if readme.exists():
        s = readme.read_text()
        a, b = "<!-- RESULTS:BEGIN -->", "<!-- RESULTS:END -->"
        if a in s and b in s:
            s = s[: s.index(a) + len(a)] + "\n" + text + "\n" + s[s.index(b):]
            readme.write_text(s)

    # ------------------------------------------------------------- figures --
    # 1. ms/iter vs grid size
    fig, ax = plt.subplots(figsize=(8, 5))
    show = ["cpu_serial/csr", "cpu_openmp16/csr", "fortran_openacc_gfortran/stencil_3loops",
            "fortran_openacc_nvfortran/stencil_3loops", "cuda/csr_scalar",
            "cuda/cusparse", "cuda/stencil", "cuda/fused_rows"]
    for i, k in enumerate([k for k in show if k in by]):
        ns = sorted(by[k])
        ax.plot([n * n for n in ns], [float(by[k][n]["ms_per_iter"]) for n in ns], marker="o", markersize=5,
                linewidth=2, color=SERIES[i], label=LABEL[k])
    ax.set_xscale("log")
    ax.set_yscale("log")
    style(ax, f"CG time per iteration ({tag})", "unknowns (n^2)", "ms per iteration (log)")
    ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(FIG / "cg_ms_per_iter.png", dpi=150)
    plt.close(fig)

    # 2. effective bandwidth at the largest size
    fig, ax = plt.subplots(figsize=(8, 4.8))
    gk = [f"cuda/{v}" for v in GPU_ORDER if f"cuda/{v}" in by]
    vals = [float(by[k][big]["eff_gbs"]) for k in gk]
    ax.barh([LABEL[k] for k in gk], vals, color=SERIES[0], height=0.6)
    for y, v in enumerate(vals):
        ax.text(v + 3, y, fmt(v, 0), va="center", fontsize=8, color=INK)
    ax.axvline(SPEC_BW, color=INK2, linestyle="--", linewidth=1)
    ax.text(SPEC_BW, len(gk) - 0.4, " 300 GB/s spec", color=INK2, fontsize=8)
    ax.axvline(best_bw, color=SERIES[1], linestyle=":", linewidth=1.5)
    ax.text(best_bw - 2, len(gk) - 0.4, f"best probe {best_bw:.0f} ", color=INK2, fontsize=8, ha="right")
    ax.invert_yaxis()
    style(ax, f"CG effective bandwidth (model bytes / time), {big}^2", "GB/s", "")
    fig.tight_layout()
    fig.savefig(FIG / "cg_bandwidth.png", dpi=150)
    plt.close(fig)

    # 3. roofline
    fig, ax = plt.subplots(figsize=(7.5, 5))
    xx = np.logspace(-2, 2, 200)
    ax.plot(xx, np.minimum(fp64, best_bw * xx), color=INK2, linewidth=1.5, label=f"roof: {best_bw:.0f} GB/s, {fp64:.0f} GFLOP/s FP64")
    pts = [("cuda/csr_scalar", 0), ("cuda/cusparse", 1), ("cuda/stencil", 2), ("cuda/fused_rows", 3)]
    for k, ci in pts:
        r = by[k][big]
        flops = 2 * float(r["nnz"]) + 10 * float(r["unknowns"])
        ai = flops / float(r["model_bytes_per_iter"])
        ax.scatter([ai], [float(r["gflops"])], s=50, color=SERIES[ci], edgecolor=SURFACE, linewidth=2, zorder=3,
                   label=LABEL[k])
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(0.03, 30)
    ax.set_ylim(5, 1000)
    style(ax, f"FP64 roofline, CG at {big}^2", "arithmetic intensity (flop / model byte)", "GFLOP/s")
    ax.legend(fontsize=8, frameon=False, loc="upper left")
    fig.tight_layout()
    fig.savefig(FIG / "roofline.png", dpi=150)
    plt.close(fig)

    # 4. Jacobi
    fig, ax = plt.subplots(figsize=(8, 4.5))
    w = 0.2
    for i, k in enumerate(jk):
        vals = [float(jby[(k, n)]["eff_gbs"]) if (k, n) in jby else 0 for n in jsizes]
        ax.bar([j + (i - 1.5) * w for j in range(len(jsizes))], vals, width=w * 0.9, color=SERIES[i],
               label={"cpu": "CPU OpenMP 16 thr", "naive": "CUDA naive", "smem_tile": "CUDA smem tile",
                      "regs_stream": "CUDA register streaming"}[k])
    ax.set_xticks(range(len(jsizes)))
    ax.set_xticklabels([f"{n}^2" for n in jsizes])
    ax.axhline(best_bw, color=INK2, linestyle=":", linewidth=1)
    style(ax, f"Jacobi sweep effective bandwidth ({tag})", "grid", "GB/s (24 B / point)")
    ax.legend(fontsize=8, frameon=False, ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.15))
    fig.tight_layout()
    fig.savefig(FIG / "jacobi_bandwidth.png", dpi=150)
    plt.close(fig)
    print(text)


if __name__ == "__main__":
    main()
