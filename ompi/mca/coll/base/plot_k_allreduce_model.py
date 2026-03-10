#!/usr/bin/env python3
"""
LogGP cost model visualization for the k_allreduce algorithm.

Two panels comparing GPU vs CPU reduction execution:
  - GPU:  γ = 1e-12 s/byte (1 TB/s) + 2 µs kernel launch per reduction call
  - CPU:  γ = 5e-11 s/byte (20 GB/s, AVX-512, memory-BW limited), no launch overhead
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

# ── Hardware configuration ──────────────────────────────────────────
N = 128
k = 8
G = N // k  # 16 nodes

# ── LogGP network parameters (same for both panels) ────────────────
alpha_1 = 2e-6          # NVLink latency  2 µs
beta_1  = 1.0 / 200e9   # NVLink per-byte 200 GB/s
alpha_2 = 5e-6          # 400GbE latency  5 µs
beta_2  = 1.0 / 50e9    # 400GbE per-byte 50 GB/s

log_k = np.log2(k)     # 3
log_G = np.log2(G)     # 4

# ── Message sizes: 16 KB to 4 MB ───────────────────────────────────
m = np.logspace(np.log10(16 * 1024), np.log10(4 * 1024 * 1024), 1000)

# ── Kernel launch counts (number of ompi_op_reduce calls) ──────────
# Phase 3 (allgather) never reduces.
# Phase 1 RD:  log₂(k) reductions
# Phase 1 RS:  log₂(k) reductions
# Phase 2 RD:  log₂(G) reductions
# Phase 2 Ring: G-1 reductions (reduce-scatter only)
launches_C1 = log_k + log_G           # RD + RD    = 7
launches_C2 = log_k + (G - 1)         # RD + Ring  = 18
launches_C3 = log_k + log_G           # RS + RD    = 7
launches_C4 = log_k + (G - 1)         # RS + Ring  = 18
launches_rd  = log_k + log_G          # flat RD    = 7
launches_rab = log_k + log_G          # flat Rab (RS phase only) = 7

# Pre-compute bandwidth coefficients from the Rabenseifner flat model
n_intra_rd = log_k
n_inter_rd = log_G
rs_inter_bw = sum(1.0 / 2**i for i in range(1, int(log_G) + 1))
rs_intra_bw = sum(1.0 / 2**i for i in range(int(log_G) + 1, int(log_k + log_G) + 1))


def compute_times(gamma, t_launch):
    """Compute wall-clock time arrays for all 6 algorithms.

    gamma    : per-byte reduction cost (s/byte)
    t_launch : fixed overhead per reduction call (s), e.g. GPU kernel launch
    """
    # k_allreduce C1: Allreduce(RD) + Allreduce(RD) + Allgather(Ring)
    T1 = ((log_k + k - 1) * alpha_1 + log_G * alpha_2
        + launches_C1 * t_launch
        + (log_k + (k - 1.0) / k) * m * beta_1 + (log_G / k) * m * beta_2
        + (log_k + log_G / k) * m * gamma)

    # k_allreduce C2: Allreduce(RD) + Ring + Allgather(Ring)
    T2 = ((log_k + k - 1) * alpha_1 + 2 * (G - 1) * alpha_2
        + launches_C2 * t_launch
        + (log_k + (k - 1.0) / k) * m * beta_1 + 2 * (1.0 / k - 1.0 / N) * m * beta_2
        + (log_k + 1.0 / k - 1.0 / N) * m * gamma)

    # k_allreduce C3: ReduceScatter(RH) + Allreduce(RD) + Allgather(Ring)
    T3 = ((log_k + k - 1) * alpha_1 + log_G * alpha_2
        + launches_C3 * t_launch
        + 2 * (1 - 1.0 / k) * m * beta_1 + (log_G / k) * m * beta_2
        + (1 - 1.0 / k + log_G / k) * m * gamma)

    # k_allreduce C4: ReduceScatter(RH) + Ring + Allgather(Ring)
    T4 = ((log_k + k - 1) * alpha_1 + 2 * (G - 1) * alpha_2
        + launches_C4 * t_launch
        + 2 * (1 - 1.0 / k) * m * beta_1 + 2 * (1.0 / k - 1.0 / N) * m * beta_2
        + (1 - 1.0 / N) * m * gamma)

    # Flat Recursive Doubling
    T_rd = (n_intra_rd * (alpha_1 + m * beta_1 + m * gamma)
          + n_inter_rd * (alpha_2 + m * beta_2 + m * gamma)
          + launches_rd * t_launch)

    # Flat Rabenseifner
    T_rab = (2 * (n_intra_rd * alpha_1 + n_inter_rd * alpha_2)
           + launches_rab * t_launch
           + 2 * (rs_intra_bw * m * beta_1 + rs_inter_bw * m * beta_2)
           + (rs_intra_bw + rs_inter_bw) * m * gamma)

    return T1, T2, T3, T4, T_rd, T_rab


def plot_panel(ax, gamma, t_launch, title_extra, palette):
    """Plot one panel."""
    T1, T2, T3, T4, T_rd, T_rab = compute_times(gamma, t_launch)

    BW1   = m / T1 / 1e9
    BW2   = m / T2 / 1e9
    BW3   = m / T3 / 1e9
    BW4   = m / T4 / 1e9
    BW_rd  = m / T_rd / 1e9
    BW_rab = m / T_rab / 1e9

    ax.semilogx(m, BW_rd,  color=palette[4], linewidth=1.4, linestyle='--',
                label='Recursive Doubling (flat)')
    ax.semilogx(m, BW_rab, color=palette[5], linewidth=1.4, linestyle='--',
                label='Rabenseifner (flat)')
    ax.semilogx(m, BW1, color=palette[0], linewidth=2.0,
                label='C1: RD + RD + Allgather')
    ax.semilogx(m, BW2, color=palette[1], linewidth=2.0,
                label='C2: RD + Ring + Allgather')
    ax.semilogx(m, BW3, color=palette[2], linewidth=2.0,
                label='C3: RS + RD + Allgather')
    ax.semilogx(m, BW4, color=palette[3], linewidth=2.0,
                label='C4: RS + Ring + Allgather')

    ax.axhline(y=50, color='gray', linestyle=':', alpha=0.5, linewidth=1)

    ax.set_xlabel('Message Size')
    ax.set_title(title_extra, fontsize=12)
    ax.set_xlim(16 * 1024, 4 * 1024**2)
    ax.set_ylim(0, 50)

    sizes  = [16384, 65536, 262144, 1048576, 4*1048576]
    labels = ['16 KB', '64 KB', '256 KB', '1 MB', '4 MB']
    ax.set_xticks(sizes)
    ax.set_xticklabels(labels, rotation=30, ha='right', fontsize=9)
    ax.xaxis.set_minor_locator(mticker.NullLocator())

    # Print asymptotic BWs
    print(f'\n  {title_extra}')
    for name, bw in [('C1: RD+RD+AG', BW1[-1]), ('C2: RD+Ring+AG', BW2[-1]),
                     ('C3: RS+RD+AG', BW3[-1]), ('C4: RS+Ring+AG', BW4[-1]),
                     ('Flat RecDouble', BW_rd[-1]), ('Flat Rabenseifner', BW_rab[-1])]:
        print(f'    {name:24s} {bw:7.1f} GB/s')


# ── Build figure with two panels ───────────────────────────────────
sns.set_theme(style="whitegrid", context="talk", font_scale=0.85)
palette = sns.color_palette("deep", 6)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7.5), sharey=True)

fig.suptitle(
    f'k_allreduce LogGP Model — DGX-like System\n'
    f'N={N}, k={k} (NVLink 200 GB/s), {G} nodes × {k}×400GbE '
    f'(50 GB/s/NIC, {k*50} GB/s aggregate)',
    fontsize=13, y=0.98)

# Left panel: GPU reduction
gamma_gpu = 1e-12      # 1 TB/s GPU reduction throughput
t_launch  = 2e-6       # 2 µs CUDA kernel launch per reduction
plot_panel(ax1, gamma_gpu, t_launch,
           'GPU reduction (γ = 1 TB/s, +2 µs kernel launch)',
           palette)
ax1.set_ylabel('Effective Allreduce Bandwidth (GB/s)')
ax1.legend(fontsize=8.5, loc='upper left', framealpha=0.9)
ax1.text(18000, 51, 'per-NIC peak (50 GB/s)', fontsize=8, color='gray')

# Right panel: CPU reduction with AVX-512
gamma_cpu = 1.0 / 20e9  # 20 GB/s (AVX-512, memory-BW limited)
t_launch_cpu = 0         # no kernel launch overhead
plot_panel(ax2, gamma_cpu, t_launch_cpu,
           'CPU reduction (γ = 20 GB/s, AVX-512)',
           palette)
ax2.text(18000, 51, 'per-NIC peak (50 GB/s)', fontsize=8, color='gray')

plt.tight_layout(rect=[0, 0, 1, 0.93])
out = 'k_allreduce_bandwidth_model.png'
plt.savefig(out, dpi=150)
print(f'\nSaved → {out}')
