# LogGP Cost Model for `ompi_coll_base_allreduce_intra_k_allreduce`

## 1. Model Parameters

### 1.1 The LogGP Model

The LogGP model [Alexandrov et al. 1997] extends the LogP model [Culler et al.
1993] with a per-byte gap parameter for long messages:

| Parameter | Meaning |
|-----------|---------|
| L | Latency: time for a single word to traverse the network (end-to-end) |
| o | Overhead: CPU time to initiate or complete a message (send or receive); the CPU cannot overlap this with computation |
| g | Gap: minimum interval between consecutive message sends (or receives) at a single processor |
| G | Gap per byte: additional per-byte transfer time for long messages (inverse of bandwidth) |
| P | Number of processors |

For a point-to-point transfer of a message of size \(s\) bytes, the time at the
receiver is:

\[
T_{\text{p2p}}(s) = 2o + L + (s - 1) \cdot G
\]

### 1.2 Simplified Notation

For collective algorithm analysis, we use the standard simplification:

| Symbol | Definition | Meaning |
|--------|------------|---------|
| \(\alpha\) | \(L + 2o\) | Per-message startup cost (latency + overhead) |
| \(\beta\) | \(G\) | Per-byte transfer cost (inverse bandwidth) |
| \(\gamma\) | — | Per-byte computation cost (reduction operation) |

A sendrecv (simultaneous bidirectional exchange) of \(s\) bytes costs:

\[
T_{\text{sendrecv}}(s) = \alpha + s \cdot \beta
\]

assuming full-duplex links.  A subsequent reduction on the received data adds
\(s \cdot \gamma\).

### 1.3 Two-Level Model (Hierarchical Networks)

For hierarchical systems (e.g., multi-node clusters), we distinguish:

| Symbol | Meaning |
|--------|---------|
| \(\alpha_1, \beta_1\) | Intra-group (e.g., intra-node) latency and bandwidth cost |
| \(\alpha_2, \beta_2\) | Inter-group (e.g., inter-node) latency and bandwidth cost |
| \(\gamma\) | Computation cost (same regardless of communication level) |

Typically \(\alpha_1 \ll \alpha_2\) and \(\beta_1 \ll \beta_2\) (shared-memory
or high-speed intra-node interconnect vs. network fabric).

### 1.4 Algorithm Parameters

| Symbol | Definition | Meaning |
|--------|------------|---------|
| \(N\) | `size` | Total number of MPI processes |
| \(k\) | `k` | Group size (number of columns in the 2D grid) |
| \(G\) | \(N/k\) | Number of groups (rows); requires \(k \mid N\) |
| \(m\) | `count × type_size` | Total message size in bytes |
| \(s\) | \(\approx m/k\) | Segment size per process (see note below) |

**Note on segment sizes**: When \(k \nmid\) count, the first `split_rank`
segments have `early_segcount` elements and the remaining have `late_segcount`.
For asymptotic analysis we use \(s = m/k\).

---

## 2. Cost Per Phase

### 2.1 Phase 1: Reduction Within Groups of k

#### Variant A — Recursive Doubling Allreduce (default)

Each of the \(\lceil\log_2 k\rceil\) steps exchanges the **full** buffer of
size \(m\) with a partner, then reduces:

\[
T_1^{(\text{RD})} = \lceil\log_2 k\rceil \cdot (\alpha + m \cdot \beta + m \cdot \gamma)
\]

For non-power-of-two \(k\), the implementation first reduces to
\(k' = 2^{\lfloor\log_2 k\rfloor}\) using one extra send/recv of size \(m\),
adding \(\alpha + m\beta + m\gamma\). Asymptotically:

\[
T_1^{(\text{RD})} = (\lceil\log_2 k\rceil + 1) \cdot \alpha + (\lceil\log_2 k\rceil + 1) \cdot m\beta + (\lceil\log_2 k\rceil + 1) \cdot m\gamma
\]

For power-of-two \(k\), the "+1" term vanishes.

**Result**: Every process in the group holds the complete group-reduced buffer.

#### Variant B — Reduce-Scatter via Recursive Halving

Each of the \(\log_2 k\) steps exchanges a **halving** portion of the buffer.
At step \(i\) (counting from 1), the exchange size is \(m / 2^i\):

| Step | Mask | Data exchanged | Reduction size |
|------|------|----------------|----------------|
| 1 | k/2 | m/2 | m/2 |
| 2 | k/4 | m/4 | m/4 |
| ... | ... | ... | ... |
| \(\log_2 k\) | 1 | m/k | m/k |

\[
T_1^{(\text{RS})} = \log_2 k \cdot \alpha
                   + \sum_{i=1}^{\log_2 k} \frac{m}{2^i} \cdot \beta
                   + \sum_{i=1}^{\log_2 k} \frac{m}{2^i} \cdot \gamma
\]

\[
= \log_2 k \cdot \alpha
  + m\left(1 - \frac{1}{k}\right) \beta
  + m\left(1 - \frac{1}{k}\right) \gamma
\]

**Result**: Process with local_rank \(p\) holds only segment \(p\) (of size
\(m/k\)), fully reduced within the group.

**Bandwidth saving over Variant A**: The factor \((1 - 1/k)\) replaces
\(\log_2 k\), a significant improvement for large \(k\).

---

### 2.2 Phase 2: Allreduce Across Groups on Segment \(p\)

Each process operates on its assigned segment of size \(s \approx m/k\).  The
\(G = N/k\) processes in the same column form the Phase 2 group.

#### Variant A — Recursive Doubling Allreduce (default)

\(\lceil\log_2 G\rceil\) steps, each exchanging \(s = m/k\) bytes:

\[
T_2^{(\text{RD})} = \lceil\log_2(N/k)\rceil \cdot \left(\alpha + \frac{m}{k}\beta + \frac{m}{k}\gamma\right)
\]

(with the same "+1" correction for non-power-of-two \(G\)).

#### Variant B — Ring Allreduce

The ring has \(G = N/k\) processes.  The segment \(s = m/k\) is further divided
into \(G\) sub-blocks of size \(\approx m/N\) each.

**Reduce-scatter ring** (\(G-1\) steps):
Each step sends and receives one sub-block of size \(m/N\), then reduces:

\[
T_{2,\text{rs}}^{(\text{Ring})} = (G-1) \cdot \alpha + (G-1) \cdot \frac{m}{N} \cdot \beta + (G-1) \cdot \frac{m}{N} \cdot \gamma
\]

**Allgather ring** (\(G-1\) steps):
Each step sends and receives one sub-block (no computation):

\[
T_{2,\text{ag}}^{(\text{Ring})} = (G-1) \cdot \alpha + (G-1) \cdot \frac{m}{N} \cdot \beta
\]

**Total Phase 2 ring**:

\[
T_2^{(\text{Ring})} = 2(G-1) \cdot \alpha + 2(G-1) \cdot \frac{m}{N} \cdot \beta + (G-1) \cdot \frac{m}{N} \cdot \gamma
\]

Substituting \(G = N/k\):

\[
T_2^{(\text{Ring})} = 2\left(\frac{N}{k}-1\right)\alpha + \frac{2m}{k}\left(1-\frac{k}{N}\right)\beta + \frac{m}{k}\left(1-\frac{k}{N}\right)\gamma
\]

---

### 2.3 Phase 3: Ring Allgather Within Groups of k

Each process starts with one reduced segment (\(m/k\) bytes).  The ring has
\(k\) processes; \(k-1\) steps, each forwarding one segment:

\[
T_3 = (k-1) \cdot \alpha + (k-1) \cdot \frac{m}{k} \cdot \beta
\]

\[
= (k-1) \cdot \alpha + m\left(1 - \frac{1}{k}\right)\beta
\]

---

## 3. Total Cost for Each Combination

Define shorthand: \(\ell_k = \log_2 k\), \(\ell_G = \log_2(N/k)\),
\(\ell_N = \log_2 N\).

### 3.1 Combination 1: Allreduce(RD) + Allreduce(RD) + Allgather(Ring)

\[
\boxed{T_1 = (\ell_k + \ell_G + k - 1)\,\alpha
       + \left(\ell_k + \frac{\ell_G + k - 1}{k}\right) m\beta
       + \left(\ell_k + \frac{\ell_G}{k}\right) m\gamma}
\]

Note: \(\ell_k + \ell_G = \ell_N\), so the latency term is \((\ell_N + k - 1)\,\alpha\).

### 3.2 Combination 2: Allreduce(RD) + Ring + Allgather(Ring)

\[
\boxed{T_2 = \left(\ell_k + \frac{2N}{k} + k - 3\right)\alpha
       + \left(\ell_k + \frac{2}{k}\left(1-\frac{k}{N}\right) + 1 - \frac{1}{k}\right) m\beta
       + \left(\ell_k + \frac{1}{k}\left(1-\frac{k}{N}\right)\right) m\gamma}
\]

### 3.3 Combination 3: ReduceScatter(RH) + Allreduce(RD) + Allgather(Ring)

\[
\boxed{T_3 = (\ell_N + k - 1)\,\alpha
       + \left(2\left(1-\frac{1}{k}\right) + \frac{\ell_G}{k}\right) m\beta
       + \left(1 - \frac{1}{k} + \frac{\ell_G}{k}\right) m\gamma}
\]

### 3.4 Combination 4: ReduceScatter(RH) + Ring + Allgather(Ring)

\[
\boxed{T_4 = \left(\ell_k + \frac{2N}{k} + k - 3\right)\alpha
       + 2\left(1 - \frac{1}{N}\right) m\beta
       + \left(1 - \frac{1}{N}\right) m\gamma}
\]

**Derivation of the bandwidth coefficient for Combination 4**:

\[
\underbrace{\left(1 - \frac{1}{k}\right)}_{\text{Phase 1}}
+ \underbrace{\frac{2}{k}\left(1 - \frac{k}{N}\right)}_{\text{Phase 2}}
+ \underbrace{\left(1 - \frac{1}{k}\right)}_{\text{Phase 3}}
= 2 - \frac{2}{k} + \frac{2}{k} - \frac{2}{N}
= 2\left(1 - \frac{1}{N}\right)
\]

This matches the theoretical lower bound for allreduce bandwidth.

---

## 4. Comparison With Standard Algorithms

### 4.1 Reference Costs

| Algorithm | Latency | Bandwidth | Computation |
|-----------|---------|-----------|-------------|
| Recursive Doubling | \(\ell_N \cdot \alpha\) | \(\ell_N \cdot m\beta\) | \(\ell_N \cdot m\gamma\) |
| Ring (RS+AG) | \(2(N-1)\,\alpha\) | \(2\frac{N-1}{N}\,m\beta\) | \(\frac{N-1}{N}\,m\gamma\) |
| Rabenseifner (RH+RD) | \(2\ell_N \cdot \alpha\) | \(2\frac{N-1}{N}\,m\beta\) | \(\frac{N-1}{N}\,m\gamma\) |

**Lower bounds** (information-theoretic):
- Bandwidth: \(2\frac{N-1}{N}\,m\beta\) (every element must be sent and
  received at least once in reduce-scatter and allgather phases)
- Computation: \(\frac{N-1}{N}\,m\gamma\) (every element reduced \(N-1\) times
  total, distributed across \(N\) processes)

### 4.2 Where Each Algorithm Excels

| Regime | Best algorithm | Why |
|--------|---------------|-----|
| Small \(m\), any \(N\) | Recursive Doubling | \(\ell_N\) latency terms, \(m\) coefficient is manageable |
| Large \(m\), moderate \(N\) | Ring or Rabenseifner | Bandwidth-optimal, \(O(N)\) vs \(O(\log N)\) latency |
| Large \(m\), large \(N\) | Rabenseifner | Bandwidth-optimal with only \(O(\log N)\) latency |
| Hierarchical network | **k_allreduce** | Minimizes inter-group communication volume |

### 4.3 The k_allreduce Sweet Spot

In a **homogeneous** network, Combination 4 achieves optimal bandwidth and
computation, with latency \((\ell_k + 2N/k + k - 3)\,\alpha\).  This latency
is always worse than Rabenseifner's \(2\ell_N \cdot \alpha\) for a single-level
network.

The algorithm's advantage emerges in **heterogeneous (two-level) networks**
where intra-group communication is significantly cheaper than inter-group.  See
Section 5.

---

## 5. Two-Level Network Analysis

### 5.1 Cost With Separate Intra/Inter Parameters

Phases 1 and 3 communicate within groups (intra-group, cost \(\alpha_1, \beta_1\)),
while Phase 2 communicates across groups (inter-group, cost \(\alpha_2, \beta_2\)).

#### Combination 3: ReduceScatter(RH) + Allreduce(RD) + Allgather(Ring)

\[
T_3 = \underbrace{(\ell_k + k - 1)\,\alpha_1 + 2\left(1-\frac{1}{k}\right) m\beta_1}_{\text{Intra-group (Phases 1+3)}}
    + \underbrace{\ell_G \cdot \alpha_2 + \frac{\ell_G}{k}\,m\beta_2}_{\text{Inter-group (Phase 2)}}
    + \left(1-\frac{1}{k} + \frac{\ell_G}{k}\right) m\gamma
\]

#### Combination 4: ReduceScatter(RH) + Ring + Allgather(Ring)

\[
T_4 = \underbrace{(\ell_k + k - 1)\,\alpha_1 + 2\left(1-\frac{1}{k}\right) m\beta_1}_{\text{Intra-group (Phases 1+3)}}
    + \underbrace{2\left(\frac{N}{k}-1\right) \alpha_2 + \frac{2}{k}\left(1-\frac{k}{N}\right) m\beta_2}_{\text{Inter-group (Phase 2)}}
    + \left(1-\frac{1}{N}\right) m\gamma
\]

### 5.2 Inter-Group Bandwidth Reduction

Compare the inter-group bandwidth term of k_allreduce with flat algorithms:

| Algorithm | Inter-group bandwidth cost | Factor |
|-----------|---------------------------|--------|
| Flat Ring / Rabenseifner | \(\approx 2m\beta_2\) (at least half the messages cross groups) | 1× |
| k_allreduce Combo 3 | \(\frac{\ell_G}{k}\,m\beta_2\) | \(\frac{\ell_G}{2k}\)× |
| k_allreduce Combo 4 | \(\frac{2}{k}\,m\beta_2\) (for \(N \gg k\)) | \(\frac{1}{k}\)× |

**Key insight**: The inter-group bandwidth is reduced by a factor of \(k\)
(Combo 4) or \(2k/\ell_G\) (Combo 3).  This is the fundamental advantage of
the 2D decomposition: by performing a full reduction within each group first,
each process only needs to send \(1/k\) of the data across the expensive
inter-group network.

### 5.3 Example: 256 Processes, 16 per Node

Parameters: \(N = 256\), \(k = 16\) (one group per node), \(G = 16\) nodes.
Typical two-level parameters: \(\alpha_2/\alpha_1 \approx 10\),
\(\beta_2/\beta_1 \approx 10\).

| Algorithm | Latency | Inter-node BW | Intra-node BW |
|-----------|---------|---------------|---------------|
| Flat Rabenseifner | \(16\,\alpha_{\text{mixed}}\) | \(\sim m\beta_2\) | \(\sim m\beta_1\) |
| k_allreduce Combo 4 | \(4\alpha_1 + 30\alpha_2 + 15\alpha_1\) | \(\frac{2}{16}\,m\beta_2 = 0.125\,m\beta_2\) | \(2 \cdot \frac{15}{16}\,m\beta_1\) |
| k_allreduce Combo 3 | \(19\alpha_1 + 4\alpha_2\) | \(\frac{4}{16}\,m\beta_2 = 0.25\,m\beta_2\) | \(2 \cdot \frac{15}{16}\,m\beta_1\) |

For Combo 3: inter-node bandwidth is reduced 4× compared to flat Rabenseifner,
with excellent inter-node latency (\(\ell_G = 4\) round trips).

For Combo 4: inter-node bandwidth is reduced **8×**, but inter-node latency is
much worse (\(2 \times 15 = 30\) round trips in the ring).

**Conclusion**: When \(\beta_2 \gg \beta_1\), the bandwidth savings dominate,
and k_allreduce significantly outperforms flat algorithms.

### 5.4 Multi-Rail Networks (Rail-Optimized NIC Assignment)

Modern GPU nodes such as the NVIDIA DGX feature multiple inter-node NICs — one
per GPU (e.g., 8 GPUs × 8 NICs on a DGX H100, each NIC providing 400 Gbps =
50 GB/s).  The aggregate inter-node bandwidth per node is therefore
\(k \times\) the per-NIC bandwidth.

#### Natural Rail Affinity of Phase 2

In Phase 2, each column group consists of processes that share the same
`local_rank = rank % k`.  On a properly configured system, GPU \(p\) is
associated with NIC \(p\).  Since Phase 2 communication only occurs between
processes with the same `local_rank`, each of the \(k\) independent column
streams naturally flows through a distinct NIC:

```
NIC 0 ← column 0 traffic (local_rank = 0)
NIC 1 ← column 1 traffic (local_rank = 1)
  ⋮
NIC 7 ← column 7 traffic (local_rank = 7)
```

This means:

- **Zero NIC contention**: Each column's data flows through its own dedicated
  NIC; no two columns share a NIC.
- **Full aggregate utilization**: All \(k\) NICs are active simultaneously
  during Phase 2, yielding an aggregate inter-node bandwidth of
  \(k \times \text{per-NIC BW}\).  For a DGX with 8 × 400GbE NICs, this is
  8 × 50 = **400 GB/s** aggregate.
- **No special configuration required**: The rail affinity follows directly
  from the 2D grid layout — it is an inherent structural property of the
  algorithm, not dependent on explicit process pinning or routing policies.

#### Per-Process Model Validity

The per-process cost model uses \(\beta_2 = 1/(\text{per-NIC bandwidth})\).
This is correct because each process exclusively uses its own NIC during
Phase 2.  The \(k\)-fold aggregate bandwidth is a consequence of \(k\)
independent, non-competing streams — it is already captured implicitly by
modeling per-process costs without a contention factor.

#### Comparison with Flat Algorithms

For flat algorithms (Recursive Doubling, Rabenseifner), rail-optimization
depends on the partner assignment.  With XOR-based partners at distance \(d\),
the partner's local rank is \((\text{rank} \oplus d) \bmod k\).  This equals
\(\text{rank} \bmod k\) only when \(d\) is a multiple of \(k\) — which happens
naturally when \(k\) is a power of two and the number of processes per node
equals \(k\).  However:

- **The k_allreduce guarantee is unconditional**: rail affinity holds for any
  valid \(k\), regardless of process layout or partner distances.
- **Flat algorithms are topology-fragile**: changing the number of processes per
  node, or using a non-power-of-two \(k\), can break the NIC alignment.

#### DGX H100 Example (N=128, k=8, 8 × 400GbE)

| Metric | k_allreduce (Combo 4) | Flat Rabenseifner |
|--------|----------------------|-------------------|
| Per-process inter-node data | \(\frac{2}{k}(1 - k/N) \cdot m \approx 0.23m\) | \(\frac{2(N-1)}{N} \cdot m \approx 1.97m\) (total, but see note) |
| NICs used simultaneously | 8 (guaranteed) | 8 (for this specific case) |
| Per-NIC load | \(0.23m\) | \(\sim 0.23m\) per step (varies by step) |
| Aggregate node BW utilized | \(8 \times 50 = 400\) GB/s | \(8 \times 50 = 400\) GB/s |

**Note**: In the flat Rabenseifner, each process sends \(\sim 1.97m\) bytes
inter-node through its own NIC — but all inter-node steps are sequential, so
the total time is proportional to the per-process volume.  In the k_allreduce,
each process sends only \(\sim 0.23m\) bytes inter-node — an **8.5× reduction**
in per-NIC traffic, which directly translates to faster inter-node phases.

---

## 6. Optimal Choice of k

### 6.1 Single-Level Network — Combination 4

Since the bandwidth and computation terms don't depend on \(k\), minimize
the latency:

\[
f(k) = \log_2 k + \frac{2N}{k} + k
\]

Taking the derivative and setting to zero:

\[
f'(k) = \frac{1}{k \ln 2} - \frac{2N}{k^2} + 1 = 0
\]

For large \(N\), the dominant balance is \(2N/k^2 \approx 1\), giving:

\[
k^* \approx \sqrt{2N}
\]

This yields \(f(k^*) \approx \frac{1}{2}\log_2(2N) + 2\sqrt{2N}\), which is
\(O(\sqrt{N})\) — worse than Rabenseifner's \(O(\log N)\) but far better than
the ring's \(O(N)\).

### 6.2 Single-Level Network — Combination 3

The latency doesn't depend on \(k\) beyond the \((k-1)\) term:
\(f(k) = \log_2 N + k - 1\). The bandwidth coefficient is
\(2(1-1/k) + \log_2(N/k)/k\).  Minimizing the total cost:

\[
\frac{\partial T_3}{\partial k} = \alpha + m\beta \cdot \frac{\partial}{\partial k}\left[\frac{2}{k} + \frac{\log_2(N/k)}{k}\right] \cdot (-1) \text{ terms} = 0
\]

The optimal \(k\) depends on the \(\alpha / (m\beta)\) ratio (message-size
dependent).

### 6.3 Two-Level Network

For the two-level model, set \(k\) to the number of processes per node
(or per NUMA domain).  This naturally aligns Phase 1 and Phase 3 with the fast
intra-node network and Phase 2 with the slower inter-node network.

This is the **primary intended use case** of the algorithm.  The optimal
\(k\) is dictated by the hardware topology, not by analytic optimization.

---

## 7. Theoretical Lower Bound Comparison

The allreduce lower bounds [Chan et al. 2007] are:

\[
T_{\text{allreduce}} \geq \lceil\log_2 N\rceil \cdot \alpha + 2\frac{N-1}{N}\,m\beta + \frac{N-1}{N}\,m\gamma
\]

No algorithm can simultaneously achieve all three optimal coefficients
(latency, bandwidth, computation).  Here is how each variant compares:

| | Latency-optimal? | BW-optimal? | Comp-optimal? |
|---|---|---|---|
| Recursive Doubling | Yes (\(\ell_N\)) | No (\(\ell_N\)) | No (\(\ell_N\)) |
| Ring | No (\(2N\)) | Yes | Yes |
| Rabenseifner | Near (\(2\ell_N\)) | Yes | Yes |
| k_allreduce Combo 3 | Near (\(\ell_N + k\)) | Near (\(2 + \ell_G/k\)) | Near (\(1 + \ell_G/k\)) |
| k_allreduce Combo 4 | No (\(\ell_k + 2N/k + k\)) | **Yes** | **Yes** |

**Combo 4 is bandwidth- and computation-optimal** for any \(k\), with tunable
latency.  Combo 3 provides a better latency/bandwidth trade-off for moderate
message sizes.

---

## 8. Summary of Trade-offs

```
                        Latency (α terms)
                 ┌─────────────────────────────────────┐
     log₂(N)    │  RecDouble          ← best latency  │
     2·log₂(N)  │  Rabenseifner                       │
     log₂(N)+k  │  k_allreduce C3     ← tunable       │
     √N         │  k_allreduce C4 (opt k)              │
     2N         │  Ring               ← worst latency  │
                 └─────────────────────────────────────┘

                        Bandwidth (m·β coefficient)
                 ┌─────────────────────────────────────┐
     2(1-1/N)   │  Ring, Rabenseifner, k_allreduce C4  │  ← optimal
     ~2         │  k_allreduce C3 (small log term)     │  ← near-optimal
     log₂(N)    │  RecDouble          ← worst BW       │
                 └─────────────────────────────────────┘
```

The k_allreduce algorithm provides a **parameterized trade-off** between
latency and bandwidth, controlled by \(k\).  Its primary advantage is in
two-level networks where it can exploit the topology to minimize the expensive
inter-node communication.

---

## References

- Alexandrov, A., Ionescu, M. F., Schauser, K. E., & Scheiman, C. (1997).
  LogGP: incorporating long messages into the LogP model. *J. Parallel and
  Distributed Computing*, 44(1), 71-79.
- Chan, E., Heimlich, M., Purkayastha, A., & van de Geijn, R. (2007).
  Collective communication: theory, practice, and experience. *Concurrency and
  Computation: Practice and Experience*, 19(13), 1749-1783.
- Culler, D. E., Karp, R. M., Patterson, D. A., et al. (1993). LogP: Towards
  a realistic model of parallel computation. *PPOPP '93*.
- Thakur, R., Rabenseifner, R., & Gropp, W. (2005). Optimization of collective
  communication operations in MPICH. *IJHPCA*, 19(1), 49-66.
