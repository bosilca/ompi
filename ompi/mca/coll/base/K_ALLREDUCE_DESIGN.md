# Design of `ompi_coll_base_allreduce_intra_k_allreduce`

## 2D Grid Allreduce: Combining Multiple Algorithms

This document captures the full design discussion and reasoning behind the
`ompi_coll_base_allreduce_intra_k_allreduce` algorithm in Open MPI.

---

## 1. High-Level Idea

Given N MPI processes and a tuning parameter k (the group size), arrange the
processes in a 2D grid of (N/k) rows × k columns:

```
             col 0    col 1    col 2    col 3      (k = 4)
  group 0:  rank 0   rank 1   rank 2   rank 3
  group 1:  rank 4   rank 5   rank 6   rank 7
  group 2:  rank 8   rank 9   rank 10  rank 11
```

- **group_id** = rank / k  (row index)
- **local_rank** = rank % k  (column index)

The allreduce proceeds in three phases:

| Phase | Scope | Operation | Communication pattern |
|-------|-------|-----------|-----------------------|
| 1 | Within each row (k processes) | Reduce (full buffer or scatter) | Recursive doubling or recursive halving |
| 2 | Within each column (N/k processes) | Allreduce on 1/k of the buffer | Recursive doubling or ring |
| 3 | Within each row (k processes) | Allgather of k segments | Ring |

### Why this works (correctness argument)

After Phase 1 (allreduce variant), every process in row g holds:
  `row_sum_g[0..count-1] = sum of all inputs in row g`

Phase 2 operates on segment `local_rank` across all rows.  Process (g, p)
computes: `global_sum[segment_p] = sum over all g of row_sum_g[segment_p]`.
Since each `row_sum_g` is the sum of k inputs, the result is the sum of all
N inputs for that segment.

Phase 3 allgathers the k segments within each row, giving every process the
complete globally-reduced buffer.

For the reduce-scatter Phase 1 variant, the reasoning is similar: each process
only computes its own segment's row-sum.  Phase 2 then reduces these partial
row-sums across groups, and Phase 3 allgathers as before.

---

## 2. Phase 1: Reduction Within Groups

### Variant A: Recursive Doubling Allreduce (default)

This is the standard recursive doubling algorithm adapted for a subgroup:

1. **Non-power-of-two handling**: If k is not a power of two, reduce it to the
   nearest lower power of two (adjsize) by pairing the first 2×extra_ranks
   processes.  Even-ranked processes send their full buffer to the odd neighbor;
   odd processes receive, reduce, and participate in the main loop with
   `newrank = local_rank / 2`.

2. **Main loop**: For distance = 1, 2, 4, ..., adjsize/2:
   - Partner = `newrank XOR distance`, mapped back to a global rank within the
     group.
   - Exchange full buffer via `sendrecv_actual`.
   - Reduce: if `rank < remote`, result = `tmpsend op tmprecv` (preserving
     left-to-right order for non-commutative operations).

3. **Non-power-of-two finish**: Odd ranks send their result back to the paired
   even ranks.

**Properties**:
- Preserves operation order → works for non-commutative operations
- Handles arbitrary k (power-of-two or not)
- Communication volume: O(count × log₂(k)) per process
- Every process ends up with the full reduced buffer

### Variant B: Reduce-Scatter via Recursive Halving

Instead of giving every process the full result, give each process only its 1/k
segment.  This saves bandwidth and feeds directly into Phase 2.

**Algorithm**: Recursive halving (mask = k/2, k/4, ..., 1):

At each step with mask m:
- Partner = `local_rank XOR m`
- Split the current working range of k-segments at the midpoint
- Lower-ranked process keeps the lower half, higher-ranked keeps the upper half
- Exchange the half you don't keep, receive into tmpbuf, reduce into rbuf

**Key design choice: mask order (halving vs doubling)**

Using mask = k/2 down to 1 (recursive halving) ensures that process `local_rank`
naturally ends up with k-segment `local_rank` at the correct offset.  This was
verified by tracing through examples:

```
k=4, count=400, local_rank=1:
  Step 0 (mask=2): partner=3, keep lower half [0,200)
  Step 1 (mask=1): partner=0, keep upper half [100,200)
  Result: segment 1 at offset 100 ✓
```

If we had used mask = 1 up to k/2 (recursive doubling), the segments would end
up in **bit-reversal order** — process 1 (binary 01) would get segment 2
(binary 10).  This is because at each step, the bit of the mask determines which
half to keep, and with increasing masks the bits are assigned MSB-first instead
of LSB-first.

**Handling uneven segment sizes**:

When count is not divisible by k, the segments have uneven sizes (early_segcount
vs late_segcount, per `COLL_BASE_COMPUTE_BLOCKCOUNT`).  The halving splits along
**segment boundaries** rather than raw element midpoints:

```c
int nmid = (block_lo + block_hi) / 2;
ptrdiff_t lo_off  = BLOCK_OFFSET(block_lo) * extent;
ptrdiff_t mid_off = BLOCK_OFFSET(nmid) * extent;
ptrdiff_t hi_off  = BLOCK_OFFSET(block_hi) * extent;
```

This ensures the element counts match the segment layout expected by Phase 2.
Verified with count=401, k=4 (split_rank=1, early=101, late=100):

```
Process 0: keeps [0, 101)  → segment 0 with 101 elements ✓
Process 1: keeps [101, 201) → segment 1 with 100 elements ✓
```

**Constraints**:
- Requires k to be a power of two (XOR pairing covers all processes exactly)
- Requires commutative operation (the Rabenseifner-style halving does not
  preserve left-to-right reduction order)
- Communication volume: O(count) total per process (count/2 + count/4 + ... ≈ count)

---

## 3. Phase 2: Allreduce Across Groups

Each process handles only its assigned 1/k segment of the buffer.  Processes
with the same `local_rank` form a **column** of `num_groups` processes.

### Variant A: Recursive Doubling (default)

Same recursive doubling pattern as Phase 1's allreduce, but:
- Operates on only `seg_count` elements (1/k of the buffer)
- Partners are in the same column: `remote = remote_group * k + local_rank`
- Handles non-power-of-two `num_groups` via the standard even/odd pairing

**Properties**:
- Preserves operation order → works for non-commutative operations
- Communication volume: O(seg_count × log₂(num_groups)) per process

### Variant B: Ring Allreduce

The ring has `num_groups` processes.  The segment is further divided into
`num_groups` sub-blocks.  Two phases:

1. **Reduce-scatter** (num_groups − 1 steps):
   At each step, send one sub-block to the right column neighbor, receive from
   the left, reduce into the local buffer.

   ```
   col_send_to   = ((group_id + 1) % num_groups) * k + local_rank
   col_recv_from = ((group_id - 1 + num_groups) % num_groups) * k + local_rank
   ```

   After completion, each process has one fully-reduced sub-block.

2. **Allgather** (num_groups − 1 steps):
   Rotate the fully-reduced sub-blocks around the ring so every column member
   has the complete segment.

**Sub-block offset computation**: Uses `COLL_BASE_COMPUTE_BLOCKCOUNT` to handle
uneven sub-block sizes:

```c
COLL_BASE_COMPUTE_BLOCKCOUNT(seg_count, num_groups, sub_split, sub_early, sub_late);
```

**Constraints**:
- Requires commutative operation (ring does not preserve reduction order)
- Requires `count >= size` so each sub-block has at least one element
  (seg_count ≈ count/k ≥ num_groups = size/k → count ≥ size)
- Communication volume: O(seg_count) per process (bandwidth-optimal)

---

## 4. Phase 3: Allgather Within Groups

Standard ring allgather among k processes in each row.  Each process starts with
its own reduced segment and sends/receives segments around the ring.

```c
sendto   = group_start + ((local_rank + 1) % k)
recvfrom = group_start + ((local_rank + k - 1) % k)
```

For step s (0 ≤ s < k-1):
- send_block = `(local_rank - s + k) % k`
- recv_block = `(local_rank - s - 1 + k) % k`

At each step, the process sends the most recently acquired block and receives the
next one.  After k-1 steps, every process has all k segments.

**Communication volume**: O(count) total per process (count/k per step × (k-1) steps).

---

## 5. Tag Safety Analysis

All three phases use `MCA_COLL_BASE_TAG_ALLREDUCE`.  This is safe because:

1. **Phase 1 ↔ Phase 2**: Group partners (rows) and column partners never
   overlap (a process can't be in the same row AND same column as another
   unless they are the same process, since group = rank/k and column = rank%k).

2. **Phase 2 ↔ Phase 3**: Column communication (Phase 2) and row communication
   (Phase 3) involve disjoint partner sets, so there is no source-rank ambiguity.

3. **Phase 1 ↔ Phase 3**: Same group members communicate in both phases, but
   Phase 2 intervenes.  Since Phase 2 uses blocking operations and involves all
   column members, a process cannot enter Phase 3 until Phase 2 is complete.
   Any early Phase 3 message will be buffered by MPI and matched correctly when
   the receiver enters Phase 3.

4. **Cross-call safety**: MPI collectives are ordered on a communicator, so
   messages from different collective invocations cannot be confused.

---

## 6. Compile-Time Configuration

Two independent `#ifdef` macros control algorithm selection:

| Macro | Phase | Default (undefined) | When defined |
|-------|-------|--------------------|----|
| `K_ALLREDUCE_REDSCAT_PHASE1` | 1 | Recursive doubling allreduce | Reduce-scatter via recursive halving |
| `K_ALLREDUCE_RING_PHASE2` | 2 | Recursive doubling allreduce | Ring allreduce |

All four combinations are valid.  Each ifdef adds its own runtime constraint
checks and falls back to plain `ompi_coll_base_allreduce_intra_recursivedoubling`
when constraints are not met:

```c
/* Always required */
if (k <= 1 || k > size || (size % k) != 0 || count < (size_t)k)
    → fallback

/* Only when K_ALLREDUCE_REDSCAT_PHASE1 is defined */
if ((k & (k - 1)) != 0 || !ompi_op_is_commute(op))
    → fallback

/* Only when K_ALLREDUCE_RING_PHASE2 is defined */
if (count < (size_t)size || !ompi_op_is_commute(op))
    → fallback
```

---

## 7. Integration Into the Tuned Component

The algorithm is registered as algorithm 8 (`k_allreduce`) in
`coll_tuned_allreduce_decision.c`.  The `k` parameter is passed via the
existing `segsize` MCA parameter:

```bash
mpirun --mca coll_tuned_use_dynamic_rules 1 \
       --mca coll_tuned_allreduce_algorithm 8 \
       --mca coll_tuned_allreduce_algorithm_segmentsize <k> \
       ./my_application
```

---

## 8. Complexity Summary

For N processes, group size k, message size M bytes, num_groups G = N/k:

| Configuration | Phase 1 | Phase 2 | Phase 3 | Total |
|---|---|---|---|---|
| Allreduce + RecDouble | M·log(k) | (M/k)·log(G) | M | M·(log(k) + log(G)/k + 1) |
| Allreduce + Ring | M·log(k) | 2·M/k | M | M·(log(k) + 2/k + 1) |
| RedScat + RecDouble | M | (M/k)·log(G) | M | M·(1 + log(G)/k + 1) |
| RedScat + Ring | M | 2·M/k | M | M·(1 + 2/k + 1) |

The reduce-scatter Phase 1 saves a factor of log(k) compared to the allreduce
Phase 1.  The ring Phase 2 is bandwidth-optimal for large messages with many
groups.

---

## 9. Files Modified

| File | Change |
|------|--------|
| `ompi/mca/coll/base/coll_base_allreduce.c` | New function implementation (~300 lines) |
| `ompi/mca/coll/base/coll_base_functions.h` | Declaration: `int ompi_coll_base_allreduce_intra_k_allreduce(ALLREDUCE_ARGS, int k);` |
| `ompi/mca/coll/tuned/coll_tuned_allreduce_decision.c` | Algorithm 8 registration and dispatch |

---

## 10. Open Design Questions and Future Work

1. **Non-power-of-two k for reduce-scatter**: The current reduce-scatter Phase 1
   requires k to be a power of two.  Supporting arbitrary k would need either
   a preliminary reduction step (Rabenseifner-style) where adjsize-segments
   don't align with k-segments, or a completely different reduce-scatter
   algorithm (e.g., ring-based).

2. **Runtime algorithm selection**: Currently the Phase 1/Phase 2 variants are
   selected at compile time via `#ifdef`.  A runtime MCA parameter could allow
   switching between them based on message size, process count, and
   commutativity.

3. **Non-blocking variant**: An `iallreduce` version using non-blocking
   point-to-point (NBC) would allow overlap with computation.

4. **Topology awareness**: The parameter k could be automatically set based on
   hardware topology (e.g., k = processes per node) to align Phase 1 with
   intra-node communication and Phase 2 with inter-node communication.
