# Iterative Design Discussion: `ompi_coll_base_allreduce_intra_k_allreduce`

This document records the step-by-step evolution of the algorithm through an
interactive design session.  It is intended as an educational companion to
`K_ALLREDUCE_DESIGN.md`, showing *how* design decisions were made and *why*
certain alternatives were explored and either adopted or rejected.

---

## Step 1: Initial Request — Three-Phase 2D Allreduce

### Request

> I would like to write a new implementation of allreduce, by combining
> multiple algorithms.  First, an allreduce on a smaller number of processes,
> k, using a recursive doubling algorithm, then k allreduce each one on a
> piece of the buffer, followed by an allgather on the same k processes.

### Interpretation and Clarification

The phrase "k allreduce each one on a piece of the buffer" was interpreted as:
each of the k positions (columns) runs an allreduce on its own 1/k segment of
the buffer, across all groups (rows).  This creates a 2D grid decomposition.

**Key design decisions at this stage:**

1. **Grid layout**: Processes are numbered sequentially, so rank r goes to row
   r/k, column r%k.  This keeps processes in the same row contiguous.

2. **Segment assignment**: The buffer is split into k segments using the
   standard Open MPI `COLL_BASE_COMPUTE_BLOCKCOUNT` macro, which handles
   count%k != 0 by assigning `count/k + 1` elements to the first
   `count%k` segments and `count/k` to the rest.

3. **Phase 1 algorithm**: Recursive doubling allreduce within each row of k
   processes.  Non-power-of-two k is handled by the standard technique of
   reducing to the nearest lower power of two.

4. **Phase 2 algorithm**: Recursive doubling allreduce within each column of
   N/k processes, operating on only the local_rank-th segment.

5. **Phase 3 algorithm**: Ring allgather within each row, distributing all k
   segments so every process has the complete result.

6. **MPI_IN_PLACE**: The function immediately copies sbuf to rbuf (or does
   nothing if sbuf == MPI_IN_PLACE), then works entirely in-place on rbuf.

7. **Temporary buffer**: A single tmpbuf of seg_count elements is allocated
   for use as a receive buffer during send/recv+reduce steps.

8. **Fallback**: If k <= 1, k > size, size%k != 0, or count < k, the function
   falls back to `ompi_coll_base_allreduce_intra_recursivedoubling`.

### Implementation

The function `ompi_coll_base_allreduce_intra_k_allreduce` was added to
`coll_base_allreduce.c`, declared in `coll_base_functions.h`, and registered
as algorithm 8 in the tuned component's decision file.

---

## Step 2: Replace Phase 2 With a Ring Algorithm

### Request

> Replace the algorithm in the second step with a ring algorithm across groups.

### Rationale

The ring algorithm is bandwidth-optimal for large messages.  Since Phase 2
operates on 1/k of the buffer, and there are N/k processes in each column, the
ring can be significantly more efficient than recursive doubling when N/k is
large and the segment is large enough to be further sub-divided.

### Design Challenges

**Sub-block computation for the ring**: The ring needs to divide the segment
into num_groups sub-blocks.  This reuses `COLL_BASE_COMPUTE_BLOCKCOUNT`:

```c
COLL_BASE_COMPUTE_BLOCKCOUNT(seg_count, num_groups, sub_split, sub_early, sub_late);
```

**Ring topology in a column**: Column members are not contiguous ranks.  The
"left" and "right" neighbors in the ring must be computed using modular
arithmetic on group_id, then mapped back to global ranks:

```c
col_send_to   = ((group_id + 1) % num_groups) * k + local_rank;
col_recv_from = ((group_id - 1 + num_groups) % num_groups) * k + local_rank;
```

**Block index tracking**: At each step of the ring, the process sends a
different sub-block.  The send_block_idx and recv_block_idx rotate:

```c
send_block_idx = (group_id - step + num_groups) % num_groups;
recv_block_idx = (group_id - step - 1 + num_groups) % num_groups;
```

### New Constraints

The ring algorithm introduced two new constraints:

1. **Commutativity required**: The ring reduce-scatter does not guarantee
   operands arrive in rank order.  For non-commutative operations, the
   reduction result would be incorrect.

2. **count >= size**: The ring divides seg_count (≈ count/k) into num_groups
   (= N/k) sub-blocks.  For each sub-block to have at least one element:
   `count/k >= N/k → count >= N = size`.

---

## Step 3: Keep Both Phase 2 Variants Under `#ifdef`

### Request

> Keep both implementations for the Phase 2, the original recursive doubling
> and the new ring, separated by an ifdef.

### Design Decision: Compile-Time vs Runtime

Using `#ifdef K_ALLREDUCE_RING_PHASE2` was chosen because:

- It has zero runtime overhead when the variant is not enabled.
- It keeps the code self-contained: each variant is a complete, readable block.
- A runtime MCA parameter could be added later as a higher-level switch that
  selects between different compiled variants (or between this function and
  other allreduce algorithms entirely).

### Constraint Placement

The ring-specific constraint checks were placed **after** the general checks,
guarded by the same `#ifdef`:

```c
#ifdef K_ALLREDUCE_RING_PHASE2
    if (count < (size_t)size || !ompi_op_is_commute(op))
        return ompi_coll_base_allreduce_intra_recursivedoubling(...);
#endif
```

This means the default (recursive doubling) variant has no commutativity
requirement and works with any count >= k, while the ring variant adds
stricter requirements.  If the stricter requirements are not met at runtime,
the entire function falls back to plain recursive doubling allreduce rather
than silently using the wrong algorithm.

---

## Step 4: Replace Phase 1 With Reduce-Scatter

### Request

> I want to replace the first phase with a reduce-scatter instead of an
> allreduce.  Use the same recursive doubling algorithm.  Keep both versions
> protected by an ifdef.

### Rationale

With the allreduce Phase 1, every process in the group ends up with the **full**
reduced buffer.  But Phase 2 only needs each process to have its own 1/k
segment.  A reduce-scatter saves bandwidth: O(count) total instead of
O(count × log(k)).

### Algorithm Design: Recursive Halving

The reduce-scatter uses "recursive halving" (sometimes called "vector halving /
distance halving"):

```
for mask = k/2, k/4, ..., 1:
    partner = local_rank XOR mask
    exchange half of current working range
    keep the half corresponding to your rank's bit at this mask position
    reduce received data into kept half
```

### Key Discussion: Why Mask Order Matters

**Question**: Should the mask iterate k/2→1 (halving) or 1→k/2 (doubling)?

**Answer**: It must be k/2→1.  Here's why:

Consider k=4, local_rank=1 (binary: 01):

With **halving** (mask = 2, then 1):
- mask=2 (bit 1): partner = 01 XOR 10 = 11 = rank 3.  Bit 1 of local_rank
  is 0 → keep lower half [segments 0,1].
- mask=1 (bit 0): partner = 01 XOR 01 = 00 = rank 0.  Bit 0 of local_rank
  is 1 → keep upper half [segment 1].
- **Result**: Process 1 gets segment 1. ✓

With **doubling** (mask = 1, then 2):
- mask=1 (bit 0): partner = 01 XOR 01 = 00 = rank 0.  Bit 0 is 1 → keep
  upper half [segments 2,3].
- mask=2 (bit 1): partner = 01 XOR 10 = 11 = rank 3.  Bit 1 is 0 → keep
  lower half [segment 2].
- **Result**: Process 1 gets segment 2. ✗ (bit-reversed order!)

The halving order processes bits MSB-first, assigning the most-significant
portion of the space at the first step.  The doubling order processes bits
LSB-first, resulting in bit-reversal permutation.

### Key Discussion: Uneven Segment Boundaries

**Problem**: When count is not divisible by k, segments have different sizes.
Naively halving the element count at each step would misalign with segment
boundaries.

**Example**: count=401, k=4.  Segments: [101, 100, 100, 100].

If we split the 401 elements at 200/201 boundaries, the split point falls in
the middle of segment 1 (which spans elements 101-200).  This would corrupt
the reduction.

**Solution**: Track halving in terms of **segment indices** rather than element
counts:

```c
int block_lo = 0, block_hi = k;       /* k-segment index range */
int nmid = (block_lo + block_hi) / 2;  /* midpoint segment index */

/* Convert segment indices to element offsets using the segment layout */
ptrdiff_t lo_off  = BLOCK_OFFSET(block_lo) * extent;
ptrdiff_t mid_off = BLOCK_OFFSET(nmid)     * extent;
ptrdiff_t hi_off  = BLOCK_OFFSET(block_hi) * extent;

size_t lower_cnt = BLOCK_COUNT(block_lo, nmid);   /* elements in lower half */
size_t upper_cnt = BLOCK_COUNT(nmid, block_hi);   /* elements in upper half */
```

Where `BLOCK_OFFSET(seg)` computes the element offset for k-segment `seg`
using early_segcount and late_segcount from `COLL_BASE_COMPUTE_BLOCKCOUNT`.

**Trace**: count=401, k=4 → split_rank=1, early=101, late=100:

```
Segment 0: offset 0, count 101    (early)
Segment 1: offset 101, count 100  (late)
Segment 2: offset 201, count 100  (late)
Segment 3: offset 301, count 100  (late)

Step 1 (mask=2): split at nmid=2
  Lower [0,2): segments 0,1 → 201 elements
  Upper [2,4): segments 2,3 → 200 elements
  ✓ Split at element 201, between segments 1 and 2

Step 2 for lower half, process 1 (mask=1): split at nmid=1
  Lower [0,1): segment 0 → 101 elements
  Upper [1,2): segment 1 → 100 elements
  ✓ Process 1 keeps segment 1 (100 elements at offset 101)
```

### New Constraints

1. **k must be a power of two**: The XOR-based partner selection with masks
   k/2, k/4, ..., 1 only covers all k processes correctly when k is a power
   of two.  For non-power-of-two k, some processes would be paired multiple
   times while others are never paired.

2. **Commutativity required**: The recursive halving combines data from
   higher-ranked and lower-ranked processes in the order determined by the
   send/receive pattern, not by the original rank order.  For non-commutative
   operations, the result would be incorrect.

---

## Design Principles Applied Throughout

### 1. Graceful Degradation

Every variant has a clean fallback path.  If the runtime constraints of a
variant (power-of-two k, commutativity, minimum count) are not met, the
function falls back to the well-tested `ompi_coll_base_allreduce_intra_recursivedoubling`
rather than returning an error or producing wrong results.

### 2. Correctness Over Performance

At each step, correctness was verified by tracing through small examples before
optimizing.  Operation order preservation was explicitly considered for each
algorithm variant.

### 3. Reuse of Existing Infrastructure

- `COLL_BASE_COMPUTE_BLOCKCOUNT` for segment arithmetic
- `ompi_coll_base_sendrecv_actual` for point-to-point exchange
- `ompi_op_reduce` for reduction with correct type handling
- `ompi_datatype_copy_content_same_ddt` for initial sbuf→rbuf copy
- Standard MCA parameter mechanism for k (via segsize)

### 4. Separation of Concerns

Each phase is implemented in its own clearly-delimited block with its own local
variables.  The `#ifdef` blocks are structured so that each variant is
self-contained and readable without needing to understand the other.

### 5. Minimal Memory Allocation

Only a single temporary buffer of max(early_segcount, late_segcount) elements
is allocated, reused across all three phases.  This is possible because each
phase completes before the next begins, and the tmpbuf is only needed during
send/recv+reduce steps.

---

## Common Pitfalls and Lessons Learned

1. **Bit-reversal in recursive algorithms**: When implementing recursive
   doubling/halving with XOR-based partner selection, the order of mask
   iteration (MSB→LSB vs LSB→MSB) fundamentally changes which segment each
   process ends up with.  Always trace through a small example.

2. **Integer division and segment boundaries**: Splitting work evenly among
   processes when the work size is not divisible requires careful bookkeeping.
   Never assume `count / k` is exact; always use `COLL_BASE_COMPUTE_BLOCKCOUNT`
   or equivalent.

3. **Commutativity matters**: Many "obvious" optimizations (ring algorithms,
   certain reduce-scatter patterns) silently produce wrong results for
   non-commutative operations.  Always check `ompi_op_is_commute(op)` and
   fall back if needed.

4. **Tag safety in multi-phase collectives**: When multiple phases use the same
   MPI tag, safety depends on the communication pattern (disjoint partner sets)
   and the blocking semantics of intermediate phases.  This must be analyzed
   explicitly.

5. **Power-of-two constraints**: XOR-based partner selection is elegant but
   fundamentally requires power-of-two process counts.  Non-power-of-two
   support requires additional pairing steps that add complexity.
