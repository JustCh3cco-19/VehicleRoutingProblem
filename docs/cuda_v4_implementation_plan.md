# Implementation Plan: CUDA Solver VRP v4 (Massive Scale Edition)

**Goal:** Transform the V3 solver into a robust, correct, and hardware-optimized architecture capable of scaling to **20,000+ nodes** using "Multicore" best practices for large datasets.

---

## 1. Scalability Architecture (N > 20,000)

### Phase 1: Shared Memory Bitmask (Memory Efficiency)
*   **Constraint:** A standard `bool` array for 20k nodes (20KB per ant) would crash occupancy (Multicore [25]).
*   **Action:** Implement a **Bitmask** for the `visited` state (1 bit per node).
*   **Impact:** Reduces memory footprint from 20KB to **2.5KB per ant**, allowing high occupancy even on large instances.
*   **Tiling:** Use **Shared Memory Tiling** to load and clear the bitmask in parallel across the warp (Multicore [2, 3]).

### Phase 2: Candidate Lists & Spatially Sorting (Complexity Reduction)
*   **Constraint:** $O(N^2)$ complexity is unsustainable for $N=20,000$.
*   **Action:** Implement **Candidate Lists** based on spatial proximity (Multicore [12, 13]). Each node will only consider its top $M$ (e.g., 32-64) nearest neighbors during the probabilistic selection.
*   **Overflow Handling:** Use an **Overflow List** in Global Memory for nodes that fall outside the candidate list (Multicore [18, 19]).

### Phase 3: Constant Memory & Parameter Broadcasting
*   **Action:** Use `__constant__` for ACO parameters and candidate list metadata.
*   **Rationale:** Enable hardware **Broadcasting** to eliminate redundant DRAM requests (Multicore [16]).

---

## 2. Low-Level Hardware Optimization

### Phase 4: L2 Cache Tuning for Sparse Access
*   **Action:** Compile the kernel with `-Xptxas -dlcm=cg` to bypass L1 cache.
*   **Rationale:** For $N=20,000$, accesses are inherently sparse. Using L2's 32-byte segments instead of L1's 128-byte lines triples bandwidth efficiency by reducing "waste" data transfers (Multicore [22, 34]).

### Phase 5: Explicit Warp Synchronization (ITS Compliance)
*   **Action:** Strict use of `__syncwarp()` after Bitmask updates.
*   **Rationale:** Ensure memory visibility across the warp on Volta+ architectures, preventing the "Duplicate Visit" bug identified in stress tests (Multicore [1]).

---

## 3. Execution Roadmap (Updated)

| Step | Task | Success Metric |
| :--- | :--- | :--- |
| **4.1** | Implement **Bitmask** logic in Shared Memory. | 2.5KB footprint for N=20k. |
| **4.2** | Integrate **Candidate Lists** (M=32) to reduce $O(N)$ loop. | Linear scaling with N. |
| **4.3** | Add `__syncwarp()` and **Non-cached loads** (`-dlcm=cg`). | Correctness + 3x Bandwidth efficiency. |
| **4.4** | Validate on 20k nodes vs PyVRP (using a smaller runtime). | Feasible solution within reasonable GPU time. |

---

**Status:** Implementation plan revised for massive scale. Ready for Bitmask implementation.
