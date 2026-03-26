---
inputDocuments: ["docs/cuda_v4_implementation_plan.md", "docs/cuda_solver_critique_report.md"]
stepsCompleted: [1]
workflowType: 'epics-stories'
project_name: 'VRP CUDA Solver V4'
---

# Epics and Stories: VRP CUDA Solver V4

## 1. Requirements Extraction

### Functional Requirements (FRs)
- FR1: The solver must implement the Ant Colony Optimization (ACO) algorithm for the Vehicle Routing Problem (VRP) on CUDA.
- FR2: The solver must scale to support instances with 20,000+ nodes.
- FR3: The solver must manage the `visited` state using a Bitmask (1 bit per node) stored in Shared Memory.
- FR4: The solver must use Candidate Lists (top M neighbors) to reduce node selection complexity.
- FR5: The solver must manage nodes not in the candidate list via an Overflow List in Global Memory.
- FR6: The solver must store ACO parameters (alpha, beta, rho, etc.) in `__constant__` memory.
- FR7: The solver must use explicit `__syncwarp()` synchronization after updating the visited state bitmask.

### Non-Functional Requirements (NFRs)
- NFR1 (Performance): The system must maintain high Streaming Multiprocessor (SM) occupancy by minimizing the Shared Memory footprint (approx. 2.5KB for 20k nodes).
- NFR2 (Performance): The system must maximize memory bandwidth by utilizing Non-cached loads (`-dlcm=cg`) for sparse access patterns.
- NFR3 (Reliability): The system must guarantee atomicity of node visits, ensuring zero duplicate visits across all vehicles and ants.
- NFR4 (Scalability): The complexity of the next-node selection must be reduced from $O(N)$ to $O(M)$, where $M$ is the candidate list size.

### Additional Technical Requirements (Architecture)
- Use `extern __shared__` for dynamic allocation of the bitmask based on problem size N.
- Apply `__launch_bounds__` to kernels to prevent the "Performance Cliff" caused by excessive register pressure.
- Implement Shared Memory Tiling for parallel loading and zeroing of the bitmask within the warp.
- Transition towards a Structure of Arrays (SoA) layout for routes and state data to favor memory coalescing.

---

## 2. Requirements Coverage Map
{{requirements_coverage_map}}

---

## 3. Epics List
{{epics_list}}
