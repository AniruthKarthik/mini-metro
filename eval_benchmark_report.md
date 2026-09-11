# Mini Metro Empirical Evaluation Report (P3-1)

**Evaluation Seeds (10)**: `[1000, 1001, 1002, 1003, 1004, 1005, 1006, 1007, 1008, 1009]`  
**Model Checkpoint**: `ml/runs/minimetro_ppo/model_final.pt`  
**Maps Evaluated**: London (Map 0), New York City (Map 1), Tokyo (Map 2)

---

## 1. Performance Summary (Score & Survival Duration)

| Map | Policy | Score (Mean ± Std) | Median [Q25 - Q75] | Min - Max | Survival Steps | Survival Time (s) |
|:---|:---|:---:|:---:|:---:|:---:|:---:|
| **London** | Model (Deterministic) | 35.2 ± 8.1 | 34.5 [29.0 - 38.2] | 27 - 56 | 458.8 ± 123.6 | 458.8s |
| **London** | Model (Stochastic) | 133.4 ± 38.5 | 123.0 [109.0 - 147.2] | 84 - 226 | 143.6 ± 21.7 | 143.6s |
| **London** | Greedy Heuristic | 123.7 ± 38.4 | 126.5 [102.8 - 139.0] | 52 - 197 | 133.0 ± 23.6 | 133.0s |
| **London** | Random Legal | 96.4 ± 38.1 | 90.5 [68.0 - 118.0] | 44 - 179 | 115.9 ± 22.7 | 115.9s |
| **New York City** | Model (Deterministic) | 40.3 ± 34.5 | 31.0 [22.2 - 37.8] | 18 - 141 | 88.2 ± 20.1 | 88.2s |
| **New York City** | Model (Stochastic) | 118.3 ± 58.4 | 120.5 [95.8 - 155.0] | 7 - 222 | 126.9 ± 32.9 | 126.9s |
| **New York City** | Greedy Heuristic | 94.6 ± 31.3 | 94.0 [67.2 - 122.5] | 51 - 142 | 118.6 ± 24.9 | 118.6s |
| **New York City** | Random Legal | 111.9 ± 44.6 | 110.5 [87.0 - 139.2] | 27 - 177 | 121.1 ± 26.0 | 121.1s |
| **Tokyo** | Model (Deterministic) | 35.1 ± 2.3 | 35.5 [33.2 - 36.0] | 32 - 40 | 500.0 ± 0.0 | 500.0s |
| **Tokyo** | Model (Stochastic) | 144.5 ± 33.9 | 148.0 [136.0 - 163.8] | 64 - 193 | 129.7 ± 14.5 | 129.7s |
| **Tokyo** | Greedy Heuristic | 136.4 ± 28.5 | 128.5 [118.5 - 163.0] | 96 - 181 | 145.3 ± 23.0 | 145.3s |
| **Tokyo** | Random Legal | 119.7 ± 36.6 | 106.0 [86.2 - 150.5] | 80 - 186 | 124.6 ± 29.1 | 124.6s |

---

## 2. Resource Utilization (Mean ± Std)

| Map | Policy | Active Lines | Trains Deployed | Tunnels Used | Interchanges Upgraded |
|:---|:---|:---:|:---:|:---:|:---:|
| **London** | Model (Deterministic) | 3.0 ± 0.0 | 3.1 ± 0.3 | 3.0 ± 0.0 | 0.0 ± 0.0 |
| **London** | Model (Stochastic) | 3.1 ± 0.7 | 4.3 ± 0.5 | 3.8 ± 1.0 | 0.3 ± 0.5 |
| **London** | Greedy Heuristic | 1.0 ± 0.0 | 4.0 ± 0.0 | 3.6 ± 1.0 | 0.4 ± 0.5 |
| **London** | Random Legal | 2.7 ± 0.5 | 4.1 ± 0.3 | 3.5 ± 0.9 | 0.2 ± 0.4 |
| **New York City** | Model (Deterministic) | 3.2 ± 0.4 | 3.8 ± 0.6 | 3.4 ± 0.8 | 0.3 ± 0.6 |
| **New York City** | Model (Stochastic) | 3.2 ± 0.4 | 4.2 ± 0.6 | 3.8 ± 1.3 | 0.2 ± 0.4 |
| **New York City** | Greedy Heuristic | 1.0 ± 0.0 | 4.0 ± 0.0 | 3.4 ± 0.8 | 0.4 ± 0.5 |
| **New York City** | Random Legal | 2.7 ± 0.5 | 4.1 ± 0.5 | 3.8 ± 1.3 | 0.3 ± 0.5 |
| **Tokyo** | Model (Deterministic) | 3.0 ± 0.0 | 3.0 ± 0.0 | 3.0 ± 0.0 | 0.0 ± 0.0 |
| **Tokyo** | Model (Stochastic) | 3.1 ± 0.3 | 4.4 ± 0.5 | 3.4 ± 0.8 | 0.5 ± 0.5 |
| **Tokyo** | Greedy Heuristic | 1.0 ± 0.0 | 4.0 ± 0.0 | 3.1 ± 0.3 | 0.4 ± 0.7 |
| **Tokyo** | Random Legal | 3.0 ± 0.6 | 4.3 ± 0.5 | 3.8 ± 0.9 | 0.2 ± 0.4 |

---

## 3. Cause of Death Distribution (Overcrowded Station Kinds)

| Map | Policy | Top Cause of Death | Full Breakdown (Shape: Count) |
|:---|:---|:---|:---|
| **London** | Model (Deterministic) | Circle (80%) | Circle: 8, Triangle: 1, Square: 1 |
| **London** | Model (Stochastic) | Circle (30%) | Circle: 3, Pentagon: 3, Triangle: 2, Square: 2 |
| **London** | Greedy Heuristic | Circle (50%) | Circle: 5, Triangle: 2, Sector: 1, Square: 1, Star: 1 |
| **London** | Random Legal | Circle (70%) | Circle: 7, Triangle: 2, Square: 1 |
| **New York City** | Model (Deterministic) | Circle (50%) | Circle: 5, Triangle: 3, Pentagon: 1, Square: 1 |
| **New York City** | Model (Stochastic) | Square (70%) | Square: 7, Triangle: 3 |
| **New York City** | Greedy Heuristic | Circle (50%) | Circle: 5, Triangle: 3, Pentagon: 2 |
| **New York City** | Random Legal | Circle (40%) | Circle: 4, Triangle: 3, Square: 2, Pentagon: 1 |
| **Tokyo** | Model (Deterministic) | Circle (90%) | Circle: 9, Triangle: 1 |
| **Tokyo** | Model (Stochastic) | Triangle (40%) | Triangle: 4, Square: 3, Circle: 3 |
| **Tokyo** | Greedy Heuristic | Triangle (50%) | Triangle: 5, Circle: 4, Pentagon: 1 |
| **Tokyo** | Random Legal | Circle (60%) | Circle: 6, Pentagon: 1, Triangle: 1, Gem: 1, Drop: 1 |

---
