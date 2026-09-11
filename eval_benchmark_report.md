# Mini Metro Empirical Evaluation Report (P3-1)

**Evaluation Seeds (2)**: `[1000, 1001]`  
**Model Checkpoint**: `ml/runs/minimetro_ppo/model_final.pt`  
**Maps Evaluated**: London (Map 0), New York City (Map 1), Tokyo (Map 2)

---

## 1. Performance Summary (Score & Survival Duration)

| Map | Policy | Score (Mean ± Std) | Median [Q25 - Q75] | Min - Max | Survival Steps | Survival Time (s) |
|:---|:---|:---:|:---:|:---:|:---:|:---:|
| **London** | Model (Deterministic) | 82.5 ± 34.5 | 82.5 [65.2 - 99.8] | 48 - 117 | 115.0 ± 10.0 | 115.0s |
| **London** | Model (Stochastic) | 101.0 ± 2.0 | 101.0 [100.0 - 102.0] | 99 - 103 | 113.0 ± 10.0 | 113.0s |
| **London** | Greedy Heuristic | 109.0 ± 14.0 | 109.0 [102.0 - 116.0] | 95 - 123 | 113.5 ± 7.5 | 113.5s |
| **London** | Random Legal | 32.5 ± 1.5 | 32.5 [31.8 - 33.2] | 31 - 34 | 82.0 ± 3.0 | 82.0s |
| **New York City** | Model (Deterministic) | 95.5 ± 23.5 | 95.5 [83.8 - 107.2] | 72 - 119 | 128.5 ± 24.5 | 128.5s |
| **New York City** | Model (Stochastic) | 143.5 ± 53.5 | 143.5 [116.8 - 170.2] | 90 - 197 | 136.5 ± 35.5 | 136.5s |
| **New York City** | Greedy Heuristic | 107.5 ± 20.5 | 107.5 [97.2 - 117.8] | 87 - 128 | 107.5 ± 1.5 | 107.5s |
| **New York City** | Random Legal | 54.0 ± 15.0 | 54.0 [46.5 - 61.5] | 39 - 69 | 102.5 ± 18.5 | 102.5s |
| **Tokyo** | Model (Deterministic) | 109.5 ± 17.5 | 109.5 [100.8 - 118.2] | 92 - 127 | 126.0 ± 14.0 | 126.0s |
| **Tokyo** | Model (Stochastic) | 109.5 ± 37.5 | 109.5 [90.8 - 128.2] | 72 - 147 | 121.5 ± 13.5 | 121.5s |
| **Tokyo** | Greedy Heuristic | 147.0 ± 21.0 | 147.0 [136.5 - 157.5] | 126 - 168 | 144.5 ± 20.5 | 144.5s |
| **Tokyo** | Random Legal | 41.5 ± 23.5 | 41.5 [29.8 - 53.2] | 18 - 65 | 103.5 ± 8.5 | 103.5s |

---

## 2. Resource Utilization (Mean ± Std)

| Map | Policy | Active Lines | Trains Deployed | Tunnels Used | Interchanges Upgraded |
|:---|:---|:---:|:---:|:---:|:---:|
| **London** | Model (Deterministic) | 1.5 ± 0.5 | 4.0 ± 0.0 | 3.5 ± 1.5 | 0.0 ± 0.0 |
| **London** | Model (Stochastic) | 1.0 ± 0.0 | 4.0 ± 0.0 | 4.0 ± 0.0 | 0.0 ± 0.0 |
| **London** | Greedy Heuristic | 1.0 ± 0.0 | 4.0 ± 0.0 | 3.5 ± 0.5 | 0.5 ± 0.5 |
| **London** | Random Legal | 3.0 ± 0.0 | 4.0 ± 0.0 | 3.5 ± 0.5 | 0.0 ± 0.0 |
| **New York City** | Model (Deterministic) | 1.5 ± 0.5 | 4.0 ± 0.0 | 2.5 ± 0.5 | 0.0 ± 0.0 |
| **New York City** | Model (Stochastic) | 1.5 ± 0.5 | 4.5 ± 0.5 | 5.0 ± 0.0 | 0.5 ± 0.5 |
| **New York City** | Greedy Heuristic | 1.0 ± 0.0 | 4.0 ± 0.0 | 4.0 ± 1.0 | 0.5 ± 0.5 |
| **New York City** | Random Legal | 3.0 ± 0.0 | 4.0 ± 0.0 | 3.0 ± 0.0 | 0.5 ± 0.5 |
| **Tokyo** | Model (Deterministic) | 1.5 ± 0.5 | 4.0 ± 0.0 | 3.5 ± 1.5 | 0.0 ± 0.0 |
| **Tokyo** | Model (Stochastic) | 1.0 ± 0.0 | 4.0 ± 0.0 | 3.0 ± 0.0 | 0.5 ± 0.5 |
| **Tokyo** | Greedy Heuristic | 1.0 ± 0.0 | 4.0 ± 0.0 | 3.5 ± 0.5 | 1.0 ± 1.0 |
| **Tokyo** | Random Legal | 2.0 ± 0.0 | 2.5 ± 0.5 | 3.5 ± 0.5 | 0.0 ± 0.0 |

---

## 3. Cause of Death Distribution (Overcrowded Station Kinds)

| Map | Policy | Top Cause of Death | Full Breakdown (Shape: Count) |
|:---|:---|:---|:---|
| **London** | Model (Deterministic) | Circle (50%) | Circle: 1, Triangle: 1 |
| **London** | Model (Stochastic) | Pentagon (50%) | Pentagon: 1, Square: 1 |
| **London** | Greedy Heuristic | Circle (50%) | Circle: 1, Square: 1 |
| **London** | Random Legal | Circle (100%) | Circle: 2 |
| **New York City** | Model (Deterministic) | Circle (50%) | Circle: 1, Pentagon: 1 |
| **New York City** | Model (Stochastic) | Square (50%) | Square: 1, Triangle: 1 |
| **New York City** | Greedy Heuristic | Circle (50%) | Circle: 1, Triangle: 1 |
| **New York City** | Random Legal | Circle (50%) | Circle: 1, Square: 1 |
| **Tokyo** | Model (Deterministic) | Square (50%) | Square: 1, Triangle: 1 |
| **Tokyo** | Model (Stochastic) | Circle (50%) | Circle: 1, Triangle: 1 |
| **Tokyo** | Greedy Heuristic | Circle (50%) | Circle: 1, Triangle: 1 |
| **Tokyo** | Random Legal | Triangle (50%) | Triangle: 1, Pentagon: 1 |

---
