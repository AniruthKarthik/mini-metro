# Mini Metro Empirical Evaluation Report (P3-1)

**Evaluation Seeds (5)**: `[1000, 1001, 1002, 1003, 1007]`  
**Model Checkpoint**: `None`  
**Maps Evaluated**: London (Map 0), New York City (Map 1), Tokyo (Map 2)

---

## 1. Performance Summary (Score & Survival Duration)

| Map | Policy | Score (Mean ± Std) | Median [Q25 - Q75] | Min - Max | Survival Steps | Survival Time (s) |
|:---|:---|:---:|:---:|:---:|:---:|:---:|
| **London** | Grandmaster Policy | 220.2 ± 61.2 | 249.0 [179.0 - 268.0] | 121 - 284 | 162.4 ± 32.5 | 162.4s |
| **New York City** | Grandmaster Policy | 200.0 ± 85.5 | 155.0 [138.0 - 296.0] | 102 - 309 | 143.4 ± 27.4 | 143.4s |
| **Tokyo** | Grandmaster Policy | 205.2 ± 21.7 | 207.0 [190.0 - 219.0] | 174 - 236 | 152.2 ± 16.3 | 152.2s |

---

## 2. Resource Utilization (Mean ± Std)

| Map | Policy | Active Lines | Trains Deployed | Tunnels Used | Interchanges Upgraded |
|:---|:---|:---:|:---:|:---:|:---:|
| **London** | Grandmaster Policy | 4.0 ± 0.6 | 4.8 ± 0.4 | 3.0 ± 0.0 | 0.4 ± 0.5 |
| **New York City** | Grandmaster Policy | 3.6 ± 0.5 | 4.4 ± 0.5 | 3.0 ± 0.0 | 0.2 ± 0.4 |
| **Tokyo** | Grandmaster Policy | 3.4 ± 0.5 | 4.8 ± 0.4 | 3.0 ± 0.0 | 0.8 ± 0.7 |

---

## 3. Cause of Death Distribution (Overcrowded Station Kinds)

| Map | Policy | Top Cause of Death | Full Breakdown (Shape: Count) |
|:---|:---|:---|:---|
| **London** | Grandmaster Policy | Circle (60%) | Circle: 3, Square: 1, Triangle: 1 |
| **New York City** | Grandmaster Policy | Square (40%) | Square: 2, Circle: 2, Triangle: 1 |
| **Tokyo** | Grandmaster Policy | Circle (40%) | Circle: 2, Star: 1, Triangle: 1, Square: 1 |

---
