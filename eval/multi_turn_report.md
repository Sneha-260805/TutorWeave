# Multi-Turn EduAgent Evaluation

This report summarizes a multi-turn tutoring evaluation in which each configuration is run over structured learning sessions that explicitly stress memory retrieval, evaluator feedback, weak-concept review, and difficulty adaptation.

## Aggregate Results

| Variant | Profile | Personalization | Adaptive Teaching | Weak Concept Recall | Difficulty Alignment | Hallucination Rate | Learning Gain | Latency (ms) |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| full | advanced | 1.000 | 0.975 | 0.946 | 0.775 | 0.150 | 0.290 | 0.0 |
| full | beginner | 1.000 | 1.000 | 0.946 | 0.800 | 0.150 | 0.275 | 0.0 |
| full | intermediate | 1.000 | 0.992 | 0.973 | 0.738 | 0.150 | 0.303 | 0.0 |
| no_classifier | advanced | 1.000 | 0.975 | 0.946 | 0.775 | 0.150 | 0.300 | 0.0 |
| no_classifier | beginner | 1.000 | 0.992 | 0.946 | 0.800 | 0.150 | 0.267 | 0.0 |
| no_classifier | intermediate | 1.000 | 1.000 | 0.973 | 0.738 | 0.150 | 0.311 | 0.0 |
| no_evaluator | advanced | 1.000 | 0.967 | 0.946 | 0.775 | 0.150 | 0.298 | 0.0 |
| no_evaluator | beginner | 1.000 | 1.000 | 0.946 | 0.800 | 0.150 | 0.275 | 0.0 |
| no_evaluator | intermediate | 1.000 | 0.992 | 0.973 | 0.738 | 0.150 | 0.303 | 0.0 |
| no_memory | advanced | 0.971 | 0.992 | 0.946 | 0.742 | 0.150 | 0.300 | 0.0 |
| no_memory | beginner | 0.971 | 0.992 | 0.946 | 0.800 | 0.150 | 0.267 | 0.0 |
| no_memory | intermediate | 0.986 | 0.984 | 0.973 | 0.738 | 0.150 | 0.295 | 0.0 |
| no_retrieval | advanced | 1.000 | 0.975 | 0.935 | 0.775 | 0.118 | 0.298 | 0.0 |
| no_retrieval | beginner | 1.000 | 0.992 | 0.946 | 0.800 | 0.134 | 0.267 | 0.0 |
| no_retrieval | intermediate | 1.000 | 1.000 | 0.952 | 0.738 | 0.124 | 0.311 | 0.0 |
| no_weak_retrieval | advanced | 1.000 | 0.975 | 0.935 | 0.775 | 0.150 | 0.300 | 0.0 |
| no_weak_retrieval | beginner | 1.000 | 1.000 | 0.946 | 0.800 | 0.150 | 0.275 | 0.0 |
| no_weak_retrieval | intermediate | 1.000 | 0.992 | 0.952 | 0.738 | 0.150 | 0.303 | 0.0 |
| plain_llm | advanced | 0.450 | 0.584 | 0.350 | 0.708 | 0.123 | 0.245 | 0.0 |
| plain_llm | beginner | 0.450 | 0.600 | 0.350 | 0.725 | 0.139 | 0.182 | 0.0 |
| plain_llm | intermediate | 0.450 | 0.600 | 0.350 | 0.700 | 0.130 | 0.218 | 0.0 |

## Interpretation

- Higher personalization and adaptive teaching indicate stronger profile-aware tutoring.
- Higher weak concept recall indicates that previously difficult concepts are revisited effectively.
- Lower hallucination rate is preferred, especially when retrieval is ablated.