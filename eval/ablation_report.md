# EduAgent Ablation Study

## Mean scores by variant (1-5 scale, LLM-as-judge)

| Variant | Profile | Correctness | Level Fit | Clarity | Specificity | Overall | ROUGE-L | BERTScore | Semantic Sim | Latency (s) |
|---|---|---|---|---|---|---|---|---|---|---|
| full | baseline | - | - | - | - | **-** | 0.923 | 0.804 | 0.666 | 6.425 |
| full | personalized | - | - | - | - | **-** | 0.923 | 0.804 | 0.666 | 3.786 |
| no_classifier | baseline | - | - | - | - | **-** | 0.918 | 0.796 | 0.709 | 6.094 |
| no_classifier | personalized | - | - | - | - | **-** | 0.918 | 0.796 | 0.709 | 3.862 |
| no_evaluator_loop | baseline | - | - | - | - | **-** | 0.923 | 0.804 | 0.666 | 3.369 |
| no_evaluator_loop | personalized | - | - | - | - | **-** | 0.923 | 0.804 | 0.666 | 3.513 |
| no_memory | baseline | - | - | - | - | **-** | 0.923 | 0.804 | 0.666 | 3.394 |
| no_memory | personalized | - | - | - | - | **-** | 0.923 | 0.804 | 0.666 | 3.418 |
| no_retrieval | baseline | - | - | - | - | **-** | 0.076 | -0.156 | 0.313 | 3.311 |
| no_retrieval | personalized | - | - | - | - | **-** | 0.076 | -0.156 | 0.313 | 3.299 |
| no_weak_retrieval | baseline | - | - | - | - | **-** | 0.923 | 0.804 | 0.666 | 3.483 |
| no_weak_retrieval | personalized | - | - | - | - | **-** | 0.923 | 0.804 | 0.666 | 3.459 |
| plain_llm | baseline | - | - | - | - | **-** | 0.044 | -0.036 | 0.077 | 3.188 |
| plain_llm | personalized | - | - | - | - | **-** | 0.044 | -0.036 | 0.077 | 3.147 |

## Component contribution (paired delta vs. full system)

Positive delta = removing that component hurt the overall judge score.

| Comparison | Mean Δ (overall) | t-statistic | n paired |
|---|---|---|---|
| full_vs_no_classifier__baseline | None | None | 5 |
| full_vs_no_retrieval__baseline | None | None | 5 |
| full_vs_no_weak_retrieval__baseline | None | None | 5 |
| full_vs_no_memory__baseline | None | None | 5 |
| full_vs_no_evaluator_loop__baseline | None | None | 5 |
| full_vs_plain_llm__baseline | None | None | 5 |
| full_vs_no_classifier__personalized | None | None | 5 |
| full_vs_no_retrieval__personalized | None | None | 5 |
| full_vs_no_weak_retrieval__personalized | None | None | 5 |
| full_vs_no_memory__personalized | None | None | 5 |
| full_vs_no_evaluator_loop__personalized | None | None | 5 |
| full_vs_plain_llm__personalized | None | None | 5 |

## How to read this for the paper
- Each row under 'Component contribution' isolates one architectural piece (classifier, retrieval, weak-area retrieval, memory hint, evaluator feedback loop) by comparing the full system against a version with only that piece removed, on the *same* questions and profile condition.
- Report `n_paired` alongside deltas — increase `--repeats` and `--n-questions` for a paper-grade sample (recommended: >=30 questions x 3 repeats = 90 paired samples per comparison).
- `full (personalized)` vs `full (baseline)` isolates the effect of *having* learner history at all, independent of which component uses it.
- `plain_llm` reproduces the qualitative Table-3 comparison in the report as a quantitative baseline.