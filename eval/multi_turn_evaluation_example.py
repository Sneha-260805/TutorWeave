from eval.multi_turn_evaluation import run_multi_turn_evaluation

if __name__ == "__main__":
    run_multi_turn_evaluation(variants=["full", "no_memory", "plain_llm"], profiles=["beginner", "intermediate"], sessions=2, turns=4)
