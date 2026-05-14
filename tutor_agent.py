import google.generativeai as genai
from example_retriever import retrieve_examples, detect_best_topic
from ml.classifier import predict_level
import os

# -----------------------------
# CONFIG
# -----------------------------
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("Set GEMINI_API_KEY in your environment before running this script.")

genai.configure(api_key=GEMINI_API_KEY)

# -----------------------------
# FORMAT EXAMPLES
# -----------------------------
def format_examples(examples_df):
    parts = []
    for i, row in enumerate(examples_df.itertuples(index=False), 1):
        parts.append(
            f"Example {i}:\n"
            f"Question: {row.question}\n"
            f"Answer: {row.answer}\n"
            f"Topic: {row.topic}"
        )
    return "\n\n".join(parts)

# -----------------------------
# GENERATE TUTOR RESPONSE
# -----------------------------
def generate_tutor_response(user_question):
    level, confidence = predict_level(user_question)
    topic = detect_best_topic(user_question, level)
    examples = retrieve_examples(user_question, level, top_n=2)
    examples_text = format_examples(examples)

    prompt = f"""
You are EduAgent, an adaptive AI tutor.

Student question:
{user_question}

Detected student level:
{level}

Detected topic:
{topic}

Retrieved knowledge base examples:
{examples_text}

Instructions:
- Your answer MUST be grounded in the retrieved examples above.
- Use the facts, explanations, and concepts from those examples as the 
  foundation of your answer. Do not ignore them.
- You may expand on them, but do not contradict or go far beyond them.
- Adapt the explanation style and difficulty to the detected level:
    - beginner: simple words, analogies, no jargon
    - intermediate: moderate detail, 1-2 key terms explained
    - advanced: technical depth, assume background knowledge
- Structure your answer clearly and keep it educational.
- Do not copy examples word for word — explain in your own words 
  but stay faithful to the retrieved content.

Now answer the student's question based on the retrieved examples above.
"""


    model = genai.GenerativeModel(model_name="gemini-2.0-flash")
    response = model.generate_content([{"role": "user", "parts": [prompt]}])
    answer = response.text
    return level, confidence, topic, examples, answer

# -----------------------------
# MAIN LOOP
# -----------------------------
if __name__ == "__main__":
    while True:
        q = input("\nAsk a question (or quit): ").strip()
        if q.lower() == "quit":
            break

        level, confidence, topic, examples, answer = generate_tutor_response(q)

        print("\n=== DETECTED LEVEL ===")
        print(level)
        print("Confidence:", confidence)

        print("\n=== DETECTED TOPIC ===")
        print(topic)

        print("\n=== RETRIEVED EXAMPLES ===")
        for i, row in enumerate(examples.itertuples(index=False), 1):
            print(f"\nExample {i}")
            print("Question:", row.question)
            print("Answer:", row.answer)
            print("Level:", row.level)
            print("Topic:", row.topic)

        print("\n=== TUTOR ANSWER ===")
        print(answer)
