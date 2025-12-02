# src/test_llm.py

from llama_index.llms.huggingface import HuggingFaceLLM

def main():
    print("🧠 Loading TinyLlama...")
    llm = HuggingFaceLLM(
        model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        tokenizer_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        context_window=2048,
        max_new_tokens=128,
        generate_kwargs={
            "temperature": 0.2,
            "top_p": 0.9,
        },
    )

    print("✅ LLM loaded. Try asking something.")
    while True:
        q = input("\nQ (or 'exit'): ").strip()
        if q.lower() in {"exit", "quit"}:
            break
        if not q:
            continue

        print("→ Generating...")
        try:
            resp = llm.complete(q)
            # resp.text is the answer string
            print("Answer:", resp.text)
        except Exception as e:
            print("❌ ERROR from LLM:", repr(e))

if __name__ == "__main__":
    main()
