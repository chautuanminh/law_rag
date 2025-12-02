import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from llama_index.llms.huggingface import HuggingFaceLLM
from llama_index.core import StorageContext, load_index_from_storage, Settings

from src.config import INDEX_DIR, EMBED_MODEL_NAME, HF_LLM_MODEL_NAME, HF_HOME
from llama_index.embeddings.huggingface import HuggingFaceEmbedding


def load_llm() -> HuggingFaceLLM:
    print(f"🤖 Loading LLM model: {HF_LLM_MODEL_NAME}")

    tokenizer = AutoTokenizer.from_pretrained(
        HF_LLM_MODEL_NAME,
        cache_dir=str(HF_HOME),
        trust_remote_code=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        HF_LLM_MODEL_NAME,
        cache_dir=str(HF_HOME),
        device_map="auto",
        torch_dtype="auto",
        trust_remote_code=True,
    )

    llm = HuggingFaceLLM(
        tokenizer=tokenizer,
        model=model,
        context_window=2048,
        max_new_tokens=512,
        is_chat_model=True,   # 🔥 IMPORTANT FOR Qwen MODELS
        generate_kwargs={
            "temperature": 0.0,   # best for law RAG
            "do_sample": False,
        }
    )
    return llm


def main() -> None:
    print("📦 Loading storage context...")
    storage_context = StorageContext.from_defaults(persist_dir=str(INDEX_DIR))

    print(f"🧠 Loading embedding model: {EMBED_MODEL_NAME}")
    embed_model = HuggingFaceEmbedding(
        model_name=EMBED_MODEL_NAME,
        cache_folder=str(HF_HOME),
    )
    Settings.embed_model = embed_model  # make sure retriever uses bge-m3

    print("📚 Loading index from disk with embedding model...")
    index = load_index_from_storage(storage_context, embed_model=embed_model)

    print("🤖 Initializing LLM...")
    llm = load_llm()

    # Simple retriever
    retriever = index.as_retriever(similarity_top_k=4)

    print("\n✅ RAG ready.")
    print("Ask about *Luật Doanh nghiệp 2020* in English or Vietnamese.")
    print("Type 'exit' to quit.\n")

    while True:
        question = input("Q (or 'exit'): ").strip()
        if not question:
            continue
        if question.lower() in {"exit", "quit"}:
            break

        print("\n🔍 Retrieving relevant chunks...")
        nodes = retriever.retrieve(question)

        for i, node in enumerate(nodes):
            score = getattr(node, "score", None)
            print(f"\n--- Chunk #{i} (score={score}) ---")
            print(node.text[:800])

        # 🔻 NEW: safely limit the context length so it fits in the LLM
        MAX_CONTEXT_CHARS = 3500  # tune this if needed
        raw_context = "\n\n".join(node.text for node in nodes)
        context_text = raw_context[:MAX_CONTEXT_CHARS]

        print(f"\n[DEBUG] Context length (chars): {len(context_text)}")

        prompt = f"""You are a helpful legal assistant.

You answer questions about the Vietnamese Enterprise Law 2020 ("Luật Doanh nghiệp 2020").
The user's question may be in English or Vietnamese.

Use ONLY the context below. If the answer is not clearly in the context,
say you cannot find it in the provided excerpts.

Context:
{context_text}

Question:
{question}

Answer in **English**, but you may quote Vietnamese terms or article numbers
(e.g., "Điều 1", "Điều 2") exactly as in the law. Give a concise answer in 3–8 sentences.
"""

        print("\n🤖 Answer:")
        try:
            resp = llm.chat([{"role": "user", "content": prompt}])
            print(resp.message.content)


        except Exception as e:
            print(f"[LLM ERROR] {e}\n")


if __name__ == "__main__":
    main()
