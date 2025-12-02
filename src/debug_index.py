# src/debug_index.py

print("DEBUG: starting debug_index.py")

from pathlib import Path
from src.config import INDEX_DIR, EMBED_MODEL_NAME
print("DEBUG: imported config")
print("DEBUG: INDEX_DIR =", INDEX_DIR)
print("DEBUG: EMBED_MODEL_NAME =", EMBED_MODEL_NAME)

from llama_index.core import StorageContext, load_index_from_storage
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
print("DEBUG: imported llama_index modules")

def main():
    print("DEBUG: in main()")

    # 1) Check index folder
    index_path = Path(INDEX_DIR)
    print("DEBUG: index path exists?", index_path.exists())
    if index_path.exists():
        print("DEBUG: index dir contents:", [p.name for p in index_path.iterdir()])
    else:
        print("DEBUG: index directory missing -> nothing was persisted")
        return

    # 2) Create embedding model
    print("DEBUG: creating HuggingFaceEmbedding...")
    embed_model = HuggingFaceEmbedding(
        model_name=EMBED_MODEL_NAME,
        embed_batch_size=8,
    )
    print("DEBUG: embedding model created")

    # 3) Load index
    print("DEBUG: creating StorageContext...")
    storage_context = StorageContext.from_defaults(persist_dir=INDEX_DIR)
    print("DEBUG: loading index from storage...")
    index = load_index_from_storage(
        storage_context,
        embed_model=embed_model,
    )
    print("DEBUG: index loaded")

    # 4) Inspect nodes
    nodes = list(index.docstore.get_all_nodes().values())
    print("Number of nodes in index:", len(nodes))

    for i, node in enumerate(nodes[:5]):
        print(f"\n--- NODE {i} ---")
        print(node.text[:500])

if __name__ == "__main__":
    main()
