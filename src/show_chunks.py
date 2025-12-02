# src/show_chunks.py

from src.config import INDEX_DIR
from llama_index.core import StorageContext


def main():
    print("DEBUG: loading StorageContext from:", INDEX_DIR)
    storage_context = StorageContext.from_defaults(persist_dir=INDEX_DIR)
    docstore = storage_context.docstore

    print("Docstore path:", f"{INDEX_DIR}\\docstore.json")
    print("Total nodes in docstore:", len(docstore.docs))

    # Print a few example nodes
    for i, (node_id, node) in enumerate(docstore.docs.items()):
        print(f"\n--- NODE {i} | id={node_id} ---")
        print(node.text[:500])
        if i >= 2:
            break


if __name__ == "__main__":
    main()
