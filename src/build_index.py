# src/build_index.py

from pathlib import Path

from src.config import DATA_PATH, INDEX_DIR, EMBED_MODEL_NAME
from llama_index.core import Document, VectorStoreIndex, StorageContext
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from pypdf import PdfReader


def load_pdf_to_documents(path: Path):
    print("📄 Reading PDF:", path)
    reader = PdfReader(str(path))

    texts = []
    for i, page in enumerate(reader.pages):
        page_text = page.extract_text() or ""
        print(f" - Page {i} characters:", len(page_text))
        texts.append(page_text)

    full_text = "\n\n".join(texts).strip()
    if not full_text:
        print("❌ No text could be extracted from the PDF.")
        return []

    print("✅ Total extracted characters:", len(full_text))
    return [Document(text=full_text, metadata={"source": str(path)})]


def load_txt_to_documents(path: Path):
    print("📄 Reading TXT:", path)
    text = path.read_text(encoding="utf-8")
    print("✅ Total characters from TXT:", len(text))

    if not text.strip():
        print("❌ TXT file empty.")
        return []

    return [Document(text=text, metadata={"source": str(path)})]


def main():
    data_path = Path(DATA_PATH)
    print("DATA_PATH:", data_path)
    print("Exists:", data_path.exists())
    if not data_path.exists():
        print("❌ DATA_PATH missing!")
        return

    print("📥 Loading documents...")
    suffix = data_path.suffix.lower()

    if suffix == ".txt":
        documents = load_txt_to_documents(data_path)
    elif suffix == ".pdf":
        documents = load_pdf_to_documents(data_path)
    else:
        print("❌ Unsupported file type:", suffix)
        return

    print("✅ Loaded", len(documents), "documents")
    if not documents:
        print("❌ No docs extracted.")
        return

    print("\n--- Preview ---")
    print(documents[0].text[:1000])

    print("\n🧠 Loading embedding model:", EMBED_MODEL_NAME)
    embed_model = HuggingFaceEmbedding(
        model_name=EMBED_MODEL_NAME,
        embed_batch_size=8,
    )
    print("✅ Embedding model ready")

    # ****** IMPORTANT FIX HERE ******  
    # Create NEW empty storage instead of loading old one
    print("\n📦 Creating NEW empty StorageContext...")
    storage_context = StorageContext.from_defaults()

    print("\n🏗  Building vector index...")
    index = VectorStoreIndex.from_documents(
        documents,
        embed_model=embed_model,
        storage_context=storage_context
    )
    print("✅ Index built")

    print("🔍 Nodes in docstore (in memory):", len(storage_context.docstore.docs))

    print("💾 Saving index to disk:", INDEX_DIR)
    storage_context.persist(persist_dir=INDEX_DIR)

    print("🎉 DONE — index successfully built and saved!")


if __name__ == "__main__":
    main()
