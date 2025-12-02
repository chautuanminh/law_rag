from sentence_transformers import SentenceTransformer

model = SentenceTransformer("BAAI/bge-m3")

text = "Hôm nay thời tiết ở Hà Nội rất đẹp."
embedding = model.encode(text, normalize_embeddings=True)

print(embedding.shape)
