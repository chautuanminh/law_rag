from pathlib import Path

DATA_PATH = "data/luatdoanhnghiep2020.txt"
INDEX_DIR = "index"

EMBED_MODEL_NAME = "BAAI/bge-m3"

# ⬇️ change this line
HF_LLM_MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"

HF_HOME = str(Path.cwd() / "hf_cache")
