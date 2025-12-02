# src/list_index_files.py

from pathlib import Path
from src.config import INDEX_DIR

def main():
    print("INDEX_DIR:", INDEX_DIR)
    index_path = Path(INDEX_DIR)

    print("Exists:", index_path.exists())
    if not index_path.exists():
        return

    print("Contents:")
    for p in index_path.iterdir():
        print(" -", p.name)

if __name__ == "__main__":
    main()
