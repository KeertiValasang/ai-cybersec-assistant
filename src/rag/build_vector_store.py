"""
Phase 5b: Build the vector store.

Loads the knowledge base JSON, embeds each entry using a sentence
embedding model, and stores it in a persistent ChromaDB collection.
This lets us later retrieve knowledge base entries by MEANING
(semantic search), not just exact keyword match -- e.g. searching
"lots of failed logins" should still find the Brute Force entry
even without matching that exact phrase.

Usage:
    python src/rag/build_vector_store.py
"""

import json
import os

import chromadb
from sentence_transformers import SentenceTransformer

KB_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "knowledge_base", "attack_knowledge_base.json")
CHROMA_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "knowledge_base", "chroma_db")

EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # small, fast, good enough for this KB size
COLLECTION_NAME = "attack_knowledge"


def entry_to_text(entry: dict) -> str:
    """Flatten a KB entry into a single text blob for embedding."""
    parts = [
        f"Attack: {entry['attack_label']}",
        f"MITRE ATT&CK: {entry['mitre_technique_id']} - {entry['mitre_technique_name']}",
        f"Description: {entry['description']}",
        "Detection indicators: " + "; ".join(entry["detection_indicators"]),
        "Mitigation: " + "; ".join(entry["mitigation"]),
    ]
    return "\n".join(parts)


def main():
    with open(KB_PATH, "r", encoding="utf-8") as f:
        kb_entries = json.load(f)
    print(f"Loaded {len(kb_entries)} knowledge base entries")

    print(f"Loading embedding model '{EMBEDDING_MODEL}' (first run downloads it, ~90MB)...")
    embedder = SentenceTransformer(EMBEDDING_MODEL)

    texts = [entry_to_text(e) for e in kb_entries]
    ids = [e["attack_label"] for e in kb_entries]
    metadatas = [
        {
            "attack_label": e["attack_label"],
            "mitre_technique_id": e["mitre_technique_id"],
            "mitre_technique_name": e["mitre_technique_name"],
        }
        for e in kb_entries
    ]

    print("Computing embeddings...")
    embeddings = embedder.encode(texts, show_progress_bar=True).tolist()

    print(f"Writing to ChromaDB at {CHROMA_DIR} ...")
    client = chromadb.PersistentClient(path=CHROMA_DIR)

    # Recreate collection fresh each time this script runs
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass
    collection = client.create_collection(COLLECTION_NAME)

    collection.add(
        ids=ids,
        embeddings=embeddings,
        documents=texts,
        metadatas=metadatas,
    )

    print(f"Done. Collection '{COLLECTION_NAME}' now has {collection.count()} entries.")


if __name__ == "__main__":
    main()
