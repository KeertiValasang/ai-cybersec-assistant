"""
Phase 5c: Retrieval -- query the knowledge base by meaning.

Given either an exact attack label (fast path) or a free-text query
(semantic search path), returns the most relevant knowledge base
entry/entries. This is the function the LLM report generator (Phase 6)
will call to ground its answers.

Usage (demo / manual test):
    python src/rag/retrieve.py
"""

import json
import os

import chromadb
from sentence_transformers import SentenceTransformer

KB_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "knowledge_base", "attack_knowledge_base.json")
CHROMA_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "knowledge_base", "chroma_db")
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
COLLECTION_NAME = "attack_knowledge"


class KnowledgeRetriever:
    def __init__(self):
        self.embedder = SentenceTransformer(EMBEDDING_MODEL)
        self.client = chromadb.PersistentClient(path=CHROMA_DIR)
        self.collection = self.client.get_collection(COLLECTION_NAME)

        with open(KB_PATH, "r", encoding="utf-8") as f:
            self.raw_entries = {e["attack_label"]: e for e in json.load(f)}

    def get_by_label(self, attack_label: str) -> dict:
        """Fast path: exact lookup when we already know the predicted attack label."""
        return self.raw_entries.get(attack_label)

    def search(self, query: str, n_results: int = 2) -> list:
        """Semantic search path: find the most relevant entries for a free-text query."""
        query_embedding = self.embedder.encode([query]).tolist()
        results = self.collection.query(
            query_embeddings=query_embedding,
            n_results=n_results,
        )
        matches = []
        for doc, meta, dist in zip(
            results["documents"][0], results["metadatas"][0], results["distances"][0]
        ):
            matches.append({"text": doc, "metadata": meta, "distance": dist})
        return matches


if __name__ == "__main__":
    retriever = KnowledgeRetriever()

    print("=" * 70)
    print("TEST 1: Exact label lookup for 'DDoS'")
    print("=" * 70)
    entry = retriever.get_by_label("DDoS")
    print(json.dumps(entry, indent=2))

    print("\n" + "=" * 70)
    print("TEST 2: Semantic search for a free-text description")
    print("=" * 70)
    query = "lots of failed login attempts against a server in a short time"
    print(f"Query: '{query}'\n")
    results = retriever.search(query, n_results=2)
    for i, r in enumerate(results, 1):
        print(f"--- Match {i} (distance={r['distance']:.4f}) ---")
        print(f"Attack: {r['metadata']['attack_label']} | "
              f"MITRE: {r['metadata']['mitre_technique_id']}")
        print()
