import sqlite3
import faiss
import numpy as np
from typing import List, Dict, Any

class LLRMemory:
    """Memory for storing and retrieving Low-Level Reflection data."""

    def __init__(self, db_path: str = "src/database/llr_reflection.db", 
                 index_path: str = "src/database/llr_embeddings.faiss"):
        self.db_path = db_path
        self.index_path = index_path
        self.index: faiss.IndexFlatL2 = None
        self._setup_sqlite()

    def _setup_sqlite(self):
        """Set up SQLite table for LLR metadata."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''CREATE TABLE IF NOT EXISTS llr_metadata
                         (id INTEGER PRIMARY KEY, ticker TEXT, date TEXT, analysis TEXT)''')
        conn.commit()
        conn.close()

    def store_llr(self, analyses: List[Dict]):
        """Store LLR analyses with random embeddings."""
        num_analyses = len(analyses)
        dummy_embeddings = np.random.rand(num_analyses, 256)  # Random 256-dim embeddings
        dimension = dummy_embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(dummy_embeddings)

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        for i, analysis in enumerate(analyses):
            cursor.execute(
                "INSERT INTO llr_metadata (id, ticker, date, analysis) VALUES (?, ?, ?, ?)",
                (i, analysis.get("ticker", ""), analysis.get("date", ""), str(analysis))
            )
        conn.commit()
        conn.close()

        faiss.write_index(self.index, self.index_path)

    def retrieve_llr(self, query_embedding: np.ndarray, k: int = 1) -> List[Dict]:
        """Retrieve LLR analyses based on query embedding."""
        if self.index is None:
            raise ValueError("No LLR index stored.")
        D, I = self.index.search(query_embedding, k)
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        results = []
        for idx in I.flatten():
            cursor.execute("SELECT ticker, date, analysis FROM llr_metadata WHERE id = ?", (idx,))
            ticker, date, analysis = cursor.fetchone()
            results.append({"ticker": ticker, "date": date, "analysis": eval(analysis)})
        conn.close()
        return results