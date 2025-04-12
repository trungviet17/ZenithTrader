import sqlite3
import faiss
import numpy as np
from typing import List, Dict, Any

class HLRMemory:
    """Memory for storing and retrieving High-Level Reflection data."""

    def __init__(self, db_path: str = "src/database/hlr_reflection.db", 
                 index_path: str = "src/database/hlr_embeddings.faiss"):
        self.db_path = db_path
        self.index_path = index_path
        self.index: faiss.IndexFlatL2 = None
        self._setup_sqlite()

    def _setup_sqlite(self):
        """Set up SQLite table for HLR metadata."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''CREATE TABLE IF NOT EXISTS hlr_metadata
                         (id INTEGER PRIMARY KEY, date TEXT, lesson TEXT, outcome TEXT)''')
        conn.commit()
        conn.close()

    def store_hlr(self, reflections: List[Dict]):
        """Store HLR reflections with random embeddings."""
        num_reflections = len(reflections)
        dummy_embeddings = np.random.rand(num_reflections, 512)  # Random 512-dim embeddings
        dimension = dummy_embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(dummy_embeddings)

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        for i, reflection in enumerate(reflections):
            cursor.execute(
                "INSERT INTO hlr_metadata (id, date, lesson, outcome) VALUES (?, ?, ?, ?)",
                (i, reflection.get("date", ""), str(reflection.get("lesson", "")), 
                 reflection.get("outcome", ""))
            )
        conn.commit()
        conn.close()

        faiss.write_index(self.index, self.index_path)

    def retrieve_hlr(self, query_embedding: np.ndarray, k: int = 1) -> List[Dict]:
        """Retrieve HLR reflections based on query embedding."""
        if self.index is None:
            raise ValueError("No HLR index stored.")
        D, I = self.index.search(query_embedding, k)
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        results = []
        for idx in I.flatten():
            cursor.execute("SELECT date, lesson, outcome FROM hlr_metadata WHERE id = ?", (idx,))
            date, lesson, outcome = cursor.fetchone()
            results.append({"date": date, "lesson": lesson, "outcome": outcome})
        conn.close()
        return results