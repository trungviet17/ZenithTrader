import sqlite3
import faiss
import numpy as np
from typing import List, Dict, Any

class LLRMemory:
    """Memory structure for storing and retrieving Low-level Reflection data with dummy embeddings."""

    def __init__(self, db_path: str = "src/database/llr_reflection.db", 
                 index_path: str = "src/database/llr_embeddings.faiss"):
        """Initialize storage configurations for LLR Memory."""
        self.db_path = db_path
        self.index_path = index_path
        self.index: faiss.IndexFlatL2 = None
        self._setup_sqlite()

    def _setup_sqlite(self):
        """Set up SQLite table for LLR metadata storage."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''CREATE TABLE IF NOT EXISTS llr_metadata
                         (embedding_id INTEGER PRIMARY KEY AUTOINCREMENT, 
                          ticker TEXT, date TEXT, pattern TEXT, insight TEXT, 
                          xml_output TEXT, queries TEXT)''')
        conn.commit()
        conn.close()

    def store_llr(self, analyses: List[Dict]):
        """Store LLR data with dummy embeddings in FAISS and metadata in SQLite."""
        num_analyses = len(analyses)
        if num_analyses == 0:
            return
        
        # Tạo embeddings giả
        dummy_embeddings = np.random.rand(num_analyses, 256)  # Random 256-dim embeddings
        dimension = dummy_embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(dummy_embeddings)

        # Lưu metadata vào SQLite
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Lấy embedding_id lớn nhất hiện có
        cursor.execute("SELECT MAX(embedding_id) FROM llr_metadata")
        max_id = cursor.fetchone()[0]
        start_id = (max_id + 1) if max_id is not None else 0

        for i, analysis in enumerate(analyses):
            cursor.execute(
                "INSERT INTO llr_metadata (ticker, date, pattern, insight, xml_output, queries) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (
                    analysis.get("ticker", ""),
                    analysis.get("date", ""),
                    analysis.get("pattern", ""),
                    analysis.get("insight", ""),
                    analysis.get("xml_output", ""),
                    str(analysis.get("queries", {}))
                )
            )
        conn.commit()
        conn.close()

        faiss.write_index(self.index, self.index_path)

    def retrieve_llr(self, query_embedding: np.ndarray, k: int = 1) -> List[Dict]:
        """Retrieve LLR metadata based on query embedding."""
        if self.index is None:
            raise ValueError("No LLR index stored.")
        D, I = self.index.search(query_embedding, k)
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        results = []
        for idx in I.flatten():
            cursor.execute(
                "SELECT ticker, date, pattern, insight, xml_output, queries "
                "FROM llr_metadata WHERE embedding_id = ?",
                (idx,)
            )
            row = cursor.fetchone()
            if row:
                ticker, date, pattern, insight, xml_output, queries = row
                results.append({
                    "ticker": ticker,
                    "date": date,
                    "pattern": pattern,
                    "insight": insight,
                    "xml_output": xml_output,
                    "queries": eval(queries) if queries else {}
                })
            else:
                results.append({
                    "ticker": "Unknown",
                    "date": "N/A",
                    "pattern": "N/A",
                    "insight": f"No data for embedding_id {idx}",
                    "xml_output": "",
                    "queries": {}
                })
        conn.close()
        return results