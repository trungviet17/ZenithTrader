import sqlite3
import pandas as pd
import faiss
import numpy as np
from typing import List, Dict, Any, Optional
from sortedcontainers import SortedList

class MIMemory:
    """Memory for storing and retrieving Market Intelligence data."""

    def __init__(self, db_path: str = "src/database/mi_memory.db", 
                 textual_index_path: str = "src/database/mi_textual_embeddings.faiss",
                 visual_index_path: str = "src/database/mi_visual_embeddings.faiss"):
        self.db_path = db_path
        self.textual_index_path = textual_index_path
        self.visual_index_path = visual_index_path
        self.numerical_data: Optional[pd.DataFrame] = None
        self.textual_index: Optional[faiss.IndexFlatL2] = None
        self.textual_metadata = SortedList(key=lambda x: x["date"])
        self.visual_index: Optional[faiss.IndexFlatL2] = None
        self._setup_sqlite()

    def _setup_sqlite(self):
        """Set up SQLite table for visual metadata."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''CREATE TABLE IF NOT EXISTS mi_visual_metadata
                         (ticker TEXT, date TEXT, embedding_id INTEGER)''')
        conn.commit()
        conn.close()

    def store_numerical(self, df: pd.DataFrame):
        """Store numerical data in pandas DataFrame."""
        self.numerical_data = df

    def store_textual(self, texts: List[str], metadata_list: List[Dict]):
        """Store textual data with random embeddings in FAISS."""
        num_texts = len(texts)
        dummy_embeddings = np.random.rand(num_texts, 384)  # Random 384-dim embeddings
        dimension = dummy_embeddings.shape[1]
        self.textual_index = faiss.IndexFlatL2(dimension)
        self.textual_index.add(dummy_embeddings)

        for i, meta in enumerate(metadata_list):
            meta["embedding_id"] = i
            self.textual_metadata.add(meta)

        faiss.write_index(self.textual_index, self.textual_index_path)

    def store_visual(self, embeddings: np.ndarray, metadata_list: List[Dict]):
        """Store visual data with random embeddings in FAISS."""
        dimension = embeddings.shape[1]
        self.visual_index = faiss.IndexFlatL2(dimension)
        self.visual_index.add(embeddings)

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        for ticker, date, embedding_id in metadata_list:
            cursor.execute("INSERT INTO mi_visual_metadata VALUES (?, ?, ?)", 
                          (ticker, date, embedding_id))
        conn.commit()
        conn.close()

        faiss.write_index(self.visual_index, self.visual_index_path)

    def retrieve_numerical(self) -> Optional[pd.DataFrame]:
        """Retrieve numerical data."""
        if self.numerical_data is None:
            raise ValueError("No numerical data stored.")
        return self.numerical_data

    def retrieve_textual(self, query_embedding: np.ndarray, k: int = 1) -> List[Dict]:
        """Retrieve textual data based on query embedding."""
        if self.textual_index is None:
            raise ValueError("No textual index stored.")
        D, I = self.textual_index.search(query_embedding, k)
        return [self.textual_metadata[i] for i in I.flatten()]

    def retrieve_visual(self, query_embedding: np.ndarray, k: int = 1) -> List[Dict]:
        """Retrieve visual metadata based on query embedding."""
        if self.visual_index is None:
            raise ValueError("No visual index stored.")
        D, I = self.visual_index.search(query_embedding, k)
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        results = []
        for idx in I.flatten():
            cursor.execute("SELECT ticker, date FROM mi_visual_metadata WHERE embedding_id = ?", (idx,))
            ticker, date = cursor.fetchone()
            results.append({"ticker": ticker, "date": date})
        conn.close()
        return results