from memory.mi_memory import MIMemory
from memory.llr_memory import LLRMemory
from memory.hlr_memory import HLRMemory
import pandas as pd
import numpy as np
from typing import List, Dict, Any

class MemoryManager:
    """Manage MI, LLR, and HLR memories."""

    def __init__(self):
        self.mi_memory = MIMemory()
        self.llr_memory = LLRMemory()
        self.hlr_memory = HLRMemory()

    # MI Memory methods
    def store_mi_numerical(self, df: pd.DataFrame):
        self.mi_memory.store_numerical(df)

    def store_mi_textual(self, texts: List[str], metadata_list: List[Dict]):
        self.mi_memory.store_textual(texts, metadata_list)

    def store_mi_visual(self, embeddings: np.ndarray, metadata_list: List[Dict]):
        self.mi_memory.store_visual(embeddings, metadata_list)

    def retrieve_mi_numerical(self):
        return self.mi_memory.retrieve_numerical()

    def retrieve_mi_textual(self, query_embedding: np.ndarray, k: int = 1) -> List[Dict]:
        return self.mi_memory.retrieve_textual(query_embedding, k)

    def retrieve_mi_visual(self, query_embedding: np.ndarray, k: int = 1) -> List[Dict]:
        return self.mi_memory.retrieve_visual(query_embedding, k)

    # LLR Memory methods
    def store_llr(self, analyses: List[Dict]):
        self.llr_memory.store_llr(analyses)

    def retrieve_llr(self, query_embedding: np.ndarray, k: int = 1) -> List[Dict]:
        return self.llr_memory.retrieve_llr(query_embedding, k)

    # HLR Memory methods
    def store_hlr(self, reflections: List[Dict]):
        self.hlr_memory.store_hlr(reflections)

    def retrieve_hlr(self, query_embedding: np.ndarray, k: int = 1) -> List[Dict]:
        return self.hlr_memory.retrieve_hlr(query_embedding, k)