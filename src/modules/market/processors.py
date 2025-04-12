import pandas as pd
import numpy as np
from typing import List, Dict, Any, Tuple

class MarketProcessor:
    """Process market data for storage."""

    def process_numerical(self, price_data: pd.DataFrame) -> pd.DataFrame:
        """Process numerical data."""
        return price_data

    def process_textual(self, news_data: List[Dict]) -> Tuple[List[str], List[Dict]]:
        """Process textual data."""
        texts = [item["text"] for item in news_data]
        return texts, news_data

    def process_visual(self, images: List[Any]) -> Tuple[np.ndarray, List[Dict]]:
        """Process visual data (dummy)."""
        num_images = len(images) or 1  # Đảm bảo ít nhất 1 để tránh lỗi
        embeddings = np.random.rand(num_images, 512)  # Random 512-dim embeddings
        metadata = [
            {"ticker": "UNKNOWN", "date": "2025-04-11", "type": "kline"}
            for _ in range(num_images)
        ]
        return embeddings, metadata