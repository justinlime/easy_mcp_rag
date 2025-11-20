import logging
import torch
from typing import List

def setup_logging(level: str = "INFO", verbose: bool = False):
    log_level = logging.DEBUG if verbose else getattr(logging, level)
    
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    
    return logging.getLogger("easy_mcp_rag")

def get_device(device_choice: str) -> str:
    if device_choice == "auto":
        if torch.cuda.is_available():
            return "cuda"
        elif torch.backends.mps.is_available():
            return "mps"
        else:
            return "cpu"
    return device_choice

def chunk_text(text: str, chunk_size: int, overlap: int) -> List[str]:
    chunks = []
    start = 0
    text_len = len(text)
    
    while start < text_len:
        end = start + chunk_size
        chunk = text[start:end]
        
        if chunk.strip():
            chunks.append(chunk)
        
        start = end - overlap if end < text_len else text_len
        
    return chunks

def sanitize_collection_name(name: str) -> str:
    # Qdrant collection names must be alphanumeric with underscores
    return "".join(c if c.isalnum() else "_" for c in name).lower()
