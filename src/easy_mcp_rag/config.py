import argparse
from dataclasses import dataclass
from typing import Optional

@dataclass
class Config:
    data_dir: str
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333
    qdrant_collection_prefix: str = "rag"
    embedding_model: str = "all-MiniLM-L6-v2"
    chunk_size: int = 512
    chunk_overlap: int = 50
    top_k: int = 5
    verbose: bool = False
    log_level: str = "INFO"
    batch_size: int = 32
    force_reindex: bool = False
    device: str = "auto"
    transport: str = "stdio"
    http_host: str = "0.0.0.0"
    http_port: int = 8000

    @classmethod
    def from_args(cls, args: Optional[list] = None) -> "Config":
        parser = argparse.ArgumentParser(
            description="MCP RAG Server with Qdrant - UV/UVX Compatible"
        )
        parser.add_argument(
            "--data-dir",
            required=True,
            help="Directory containing subdirectories with documents"
        )
        parser.add_argument(
            "--qdrant-host",
            default="localhost",
            help="Qdrant server host (default: localhost)"
        )
        parser.add_argument(
            "--qdrant-port",
            type=int,
            default=6333,
            help="Qdrant server port (default: 6333)"
        )
        parser.add_argument(
            "--qdrant-collection-prefix",
            default="rag",
            help="Prefix for Qdrant collection names (default: rag)"
        )
        parser.add_argument(
            "--embedding-model",
            default="all-MiniLM-L6-v2",
            help="Sentence transformer model (default: all-MiniLM-L6-v2)"
        )
        parser.add_argument(
            "--chunk-size",
            type=int,
            default=512,
            help="Text chunk size in characters (default: 512)"
        )
        parser.add_argument(
            "--chunk-overlap",
            type=int,
            default=50,
            help="Overlap between chunks (default: 50)"
        )
        parser.add_argument(
            "--top-k",
            type=int,
            default=5,
            help="Number of results to return per search (default: 5)"
        )
        parser.add_argument(
            "--verbose",
            action="store_true",
            help="Enable verbose logging"
        )
        parser.add_argument(
            "--log-level",
            default="INFO",
            choices=["DEBUG", "INFO", "WARNING", "ERROR"],
            help="Logging level (default: INFO)"
        )
        parser.add_argument(
            "--batch-size",
            type=int,
            default=32,
            help="Batch size for embedding generation (default: 32)"
        )
        parser.add_argument(
            "--force-reindex",
            action="store_true",
            help="Force reindexing of all documents"
        )
        parser.add_argument(
            "--device",
            default="auto",
            choices=["auto", "cpu", "cuda", "mps"],
            help="Device for embeddings: auto, cpu, cuda (GPU), or mps (Apple Silicon)"
        )
        parser.add_argument(
            "--transport",
            default="stdio",
            choices=["stdio", "http"],
            help="MCP transport type: stdio or http (default: stdio)"
        )
        parser.add_argument(
            "--http-host",
            default="0.0.0.0",
            help="HTTP server host (only for http transport, default: 0.0.0.0)"
        )
        parser.add_argument(
            "--http-port",
            type=int,
            default=8000,
            help="HTTP server port (only for http transport, default: 8000)"
        )
        
        parsed_args = parser.parse_args(args)
        return cls(**vars(parsed_args))
