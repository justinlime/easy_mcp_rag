import logging
import gc
from pathlib import Path
from typing import List, Dict, Any
from mcp.server import Server
from mcp.types import Tool, TextContent
from .config import Config
from .document_loader import DocumentLoader
from .vectorstore import VectorStore
from .utils import setup_logging, chunk_text, sanitize_collection_name, get_device

logger = logging.getLogger("easy_mcp_rag.server")

class MCPRAGServer:
    def __init__(self, config: Config):
        self.config = config
        self.logger = setup_logging(config.log_level, config.verbose)
        
        # Determine device
        device = get_device(config.device)
        self.logger.info(f"Using device: {device}")
        
        self.server = Server("rag-server")
        self.loader = DocumentLoader()
        self.vectorstore = VectorStore(
            config.qdrant_host,
            config.qdrant_port,
            config.embedding_model,
            device
        )
        
        self.collections: Dict[str, str] = {}
        self._setup_handlers()
    
    def _setup_handlers(self):
        @self.server.list_tools()
        async def list_tools() -> List[Tool]:
            return [
                Tool(
                    name=tool_name,
                    description=f"Search documents in the '{subdir}' collection",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "Search query"
                            },
                            "top_k": {
                                "type": "number",
                                "description": f"Number of results (default: {self.config.top_k})"
                            }
                        },
                        "required": ["query"]
                    }
                )
                for tool_name, subdir in self.collections.items()
            ]
        
        @self.server.call_tool()
        async def call_tool(name: str, arguments: dict) -> List[TextContent]:
            if name not in self.collections:
                return [TextContent(type="text", text=f"Tool '{name}' not found")]
            
            query = arguments.get("query", "")
            top_k = arguments.get("top_k", self.config.top_k)
            
            self.logger.info(f"Tool accessed: {name}")
            self.logger.info(f"Query: {query}")
            
            collection_name = self.collections[name]
            results = self.vectorstore.search(collection_name, query, int(top_k))
            
            self.logger.info(f"Results returned: {len(results)}")
            
            if self.config.verbose:
                for i, result in enumerate(results, 1):
                    self.logger.debug(f"Result {i}: score={result['score']:.4f}, "
                                    f"source={result['metadata'].get('source', 'unknown')}")
            
            response_text = self._format_results(results, query)
            return [TextContent(type="text", text=response_text)]
    
    def _format_results(self, results: List[Dict[str, Any]], query: str) -> str:
        if not results:
            return f"No results found for query: '{query}'"
        
        lines = [f"Found {len(results)} results for query: '{query}'\n"]
        
        for i, result in enumerate(results, 1):
            lines.append(f"Result {i} (score: {result['score']:.4f}):")
            lines.append(f"Source: {result['metadata'].get('source', 'unknown')}")
            lines.append(f"Content: {result['content'][:500]}...")
            lines.append("")
        
        return "\n".join(lines)
    
    def initialize(self):
        data_dir = Path(self.config.data_dir)
        
        if not data_dir.exists():
            raise ValueError(f"Data directory does not exist: {data_dir}")
        
        subdirs = [d for d in data_dir.iterdir() if d.is_dir()]
        
        if not subdirs:
            raise ValueError(f"No subdirectories found in: {data_dir}")
        
        self.logger.info(f"Found {len(subdirs)} subdirectories to process")
        
        for subdir in subdirs:
            self._process_subdirectory(subdir)
            # Force garbage collection after each subdirectory
            gc.collect()
        
        self.logger.info(f"Initialized {len(self.collections)} RAG collections")
    
    def _process_subdirectory(self, subdir: Path):
        """
        FIXED: Process subdirectory with streaming to avoid memory accumulation.
        """
        subdir_name = subdir.name
        tool_name = f"{subdir_name}_search"
        collection_name = f"{self.config.qdrant_collection_prefix}_{sanitize_collection_name(subdir_name)}"

        self.logger.info(f"Processing subdirectory: {subdir_name}")

        # Create collection first (respects force_reindex inside VectorStore impl)
        self.vectorstore.create_collection(collection_name, self.config.force_reindex)

        # Get list of files to process (just paths, not content)
        supported_extensions = {
            '.pdf', '.docx', '.doc', '.txt', '.md', '.py', '.js',
            '.json', '.xml', '.html', '.css', '.csv', '.xlsx', '.xls'
        }
        
        files = [
            f for f in subdir.rglob('*')
            if f.is_file() and f.suffix.lower() in supported_extensions
        ]
        
        if not files:
            self.logger.warning(f"No supported documents found in {subdir_name}")
            return

        self.logger.info(f"Found {len(files)} files to process in {subdir_name}")

        batch: List[Dict[str, Any]] = []
        total_chunks = 0
        total_docs = 0

        # CRITICAL FIX: Process ONE file at a time, not all at once
        for file_idx, file_path in enumerate(files, 1):
            try:
                self.logger.debug(f"Processing file {file_idx}/{len(files)}: {file_path.name}")
                
                # Load ONLY this single file
                doc = self.loader.load_single_file(file_path)
                
                if not doc:
                    self.logger.warning(f"Could not load {file_path.name}")
                    continue
                
                total_docs += 1
                content = doc.get("content", "")
                metadata = doc.get("metadata", {})
                
                # Chunk this single document
                chunks = chunk_text(content, self.config.chunk_size, self.config.chunk_overlap)
                
                # Clear the original content immediately after chunking
                del content
                del doc
                
                # Process chunks from this file
                for chunk in chunks:
                    total_chunks += 1
                    batch.append({
                        "content": chunk,
                        "metadata": metadata
                    })

                    # Index when batch is full
                    if len(batch) >= max(1, int(self.config.batch_size)):
                        self._index_batch(collection_name, batch, total_chunks)
                        # Clear batch after indexing
                        del batch[:]
                        gc.collect()
                
                # Clear chunks list after processing this file
                del chunks
                
                # Periodic garbage collection every 5 files
                if file_idx % 5 == 0:
                    gc.collect()
                    self.logger.debug(f"Progress: {file_idx}/{len(files)} files, {total_chunks} chunks")
                
            except Exception as e:
                self.logger.error(f"Error processing {file_path.name}: {e}")
                # Continue with next file instead of crashing
                continue

        # Index any remaining chunks
        if batch:
            self._index_batch(collection_name, batch, total_chunks)
            del batch
            gc.collect()

        self.collections[tool_name] = collection_name
        self.logger.info(f"Created tool: {tool_name} -> collection: {collection_name}")
        self.logger.info(f"Processed {total_docs} documents into {total_chunks} chunks for collection {collection_name}")
    
    def _index_batch(self, collection_name: str, batch: List[Dict[str, Any]], total_chunks: int):
        """Helper method to index a batch and handle errors."""
        try:
            self.logger.debug(f"Indexing batch of {len(batch)} chunks (total={total_chunks})")
            self.vectorstore.index_documents(collection_name, batch, self.config.batch_size)
        except Exception as e:
            self.logger.exception(f"Error indexing batch: {e}")
            raise
    
    def run(self):
        import asyncio
        
        if self.config.transport == "stdio":
            self._run_stdio()
        elif self.config.transport == "http":
            self._run_http()
    
    def _run_stdio(self):
        import asyncio
        from mcp.server.stdio import stdio_server
        
        async def main():
            async with stdio_server() as (read_stream, write_stream):
                await self.server.run(
                    read_stream,
                    write_stream,
                    self.server.create_initialization_options()
                )
        
        asyncio.run(main())
    
    def _run_http(self):
        import uvicorn
        from starlette.applications import Starlette
        from starlette.routing import Route
        from mcp.server.sse import SseServerTransport
        
        self.logger.info(f"Starting HTTP server on {self.config.http_host}:{self.config.http_port}")
        
        async def handle_sse(request):
            async with SseServerTransport("/messages") as transport:
                await self.server.run(
                    transport.read_stream,
                    transport.write_stream,
                    self.server.create_initialization_options()
                )
        
        app = Starlette(
            routes=[
                Route("/sse", endpoint=handle_sse),
            ]
        )
        
        uvicorn.run(
            app,
            host=self.config.http_host,
            port=self.config.http_port,
            log_level="info"
        )
