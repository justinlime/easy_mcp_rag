# Easy MCP RAG 🚀

A high-performance Model Context Protocol (MCP) server for RAG using Qdrant. Built for UV/UVX with CPU/GPU support and HTTP transport.

## ✨ Features

- 🔍 **Automatic Document Indexing** - Scan directories and index all documents
- 📁 **Smart Organization** - Each subdirectory becomes its own searchable dataset
- 🛠️ **Dynamic MCP Tools** - Auto-generated tools for each collection
- 📄 **Multi-Format Support** - PDF, DOCX, CSV, XLSX, TXT, Markdown, and more
- ⚡ **GPU Acceleration** - Optional CUDA/MPS support for faster embeddings
- 🌐 **HTTP Transport** - Run as HTTP server or stdio
- 📦 **UV/UVX Ready** - Install and run with a single command
- 📊 **Verbose Logging** - Detailed query tracking and monitoring

## 🚀 Quick Start

### Install with UVX (Recommended)

Run directly from GitHub without installation:

```bash
uvx --from git+https://github.com/yourusername/easy_mcp_rag.git easy_mcp_rag --data-dir ./documents
```

### Install with UV

```bash
# Install from GitHub
uv pip install git+https://github.com/yourusername/easy_mcp_rag.git

# Or clone and install locally
git clone https://github.com/yourusername/easy_mcp_rag.git
cd easy_mcp_rag
uv pip install -e .
```

## 📋 Prerequisites

1. **Start Qdrant** (using Docker):
```bash
docker run -p 6333:6333 qdrant/qdrant
```

2. **Prepare your documents**:
```
documents/
├── legal_docs/
│   ├── contract.pdf
│   └── terms.docx
├── research/
│   ├── paper1.pdf
│   └── notes.txt
└── data/
    └── analysis.csv
```

## 💻 Usage

### Basic Usage (stdio)

```bash
# With UVX
uvx --from git+https://github.com/yourusername/easy_mcp_rag.git easy_mcp_rag --data-dir ./documents

# With UV
uv run easy_mcp_rag --data-dir ./documents

# After installation
easy_mcp_rag --data-dir ./documents
```

### HTTP Mode

```bash
easy_mcp_rag --data-dir ./documents --transport http --http-port 8000
```

### GPU Acceleration

```bash
# Auto-detect GPU
easy_mcp_rag --data-dir ./documents --device auto

# Force CUDA (NVIDIA GPU)
easy_mcp_rag --data-dir ./documents --device cuda

# Force MPS (Apple Silicon)
easy_mcp_rag --data-dir ./documents --device mps

# Force CPU
easy_mcp_rag --data-dir ./documents --device cpu
```

### Advanced Configuration

```bash
easy_mcp_rag \
  --data-dir ./documents \
  --qdrant-host localhost \
  --qdrant-port 6333 \
  --device cuda \
  --embedding-model all-mpnet-base-v2 \
  --chunk-size 1024 \
  --chunk-overlap 100 \
  --top-k 10 \
  --batch-size 64 \
  --verbose \
  --force-reindex
```

## 🔧 Configuration Options

| Flag | Description | Default |
|------|-------------|---------|
| `--data-dir` | Directory with document subdirectories | **Required** |
| `--qdrant-host` | Qdrant server host | `localhost` |
| `--qdrant-port` | Qdrant server port | `6333` |
| `--device` | Device: auto, cpu, cuda, mps | `auto` |
| `--transport` | Transport type: stdio, http | `stdio` |
| `--http-host` | HTTP server host | `0.0.0.0` |
| `--http-port` | HTTP server port | `8000` |
| `--embedding-model` | Sentence transformer model | `all-MiniLM-L6-v2` |
| `--chunk-size` | Text chunk size (chars) | `512` |
| `--chunk-overlap` | Chunk overlap (chars) | `50` |
| `--top-k` | Results per search | `5` |
| `--batch-size` | Embedding batch size | `32` |
| `--verbose` | Enable verbose logging | `False` |
| `--log-level` | Log level | `INFO` |
| `--force-reindex` | Force reindex all docs | `False` |

## 🎯 MCP Client Configuration

### Claude Desktop / Cline / Other MCP Clients

Add to your MCP client config:

```json
{
  "mcpServers": {
    "rag-server": {
      "command": "uvx",
      "args": [
        "--from",
        "git+https://github.com/yourusername/easy_mcp_rag.git",
        "easy_mcp_rag",
        "--data-dir",
        "/path/to/your/documents",
        "--device",
        "auto",
        "--verbose"
      ]
    }
  }
}
```

### With HTTP Transport

```json
{
  "mcpServers": {
    "rag-server": {
      "command": "uvx",
      "args": [
        "--from",
        "git+https://github.com/yourusername/easy_mcp_rag.git",
        "easy_mcp_rag",
        "--data-dir",
        "/path/to/your/documents",
        "--transport",
        "http",
        "--http-port",
        "8000"
      ]
    }
  }
}
```

## 🛠️ How It Works

1. **Scan** - Discovers all subdirectories in your data directory
2. **Load** - Extracts text from all supported file types
3. **Chunk** - Splits documents into overlapping chunks
4. **Embed** - Generates vector embeddings (CPU or GPU)
5. **Index** - Stores in Qdrant (one collection per subdirectory)
6. **Serve** - Creates MCP tools for each collection

### Example

```
documents/
├── legal_docs/      → Creates "legal_docs_search" tool
├── research/        → Creates "research_search" tool
└── data/            → Creates "data_search" tool
```

## 📄 Supported File Types

| Category | Extensions |
|----------|-----------|
| Text | `.txt`, `.md`, `.py`, `.js`, `.json`, `.xml`, `.html`, `.css` |
| PDF | `.pdf` |
| Word | `.docx`, `.doc` |
| Spreadsheet | `.csv`, `.xlsx`, `.xls` |

## 🎨 Embedding Models

Choose based on your needs:

| Model | Dimensions | Speed | Quality | Use Case |
|-------|-----------|-------|---------|----------|
| `all-MiniLM-L6-v2` | 384 | ⚡⚡⚡ | Good | Default, fast |
| `all-MiniLM-L12-v2` | 384 | ⚡⚡ | Better | Balanced |
| `all-mpnet-base-v2` | 768 | ⚡ | Best | Quality |

## 🐛 Troubleshooting

### Qdrant Connection Failed
```bash
# Check if Qdrant is running
curl http://localhost:6333

# Start Qdrant
docker run -p 6333:6333 qdrant/qdrant
```

### GPU Not Detected
```bash
# Check PyTorch GPU support
python -c "import torch; print(torch.cuda.is_available())"

# Install with GPU support
uv pip install -e ".[gpu]"
```

### Out of Memory
```bash
# Use smaller model
--embedding-model all-MiniLM-L6-v2

# Reduce batch size
--batch-size 16

# Use CPU
--device cpu
```

## 📊 Logging

Enable verbose logging to see detailed information:

```bash
easy_mcp_rag --data-dir ./documents --verbose
```

Output includes:
- ✅ Tool access events
- 🔍 Query details
- 📈 Result counts
- 🎯 Relevance scores
- 📁 Source files

Example:
```
2024-01-20 10:30:15 - easy_mcp_rag.server - INFO - Tool accessed: legal_docs_search
2024-01-20 10:30:15 - easy_mcp_rag.server - INFO - Query: contract terms
2024-01-20 10:30:15 - easy_mcp_rag.server - INFO - Results returned: 5
2024-01-20 10:30:15 - easy_mcp_rag.server - DEBUG - Result 1: score=0.8542
```

## 🔐 Security Notes

- HTTP mode exposes the server on the network
- Use `--http-host 127.0.0.1` for local-only access
- Consider authentication for production deployments

## 📝 Development

```bash
# Clone repository
git clone https://github.com/yourusername/easy_mcp_rag.git
cd easy_mcp_rag

# Install with dev dependencies
uv pip install -e ".[dev]"

# Run tests
pytest

# Format code
black src/

# Lint
ruff src/
```

## 🤝 Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## 📜 License

MIT License - see LICENSE file

## 🙏 Credits

Built with:
- [MCP](https://github.com/anthropics/mcp) - Model Context Protocol
- [Qdrant](https://qdrant.tech/) - Vector database
- [Sentence Transformers](https://www.sbert.net/) - Embeddings
- [UV](https://github.com/astral-sh/uv) - Package manager
