import logging
from pathlib import Path
from typing import Dict, Any, Optional, List, Iterator
import PyPDF2
import docx
import pandas as pd

logger = logging.getLogger("easy_mcp_rag.document_loader")


class DocumentLoader:
    """Load documents from various file formats."""
    
    def __init__(self):
        self.supported_extensions = {
            '.pdf', '.docx', '.doc', '.txt', '.md', '.py', '.js',
            '.json', '.xml', '.html', '.css', '.csv', '.xlsx', '.xls'
        }
    
    def load_single_file(self, file_path: Path) -> Optional[Dict[str, Any]]:
        """
        Load a single file and return its content with metadata.
        Returns None if file cannot be loaded.
        
        This is the KEY method for memory efficiency - load ONE file at a time.
        """
        if not file_path.exists() or not file_path.is_file():
            logger.warning(f"File not found or not a file: {file_path}")
            return None
        
        suffix = file_path.suffix.lower()
        
        if suffix not in self.supported_extensions:
            logger.warning(f"Unsupported file type: {suffix}")
            return None
        
        try:
            content = self._extract_text(file_path, suffix)
            
            if not content or not content.strip():
                logger.warning(f"No content extracted from {file_path.name}")
                return None
            
            return {
                "content": content,
                "metadata": {
                    "source": str(file_path),
                    "filename": file_path.name,
                    "file_type": suffix,
                }
            }
        except Exception as e:
            logger.error(f"Error loading {file_path.name}: {e}")
            return None
    
    def _extract_text(self, file_path: Path, suffix: str) -> str:
        """Extract text based on file type."""
        if suffix == '.pdf':
            return self._extract_pdf(file_path)
        elif suffix in ['.docx', '.doc']:
            return self._extract_docx(file_path)
        elif suffix in ['.txt', '.md', '.py', '.js', '.json', '.xml', '.html', '.css']:
            return self._extract_plain_text(file_path)
        elif suffix in ['.csv', '.xlsx', '.xls']:
            return self._extract_spreadsheet(file_path)
        else:
            return ""
    
    def _extract_pdf(self, file_path: Path) -> str:
        """Extract text from PDF."""
        text_parts = []
        with open(file_path, 'rb') as f:
            reader = PyPDF2.PdfReader(f)
            for page in reader.pages:
                text = page.extract_text()
                if text:
                    text_parts.append(text)
        return "\n".join(text_parts)
    
    def _extract_docx(self, file_path: Path) -> str:
        """Extract text from DOCX."""
        doc = docx.Document(file_path)
        return "\n".join([para.text for para in doc.paragraphs if para.text.strip()])
    
    def _extract_plain_text(self, file_path: Path) -> str:
        """Extract plain text."""
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            return f.read()
    
    def _extract_spreadsheet(self, file_path: Path) -> str:
        """Extract text from spreadsheet."""
        try:
            if file_path.suffix.lower() == '.csv':
                df = pd.read_csv(file_path)
            else:
                df = pd.read_excel(file_path)
            return df.to_string()
        except Exception as e:
            logger.error(f"Error reading spreadsheet {file_path.name}: {e}")
            return ""
    
    def load_directory(self, directory: Path) -> Iterator[Dict[str, Any]]:
        """
        DEPRECATED but kept for backwards compatibility.
        
        WARNING: This returns an iterator now, not a list!
        If you iterate over this, it yields documents one at a time.
        
        BETTER APPROACH: Use load_single_file() directly in your server code
        so you control exactly when each file is loaded and freed.
        """
        logger.warning("load_directory() is deprecated. Use load_single_file() for better memory control.")
        
        for file_path in directory.rglob('*'):
            if file_path.is_file() and file_path.suffix.lower() in self.supported_extensions:
                doc = self.load_single_file(file_path)
                if doc:
                    yield doc
    
    def load_directory_list(self, directory: Path) -> List[Dict[str, Any]]:
        """
        Load all documents from directory into a list.
        
        WARNING: This loads everything into memory at once!
        Only use this for small directories or testing.
        For production, use load_single_file() in a loop instead.
        """
        return list(self.load_directory(directory))
