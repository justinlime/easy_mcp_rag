import logging
from pathlib import Path
from typing import List, Dict, Any
import pypdf
import docx
import pandas as pd

logger = logging.getLogger("easy_mcp_rag.document_loader")

class DocumentLoader:
    SUPPORTED_EXTENSIONS = {
        ".txt", ".md", ".py", ".js", ".json", ".xml", ".html", ".css",
        ".pdf", ".docx", ".doc", ".csv", ".xlsx", ".xls"
    }
    
    def __init__(self):
        pass
    
    def load_directory(self, directory: Path) -> List[Dict[str, Any]]:
        documents = []
        
        for file_path in directory.rglob("*"):
            if file_path.is_file() and file_path.suffix.lower() in self.SUPPORTED_EXTENSIONS:
                try:
                    content = self.load_file(file_path)
                    if content:
                        documents.append({
                            "content": content,
                            "metadata": {
                                "source": str(file_path),
                                "filename": file_path.name,
                                "extension": file_path.suffix
                            }
                        })
                        logger.info(f"Loaded: {file_path}")
                except Exception as e:
                    logger.error(f"Error loading {file_path}: {e}")
        
        return documents
    
    def load_file(self, file_path: Path) -> str:
        ext = file_path.suffix.lower()
        
        if ext in [".txt", ".md", ".py", ".js", ".json", ".xml", ".html", ".css"]:
            return self._load_text(file_path)
        elif ext == ".pdf":
            return self._load_pdf(file_path)
        elif ext in [".docx", ".doc"]:
            return self._load_docx(file_path)
        elif ext == ".csv":
            return self._load_csv(file_path)
        elif ext in [".xlsx", ".xls"]:
            return self._load_excel(file_path)
        
        return ""
    
    def _load_text(self, file_path: Path) -> str:
        try:
            return file_path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            return file_path.read_text(encoding="latin-1")
    
    def _load_pdf(self, file_path: Path) -> str:
        text = []
        with open(file_path, "rb") as f:
            pdf = pypdf.PdfReader(f)
            for page in pdf.pages:
                text.append(page.extract_text())
        return "\n".join(text)
    
    def _load_docx(self, file_path: Path) -> str:
        doc = docx.Document(file_path)
        return "\n".join([para.text for para in doc.paragraphs])
    
    def _load_csv(self, file_path: Path) -> str:
        df = pd.read_csv(file_path)
        return df.to_string()
    
    def _load_excel(self, file_path: Path) -> str:
        df = pd.read_excel(file_path)
        return df.to_string()
