import os 
import pickle
import hashlib 
from datetime import datetime,timedelta 
from pathlib import Path 
from typing import List

from docling.document_converter import DocumentConverter
from langchain_text_splitters import MarkdownHeaderTextSplitter
from langchain_core.documents import Document

from src.config.settings import settings 
from src.config.constants import *
from src.custom_logger.logger import logger

class DocumentProcessor:
    def __init__(self):
        self.headers = [("#","Header 1"),("##","Header 2")]
        self.cache_dir = Path(settings.CACHE_DIR)
        self.cache_dir.mkdir(parents=True,exist_ok=True)
    
    def validate_files(self,files: List) -> None: 
        total_size = sum(os.path.getsize(f) for f in files)
        if total_size > MAX_TOTAL_SIZE:
            raise ValueError(f"Total size exceeds {MAX_TOTAL_SIZE // 1024 // 1024} MB limit")
    
    def process(self,file_paths):
        """Convert a list of file paths -> deduplicated LangChain Doc chunks
        Caches results per file hash so repeated queries on unchanged docs are instant"""

        self.validate_files(file_paths)
        
        all_chunks: List[Document] = []
        seen_hashes: set = set()

        for path in file_paths: 
            try:
                with open(path,"rb") as f:
                    file_hash = self._generate_hash(f.read())
                
                cache_path = self.cache_dir / f"{file_hash}.pkl"

                if self._is_cache_valid(cache_path):
                    logger.info(f"Loading from cache: {path}")
                    chunks = self._load_from_cache(cache_path)
                
                else:
                    logger.info(f"Processing and caching: {path}")
                    chunks = self._process_file(path)
                    self._save_to_cache(chunks,cache_path)

                for chunk in chunks: 
                    chunk_hash = self._generate_hash(chunk.page_content.encode())
                    if chunk_hash not in seen_hashes:
                        all_chunks.append(chunk)
                        seen_hashes.add(chunk_hash)

            except Exception as e:
                logger.error(f"Failed to process {path}:{e}")
                continue

        logger.info(f"Total unique chunks: {len(all_chunks)}")
        
        return all_chunks 
    
    def _process_file(self, path: str) -> List[Document]:
        """Run Docling -> Markdown -> MarkdownHeaderTextSplitter."""
        ext = Path(path).suffix.lower()
        if ext not in (".pdf", ".docx", ".txt", ".md"):
            logger.warning(f"Skipping unsupported file type: {path}")
            return []

        converter = DocumentConverter()
        # converter converts files to structured amrkdown
        markdown = converter.convert(path).document.export_to_markdown()
        splitter = MarkdownHeaderTextSplitter(self.headers)
        return splitter.split_text(markdown)

    @staticmethod
    def _generate_hash(content: bytes) -> str:
        # creates a cryptographic hash of content 
        return hashlib.sha256(content).hexdigest()
    
    def _is_cache_valid(self,cache_path: Path) -> bool:
        if not cache_path.exists():
            return False
        age = datetime.now() - datetime.fromtimestamp(cache_path.stat().st_mtime)
        return age < timedelta(days=settings.CACHE_EXPIRE_DAYS) 
    
    def _load_from_cache(self,cache_path: Path) -> List[Document]:
        with open(cache_path,"rb") as f:
            return pickle.load(f)['chunks']
        
    def _save_to_cache(self,chunks: List[Document],cache_path: Path) -> None:
        with open(cache_path,"wb") as f:
            pickle.dump({"timestamp":datetime.now().timestamp(),
                         "chunks":chunks},f)
