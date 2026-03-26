import os
import pickle
import hashlib
import asyncio

from datetime import datetime, timedelta
from pathlib import Path
from typing import List

from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from src.config.settings import settings
from src.config.constants import *
from src.custom_logger.logger import logger


class DocumentProcessor:

    def __init__(self):

        self.cache_dir = Path(settings.CACHE_DIR)
        self.cache_dir.mkdir(parents=True,exist_ok=True)



    def validate_files(self,files: List[str]) -> None:
        total_size = sum(os.path.getsize(f) for f in files)

        if total_size > MAX_TOTAL_SIZE:
            raise ValueError("Total size exceeded")

        for f in files:
            if not os.path.exists(f):
                raise FileNotFoundError(f)

            if os.path.getsize(f) > MAX_FILE_SIZE:
                raise ValueError(f"{f} too large")


    async def process(self,file_paths: List[str]) -> List[Document]:

        self.validate_files(file_paths)

        tasks = [self._process_single_file(path)
                for path in file_paths]

        results = await asyncio.gather(*tasks,return_exceptions=True)

        all_chunks = []
        seen_hashes = set()

        for result in results:

            if isinstance(result,Exception):
                logger.warning(result)
                continue

            for chunk in result:
                chunk_hash = self._generate_hash(chunk.page_content.encode())

                if chunk_hash not in seen_hashes:
                    seen_hashes.add(chunk_hash)
                    all_chunks.append(chunk)

        logger.info(f"Total unique chunks: {len(all_chunks)}")

        return all_chunks



    async def _process_single_file(self,path:str):

        try:
            file_hash = await asyncio.to_thread(self._hash_file,path)
            
            cache_path = self.cache_dir /f"{file_hash}.pkl"

            if self._is_cache_valid(cache_path):   
                logger.info(f"Loading cache {path}")

                return await asyncio.to_thread(self._load_from_cache,
                                               cache_path)

            logger.info(f"Processing {path}")

            chunks = await asyncio.to_thread(self._process_file,path)

            await asyncio.to_thread(self._save_to_cache,
                                    chunks,
                                    cache_path)

            return chunks

        except Exception as e:
            logger.warning(f"Failed {path}: {e}")
            return []


    def _process_file(self,path: str) -> List[Document]:

        ext = Path(path).suffix.lower()

        if ext not in settings.ALLOWED_TYPES:
            return []

        try:

            if ext == ".pdf":

                pipeline = PdfPipelineOptions()
                pipeline.do_ocr = False

                converter = DocumentConverter(

                    format_options={

                        "pdf": PdfFormatOption(

                            pipeline_options=pipeline

                        )

                    }

                )

            else:

                converter = DocumentConverter()

            markdown = (

                converter

                .convert(path)

                .document

                .export_to_markdown()

            )

        except Exception:

            converter = DocumentConverter()

            markdown = (

                converter

                .convert(path)

                .document

                .export_to_markdown()

            )

        splitter = RecursiveCharacterTextSplitter(chunk_size=800,
                                                  chunk_overlap=150,
                                                  separators=["\n\n","\n","."," ",""])

        texts = splitter.split_text(markdown)

        chunks = []
        for i,text in enumerate(texts):
            text = text.strip()

            if len(text) < 50:
                continue

            chunks.append(Document(page_content=text,
                                   metadata={"source":path,
                                             "chunk_id":i}))

        return chunks


    def _hash_file(self,path:str):
        sha = hashlib.sha256()

        with open(path,"rb") as f:

            while True:
                data = f.read(8192)

                if not data:
                    break

                sha.update(data)

        return sha.hexdigest()


    @staticmethod
    def _generate_hash(content:bytes):
        return hashlib.sha256(content).hexdigest()


    def _is_cache_valid(self,cache_path: Path):
        if not cache_path.exists():

            return False

        age = (datetime.now() - datetime.fromtimestamp(cache_path.stat().st_mtime))

        return age < timedelta(days=settings.CACHE_EXPIRE_DAYS)


    def _load_from_cache(self,cache_path: Path):

        try:
            with open(cache_path,"rb") as f:
                data = pickle.load(f)
                return data["chunks"]

        except:
            return []


    def _save_to_cache(self,chunks,cache_path):

        with open(cache_path,"wb") as f:
            pickle.dump({"timestamp":datetime.now().timestamp(),
                         "chunks":chunks},f)