import os
import pytest 
from src.document_processor.file_handler import DocumentProcessor

# fixtures help prepare data,objects or env before running tests 
# run pytest tests/test_file_handler.py -v

@pytest.fixture
def processor():
    return DocumentProcessor()

@pytest.mark.asyncio
async def test_pdf_processing(processor):
    docs = await processor.process([r"examples\gpt4o_report.pdf"])
    assert docs is not None 
    assert len(docs) > 0 
    assert docs[0].page_content != ""
    assert "source" in docs[0].metadata
