import os 
import tempfile 
from typing import List 

from fastapi import APIRouter,UploadFile,File,HTTPException,Depends

from api.schemas import (ChatRequest,
                         ChatResponse,
                         UploadResponse,
                         HealthResponse)

from api.dependencies import (get_builder,
                              get_workflow,
                              rebuild_workflow)

from src.document_processor.file_handler import DocumentProcessor
from src.tools.mcp_client import is_ready as mcp_is_ready
from src.custom_logger.logger import logger

router = APIRouter()

processor = DocumentProcessor()

@router.get("/health",response_model=HealthResponse)
async def health():
    workflow = get_workflow()
    return HealthResponse(status="ok",
                          mcp_ready=mcp_is_ready(),
                          retriever_ready=not isinstance(workflow.retriever,type(None)))

@router.post("/documents/upload",response_model=UploadResponse)
async def upload_documents(files: List[UploadFile] = File(...)):
    if not files:
        raise HTTPException(status_code=400,detail="No files provided")
    
    tmp_dir = tempfile.mkdtemp()
    paths = []

    for f in files:
        if not any(f.filename.endswith(ext) for ext in [".pdf",".txt",".docx",".md"]):
            raise HTTPException(status_code=400,detail=f"Unsupported file type: {f.filename}")

        path = os.path.join(tmp_dir,f.filename)
        content = await f.read()
        with open(path,"wb") as out:
            out.write(content)

        paths.append(path)

    try:
        docs = await processor.process(paths)
        await rebuild_workflow(docs)
        return UploadResponse(message="Documents indexed successfully",
                              chunks_indexed=len(docs),
                              files=[f.filename for f in files])

    except Exception as e:
        logger.error(f"Document processing failed: {e}")
        raise HTTPException(status_code=500,detail=str(e))

@router.post("/chat/invoke",response_model=ChatResponse)
async def chat(request: ChatRequest):
    if not request.question.strip():
        raise HTTPException(status_code=400,detail="Question cannot be empty")
    
    workflow = get_workflow()

    try:
        result = await workflow.run(request.question)
        return ChatResponse(answer=result['final_answer'],
                            reasoning_steps=result['reasoning_steps'],
                            delegated=result.get("delegated",False),
                            web_used=result.get("web_used",False),
                            session_id=request.session_id)
    
    except Exception as e:
        logger.error(f"Workflow error: {e}")
        raise HTTPException(status_code=500,detail="Agent pipeline failed")