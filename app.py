import asyncio
import tempfile
import os

import nest_asyncio          
nest_asyncio.apply()

import streamlit as st

from src.tools.mcp_client import startup, shutdown
from src.document_processor.file_handler import DocumentProcessor
from src.retriever.builder import RetrieverBuilder
from src.agents.workflow import AgentWorkflow

st.set_page_config(page_title="DocChat", layout="wide")
st.title("DocChat - Agentic RAG")

uploaded = st.file_uploader(
    "Upload documents",
    type=["pdf", "txt", "docx", "md"],
    accept_multiple_files=True,
)

question = st.text_input("Ask a question about your documents")

if st.button("Run") and uploaded and question:

    async def run_pipeline():
        paths = []
        tmp_dir = tempfile.mkdtemp()  

        for f in uploaded:
            path = os.path.join(tmp_dir, f.name)
            with open(path, "wb") as out:
                out.write(f.read())
            paths.append(path)

        await startup()

        processor = DocumentProcessor()
        builder = RetrieverBuilder()

        with st.spinner("Processing documents..."):
            docs = await processor.process(paths)
            retriever = await builder.build_hybrid_retriever(docs)

        workflow = AgentWorkflow(retriever=retriever, builder=builder)

        with st.spinner("Running agent..."):
            result = await workflow.run(question)

        await shutdown()
        return result

    result = asyncio.run(run_pipeline())

    st.subheader("Answer")
    st.write(result["final_answer"])

    with st.expander("Reasoning steps"):
        for step in result["reasoning_steps"]:
            st.write(f"- {step}")

    with st.expander("Draft history"):
        for i, draft in enumerate(result["draft_history"], 1):
            st.write(f"Draft {i}: {draft}")
