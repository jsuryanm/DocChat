import asyncio
import tempfile
import os

import nest_asyncio
nest_asyncio.apply()

import streamlit as st

from src.tools.mcp_client import startup as mcp_startup, shutdown as mcp_shutdown
from src.a2a.client import startup as a2a_startup, shutdown as a2a_shutdown, ping_remote_agent
from src.document_processor.file_handler import DocumentProcessor
from src.retriever.builder import RetrieverBuilder
from src.agents.workflow import AgentWorkflow
from src.config.settings import settings

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

        # Start MCP client and A2A HTTP client
        await mcp_startup()
        await a2a_startup()

        # Fix 5: probe REMOTE_AGENT_URL at boot so the user gets an
        # early warning instead of a silent fallback answer mid-query.
        if settings.REMOTE_AGENT_URL:
            reachable = await ping_remote_agent(settings.REMOTE_AGENT_URL)
            if not reachable:
                st.warning(
                    f"Remote agent at `{settings.REMOTE_AGENT_URL}` is not reachable. "
                    "Questions that require delegation will fall back to an error message."
                )

        processor = DocumentProcessor()
        builder = RetrieverBuilder()

        with st.spinner("Processing documents..."):
            docs = await processor.process(paths)
            retriever = await builder.build_hybrid_retriever(docs)

        workflow = AgentWorkflow(retriever=retriever, builder=builder)

        with st.spinner("Running agent..."):
            result = await workflow.run(question)

        # Always shut down both clients cleanly
        await mcp_shutdown()
        await a2a_shutdown()

        return result

    result = asyncio.run(run_pipeline())

    st.subheader("Answer")
    st.write(result["final_answer"])

    if result.get("delegated"):
        st.info("ℹ This answer was sourced from a remote specialist agent.")

    with st.expander("Reasoning steps"):
        for step in result["reasoning_steps"]:
            st.write(f"- {step}")

    with st.expander("Draft history"):
        for i, draft in enumerate(result["draft_history"], 1):
            st.write(f"Draft {i}: {draft}")