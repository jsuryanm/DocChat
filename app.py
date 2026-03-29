import asyncio
import tempfile
import os

import nest_asyncio
nest_asyncio.apply()

import streamlit as st

from src.tools.mcp_client import startup as mcp_startup
from src.a2a.client import startup as a2a_startup
from src.document_processor.file_handler import DocumentProcessor
from src.retriever.builder import RetrieverBuilder
from src.agents.workflow import AgentWorkflow
from src.config.settings import settings

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(page_title="DocChat", layout="wide")

# ── Session state defaults ────────────────────────────────────────────────────
_DEFAULTS = {
    "workflow": None,
    "builder": None,
    "infra_ready": False,
    "indexed_files": [],
    "result": None,
    "error": None,
}
for k, v in _DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v


# ── Infrastructure startup (once per session) ─────────────────────────────────
async def _ensure_infra():
    if not st.session_state.infra_ready:
        await mcp_startup()
        await a2a_startup()
        st.session_state.infra_ready = True

asyncio.run(_ensure_infra())


# ── Helpers ───────────────────────────────────────────────────────────────────
async def _index_files(uploaded_files):
    incoming = sorted(f.name for f in uploaded_files)
    if incoming == sorted(st.session_state.indexed_files):
        return  # nothing changed

    processor = DocumentProcessor()
    if st.session_state.builder is None:
        st.session_state.builder = RetrieverBuilder()

    tmp_dir = tempfile.mkdtemp()
    paths = []
    for f in uploaded_files:
        path = os.path.join(tmp_dir, f.name)
        with open(path, "wb") as out:
            out.write(f.read())
        paths.append(path)

    docs = await processor.process(paths)
    retriever = await st.session_state.builder.build_hybrid_retriever(docs)
    st.session_state.workflow = AgentWorkflow(
        retriever=retriever,
        builder=st.session_state.builder,
    )
    st.session_state.indexed_files = incoming
    st.success(f"Indexed {len(incoming)} document(s).")


async def _run_pipeline(question: str) -> dict:
    return await st.session_state.workflow.run(question)


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("System Status")
    st.write("MCP tools:", "Ready" if st.session_state.infra_ready else "Starting")
    st.write("A2A client:", "Ready" if st.session_state.infra_ready else "Starting")
    st.divider()

    if st.session_state.indexed_files:
        st.header("Indexed Documents")
        for name in st.session_state.indexed_files:
            st.write(f"• {name}")
    else:
        st.info("No documents indexed yet.")

    st.divider()
    if st.button("Reset Session", use_container_width=True):
        for k, v in _DEFAULTS.items():
            st.session_state[k] = v
        st.rerun()


# ── Main area ─────────────────────────────────────────────────────────────────
st.title("DocChat — Agentic RAG")
st.caption("Upload documents, ask questions. Answers are grounded and verified.")

st.subheader("Upload Documents")
uploaded_files = st.file_uploader(
    "Upload documents",
    type=["pdf", "txt", "docx", "md"],
    accept_multiple_files=True,
    label_visibility="collapsed",
)

if uploaded_files:
    asyncio.run(_index_files(uploaded_files))

st.divider()

# ── Question input ────────────────────────────────────────────────────────────
question = st.text_input(
    "Ask a question about your documents",
    placeholder="e.g. What were the key findings in the Q3 report?",
)

if st.button("Run", disabled=not (uploaded_files and question.strip()), type="primary"):
    st.session_state.result = None
    st.session_state.error = None

    if st.session_state.workflow is None:
        st.error("Please upload documents first.")
    else:
        with st.spinner("Running agent pipeline…"):
            try:
                st.session_state.result = asyncio.run(_run_pipeline(question))
            except Exception as exc:
                st.session_state.error = str(exc)

# ── Results ───────────────────────────────────────────────────────────────────
if st.session_state.error:
    st.error(f"Pipeline error: {st.session_state.error}")

if st.session_state.result:
    result = st.session_state.result

    st.subheader("Answer")
    st.write(result["final_answer"])

    if result.get("delegated"):
        st.info("Answer sourced from a remote specialist agent.")
    if result.get("web_used"):
        st.info("Web search was used (no relevant local documents).")

    retries = result.get("retries", 0)
    if retries:
        st.caption(f"Pipeline retried {retries} time(s).")

    with st.expander("Reasoning steps"):
        for step in result.get("reasoning_steps", []):
            st.write(f"• {step}")

    with st.expander("Draft history"):
        for i, draft in enumerate(result.get("draft_history", []), 1):
            st.write(f"**Draft {i}:** {draft}")