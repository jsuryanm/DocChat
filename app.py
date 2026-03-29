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

# ─────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="DocChat",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# Session state helpers
# ─────────────────────────────────────────────
_SESSION_DEFAULTS: dict = {
    "workflow": None,
    "builder": None,
    "mcp_ready": False,
    "a2a_ready": False,
    "indexed_files": [],
    "result": None,
    "error": None,
    "remote_agent_ok": None,
    # FIX: track indexing feedback in session state so it can be
    # rendered AFTER the sidebar (which reads indexed_files).
    "_index_changed": False,
    "_index_err": "",
}

for _k, _v in _SESSION_DEFAULTS.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v


# ─────────────────────────────────────────────
# Infrastructure startup
# ─────────────────────────────────────────────
async def _ensure_infra() -> None:
    """Start MCP and A2A clients exactly once per Streamlit session."""
    if not st.session_state.mcp_ready:
        await mcp_startup()
        st.session_state.mcp_ready = True

    if not st.session_state.a2a_ready:
        await a2a_startup()
        st.session_state.a2a_ready = True

    if (
        settings.REMOTE_AGENT_URL
        and st.session_state.remote_agent_ok is None
    ):
        st.session_state.remote_agent_ok = await ping_remote_agent(
            settings.REMOTE_AGENT_URL
        )


# ─────────────────────────────────────────────
# Document processing
# ─────────────────────────────────────────────
async def _maybe_rebuild_workflow(uploaded_files) -> tuple[bool, str]:
    """Re-index documents only when the uploaded file set has changed."""
    incoming_names = sorted(f.name for f in uploaded_files)
    if incoming_names == sorted(st.session_state.indexed_files):
        return False, ""

    processor = DocumentProcessor()

    if st.session_state.builder is None:
        st.session_state.builder = RetrieverBuilder()

    tmp_dir = tempfile.mkdtemp()
    paths: list[str] = []

    for f in uploaded_files:
        path = os.path.join(tmp_dir, f.name)
        with open(path, "wb") as out:
            out.write(f.read())
        paths.append(path)

    try:
        docs = await processor.process(paths)
    except Exception as exc:
        return False, f"Document processing failed: {exc}"

    try:
        retriever = await st.session_state.builder.build_hybrid_retriever(docs)
    except Exception as exc:
        return False, f"Retriever build failed: {exc}"

    st.session_state.workflow = AgentWorkflow(
        retriever=retriever,
        builder=st.session_state.builder,
    )
    st.session_state.indexed_files = incoming_names
    return True, ""


# ─────────────────────────────────────────────
# Pipeline runner
# ─────────────────────────────────────────────
async def _run_pipeline(question: str) -> dict:
    await _ensure_infra()

    if st.session_state.workflow is None:
        raise RuntimeError(
            "No documents indexed yet. Please upload documents before asking a question."
        )

    return await st.session_state.workflow.run(question)


# ═════════════════════════════════════════════
# FIX 1: Run infra startup on EVERY page load,
# before anything reads mcp_ready / a2a_ready.
# _ensure_infra() is idempotent — safe to call
# every rerun; it no-ops when already started.
# ═════════════════════════════════════════════
asyncio.run(_ensure_infra())

# ═════════════════════════════════════════════
# FIX 2: Process uploaded files BEFORE the
# sidebar renders, so indexed_files is already
# populated when the sidebar reads it.
# Streamlit renders top-to-bottom in one pass;
# anything that mutates session state must run
# before the UI that reads that state.
# ═════════════════════════════════════════════
# We must create the uploader widget here to
# get the file objects early, but we hide the
# label rendering until the main area below.
uploaded = st.session_state.get("_uploaded_files_cache")  # not used for logic

# Use a top-level file_uploader call result via session state trick:
# Streamlit always re-runs the full script, so we just move the
# _maybe_rebuild_workflow call to before the sidebar block.
_uploaded_early = st.file_uploader(
    "Upload documents",
    type=["pdf", "txt", "docx", "md"],
    accept_multiple_files=True,
    help="Supported: PDF, TXT, DOCX, Markdown. Files are cached — re-uploading the same file is instant.",
    key="file_uploader_widget",
    label_visibility="collapsed",   # rendered properly in main area below
)

if _uploaded_early:
    _changed, _err = asyncio.run(_maybe_rebuild_workflow(_uploaded_early))
    st.session_state._index_changed = _changed
    st.session_state._index_err = _err
else:
    # Reset feedback flags when no files are uploaded
    st.session_state._index_changed = False
    st.session_state._index_err = ""

# ─────────────────────────────────────────────
# UI — Sidebar (reads indexed_files AFTER it
# has already been updated above)
# ─────────────────────────────────────────────
st.title("DocChat — Agentic RAG")
st.caption("Upload documents, ask questions. Answers are grounded and verified.")

with st.sidebar:
    st.header("System Status")

    mcp_status = "Ready" if st.session_state.mcp_ready else "Not started"
    a2a_status = "Ready" if st.session_state.a2a_ready else "Not started"
    st.markdown(f"MCP tools: {mcp_status}")
    st.markdown(f"A2A client: {a2a_status}")

    if settings.REMOTE_AGENT_URL:
        if st.session_state.remote_agent_ok is True:
            st.markdown("Remote agent:Reachable")
        elif st.session_state.remote_agent_ok is False:
            st.markdown("Remote agent:Unreachable")
        else:
            st.markdown("Remote agent:Unchecked")

    st.divider()

    if st.session_state.indexed_files:
        st.header("Indexed Documents")
        for fname in st.session_state.indexed_files:
            st.markdown(f"- `{fname}`")
    else:
        st.info("No documents indexed yet.")

    st.divider()

    if st.button("Reset Session", use_container_width=True):
        for _k, _v in _SESSION_DEFAULTS.items():
            st.session_state[_k] = _v
        st.rerun()

# ─────────────────────────────────────────────
# Main area — upload label + feedback messages
# ─────────────────────────────────────────────
st.markdown("**Upload documents**")
# The actual widget was rendered above (collapsed label); show feedback here.
if st.session_state._index_err:
    st.error(st.session_state._index_err)
elif st.session_state._index_changed:
    n = len(st.session_state.indexed_files)
    st.success(f"Indexed {n} document{'s' if n != 1 else ''}.")

if (
    settings.REMOTE_AGENT_URL
    and st.session_state.remote_agent_ok is False
):
    st.warning(
        f"Remote agent at `{settings.REMOTE_AGENT_URL}` is unreachable. "
        "Questions requiring delegation will return an error message."
    )

# ─────────────────────────────────────────────
# Question input + Run button
# ─────────────────────────────────────────────
question = st.text_input(
    "Ask a question about your documents",
    placeholder="e.g. What were the key findings in the Q3 report?",
)

run_disabled = not (_uploaded_early and question.strip())
run_clicked = st.button("▶ Run", disabled=run_disabled, type="primary")

# ─────────────────────────────────────────────
# Run pipeline
# ─────────────────────────────────────────────
if run_clicked and _uploaded_early and question.strip():
    st.session_state.result = None
    st.session_state.error = None

    with st.spinner("Running agent pipeline…"):
        try:
            result = asyncio.run(_run_pipeline(question))
            st.session_state.result = result
        except Exception as exc:
            st.session_state.error = str(exc)

# ─────────────────────────────────────────────
# Results
# ─────────────────────────────────────────────
if st.session_state.error:
    st.error(f"Pipeline error: {st.session_state.error}")

if st.session_state.result:
    result = st.session_state.result

    st.subheader("Answer")
    st.write(result["final_answer"])

    col1, col2 = st.columns(2)
    with col1:
        if result.get("delegated"):
            st.info("Answer sourced from a remote specialist agent.")
    with col2:
        if result.get("web_used"):
            st.info("Web search was used (no relevant local documents).")

    retries = result.get("retries", 0)
    if retries > 0:
        st.caption(f"Pipeline retried {retries} time{'s' if retries != 1 else ''}.")

    with st.expander("Reasoning steps"):
        for step in result.get("reasoning_steps", []):
            st.write(f"- {step}")

    with st.expander("Draft history"):
        drafts = result.get("draft_history", [])
        if drafts:
            for i, draft in enumerate(drafts, 1):
                st.write(f"**Draft {i}:** {draft}")
        else:
            st.write("No draft history available.")