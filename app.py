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
# Initialise all keys once so we never KeyError
# ─────────────────────────────────────────────
_SESSION_DEFAULTS: dict = {
    "workflow": None,           # AgentWorkflow — rebuilt only when docs change
    "builder": None,            # RetrieverBuilder — shared across queries
    "mcp_ready": False,         # True after mcp_startup() succeeds
    "a2a_ready": False,         # True after a2a_startup() succeeds
    "indexed_files": [],        # names of currently-indexed files
    "result": None,             # last pipeline result dict
    "error": None,              # last pipeline error string or None
    "remote_agent_ok": None,    # None=unchecked, True/False
}

for _k, _v in _SESSION_DEFAULTS.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v


# ─────────────────────────────────────────────
# Infrastructure startup (runs once per session)
# ─────────────────────────────────────────────
async def _ensure_infra() -> None:
    """Start MCP and A2A clients exactly once per Streamlit session."""
    if not st.session_state.mcp_ready:
        await mcp_startup()
        st.session_state.mcp_ready = True

    if not st.session_state.a2a_ready:
        await a2a_startup()
        st.session_state.a2a_ready = True

    # Probe remote agent once per session, not on every run
    if (
        settings.REMOTE_AGENT_URL
        and st.session_state.remote_agent_ok is None
    ):
        st.session_state.remote_agent_ok = await ping_remote_agent(
            settings.REMOTE_AGENT_URL
        )


# ─────────────────────────────────────────────
# Document processing — only when file set changes
# ─────────────────────────────────────────────
async def _maybe_rebuild_workflow(uploaded_files) -> tuple[bool, str]:
    """
    Re-index documents only when the uploaded file set has changed.

    Returns (changed: bool, error_message: str).
    `changed` is True if a new workflow was built.
    `error_message` is non-empty on failure.
    """
    incoming_names = sorted(f.name for f in uploaded_files)
    if incoming_names == sorted(st.session_state.indexed_files):
        # Same files — reuse cached workflow
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


# ─────────────────────────────────────────────
# UI
# ─────────────────────────────────────────────
st.title("DocChat — Agentic RAG")
st.caption("Upload documents, ask questions. Answers are grounded and verified.")

# ── Sidebar: status indicators ───────────────
with st.sidebar:
    st.header("System Status")

    mcp_status = "Ready" if st.session_state.mcp_ready else "Not started"
    a2a_status = "Ready" if st.session_state.a2a_ready else "Not started"
    st.markdown(f"**MCP tools:** {mcp_status}")
    st.markdown(f"**A2A client:** {a2a_status}")

    if settings.REMOTE_AGENT_URL:
        if st.session_state.remote_agent_ok is True:
            st.markdown("**Remote agent:** Reachable")
        elif st.session_state.remote_agent_ok is False:
            st.markdown("**Remote agent:** Unreachable")
        else:
            st.markdown("**Remote agent:** Unchecked")

    st.divider()

    if st.session_state.indexed_files:
        st.header("Indexed Documents")
        for fname in st.session_state.indexed_files:
            st.markdown(f"- `{fname}`")
    else:
        st.info("No documents indexed yet.")

    st.divider()

    # Allow operator to reset session cleanly
    if st.button("Reset Session", use_container_width=True):
        for _k, _v in _SESSION_DEFAULTS.items():
            st.session_state[_k] = _v
        # MCP/A2A shutdown is best-effort; Streamlit will GC the process anyway
        st.rerun()

# ── Main area ────────────────────────────────
uploaded = st.file_uploader(
    "Upload documents",
    type=["pdf", "txt", "docx", "md"],
    accept_multiple_files=True,
    help="Supported: PDF, TXT, DOCX, Markdown. Files are cached — re-uploading the same file is instant.",
)

question = st.text_input(
    "Ask a question about your documents",
    placeholder="e.g. What were the key findings in the Q3 report?",
)

run_disabled = not (uploaded and question.strip())
run_clicked = st.button("▶ Run", disabled=run_disabled, type="primary")

# ── Remote agent warning (shown immediately after first infra startup) ────
if (
    settings.REMOTE_AGENT_URL
    and st.session_state.remote_agent_ok is False
):
    st.warning(
        f"Remote agent at `{settings.REMOTE_AGENT_URL}` is unreachable. "
        "Questions requiring delegation will return an error message."
    )

# ── Document indexing (triggered when files change, independently of Run) ─
if uploaded:
    changed, err = asyncio.run(_maybe_rebuild_workflow(uploaded))
    if err:
        st.error(err)
    elif changed:
        n = len(st.session_state.indexed_files)
        st.success(f"Indexed {n} document{'s' if n != 1 else ''}.")

# ── Run pipeline ──────────────────────────────
if run_clicked and uploaded and question.strip():
    st.session_state.result = None
    st.session_state.error = None

    # Ensure infra is up (idempotent)
    asyncio.run(_ensure_infra())

    # Show remote-agent warning if discovered after first infra startup
    if settings.REMOTE_AGENT_URL and st.session_state.remote_agent_ok is False:
        st.warning(
            f"Remote agent at `{settings.REMOTE_AGENT_URL}` is unreachable."
        )

    with st.spinner("Running agent pipeline…"):
        try:
            result = asyncio.run(_run_pipeline(question))
            st.session_state.result = result
        except Exception as exc:
            st.session_state.error = str(exc)

# ── Results ───────────────────────────────────
if st.session_state.error:
    st.error(f"Pipeline error: {st.session_state.error}")

if st.session_state.result:
    result = st.session_state.result

    st.subheader("Answer")
    st.write(result["final_answer"])

    # Attribution badges
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