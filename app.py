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

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="DocChat",
    layout="wide",
)

# ── Session state defaults ────────────────────────────────────────────────────
_DEFAULTS = {
    "workflow": None,
    "builder": None,
    "infra_ready": False,
    "indexed_files": [],
    "messages": [],
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

    # Reset chat history when documents change —
    # old answers no longer apply to the new corpus
    st.session_state.messages = []

    st.success(f"Indexed {len(incoming)} document(s). You can now ask questions.")


async def _run_pipeline(question: str) -> dict:
    return await st.session_state.workflow.run(question)


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ System Status")
    mcp_status = "Ready" if st.session_state.infra_ready else "Starting"
    a2a_status = "Ready" if st.session_state.infra_ready else "Starting"
    st.write(f"MCP tools: {mcp_status}")
    st.write(f"A2A client: {a2a_status}")

    st.divider()

    st.header("Upload Documents")
    uploaded_files = st.file_uploader(
        "Upload documents",
        type=["pdf", "txt", "docx", "md"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )

    if uploaded_files:
        asyncio.run(_index_files(uploaded_files))

    st.divider()

    if st.session_state.indexed_files:
        st.header("Indexed Documents")
        for name in st.session_state.indexed_files:
            st.write(f"• {name}")
    else:
        st.info("No documents indexed yet.\nUpload files above to get started.")

    st.divider()

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Clear Chat", use_container_width=True):
            st.session_state.messages = []
            st.rerun()
    with col2:
        if st.button("Reset All", use_container_width=True):
            for k, v in _DEFAULTS.items():
                st.session_state[k] = v
            st.rerun()


# ── Main area ─────────────────────────────────────────────────────────────────
st.title("DocChat — Agentic RAG")
st.caption("Upload documents in the sidebar, then ask questions below.")

# Show a welcome message if no conversation yet
if not st.session_state.messages:
    if not st.session_state.indexed_files:
        st.info("Upload documents in the sidebar to get started.")
    else:
        st.info("Documents indexed. Ask a question below!")

# ── Render existing chat messages ─────────────────────────────────────────────
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

        # Show metadata only for assistant messages
        if msg["role"] == "assistant" and msg.get("meta"):
            meta = msg["meta"]

            if meta.get("web_used"):
                st.info("Web search was used (no relevant local documents found).")
            if meta.get("delegated"):
                st.info("Answer sourced from a remote specialist agent.")
            if meta.get("retries", 0):
                st.caption(f"Pipeline retried {meta['retries']} time(s).")

            with st.expander("Reasoning steps"):
                for step in meta.get("reasoning_steps", []):
                    st.write(f"• {step}")

            if meta.get("draft_history"):
                with st.expander("Draft history"):
                    for i, draft in enumerate(meta["draft_history"], 1):
                        st.write(f"Draft {i}: {draft}")

# ── Chat input ────────────────────────────────────────────────────────────────
if prompt := st.chat_input(
    "Ask a question about your documents...",
    disabled=not st.session_state.indexed_files,
):
    # Immediately render user message
    with st.chat_message("user"):
        st.write(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    if st.session_state.workflow is None:
        with st.chat_message("assistant"):
            st.error("Please upload and index documents first.")
        st.session_state.messages.append({
            "role": "assistant",
            "content": "Please upload and index documents first.",
            "meta": None,
        })
    else:
        with st.chat_message("assistant"):
            with st.spinner("Running agent pipeline…"):
                try:
                    result = asyncio.run(_run_pipeline(prompt))
                    answer = result["final_answer"]

                    st.write(answer)

                    # Inline metadata badges
                    if result.get("web_used"):
                        st.info("Web search was used (no relevant local documents found).")
                    if result.get("delegated"):
                        st.info("Answer sourced from a remote specialist agent.")
                    if result.get("retries", 0):
                        st.caption(f"Pipeline retried {result['retries']} time(s).")

                    with st.expander("Reasoning steps"):
                        for step in result.get("reasoning_steps", []):
                            st.write(f"• {step}")

                    if result.get("draft_history"):
                        with st.expander("Draft history"):
                            for i, draft in enumerate(result["draft_history"], 1):
                                st.write(f"**Draft {i}:** {draft}")

                    # Persist to session
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer,
                        "meta": result,
                    })

                except Exception as exc:
                    error_msg = f"Pipeline error: {exc}"
                    st.error(error_msg)
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": f"{error_msg}",
                        "meta": None,
                    })