import os
import tempfile
from datetime import datetime
from typing import List

import streamlit as st
import bs4
from agno.agent import Agent
from agno.models.openai import OpenAIChat
from langchain_community.document_loaders import PyPDFLoader, WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
from langchain_core.embeddings import Embeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from agno.tools.exa import ExaTools
from langchain_community.vectorstores import FAISS


# ---------- Streamlit UI: prettier header + styled sidebar ----------
st.set_page_config(
    page_title="Agentic RAG — OpenRouter + Agno",
    page_icon="🧭",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Simple custom CSS for nicer look
CUSTOM_CSS = """
<style>
:root {
    --sb-dark: #845BB3;      /* seaborn primary blue */
    --sb-accent: #DD8452;    /* seaborn orange */
    --sb-teal: #55A868;      /* seaborn green/teal */
    --sb-grey: #D0BDF4;
    --text-light: #D0BDF4;
    --panel: #D0BDF4;
}

/* Main background */
.stApp {
    background-color: #494D5F;
}

/* Sidebar background */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, var(--sb-dark), #2E4A7B);
    color: var(--text-light);
}

/* Sidebar text */
section[data-testid="stSidebar"] * {
    color: var(--text-light) !important;
}

/* Cards */
.card {
    background: var(--panel);
    border-left: 5px solid var(--sb-dark);
    border-radius: 10px;
    padding: 14px;
    box-shadow: 0 4px 12px rgba(0,0,0,0.08);
    margin-bottom: 12px;
}

/* Header */
.h1 {
    font-size: 30px;
    font-weight: 800;
    color: var(--sb-dark);
}
.h2 {
    font-size: 15px;
    color: #D0BDF4;
}
.small-muted {
    color: #D0BDF4;
    font-size: 13px;
}

/* Buttons */
.stButton>button {
    background-color: var(--sb-dark);
    color: var(--text-light);
    border-radius: 8px;
    border: none;
}
.stButton>button:hover {
    background-color: var(--sb-accent);
    color: white;
}

/* File uploader */
.css-1cpxqw2, .stTextArea textarea, .stTextInput input {
    border-radius: 8px !important;
}

/* Metrics */
[data-testid="stMetricValue"] {
    color: var(--sb-teal) !important;
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
# -------------------------------------------------------------------

# Constants
COLLECTION_NAME = "agentic-rag-openrouter"

# Session State Initialization (ensure defaults before UI reads them)
if "openrouter_api_key" not in st.session_state:
    st.session_state.openrouter_api_key = ""
if "qdrant_api_key" not in st.session_state:
    st.session_state.qdrant_api_key = ""
if "qdrant_url" not in st.session_state:
    st.session_state.qdrant_url = ""
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "processed_documents" not in st.session_state:
    st.session_state.processed_documents = []
if "history" not in st.session_state:
    st.session_state.history = []
if "exa_api_key" not in st.session_state:
    st.session_state.exa_api_key = ""
if "use_web_search" not in st.session_state:
    st.session_state.use_web_search = False
if "force_web_search" not in st.session_state:
    st.session_state.force_web_search = False
if "similarity_threshold" not in st.session_state:
    st.session_state.similarity_threshold = 0.7
if "chat_model" not in st.session_state:
    st.session_state.chat_model = "openrouter/polaris-alpha"
if "embed_model" not in st.session_state:
    st.session_state.embed_model = "sentence-transformers/all-MiniLM-L6-v2"

# Header area with logo, title and description
logo_url = None  # Replace with your logo url or keep None
with st.container():
    left, mid, right = st.columns([1, 6, 2])
    with left:
        if logo_url:
            st.image(logo_url, width=72)
        else:
            st.markdown("<div style='font-size:42px'>🧭</div>", unsafe_allow_html=True)
    with mid:
        st.markdown("<div class='h1'>Agentic RAG — OpenRouter & Agno</div>", unsafe_allow_html=True)
        st.markdown("<div class='h2'>Retrieval-Augmented Generation with autonomous agents, semantic search, and LLM reasoning.</div>", unsafe_allow_html=True)
        st.markdown("<div class='small-muted'>Upload PDFs or URLs, tune similarity, and interact with the knowledge base.</div>", unsafe_allow_html=True)
    with right:
        st.metric(label="Docs indexed", value=len(st.session_state.processed_documents))
        st.markdown("")

st.markdown("---")

# Nice stats row / quick controls
with st.container():
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown("<div class='card'><strong>Vector Store</strong><div class='small-muted'>Active: {}</div></div>".format("Qdrant" if st.session_state.qdrant_url else "FAISS (local)"), unsafe_allow_html=True)
    with c2:
        st.markdown("<div class='card'><strong>Embedding model</strong><div class='small-muted'>{}</div></div>".format(st.session_state.embed_model), unsafe_allow_html=True)
    with c3:
        st.markdown("<div class='card'><strong>LLM</strong><div class='small-muted'>{}</div></div>".format(st.session_state.chat_model), unsafe_allow_html=True)
    with c4:
        st.markdown("<div class='card'><strong>Similarity</strong><div class='small-muted'>{:.2f}</div></div>".format(st.session_state.similarity_threshold), unsafe_allow_html=True)

# Sidebar: group into visually separated cards (all widgets have unique keys)
st.sidebar.markdown("<div class='card'><strong>🔑 API Configuration</strong></div>", unsafe_allow_html=True)
openrouter_api_key = st.sidebar.text_input("OpenRouter API Key", type="password", value=st.session_state.openrouter_api_key, placeholder="sk-...", key="openrouter_api_key_input")
qdrant_api_key = st.sidebar.text_input("Qdrant API Key", type="password", value=st.session_state.qdrant_api_key, placeholder="qdrant_key...", key="qdrant_api_key_input")
qdrant_url = st.sidebar.text_input("Qdrant URL", value=st.session_state.qdrant_url, placeholder="https://xxx-xxx.cloud.qdrant.io:6333", key="qdrant_url_input")

st.sidebar.markdown("<div class='card'><strong>🧠 Model Settings</strong></div>", unsafe_allow_html=True)
st.session_state.chat_model = st.sidebar.text_input(
    "Chat model (OpenRouter id)", value=st.session_state.chat_model,
    help="e.g., openrouter/polaris-alpha", key="chat_model_input"
)
st.session_state.embed_model = st.sidebar.text_input(
    "Embedding model (local)", value=st.session_state.embed_model,
    help="HuggingFace sentence-transformers model; default is all-MiniLM-L6-v2", key="embed_model_input"
)

st.sidebar.markdown("<div class='card'><strong>🌐 Web Search</strong></div>", unsafe_allow_html=True)
st.session_state.use_web_search = st.sidebar.checkbox("Enable Web Search Fallback", value=st.session_state.use_web_search, key="web_search_checkbox")
if st.session_state.use_web_search:
    exa_api_key = st.sidebar.text_input("Exa AI API Key", type="password", value=st.session_state.exa_api_key, key="exa_api_key_input")

st.sidebar.markdown("<div class='card'><strong>🎯 Search Configuration</strong></div>", unsafe_allow_html=True)
st.session_state.similarity_threshold = st.sidebar.slider(
    "Document Similarity Threshold", min_value=0.0, max_value=1.0, value=st.session_state.similarity_threshold, key="similarity_slider"
)

# Clear Chat Button (single unique key to avoid duplicates)
if st.sidebar.button("🗑️ Clear Chat History", key="clear_chat_btn"):
    st.session_state.history = []
    st.rerun()

# Update session state (persist changes)
st.session_state.openrouter_api_key = openrouter_api_key
st.session_state.qdrant_api_key = qdrant_api_key
st.session_state.qdrant_url = qdrant_url

# Friendly upload card in main area (replaces basic file_uploader placement)
st.markdown("<div class='card'><strong>📁 Upload & Index</strong></div>", unsafe_allow_html=True)
upload_col1, upload_col2 = st.columns([3, 1])
with upload_col1:
    uploaded_file = st.file_uploader("Upload PDF to index (or drag & drop)", type=["pdf"], key="file_uploader")
    web_url = st.text_input("Or enter a URL to index", placeholder="https://example.com/article", key="web_url_input")
with upload_col2:
    st.markdown("<div style='padding-top:16px;'>Use this to add docs to your knowledge base.<br>Processed files show in the sidebar.</div>", unsafe_allow_html=True)

st.markdown("---")
# ---------------- end UI snippet ----------------


# ---------------- Embeddings / Vector helpers ----------------
class LocalEmbedder(Embeddings):
    """Local embeddings via Sentence-Transformers (no API needed)."""
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model_name = model_name
        # Force CPU & safe kwargs to avoid meta/GPU issues
        self._emb = HuggingFaceEmbeddings(
            model_name=self.model_name,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self._emb.embed_documents(texts)

    def embed_query(self, text: str) -> List[float]:
        return self._emb.embed_query(text)


def init_qdrant():
    """Initialize Qdrant client with configured settings."""
    if not all([st.session_state.qdrant_api_key, st.session_state.qdrant_url]):
        return None
    try:
        return QdrantClient(
            url=st.session_state.qdrant_url,
            api_key=st.session_state.qdrant_api_key,
            timeout=60,
        )
    except Exception as e:
        st.error(f"🔴 Qdrant connection failed: {str(e)}")
        return None


def process_pdf(file) -> List:
    """Process PDF file and add source metadata."""
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(file.getvalue())
            loader = PyPDFLoader(tmp_file.name)
            documents = loader.load()

            for doc in documents:
                doc.metadata.update(
                    {
                        "source_type": "pdf",
                        "file_name": file.name,
                        "timestamp": datetime.now().isoformat(),
                    }
                )

            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000, chunk_overlap=200
            )
            return text_splitter.split_documents(documents)
    except Exception as e:
        st.error(f"📄 PDF processing error: {str(e)}")
        return []


def process_web(url: str) -> List:
    """Process web URL and add source metadata."""
    try:
        loader = WebBaseLoader(
            web_paths=(url,),
            bs_kwargs=dict(
                parse_only=bs4.SoupStrainer(
                    class_=(
                        "post-content",
                        "post-title",
                        "post-header",
                        "content",
                        "main",
                    )
                )
            ),
        )
        documents = loader.load()

        for doc in documents:
            doc.metadata.update(
                {"source_type": "url", "url": url, "timestamp": datetime.now().isoformat()}
            )

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=200
        )
        return text_splitter.split_documents(documents)
    except Exception as e:
        st.error(f"🌐 Web processing error: {str(e)}")
        return []


def _infer_embedding_size(embedding: Embeddings) -> int:
    try:
        vec = embedding.embed_query("hello world")
        return len(vec)
    except Exception as e:
        st.error(f"Failed to infer embedding size: {e}")
        # Reasonable default for many models
        return 768


# Vector Store Management
def create_vector_store(client, texts, embedding: Embeddings):
    """Create and initialize vector store with documents."""
    try:
        vec_size = _infer_embedding_size(embedding)

        # Create collection if needed
        try:
            client.create_collection(
                collection_name=COLLECTION_NAME,
                vectors_config=VectorParams(size=vec_size, distance=Distance.COSINE),
            )
            st.success(f"📚 Created new collection: {COLLECTION_NAME} ({vec_size} dims)")
        except Exception as e:
            if "already exists" not in str(e).lower():
                raise e

        # Initialize vector store
        vector_store = QdrantVectorStore(
            client=client, collection_name=COLLECTION_NAME, embedding=embedding
        )

        # Add documents
        with st.spinner("📤 Uploading documents to Qdrant..."):
            vector_store.add_documents(texts)
            st.success("✅ Documents stored successfully!")
            return vector_store

    except Exception as e:
        st.error(f"🔴 Vector store error: {str(e)}")
        return None


# FAISS fallback helper (create or upsert)
def upsert_docs(texts, embedding, qdrant_client):
    if not texts:
        return
    if st.session_state.vector_store is None:
        if qdrant_client:
            st.session_state.vector_store = create_vector_store(qdrant_client, texts, embedding)
        else:
            st.session_state.vector_store = FAISS.from_documents(texts, embedding)
            st.success("✅ Created local FAISS index (no Qdrant configured)")
    else:
        st.session_state.vector_store.add_documents(texts)
        st.success("✅ Added documents to active vector store")


# Agents
def get_query_rewriter_agent() -> Agent:
    return Agent(
        name="Query Rewriter",
        model=OpenAIChat(id=st.session_state.chat_model),
        instructions=(
            "You are an expert at reformulating questions to be more precise and detailed.\n"
            "1) Analyze the user's question. 2) Rewrite it to be specific and search-friendly.\n"
            "3) Expand acronyms. 4) Return ONLY the rewritten query."
        ),
        markdown=True,
    )


def get_web_search_agent() -> Agent:
    # search_domains might be defined in sidebar block; fall back safely
    include_domains = locals().get("search_domains", None)
    return Agent(
        name="Web Search Agent",
        model=OpenAIChat(id=st.session_state.chat_model),
        tools=[
            ExaTools(
                api_key=st.session_state.exa_api_key,
                include_domains=include_domains,
                num_results=5,
            )
        ],
        instructions=(
            "Search the web for the query, summarize the most relevant information, and include sources."
        ),
        markdown=True,
    )


def get_rag_agent() -> Agent:
    return Agent(
        name="RAG Agent",
        model=OpenAIChat(id=st.session_state.chat_model),
        instructions=(
            "You answer accurately based on provided context.\n"
            "If documents are provided, cite details from them.\n"
            "If web results are provided, clearly mark them as from the web."
        ),
        markdown=True,
    )


# Main Application Flow
if st.session_state.openrouter_api_key:
    # Configure OpenRouter (OpenAI-compatible)
    os.environ["OPENAI_API_KEY"] = st.session_state.openrouter_api_key
    os.environ["OPENAI_BASE_URL"] = "https://openrouter.ai/api/v1"

    qdrant_client = init_qdrant()

    # Prepare embedding instance (so we can infer dims early)
    embedding = LocalEmbedder(model_name=st.session_state.embed_model)

    # Process documents
    if uploaded_file:
        file_name = uploaded_file.name
        if file_name not in st.session_state.processed_documents:
            with st.spinner("Processing PDF..."):
                texts = process_pdf(uploaded_file)
                # Use upsert_docs which handles Qdrant vs FAISS
                upsert_docs(texts, embedding, qdrant_client)
                st.session_state.processed_documents.append(file_name)
                st.success(f"✅ Added PDF: {file_name}")

    if web_url:
        if web_url not in st.session_state.processed_documents:
            with st.spinner("Processing URL..."):
                texts = process_web(web_url)
                upsert_docs(texts, embedding, qdrant_client)
                st.session_state.processed_documents.append(web_url)
                st.success(f"✅ Added URL: {web_url}")

    # Display sources in sidebar
    if st.session_state.processed_documents:
        st.sidebar.header("📚 Processed Sources")
        for source in st.session_state.processed_documents:
            if source.endswith(".pdf"):
                st.sidebar.text(f"📄 {source}")
            else:
                st.sidebar.text(f"🌐 {source}")

    # Chat Interface
    chat_col, toggle_col = st.columns([0.9, 0.1])

    with chat_col:
        prompt = st.chat_input("Ask about your documents...", key="chat_input")

    with toggle_col:
        # using a checkbox toggle for clarity and unique key
        st.session_state.force_web_search = st.checkbox("🌐", key="force_web_search_checkbox", help="Force web search")

    if prompt:
        st.session_state.history.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)

        # Step 1: Rewrite the query for better retrieval
        with st.spinner("🤔 Reformulating query..."):
            try:
                query_rewriter = get_query_rewriter_agent()
                rewritten_query = query_rewriter.run(prompt).content
                with st.expander("🔄 See rewritten query"):
                    st.write(f"Original: {prompt}")
                    st.write(f"Rewritten: {rewritten_query}")
            except Exception as e:
                st.error(f"❌ Error rewriting query: {str(e)}")
                rewritten_query = prompt

        # Step 2: Choose search strategy
        context = ""
        docs = []
        safe_query = (
            rewritten_query if isinstance(rewritten_query, str) and rewritten_query else prompt
        )
        if not st.session_state.force_web_search and st.session_state.vector_store:
            # use similarity search; adjust params for compatibility
            try:
                retriever = st.session_state.vector_store.as_retriever(
                    search_type="similarity",
                    search_kwargs={"k": 5},
                )
                docs = retriever.invoke(safe_query)
                if docs:
                    context = "\n\n".join([d.page_content for d in docs])
                    st.info(
                        f"📊 Found {len(docs)} relevant documents (similarity threshold {st.session_state.similarity_threshold})"
                    )
                elif st.session_state.use_web_search:
                    st.info("🔄 No relevant documents found in database, falling back to web search...")
            except Exception as e:
                st.warning(f"⚠️ Retriever error: {e}")
                if st.session_state.use_web_search:
                    st.info("🔄 Falling back to web search due to retrieval error.")

        # Step 3: Web search if needed
        if (
            (st.session_state.force_web_search or not context)
            and st.session_state.use_web_search
            and st.session_state.exa_api_key
        ):
            with st.spinner("🔍 Searching the web..."):
                try:
                    web_search_agent = get_web_search_agent()
                    web_results = web_search_agent.run(safe_query).content
                    if web_results:
                        context = f"Web Search Results:\n{web_results}"
                        if st.session_state.force_web_search:
                            st.info("ℹ️ Using web search as requested via toggle.")
                        else:
                            st.info(
                                "ℹ️ Using web search as fallback since no relevant documents were found."
                            )
                except Exception as e:
                    st.error(f"❌ Web search error: {str(e)}")

        # Step 4: Answer
        with st.spinner("🤖 Thinking..."):
            try:
                rag_agent = get_rag_agent()

                if context:
                    full_prompt = f"""Context: {context}

Original Question: {prompt}
Rewritten Question: {safe_query}

Provide a comprehensive answer based on the available information."""
                else:
                    full_prompt = f"Original Question: {prompt}\nRewritten Question: {safe_query}"
                    st.info("ℹ️ No relevant information found in documents or web search.")

                response = rag_agent.run(full_prompt)

                st.session_state.history.append(
                    {"role": "assistant", "content": response.content}
                )

                with st.chat_message("assistant"):
                    st.write(response.content)

                    if not st.session_state.force_web_search and "docs" in locals() and docs:
                        with st.expander("🔍 See document sources"):
                            for i, doc in enumerate(docs, 1):
                                source_type = doc.metadata.get("source_type", "unknown")
                                source_icon = "📄" if source_type == "pdf" else "🌐"
                                source_name = doc.metadata.get(
                                    "file_name" if source_type == "pdf" else "url", "unknown"
                                )
                                st.write(f"{source_icon} Source {i} from {source_name}:")
                                st.write(f"{doc.page_content[:200]}...")

            except Exception as e:
                st.error(f"❌ Error generating response: {str(e)}")

else:
    st.warning("⚠️ Please enter your OpenRouter API Key to continue")
