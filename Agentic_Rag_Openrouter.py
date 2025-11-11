import os
import tempfile
from datetime import datetime
from typing import List

import streamlit as st
import bs4
from agno.agent import Agent
from agno.models.openai import OpenAIChat
from langchain_community.document_loaders import PyPDFLoader, WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
from langchain_core.embeddings import Embeddings
from langchain_openai import OpenAIEmbeddings
from agno.tools.exa import ExaTools


class OpenRouterEmbedder(Embeddings):
    """Embeddings via OpenRouter (OpenAI-compatible API).

    Default model uses 768d vectors. You can change it from the sidebar.
    """
    def __init__(self, model_name: str = "nomic-ai/nomic-embed-text-v1.5"):
        self.model_name = model_name
        self._emb = OpenAIEmbeddings(
            model=self.model_name,
            openai_api_key=st.session_state.openrouter_api_key,
            openai_api_base="https://openrouter.ai/api/v1",
        )

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self._emb.embed_documents(texts)

    def embed_query(self, text: str) -> List[float]:
        return self._emb.embed_query(text)


# Constants
COLLECTION_NAME = "agentic-rag-openrouter"


# Streamlit App Initialization
st.title("🧭 Agentic RAG with OpenRouter and Agno")

# Session State Initialization
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
    st.session_state.chat_model = "meta-llama/llama-3.1-70b-instruct"
if "embed_model" not in st.session_state:
    st.session_state.embed_model = "nomic-ai/nomic-embed-text-v1.5"


# Sidebar Configuration
st.sidebar.header("🔑 API Configuration")
openrouter_api_key = st.sidebar.text_input(
    "OpenRouter API Key", type="password", value=st.session_state.openrouter_api_key
)
qdrant_api_key = st.sidebar.text_input(
    "Qdrant API Key", type="password", value=st.session_state.qdrant_api_key
)
qdrant_url = st.sidebar.text_input(
    "Qdrant URL",
    placeholder="https://your-cluster.cloud.qdrant.io:6333",
    value=st.session_state.qdrant_url,
)

# Model choices
st.sidebar.header("🧠 Model Settings")
st.session_state.chat_model = st.sidebar.text_input(
    "Chat model (OpenRouter id)", value=st.session_state.chat_model,
    help="e.g., meta-llama/llama-3.1-70b-instruct, qwen/qwen-2.5-72b-instruct, deepseek/deepseek-r1"
)
st.session_state.embed_model = st.sidebar.text_input(
    "Embedding model (OpenRouter id)", value=st.session_state.embed_model,
    help="Default is nomic-ai/nomic-embed-text-v1.5 (768 dims)"
)

# Clear Chat Button
if st.sidebar.button("🗑️ Clear Chat History"):
    st.session_state.history = []
    st.rerun()

# Update session state
st.session_state.openrouter_api_key = openrouter_api_key
st.session_state.qdrant_api_key = qdrant_api_key
st.session_state.qdrant_url = qdrant_url

# Web search config
st.sidebar.header("🌐 Web Search Configuration")
st.session_state.use_web_search = st.sidebar.checkbox(
    "Enable Web Search Fallback", value=st.session_state.use_web_search
)

if st.session_state.use_web_search:
    exa_api_key = st.sidebar.text_input(
        "Exa AI API Key",
        type="password",
        value=st.session_state.exa_api_key,
        help="Required for web search fallback when no relevant documents are found",
    )
    st.session_state.exa_api_key = exa_api_key

    # Optional domain filtering
    default_domains = ["arxiv.org", "wikipedia.org", "github.com", "medium.com"]
    custom_domains = st.sidebar.text_input(
        "Custom domains (comma-separated)",
        value=",".join(default_domains),
        help="Enter domains to search from, e.g.: arxiv.org,wikipedia.org",
    )
    search_domains = [d.strip() for d in custom_domains.split(",") if d.strip()]

# Search threshold
st.sidebar.header("🎯 Search Configuration")
st.session_state.similarity_threshold = st.sidebar.slider(
    "Document Similarity Threshold",
    min_value=0.0,
    max_value=1.0,
    value=0.7,
    help="Lower values will return more documents but might be less relevant. Higher values are more strict.",
)


# Utility Functions
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
    return Agent(
        name="Web Search Agent",
        model=OpenAIChat(id=st.session_state.chat_model),
        tools=[
            ExaTools(
                api_key=st.session_state.exa_api_key,
                include_domains=search_domains,
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

    # File/URL Upload Section
    st.sidebar.header("📁 Data Upload")
    uploaded_file = st.sidebar.file_uploader("Upload PDF", type=["pdf"])
    web_url = st.sidebar.text_input("Or enter URL")

    # Prepare embedding instance (so we can infer dims early)
    embedding = OpenRouterEmbedder(model_name=st.session_state.embed_model)

    # Process documents
    if uploaded_file:
        file_name = uploaded_file.name
        if file_name not in st.session_state.processed_documents:
            with st.spinner("Processing PDF..."):
                texts = process_pdf(uploaded_file)
                if texts and qdrant_client:
                    if st.session_state.vector_store:
                        st.session_state.vector_store.add_documents(texts)
                    else:
                        st.session_state.vector_store = create_vector_store(
                            qdrant_client, texts, embedding
                        )
                    st.session_state.processed_documents.append(file_name)
                    st.success(f"✅ Added PDF: {file_name}")

    if web_url:
        if web_url not in st.session_state.processed_documents:
            with st.spinner("Processing URL..."):
                texts = process_web(web_url)
                if texts and qdrant_client:
                    if st.session_state.vector_store:
                        st.session_state.vector_store.add_documents(texts)
                    else:
                        st.session_state.vector_store = create_vector_store(
                            qdrant_client, texts, embedding
                        )
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
        prompt = st.chat_input("Ask about your documents...")

    with toggle_col:
        st.session_state.force_web_search = st.toggle(
            "🌐", help="Force web search"
        )

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
            retriever = st.session_state.vector_store.as_retriever(
                search_type="similarity_score_threshold",
                search_kwargs={
                    "k": 5,
                    "score_threshold": st.session_state.similarity_threshold,
                },
            )
            docs = retriever.invoke(safe_query)
            if docs:
                context = "\n\n".join([d.page_content for d in docs])
                st.info(
                    f"📊 Found {len(docs)} relevant documents (similarity > {st.session_state.similarity_threshold})"
                )
            elif st.session_state.use_web_search:
                st.info("🔄 No relevant documents found in database, falling back to web search...")

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
