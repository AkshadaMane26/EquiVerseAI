import os
import socket
import requests
import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQAWithSourcesChain
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from streamlit_lottie import st_lottie
from streamlit_card import card
from urllib3.exceptions import NewConnectionError, MaxRetryError

# === Streamlit page config ===
st.set_page_config(page_title="EquiVerse AI", layout="wide", page_icon="🧠")

# === Load env ===
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    st.error("⚠️ GOOGLE_API_KEY not found. Please set it in your .env file.")
    st.stop()

# === CSS Styling ===
st.markdown("""
<style>
    .stApp { background-color: #f8fafc; }
    .question-box {
        background-color: #e8f0fe;
        border-radius: 12px;
        padding: 10px 15px;
    }
    .answer-box {
        background-color: #f1f3f4;
        border-radius: 12px;
        padding: 15px;
        font-size: 17px;
    }
</style>
""", unsafe_allow_html=True)

# === Load Lottie animation ===
def load_lottieurl(url: str):
    try:
        r = requests.get(url, timeout=10)
        if r.status_code != 200:
            return None
        return r.json()
    except Exception:
        return None

lottie_ai = load_lottieurl("https://assets4.lottiefiles.com/packages/lf20_4kx2q32n.json")

# === Header and animation ===
col1, col2 = st.columns([3, 1])
with col1:
    st.markdown("<h1 style='color: #1a73e8;'>🧠 EquiVerse AI: Your AI-powered edge in equity analysis.</h1>", unsafe_allow_html=True)
    st.markdown("Analyze equity news and ask questions. Get sharp, reliable insights backed by trusted sources.")
with col2:
    if lottie_ai:
        st_lottie(lottie_ai, speed=1, height=120)

st.divider()

# === Sidebar URL input ===
st.sidebar.header("🔗 Enter up to 3 News URLs")
with st.sidebar.form("url_form"):
    urls = [st.text_input(f"URL {i+1}", key=f"url_{i}") for i in range(3)]
    process_url_clicked = st.form_submit_button("⚙️ Process Articles")

# === Session state ===
if "qa_history" not in st.session_state:
    st.session_state.qa_history = []

faiss_folder = "faiss_store_gemini"

# === Initialize LLM and embeddings ===
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", temperature=0.7, max_output_tokens=1024)
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

# === Helper: Check network error ===
def is_network_error(exc):
    network_errors = (
        socket.gaierror,
        socket.herror,
        socket.timeout,
        requests.exceptions.ConnectionError,
        requests.exceptions.Timeout,
        MaxRetryError,
        NewConnectionError,
    )
    return isinstance(exc, network_errors)

# === Process URLs ===
if process_url_clicked:
    valid_urls = [url.strip() for url in urls if url.strip()]
    if not valid_urls:
        st.sidebar.warning("⚠️ Enter at least one valid URL.")
    else:
        all_docs = []
        error_urls = []

        with st.spinner("🔄 Starting article processing..."):
            for idx, url in enumerate(valid_urls, 1):
                status_text = st.empty()
                status_text.info(f"({idx}/{len(valid_urls)}) Loading URL: {url}")
                try:
                    loader = WebBaseLoader(url)
                    status_text.info(f"({idx}/{len(valid_urls)}) Extracting text from URL...")
                    docs = loader.load()
                    if docs:
                        all_docs.extend(docs)
                        status_text.success(f"({idx}/{len(valid_urls)}) Loaded and extracted {len(docs)} document(s).")
                    else:
                        status_text.warning(f"({idx}/{len(valid_urls)}) No documents extracted.")
                except Exception as e:
                    if is_network_error(e):
                        status_text.error(f"({idx}/{len(valid_urls)}) Network error loading URL: {url}\nDetails: {e}")
                    else:
                        status_text.error(f"({idx}/{len(valid_urls)}) Could not load URL: {url}\nDetails: {e}")
                    error_urls.append(url)

            if not all_docs:
                st.error("❌ No valid articles could be loaded. Please check URLs and your network connection.")
                st.stop()

            # Split documents
            splitting_msg = st.empty()
            splitting_msg.info("Splitting documents into chunks for embedding...")
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=100,
                separators=["\n\n", "\n", ".", ","]
            )
            split_docs = splitter.split_documents(all_docs)
            splitting_msg.success(f"Split into {len(split_docs)} chunks.")

            # Create vectorstore
            vector_msg = st.empty()
            vector_msg.info("Creating vector store from document chunks...")
            vectorstore = FAISS.from_documents(split_docs, embeddings)
            os.makedirs(faiss_folder, exist_ok=True)
            vectorstore.save_local(faiss_folder)
            vector_msg.success("Vector store created and saved successfully.")

            st.success("✅ All articles processed successfully.")

# === Cache vectorstore loading ===
@st.cache_resource
def load_vectorstore():
    return FAISS.load_local(faiss_folder, embeddings, allow_dangerous_deserialization=True)

# === Question input and handling ===
st.markdown("### 💬 Ask InsightBot")
chat_col, btn_col = st.columns([5, 1])
with chat_col:
    question = st.text_input("Type your question here:", key="chat_q")
with btn_col:
    ask = st.button("🚀 Ask")

if ask and question:
    if not os.path.exists(faiss_folder):
        st.warning("⚠️ Please process URLs first by entering article links and clicking 'Process Articles'.")
    else:
        try:
            with st.spinner("🔍 Getting answer..."):
                vectorstore = load_vectorstore()
                chain = RetrievalQAWithSourcesChain.from_llm(llm=llm, retriever=vectorstore.as_retriever())

                result = chain.invoke({"question": question}, return_only_outputs=True)

                # Save Q&A history
                st.session_state.qa_history.append({
                    "question": question,
                    "answer": result.get("answer", ""),
                    "sources": result.get("sources") or ""
                })

                tab1, tab2 = st.tabs(["🧠 Answer", "🔗 Sources"])
                with tab1:
                    st.markdown(f"<div class='answer-box'>{result.get('answer', '')}</div>", unsafe_allow_html=True)
                with tab2:
                    sources_text = result.get("sources") or ""
                    if sources_text.strip():
                        for source in sources_text.split("\n"):
                            if source.strip():
                                card(title="Source", text=source.strip(), url=source.strip())
                    else:
                        st.info("No sources available.")

        except Exception as e:
            st.error(f"❌ Error during question answering: {e}")

# === Display Q&A history ===
if st.session_state.qa_history:
    with st.expander("🕘 View Question & Answer History"):
        for i, qa in enumerate(reversed(st.session_state.qa_history), 1):
            st.markdown(f"**{i}. Question:** {qa['question']}")
            st.markdown(f"<div class='answer-box'>{qa['answer']}</div>", unsafe_allow_html=True)
            if qa['sources']:
                st.markdown("**Sources:**")
                for src in qa['sources'].split("\n"):
                    if src.strip():
                        st.markdown(f"- {src.strip()}")
            st.markdown("---")
