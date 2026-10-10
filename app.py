import io
import os
from typing import Dict, List

import numpy as np
import streamlit as st
from groq import Groq
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

st.set_page_config(page_title="DocMind AI", page_icon="📚", layout="wide")

st.markdown("""
<style>
.block-container {padding-top: 1.6rem; max-width: 1200px;}
.hero {padding: 1.4rem 1.6rem; border: 1px solid rgba(128,128,128,.25);
       border-radius: 18px; background: linear-gradient(120deg, rgba(90,110,240,.12), rgba(60,190,170,.08));}
.small-muted {color: #7a8190; font-size: .92rem;}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero">
  <h1 style="margin:0">📚 DocMind AI</h1>
  <p style="font-size:1.08rem;margin:.45rem 0 0">Chat with your documents. Get answers with evidence.</p>
  <p class="small-muted" style="margin:.45rem 0 0">RAG-powered document Q&A · Source-grounded answers · Instant insights</p>
</div>
""", unsafe_allow_html=True)

def get_secret(name: str, default: str = "") -> str:
    try:
        return st.secrets.get(name, default) or os.getenv(name, default)
    except Exception:
        return os.getenv(name, default)

api_key = get_secret("GROQ_API_KEY")
CHAT_MODEL = get_secret("GROQ_MODEL", "openai/gpt-oss-20b")

with st.sidebar:
    st.markdown("## 📚 DocMind AI")
    st.caption("Document intelligence, grounded in your files")
    uploaded_files = st.file_uploader(
        "Upload PDF documents", type=["pdf"], accept_multiple_files=True,
        help="For best results, start with text-based PDFs."
    )
    st.divider()
    st.markdown("**What you can do**")
    st.markdown("- Ask questions across PDFs\n- Get page-level citations\n- Summarize key findings\n- Extract action items")
    st.divider()
    st.caption("Uploaded content is processed for your current app session.")

if not api_key:
    st.info("The app is deployed, but its AI service needs configuration.")
    st.markdown("To enable Q&A, add a Groq API key in your Streamlit app settings under **Secrets**.")
    st.code('GROQ_API_KEY = "your-groq-api-key"\nGROQ_MODEL = "openai/gpt-oss-20b"', language="toml")
    st.markdown("Create/manage a key at [Groq Console](https://console.groq.com/keys).")
    st.stop()

client = Groq(api_key=api_key)

@st.cache_resource(show_spinner="Loading the text-embedding model...")
def load_embedder():
    return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

def split_text(text: str, chunk_size: int = 1100, overlap: int = 180) -> List[str]:
    text = " ".join(text.split())
    if not text:
        return []
    chunks, start = [], 0
    while start < len(text):
        end = min(len(text), start + chunk_size)
        if end < len(text):
            boundary = max(text.rfind(". ", start, end), text.rfind("? ", start, end), text.rfind("! ", start, end))
            if boundary > start + chunk_size * 0.55:
                end = boundary + 1
        chunks.append(text[start:end].strip())
        if end >= len(text):
            break
        start = max(end - overlap, start + 1)
    return chunks

@st.cache_data(show_spinner=False)
def extract_pdf(file_bytes: bytes, filename: str) -> List[Dict]:
    reader = PdfReader(io.BytesIO(file_bytes))
    result = []
    for page_no, page in enumerate(reader.pages, start=1):
        try:
            page_text = page.extract_text() or ""
        except Exception:
            page_text = ""
        for idx, chunk in enumerate(split_text(page_text), start=1):
            result.append({"text": chunk, "source": filename, "page": page_no, "chunk": idx})
    return result

def embed_texts(texts: List[str]) -> np.ndarray:
    model = load_embedder()
    vectors = model.encode(texts, convert_to_numpy=True, normalize_embeddings=True,
                           show_progress_bar=False, batch_size=32)
    return np.asarray(vectors, dtype=np.float32)

def retrieve(question: str, chunks: List[Dict], matrix: np.ndarray, k: int = 5):
    qvec = embed_texts([question])[0]
    scores = matrix @ qvec
    indices = np.argsort(scores)[::-1][:min(k, len(chunks))]
    return [(chunks[int(i)], float(scores[int(i)])) for i in indices]

def ask_model(system_prompt: str, user_prompt: str) -> str:
    response = client.chat.completions.create(
        model=CHAT_MODEL,
        temperature=0.2,
        max_completion_tokens=1200,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )
    return response.choices[0].message.content or "I couldn't generate a response."

def answer_question(question: str, retrieved) -> str:
    evidence = "\n\n".join(
        f"[Source {i}: {item['source']}, page {item['page']}]\n{item['text']}"
        for i, (item, score) in enumerate(retrieved, start=1)
    )
    return ask_model(
        "You are DocMind, a careful document intelligence assistant. Answer using only the supplied document excerpts. "
        "If the excerpts do not contain enough evidence, say you could not find the answer in the uploaded documents. "
        "Cite factual claims using [Source 1], [Source 2], etc. Never invent citations.",
        f"Question: {question}\n\nDocument excerpts:\n{evidence}"
    )

def generate_insight(kind: str, chunks: List[Dict]) -> str:
    selected = chunks[:30]
    context = "\n\n".join(f"[{c['source']}, page {c['page']}]\n{c['text']}" for c in selected)
    prompts = {
        "Executive summary": "Create a concise executive summary with the document's purpose, main points, and conclusion.",
        "Key findings": "List the most important findings as concise bullets. Cite source filename and page for each finding.",
        "Action items": "Extract concrete action items. Include owner and deadline only if explicitly stated; otherwise write 'Not specified'. Cite source filename and page.",
        "Important dates": "Extract important dates, deadlines, milestones, and associated events. If none are found, say so. Cite source filename and page."
    }
    return ask_model(
        "Summarize documents faithfully. Never invent facts, dates, owners, or deadlines. State when information is not specified.",
        f"{prompts[kind]}\n\nDocument excerpts:\n{context}"
    )

if not uploaded_files:
    left, right = st.columns([1.25, 1])
    with left:
        st.subheader("Turn long PDFs into useful answers")
        st.write("Upload a PDF to begin. DocMind extracts page text, retrieves relevant passages, and generates answers grounded in that evidence.")
        st.markdown("- **Grounded Q&A:** answers use retrieved passages\n- **Traceability:** source pages help verify answers\n- **Document insights:** summaries, findings, action items, and dates")
    with right:
        st.subheader("Demo checklist")
        st.markdown("1. Upload a text-based PDF.\n2. Ask a specific question.\n3. Open the retrieved evidence.\n4. Generate an executive summary.")
    st.stop()

with st.spinner("Reading and indexing your documents..."):
    all_chunks = []
    for f in uploaded_files:
        all_chunks.extend(extract_pdf(f.getvalue(), f.name))

if not all_chunks:
    st.error("No selectable text was found in these PDFs. Try a text-based PDF; scanned PDFs need OCR.")
    st.stop()

fingerprint = tuple((f.name, len(f.getvalue())) for f in uploaded_files)
if st.session_state.get("fingerprint") != fingerprint:
    st.session_state["fingerprint"] = fingerprint
    st.session_state["chunks"] = all_chunks
    st.session_state["embeddings"] = None
    st.session_state["chat"] = []

if st.session_state.get("embeddings") is None:
    with st.spinner(f"Creating embeddings for {len(all_chunks)} passages..."):
        st.session_state["embeddings"] = embed_texts([c["text"] for c in all_chunks])

chunks = st.session_state["chunks"]
matrix = st.session_state["embeddings"]

m1, m2, m3 = st.columns(3)
m1.metric("Documents", len(uploaded_files))
m2.metric("Indexed passages", len(chunks))
m3.metric("Citation style", "Page-level")

tab_chat, tab_insights, tab_sources = st.tabs(["💬 Ask documents", "✨ Generate insights", "📑 Indexed sources"])

with tab_chat:
    st.subheader("Ask your documents")
    for msg in st.session_state.get("chat", []):
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("sources"):
                with st.expander("View retrieved evidence"):
                    for i, item in enumerate(msg["sources"], start=1):
                        st.markdown(f"**[Source {i}] {item['source']} · Page {item['page']}**")
                        st.write(item["text"])
    question = st.chat_input("Ask something about your uploaded PDFs...")
    if question:
        st.session_state["chat"].append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)
        with st.chat_message("assistant"):
            with st.spinner("Searching documents and drafting an evidence-based answer..."):
                retrieved = retrieve(question, chunks, matrix)
                try:
                    answer = answer_question(question, retrieved)
                except Exception as exc:
                    answer = f"Sorry, the AI request failed. Check the Groq model/key configuration. Details: {exc}"
                st.markdown(answer)
                with st.expander("View retrieved evidence", expanded=True):
                    for i, (item, score) in enumerate(retrieved, start=1):
                        st.markdown(f"**[Source {i}] {item['source']} · Page {item['page']}**")
                        st.caption(f"Retrieval similarity: {score:.3f}")
                        st.write(item["text"])
        st.session_state["chat"].append({"role": "assistant", "content": answer,
                                         "sources": [item for item, score in retrieved]})

with tab_insights:
    st.subheader("Generate document insights")
    kind = st.selectbox("Choose an insight", ["Executive summary", "Key findings", "Action items", "Important dates"])
    if st.button("Generate insight", type="primary"):
        with st.spinner("Analyzing document content..."):
            try:
                insight = generate_insight(kind, chunks)
                st.markdown(insight)
                st.download_button("Download insight as TXT", insight,
                                   file_name=f"docmind_{kind.lower().replace(' ', '_')}.txt")
            except Exception as exc:
                st.error(f"Couldn't generate the insight. Check the Groq configuration. Details: {exc}")

with tab_sources:
    st.subheader("Indexed source passages")
    st.caption("Text passages are extracted and processed in memory for this app session.")
    for c in chunks:
        with st.expander(f"{c['source']} · Page {c['page']} · Passage {c['chunk']}"):
            st.write(c["text"])

st.divider()
st.caption("DocMind AI · Prototype for demonstration. Verify important information against the original document. Scanned PDFs require OCR.")
