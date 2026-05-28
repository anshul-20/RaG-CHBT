import streamlit as st
import requests
import uuid
from typing import Optional

# ========================= CONFIG =========================
st.set_page_config(
    page_title="RAG Chatbot",
    page_icon="📚",
    layout="wide",           
    initial_sidebar_state="expanded"
)

API_BASE = "http://localhost:8000"

# ====================== SESSION STATE ======================
if "session_id" not in st.session_state:
    st.session_state.session_id = None

if "doc_id" not in st.session_state:
    st.session_state.doc_id = None

if "messages" not in st.session_state:
    st.session_state.messages = []

if "uploaded_file_key" not in st.session_state:
    st.session_state.uploaded_file_key = 0

# ====================== HELPER FUNCTIONS ======================
def get_new_session():
    try:
        r = requests.post(f"{API_BASE}/session/new")
        r.raise_for_status()
        return r.json()["session_id"]
    except:
        return str(uuid.uuid4())

def upload_pdf(file):
    try:
        with st.spinner("Processing PDF..."):
            files = {"file": (file.name, file.getvalue(), "application/pdf")}
            r = requests.post(f"{API_BASE}/ingest", files=files)
            r.raise_for_status()
            return r.json()
    except Exception as e:
        st.error(f"Upload failed: {e}")
        return None

def chat_query(question: str) -> Optional[dict]:
    try:
        payload = {
            "session_id": st.session_state.session_id,
            "question": question,
            "doc_id": st.session_state.doc_id
        }
        r = requests.post(f"{API_BASE}/chat", json=payload)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        st.error(f"Chat error: {e}")
        return None

def clear_document_and_session():
    if st.session_state.doc_id:
        try:
            requests.delete(f"{API_BASE}/document/{st.session_state.doc_id}")
        except:
            pass
    if st.session_state.session_id:
        try:
            requests.post(f"{API_BASE}/session/clear", json={"session_id": st.session_state.session_id})
        except:
            pass

    st.session_state.session_id = None
    st.session_state.doc_id = None
    st.session_state.messages = []
    st.session_state.uploaded_file_key += 1
    st.rerun()

# ====================== SIDEBAR ======================
with st.sidebar:
    st.header("📄 Document")
    
    uploaded_file = st.file_uploader(
        "Upload PDF", 
        type=["pdf"], 
        key=f"pdf_uploader_{st.session_state.uploaded_file_key}"
    )
    
    if uploaded_file and st.button("Process Document", type="primary", use_container_width=True):
        result = upload_pdf(uploaded_file)
        if result:
            st.success(f"✅ {result['source']} processed!")
            st.caption(f"Pages: {result.get('total_pages')} | Chunks: {result.get('total_chunks')}")
            
            st.session_state.doc_id = result["doc_id"]
            
            if not st.session_state.session_id:
                st.session_state.session_id = get_new_session()
            
            if not st.session_state.messages:
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": f"Hi! I've loaded **{result['source']}**. Ask me anything about this document."
                })
            st.rerun()

    st.divider()
    
    if st.button("🗑️ Clear Document & Start New", type="secondary", use_container_width=True):
        clear_document_and_session()

    if st.session_state.session_id:
        st.caption(f"Session ID: `{st.session_state.session_id[:8]}...`")

# ====================== MAIN CHAT AREA (Full Screen) ======================
st.title("RAG Chatbot")
st.markdown("**Ask questions grounded in the uploaded document**")

if not st.session_state.doc_id:
    st.info("👆 Please upload a PDF document from the sidebar to start chatting.", icon="📌")
    
else:
    # Display chat history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Chat Input - Full width
    if prompt := st.chat_input("Ask a question about the document..."):
        # User message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Assistant response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = chat_query(prompt)
                if response and "answer" in response:
                    answer = response["answer"]
                    st.markdown(answer)
                    st.session_state.messages.append({"role": "assistant", "content": answer})
                else:
                    st.error("Sorry, I couldn't get a response.")