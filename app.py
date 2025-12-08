# app.py
"""
Project 2: Abbreviation Extractor (Gemini Version)
Uses Google Gemini (Closed-Source) to satisfy Question 4.
"""
import re
import tempfile
import streamlit as st
import docx2txt
from bs4 import BeautifulSoup
import fitz  # PyMuPDF

# REPLACE GROQ IMPORT WITH GOOGLE
from langchain_google_genai import ChatGoogleGenerativeAI

# ----------------------------
# 1. Model Configuration (Closed-Source API)
# ----------------------------
# Get key from secrets
api_key = st.secrets["GOOGLE_API_KEY"]

# Initialize Gemini (Closed-Source Model)
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash", 
    temperature=0.1,
    google_api_key=api_key
)

# ----------------------------
# 2. PDF Extraction
# ----------------------------
def extract_pdf_columns(file_bytes: bytes) -> str:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(file_bytes)
        pdf_path = tmp.name

    doc = fitz.open(pdf_path)
    page_texts = []
    for page in doc:
        blocks = page.get_text("blocks")
        blocks = [b for b in blocks if isinstance(b[4], str) and b[4].strip()]
        blocks_sorted = sorted(blocks, key=lambda b: (round(b[1], 1), round(b[0], 1)))
        page_lines = []
        for b in blocks_sorted:
            txt = re.sub(r'\n+', ' ', b[4].strip())
            page_lines.append(txt)
        page_texts.append("\n\n".join(page_lines))
    doc.close()
    return "\n\n".join(page_texts)

def clean_extracted_text(text: str) -> str:
    if not text: return ""
    text = re.sub(r"(\w)-\s+([a-z])", r"\1\2", text) # Fix hyphenation
    text = text.replace("\xa0", " ")
    return text.strip()

# ----------------------------
# 3. File Parser
# ----------------------------
def parse_file(uploaded_file) -> str:
    if not uploaded_file: return ""
    data = uploaded_file.getvalue()
    filename = uploaded_file.name.lower()

    if filename.endswith(".pdf"):
        return clean_extracted_text(extract_pdf_columns(data))
    if filename.endswith(".txt"):
        return data.decode("utf-8", errors="replace")
    if filename.endswith(".docx"):
        with tempfile.NamedTemporaryFile(suffix=".docx", delete=False) as tf:
            tf.write(data)
            return (docx2txt.process(tf.name) or "").strip()
    return ""

# ----------------------------
# 4. Extraction Logic
# ----------------------------
def get_abbreviations(text: str) -> str:
    if not text.strip(): return "No text found."
    
    doc_clip = text[:50000] 

    prompt = (
        "TASK: Extract scientific/technical abbreviations and definitions.\n"
        "RULES:\n"
        "1. Identify abbreviations (2-10 uppercase letters) defined in the text.\n"
        "2. Format strictly as a bulleted list: '• ABBR: Full Definition'\n"
        "3. Sort alphabetically.\n"
        "4. Do NOT summarize. ONLY output the list.\n\n"
        "--- START TEXT ---\n"
        f"{doc_clip}\n"
        "--- END TEXT ---\n\n"
        "REMINDER: Output ONLY the list in format '• ABBR: Definition'."
    )
    
    try:
        response = llm.invoke(prompt)
        return response.content.strip()
    except Exception as e:
        return f"Error connecting to Cloud API: {e}"

# ----------------------------
# 5. UI Layout
# ----------------------------
st.set_page_config(page_title="Project 2 — Gemini Extractor", layout="wide")
st.title("Doc Chat (Project 2) — Abbreviation Extractor")

if "messages" not in st.session_state: st.session_state.messages = []

with st.sidebar:
    st.header("Upload Document")
    uploaded_file = st.file_uploader("Choose file", type=["pdf", "docx", "txt"])
    if st.button("Clear Chat"):
        st.session_state.messages = []
        st.rerun()

file_content = ""
if uploaded_file:
    with st.spinner("Processing file..."):
        file_content = parse_file(uploaded_file)
    st.success(f"Loaded: {uploaded_file.name}")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Type 'Extract abbreviations' or ask a question..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"): st.markdown(prompt)

    response_text = ""
    with st.chat_message("assistant"):
        is_extract = any(x in prompt.lower() for x in ["abbreviation", "extract", "list"])
        
        if is_extract and file_content:
            with st.spinner("Analyzing with Gemini..."):
                response_text = get_abbreviations(file_content)
                st.markdown(response_text)
                st.download_button("Download Index", response_text, "index.txt")
        elif not file_content:
            response_text = "Please upload a document first."
            st.markdown(response_text)
        else:
            with st.spinner("Thinking..."):
                rag_prompt = f"Answer based on context:\n\nQuestion: {prompt}\n\nContext: {file_content[:20000]}"
                response_text = llm.invoke(rag_prompt).content
                st.markdown(response_text)

    st.session_state.messages.append({"role": "assistant", "content": response_text})
