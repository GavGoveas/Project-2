# app(ollama).py
"""
Project 2: Web-Based LLM App for Abbreviation Extraction
Refined with "Sandwich Prompting" to force strict formatting and fixed for Windows networking.
"""
import re
import tempfile
from io import BytesIO

import streamlit as st
import docx2txt
from bs4 import BeautifulSoup
import fitz  # PyMuPDF

# LLM wrapper
from langchain_ollama import ChatOllama

# ----------------------------
# 1. Model Configuration (Fixed for Windows)
# ----------------------------
# We explicitly set base_url to 127.0.0.1 to avoid [WinError 10049] on Windows.
llm = ChatOllama(
    model="llama3.2:latest", 
    temperature=0.1,
    base_url="http://127.0.0.1:11434"
)

# ----------------------------
# 2. PDF Column-Aware Extraction
# ----------------------------
def extract_pdf_columns(file_bytes: bytes) -> str:
    """
    Extract text using bounding-box block extraction to reconstruct reading order
    for 2-column academic PDFs.
    """
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(file_bytes)
        pdf_path = tmp.name

    doc = fitz.open(pdf_path)
    page_texts = []

    for page in doc:
        blocks = page.get_text("blocks")  # (x0, y0, x1, y1, text, block_no)
        # Filter empty blocks
        blocks = [b for b in blocks if isinstance(b[4], str) and b[4].strip()]
        # Sort top-to-bottom then left-to-right (standard reading order)
        blocks_sorted = sorted(blocks, key=lambda b: (round(b[1], 1), round(b[0], 1)))
        
        page_lines = []
        for b in blocks_sorted:
            txt = b[4].strip()
            # Replace multiple newlines inside block to single space to fix mid-sentence breaks
            txt = re.sub(r'\n+', ' ', txt) 
            page_lines.append(txt)
        page_texts.append("\n\n".join(page_lines))

    doc.close()
    return "\n\n".join(page_texts)

# ----------------------------
# 3. Text Cleanup
# ----------------------------
def clean_extracted_text(text: str) -> str:
    """
    Clean typical academic PDF extraction artifacts.
    """
    if not text:
        return ""
    
    # Remove hyphenation at line breaks (e.g. "organi- zation" -> "organization")
    text = re.sub(r"(\w)-\s+([a-z])", r"\1\2", text)
    
    # Fix spacing
    text = text.replace("\xa0", " ")
    text = re.sub(r"[ \t]{2,}", " ", text)
    
    return text.strip()

# ----------------------------
# 4. File Parsing Wrapper
# ----------------------------
def parse_file(uploaded_file) -> str:
    if uploaded_file is None:
        return ""
    filename = uploaded_file.name.lower()
    data = uploaded_file.getvalue()

    if filename.endswith(".pdf"):
        raw = extract_pdf_columns(data)
        return clean_extracted_text(raw)

    if filename.endswith(".txt"):
        return data.decode("utf-8", errors="replace")

    if filename.endswith(".docx"):
        with tempfile.NamedTemporaryFile(suffix=".docx", delete=False) as tf:
            tf.write(data)
            path = tf.name
        return (docx2txt.process(path) or "").strip()

    if filename.endswith(".html") or filename.endswith(".htm"):
        html = data.decode("utf-8", errors="replace")
        soup = BeautifulSoup(html, "html.parser")
        return soup.get_text(separator="\n", strip=True)

    return ""

# ----------------------------
# 5. LLM Extraction Logic (Improved Prompting)
# ----------------------------
def get_abbreviations_from_llm(text: str) -> str:
    """
    Uses the LLM to extract abbreviations.
    Uses 'Sandwich Prompting' and Examples to force Llama 3.2 to follow the format.
    """
    if not text.strip():
        return "No text to analyze."

    # Limit context (approx 7k tokens) to keep the model focused
    # Most abbreviations are defined in the Introduction
    doc_clip = text[:30000] 

    # Stronger prompt with examples and repeated instructions
    prompt = (
        "TASK: Extract a list of scientific/technical abbreviations and their definitions from the text below.\n"
        "RULES:\n"
        "1. Identify abbreviations (2-10 uppercase letters) explicitly defined in the text (e.g., 'Artificial Intelligence (AI)').\n"
        "2. Format the output strictly as a bulleted list: '• ABBR: Full Definition'.\n"
        "3. Sort alphabetically.\n"
        "4. Do NOT summarize the document. Do NOT output sentences. ONLY output the list.\n"
        "5. If a definition is not found, exclude the abbreviation.\n\n"
        "EXAMPLE OUTPUT:\n"
        "• AI: Artificial Intelligence\n"
        "• NASA: National Aeronautics and Space Administration\n\n"
        "--- START OF DOCUMENT TEXT ---\n"
        f"{doc_clip}\n"
        "--- END OF DOCUMENT TEXT ---\n\n"
        "REMINDER: Output ONLY the alphabetical list of abbreviations in the format '• ABBR: Definition'. Do not add introduction text."
    )

    try:
        if hasattr(llm, "invoke"):
            out = llm.invoke(prompt)
            result = getattr(out, "content", str(out))
        else:
            result = llm(prompt)
        
        return result.strip()
    except Exception as e:
        return f"Error querying LLM: {e}"

# ----------------------------
# 6. Streamlit UI
# ----------------------------
st.set_page_config(page_title="Project 2 — Abbreviation Extractor", layout="wide")
st.title("Doc Chat (Project 2) — Abbreviation Index Generator")

if "messages" not in st.session_state:
    st.session_state.messages = []

with st.sidebar:
    st.header("Upload Document")
    uploaded_file = st.file_uploader("Choose file (PDF, DOCX, TXT)", type=["pdf", "docx", "txt", "html"])
    if st.button("Clear chat"):
        st.session_state.messages = []
        st.rerun()

# Parse file immediately upon upload
file_content = ""
if uploaded_file is not None:
    with st.spinner("Processing file..."):
        file_content = parse_file(uploaded_file)
    st.success(f"Loaded: {uploaded_file.name}")

# Display Chat History
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Chat Input
if question := st.chat_input("Ask about the document (or type 'Extract abbreviations')..."):
    # User message
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    # Detect if user wants the abbreviation index
    triggers = ["abbreviation", "index", "extract", "list"]
    is_abbr_task = any(t in question.lower() for t in triggers)

    response_text = ""
    
    with st.chat_message("assistant"):
        if is_abbr_task and file_content:
            with st.spinner("Extracting and formatting abbreviations..."):
                response_text = get_abbreviations_from_llm(file_content)
                st.markdown(response_text)
                
                # Provide Download Button for the Case Study
                st.download_button(
                    label="Download Index (.txt)",
                    data=response_text,
                    file_name="abbreviations.txt",
                    mime="text/plain"
                )
        elif is_abbr_task and not file_content:
             response_text = "Please upload a document first."
             st.markdown(response_text)
        elif not file_content:
             # Basic chat without context
             with st.spinner("Thinking..."):
                if hasattr(llm, "invoke"):
                    out = llm.invoke(question)
                    response_text = getattr(out, "content", str(out))
                else:
                    response_text = llm(question)
                st.markdown(response_text)
        else:
            # RAG-style Q&A
            prompt = f"Answer this question based on the document:\n\nQuestion: {question}\n\nContext: {file_content[:50000]}"
            with st.spinner("Thinking..."):
                try:
                    if hasattr(llm, "invoke"):
                        out = llm.invoke(prompt)
                        response_text = getattr(out, "content", str(out))
                    else:
                        response_text = llm(prompt)
                    st.markdown(response_text)
                except Exception as e:
                    st.error(f"Error: {e}")
                    response_text = "I encountered an error connecting to the model."

    st.session_state.messages.append({"role": "assistant", "content": response_text})