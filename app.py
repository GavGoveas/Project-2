# app.py
"""
Project 2: Abbreviation Extractor (Gemini Robust Version)
Uses Google's Native SDK with Auto-Model Selection to prevent 404 errors.
"""
import re
import tempfile
import streamlit as st
import docx2txt
from bs4 import BeautifulSoup
import fitz  # PyMuPDF
import google.generativeai as genai

# ----------------------------
# 1. Model Configuration (Auto-Selector)
# ----------------------------
# Initialize a global variable for the model
model = None

try:
    # Configure with the key
    genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])
    
    # SMART SELECTION: Ask the API what models are actually available to this key
    # This prevents the "404 Not Found" error by never guessing.
    available = [m.name for m in genai.list_models() if 'generateContent' in m.supported_generation_methods]
    
    # Priority list: Try Flash first, then Pro, then whatever is available
    model_name = None
    if 'models/gemini-1.5-flash' in available:
        model_name = 'gemini-1.5-flash'
    elif 'models/gemini-pro' in available:
        model_name = 'gemini-pro'
    elif available:
        model_name = available[0] # Fallback to the first available model
        
    if model_name:
        model = genai.GenerativeModel(model_name)
        # Store the name in session state to display it (optional debugging)
        if "model_name" not in st.session_state:
            st.session_state.model_name = model_name
    else:
        st.error("Error: No compatible Gemini models found for this API key.")

except Exception as e:
    st.error(f"Configuration Error: {e}")

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
    if not model: return "Error: Model not loaded."
    
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
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        return f"Error connecting to Google API: {e}"

# ----------------------------
# 5. UI Layout
# ----------------------------
st.set_page_config(page_title="Project 2 — Gemini Extractor", layout="wide")
st.title("Doc Chat (Project 2) — Abbreviation Extractor")

if "messages" not in st.session_state: st.session_state.messages = []

# --- TEAM CREDITS ---
with st.sidebar:
    st.header("Team Members")
    st.write("• Regan DeQuasie")
    st.write("• Kevin Magallon")
    st.write("• Gavrel Goveas")
    st.divider()
    
    # Debug info to verify which model was picked
    if "model_name" in st.session_state:
        st.caption(f"Running on: {st.session_state.model_name}")
    
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
                if model:
                    rag_prompt = f"Answer based on context:\n\nQuestion: {prompt}\n\nContext: {file_content[:20000]}"
                    try:
                        response = model.generate_content(rag_prompt)
                        response_text = response.text
                    except Exception as e:
                        response_text = f"Error: {e}"
                else:
                    response_text = "Model failed to initialize. Check API Key."
                st.markdown(response_text)

    st.session_state.messages.append({"role": "assistant", "content": response_text})
