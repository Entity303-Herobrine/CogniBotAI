import streamlit as st
import sqlite3
import requests
import chromadb
import base64
import os
import re
from datetime import datetime

# --- CONFIGURATION & CONSTANTS ---
OLLAMA_API = "https://lhr.life"
CHAT_MODEL = "qwen3:8b"
EMBED_MODEL = "nomic-embed-text"

st.set_page_config(page_title="Cogni AI", page_icon="🤖", layout="wide")

# --- CUSTOM CSS FOR CHATGPT LOOK & FEEL ---
st.markdown("""
<style>
/* Hide default Streamlit elements for a clean app feel */
#MainMenu, footer, header {visibility: hidden;}
.block-container {padding-top: 2rem; padding-bottom: 2rem; max-width: 900px;}

/* Main Layout Containers */
.chat-container {
    display: flex;
    flex-direction: column;
    gap: 24px;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    margin-bottom: 120px;
}

/* Message Rows */
.message-row {
    display: flex;
    align-items: flex-start;
    gap: 16px;
    padding: 12px;
    border-radius: 8px;
}
.user-row {
    background-color: transparent;
}
.ai-row {
    background-color: #f7f7f8;
    border: 1px solid #e5e5e5;
}

/* Avatars */
.avatar {
    width: 36px;
    height: 36px;
    border-radius: 50%;
    object-fit: cover;
    flex-shrink: 0;
}
.user-avatar {
    background-color: #543A3A;
    display: flex;
    align-items: center;
    justify-content: center;
    color: white;
    font-weight: bold;
    font-size: 14px;
}

/* Text Contents */
.message-content {
    color: #2d3748;
    font-size: 16px;
    line-height: 1.6;
    width: 100%;
}
.message-content p {
    margin-bottom: 8px;
}
.message-content h1, .message-content h2, .message-content h3 {
    margin-top: 16px;
    margin-bottom: 8px;
    color: #1a202c;
}

/* ChatGPT Code Block Styling */
pre {
    background-color: #1e1e1e !important;
    color: #f8f8f2 !important;
    padding: 16px !important;
    border-radius: 8px !important;
    overflow-x: auto !important;
    font-family: 'Fira Code', Consolas, Monaco, monospace !important;
    margin: 12px 0 !important;
    border: 1px solid #333;
}
code {
    background-color: #f1f1f1;
    padding: 2px 6px;
    border-radius: 4px;
    font-family: monospace;
    font-size: 14px;
}
pre code {
    background-color: transparent !important;
    padding: 0 !important;
    font-size: 14px !important;
}
</style>
""", unsafe_allow_html=True)

# --- HELPER: BASE64 LOGO ENCODER ---
def get_base64_image(image_path):
    if os.path.exists(image_path):
        with open(image_path, "rb") as img_file:
            return f"data:image/png;base64,{base64.b64encode(img_file.read()).decode()}"
    # Fallback placeholder icon if logo.png is missing
    return "https://flaticon.com"

LOGO_BASE64 = get_base64_image("logo.png")

# --- BACKEND DB INITIALIZATION ---
@st.cache_resource
def init_databases():
    chroma_client = chromadb.PersistentClient(path="./cogni_memory")
    vector_col = chroma_client.get_or_create_collection(name="long_term_store")
    
    conn = sqlite3.connect("cogni_chats.db", check_same_thread=False)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS system_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT,
            role TEXT,
            content TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    return conn, vector_col

db_conn, vector_collection = init_databases()

# --- BACKEND MEMORY ENGINE ---
class CogniMemory:
    @staticmethod
    def get_embedding(text):
        try:
            res = requests.post(f"{OLLAMA_API}/embeddings", json={"model": EMBED_MODEL, "prompt": text})
            return res.json()["embedding"]
        except:
            return [0.0] * 768  # Fallback blank array vector

    @staticmethod
    def save_chat(session_id, role, content):
        cursor = db_conn.cursor()
        cursor.execute("INSERT INTO system_logs (session_id, role, content) VALUES (?, ?, ?)", (session_id, role, content))
        db_conn.commit()

    @staticmethod
    def get_short_term(session_id, limit=6):
        cursor = db_conn.cursor()
        cursor.execute("SELECT role, content FROM system_logs WHERE session_id = ? ORDER BY timestamp DESC LIMIT ?", (session_id, limit))
        return [{"role": r, "content": c} for r, c in reversed(cursor.fetchall())]

    @classmethod
    def get_long_term(cls, query):
        try:
            q_vector = cls.get_embedding(query)
            results = vector_collection.query(query_embeddings=[q_vector], n_results=2)
            return [doc for sublist in results['documents'] for doc in sublist] if results['documents'] else []
        except:
            return []

    @classmethod
    def archive(cls, session_id, user_p, ai_r):
        combined = f"User: {user_p} \nCogni: {ai_r}"
        vec = cls.get_embedding(combined)
        doc_id = f"{session_id}_{int(datetime.utcnow().timestamp())}"
        vector_collection.add(embeddings=[vec], documents=[combined], ids=[doc_id])

# --- DYNAMIC LENGTH SYSTEM ENGINE ENGINE ---
def analyze_prompt_complexity(prompt):
    """
    Evaluates prompt layout to dynamically adjust output boundaries via instructions.
    """
    programming_signals = ['code', 'program', 'python', 'script', 'function', 'html', 'write a', 'c++', 'java', 'bug', 'compile']
    complex_signals = ['explain', 'why', 'how to', 'math', 'calculate', 'prove', 'step by step', 'solve', 'difference between']
    
    prompt_lower = prompt.lower()
    is_programming = any(sig in prompt_lower for sig in programming_signals)
    is_complex = any(sig in prompt_lower for sig in complex_signals)
    
    if is_programming:
        return "COMPLEX_PROGRAMMING"
    elif is_complex or len(prompt.split()) > 12:
        return "COMPLEX_EXPLANATION"
    else:
        return "SIMPLE_CONVERSATION"

# --- RENDER ENGINE FOR THE INTERFACE ---
def render_chat_history(history):
    chat_html = '<div class="chat-container">'
    for msg in history:
        if msg["role"] == "user":
            chat_html += f"""
            <div class="message-row user-row">
                <div class="avatar user-avatar">U</div>
                <div class="message-content"><strong>You</strong><br>{msg['content']}</div>
            </div>
            """
        elif msg["role"] == "assistant":
            pass
    chat_html += '</div>'
    return chat_html

# --- MAIN ENGINE PROCESS RUNNER ---
def run_cogni_core():
    # Session identity instantiation
    if "session_id" not in st.session_state:
        st.session_state.session_id = "session_" + str(int(datetime.utcnow().timestamp()))
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Application Header Title
    st.markdown("<h2 style='text-align: center; color: #1a202c;'>Cogni AI Engine Workspace</h2>", unsafe_allow_html=True)

    # Render loop for existing items inside State
    for msg in st.session_state.chat_history:
        if msg["role"] == "user":
            st.markdown(f"""
            <div class="message-row user-row">
                <div class="avatar user-avatar">U</div>
                <div class="message-content"><strong>You</strong><br>{msg['content']}</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            # Container boundary layout wrapper
            st.markdown(f"""
            <div class="message-row ai-row">
                <img src="{LOGO_BASE64}" class="avatar">
                <div class="message-content"><strong>Cogni</strong>
            """, unsafe_allow_html=True)
            st.markdown(msg['content']) # Native rendering handles backticks, blocks, headings cleanly
            st.markdown("</div></div>", unsafe_allow_html=True)

    # Base prompt input tracking footer configuration layout
    with st.sidebar:
        st.markdown(f"<img src='{LOGO_BASE64}' style='width:80px; display:block; margin:auto;'>", unsafe_allow_html=True)
        st.markdown("<h3 style='text-align:center;'>Cogni Configuration</h3>", unsafe_allow_html=True)
        st.write("Engine state online. Ready to accept structural instructions.")
        if st.button("Clear Chat State"):
            st.session_state.chat_history = []
            st.rerun()

    # Sticky Input Layout at bottom using standard structural form components
    with st.form(key="chat_form", clear_on_submit=True):
        user_input = st.text_input("Message Cogni...", placeholder="Ask a question or provide code layout updates...", key="input_field")
        submit_button = st.form_submit_button(label="Send")

    if submit_button and user_input:
        # Append User Input immediately
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        CogniMemory.save_chat(st.session_state.session_id, "user", user_input)

        # Analyze Complexity & Dynamically Frame Guidelines
        complexity = analyze_prompt_complexity(user_input)
        if complexity == "COMPLEX_PROGRAMMING":
            length_instruction = "The user is asking for a program/script. Provide a thorough response. Write clear, complete, and functional code blocks inside standard Markdown triple backticks. Explain what the variables and core structures do step-by-step."
        elif complexity == "COMPLEX_EXPLANATION":
            length_instruction = "The user is asking a complex conceptual or mathematical question. Provide a comprehensive, structured response with definitions, deep analytical explanations, and bold headings to break down items."
        else:
            length_instruction = "The user is asking a simple question or greeting you. Keep your response brief, conversational, and direct to the point. Do not exceed 2-3 sentences."

        # Fetch relevant historical contexts
        past_memories = CogniMemory.get_long_term(user_input)
        memory_context = "\n".join(past_memories)

        # Inject context + dynamic instructions directly into system layout
        system_prompt = f"""You are Cogni, a highly capable, structurally refined AI assistant.
You use formatting beautifully: use bolding (** text **), clear headers (### Header), and emojis where appropriate to make items readable.

CRITICAL OUTPUT LENGTH INSTRUCTION: {length_instruction}

Historical Context Records:
{memory_context}"""

        # Generate request message architecture payload
        messages_payload = [{"role": "system", "content": system_prompt}]
        messages_payload.extend(CogniMemory.get_short_term(st.session_state.session_id))
        messages_payload.append({"role": "user", "content": user_input})

        # Process response with spinning status indicator
        with st.spinner("Cogni is processing..."):
            try:
                res = requests.post(
                    f"{OLLAMA_API}/chat", 
                    json={
                        "model": CHAT_MODEL,
                        "messages": messages_payload,
                        "stream": False
                    }
                )
                ai_response = res.json()["message"]["content"]
            except Exception as e:
                ai_response = f"⚠️ Engine Link Exception: Could not establish a connection to local Ollama runtime engine pipeline server. Details: {str(e)}"

        # Commit updates
        st.session_state.chat_history.append({"role": "assistant", "content": ai_response})
        CogniMemory.save_chat(st.session_state.session_id, "assistant", ai_response)
        CogniMemory.archive(st.session_state.session_id, user_input, ai_response)
        st.rerun()

if __name__ == "__main__":
    run_cogni_core()

   
