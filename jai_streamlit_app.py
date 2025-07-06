import os
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain.memory import ConversationBufferMemory
from langchain_openai import ChatOpenAI

# === CONFIGURATION ===
os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
PDF_PATH = "Johnson-Tile-Guide-2023-Final-Complete-With-Tables.pdf"

# === VECTORSTORE PREPARATION ===
@st.cache_resource
def prepare_vectorstore():
    loader = PyPDFLoader(PDF_PATH)
    pages = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    split_docs = splitter.split_documents(pages)

    embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")
    return FAISS.from_documents(split_docs, embeddings)

# === RICH RESPONSE BUILDER ===
def get_formatted_response(prompt):
    retrieved_docs = vectorstore.similarity_search(prompt, k=4)
    combined_context = "\n\n".join([doc.page_content for doc in retrieved_docs])

    rich_prompt = f"""
You are JAI — Johnson Tiles AI assistant.

Answer the user's question using the following context:
----------------
{combined_context}
----------------

Please reply in a **friendly, markdown-formatted** style with:
- A clear title or heading
- **Bold labels** where needed
- ✅ Bullet points for features or tips
- 🧱 Emojis for warmth and clarity
- Line breaks for readability
- Markdown only (no HTML)

User's question:
\"{prompt}\"
"""
    return llm.predict(rich_prompt)

# === SMART SUGGESTION GENERATOR ===
def generate_suggestions(user_input):
    lower = user_input.lower()
    if "bathroom" in lower:
        return ["What size tiles are best for bathrooms?", "Are bathroom tiles slip-resistant?", "Glossy or matte for bathroom walls?"]
    elif "parking" in lower:
        return ["Which tiles are durable for parking areas?", "Do you have anti-skid parking tiles?", "Best color tiles for parking?"]
    elif "living room" in lower:
        return ["Best designs for living room tiles?", "Which finish suits living room flooring?", "Is glossy suitable for living rooms?"]
    elif "swimming pool" in lower:
        return ["Tiles suitable for pool decks?", "Are pool tiles anti-slip?", "Can Johnson tiles be used underwater?"]
    elif "industrial" in lower:
        return ["Best tiles for industrial use?", "Can tiles withstand heavy machinery?", "Are Endura tiles chemical resistant?"]
    elif "cool roof" in lower:
        return ["How do cool roof tiles work?", "Do they reduce temperature indoors?", "Which tiles for summer heat?"]
    else:
        return [
            "Which Johnson tiles are best for outdoors?",
            "Where can I buy Johnson tiles?",
            "How do I clean my Johnson tiles?",
            "Do you have eco-friendly Johnson tiles?",
            "Best tiles from Johnson for a modern kitchen?"
        ]

# === STREAMLIT UI SETUP ===
st.set_page_config(page_title="JAI - Johnson AI", page_icon="🧱", layout="centered")
st.markdown("""
    <style>
    body {
        background-image: url('https://www.hrjohnsonindia.com/images/product/bg_wall_tile.jpg');
        background-size: cover;
        background-position: center;
    }
    .block-container {
        background-color: rgba(255, 255, 255, 0.9);
        padding: 2rem;
        border-radius: 12px;
        box-shadow: 0 4px 10px rgba(0, 0, 0, 0.1);
    }
    </style>
    <link rel="icon" href="https://www.hrjohnsonindia.com/favicon.ico" type="image/x-icon">
    <h1 style='text-align: center;'>🤖 JAI — Johnson AI</h1>
    <p style='text-align: center;'>Your smart assistant and tile advisor</p>
    <hr style='border:1px solid #ddd;'>
""", unsafe_allow_html=True)

# === LOAD MODELS ===
vectorstore = prepare_vectorstore()
llm = ChatOpenAI(model_name="gpt-4-1106-preview")
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# === SESSION STATE INITIALIZATION ===
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "user_context" not in st.session_state:
    st.session_state.user_context = {}
if "last_input" not in st.session_state:
    st.session_state.last_input = ""
if "show_suggestions" not in st.session_state:
    st.session_state.show_suggestions = False
if "pending_followups" not in st.session_state:
    st.session_state.pending_followups = []

# === CLEAR CHAT BUTTON ===
if st.button("🗑️ Clear Chat"):
    st.session_state.chat_history = []
    st.session_state.user_context = {}
    st.session_state.show_suggestions = False
    st.session_state.pending_followups = []
    st.rerun()

# === DISPLAY PAST MESSAGES ===
for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"], unsafe_allow_html=True)

# === CHAT INPUT HANDLING ===
prompt = st.chat_input("Ask me anything about Johnson tiles or share your requirement...")

if prompt:
    st.session_state.chat_history.append({"role": "user", "content": prompt})

    allowed_keywords = [
        "johnson", "tiles", "endura", "marbonite", "porselano", "dealer", "showroom",
        "cool roof", "parking", "bathroom", "floor", "wall", "tactile", "industrial",
        "anti-skid", "ceramic", "glazed", "tile selection"
    ]

    if not any(word in prompt.lower() for word in allowed_keywords):
        response = (
            "⚠️ I can only assist with queries related to <b>Johnson Tiles</b>, including product details, design help, or dealer locations.<br><br>"
            "Please ask something like:<br>"
            "• What are the best tiles for bathrooms?<br>"
            "• Where can I find a Johnson Tiles dealer near me?<br>"
            "• Tell me about Endura tiles for industrial use."
        )
    elif st.session_state.pending_followups:
        last_question = st.session_state.pending_followups.pop(0)
        st.session_state.user_context[last_question] = prompt
        followup_query = " ".join([f"{k}: {v}" for k, v in st.session_state.user_context.items()])
        followup_prompt = f"""
You are JAI — Johnson Tiles AI assistant.

User details: {followup_query}

Based on this, suggest the best Johnson tiles with:
- A friendly tone
- Markdown headings and **bold** labels
- ✅ Bullet points
- 🧱 Emojis
- Clean formatting

User follow-up: "{prompt}"
"""
        response = llm.predict(followup_prompt)
    else:
        try:
            response = get_formatted_response(prompt)
        except Exception:
            response = "⚠️ Sorry, I couldn’t understand that. Please ask something related to Johnson Tiles."

    st.session_state.chat_history.append({"role": "assistant", "content": response})
    with st.chat_message("assistant"):
        st.markdown(response, unsafe_allow_html=True)

    st.session_state.last_input = prompt
    st.session_state.show_suggestions = True

# === SHOW SMART SUGGESTIONS ===
if st.session_state.show_suggestions:
    suggestions = generate_suggestions(st.session_state.last_input)
    if suggestions:
        st.markdown("##### 🔍 Suggested Questions:")
        cols = st.columns(min(len(suggestions), 5))
        for i, suggestion in enumerate(suggestions):
            with cols[i % len(cols)]:
                if st.button(suggestion, key=f"suggestion_{i}"):
                    st.session_state.chat_history.append({"role": "user", "content": suggestion})
                    with st.spinner("JAI is typing..."):
                        try:
                            response = get_formatted_response(suggestion)
                        except Exception:
                            response = "⚠️ Sorry, I couldn’t understand that. Please ask something related to Johnson Tiles."
                    st.session_state.chat_history.append({"role": "assistant", "content": response})
                    st.rerun()
