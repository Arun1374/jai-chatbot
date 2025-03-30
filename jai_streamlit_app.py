# Enhanced JAI Streamlit UI with better layout, styling, and icons

import os
import random
import fitz
import pandas as pd
import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.schema import Document
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import RetrievalQA

# === CONFIGURATION ===
os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
PDF_PATH = "Johnson-Tile-Guide-2023.pdf"
IMAGE_FOLDER = "extracted_images"

# === TILE TOPIC TO PAGE MAP ===
topic_page_map = {
    "bathroom": 14,
    "parking": 22,
    "cool roof": 30,
    "swimming pool": 24,
    "living room": 18,
    "hospital": 27,
    "industrial": 25
}

TILE_SONGS = [
    "🎵 I'm a tile and I shine so bright, step on me, your room's just right!",
    "🎶 Glossy, matte, or slip-resistant too, Johnson Tiles are made for you!",
    "🎵 Stick with me, and never fall, I grip the ground and beat them all!",
    "🎶 On the floor or on the wall, Johnson Tiles stand tall for all!",
    "🎵 From your kitchen to your bath, I pave the perfect tiled path!"
]

# === FUNCTIONS ===
def extract_images_from_pdf(pdf_path):
    if not os.path.exists(IMAGE_FOLDER):
        os.makedirs(IMAGE_FOLDER)
    if os.listdir(IMAGE_FOLDER):
        return
    doc = fitz.open(pdf_path)
    for page_index in range(len(doc)):
        page = doc[page_index]
        images = page.get_images(full=True)
        for img_index, img in enumerate(images):
            xref = img[0]
            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]
            ext = base_image["ext"]
            filename = f"page_{page_index + 1}_img_{img_index + 1}.{ext}"
            with open(os.path.join(IMAGE_FOLDER, filename), "wb") as f:
                f.write(image_bytes)

def prepare_vectorstore():
    loader = PyPDFLoader(PDF_PATH)
    documents = loader.load()
    splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    pdf_docs = splitter.split_documents(documents)
    embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")
    return FAISS.from_documents(pdf_docs, embeddings)

def generate_suggestions(prompt):
    prompt = prompt.lower()
    base = [
        "Where can I buy tiles?",
        "What tiles are best for bathrooms?",
        "Tell me about anti-skid tiles.",
        "Show tile options for living room.",
        "What’s the size of cool roof tiles?"
    ]
    if "bathroom" in prompt:
        return ["Best slip-resistant bathroom tiles?", "Sizes available for bathroom floors?", "Should I use glossy or matte?"]
    elif "parking" in prompt:
        return ["Best tiles for heavy vehicles?", "Are parking tiles anti-skid?", "Colors available for parking area?"]
    return random.sample(base, 3)

# === STREAMLIT UI ===
st.set_page_config(page_title="JAI - Johnson AI", page_icon="🧱", layout="wide")
st.markdown("""
    <div style='text-align: center;'>
        <h1 style='font-size: 2.5rem;'>🤖 JAI — Johnson AI</h1>
        <p style='color: gray;'>Your smart assistant for tile solutions</p>
    </div>
    <hr style='border:1px solid #eee;'>
""", unsafe_allow_html=True)

extract_images_from_pdf(PDF_PATH)
vectorstore = prepare_vectorstore()
qa = RetrievalQA.from_chain_type(llm=ChatOpenAI(model_name="gpt-3.5-turbo"), chain_type="stuff", retriever=vectorstore.as_retriever())

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
    st.session_state.suggestions = []
    st.session_state.last_prompt = ""

col1, col2 = st.columns([6, 1])
with col2:
    if st.button("🗑️ Clear Chat"):
        st.session_state.chat_history = []
        st.rerun()

for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"], unsafe_allow_html=True)

prompt = st.chat_input("Ask me anything about tiles...")
if prompt:
    st.session_state.chat_history.append({"role": "user", "content": prompt})
    query = prompt.lower()
    response = ""

    if query in ["hi", "hello", "hi jai", "hello jai"]:
        response = "Hello! I'm JAI 😊 — happy to help you with tile advice. What would you like to know?"
    elif "your name" in query:
        response = "My name is <b>JAI — Johnson AI</b> 🤖. I'm your smart assistant for all things tiles!"
    elif "who are you" in query:
        response = "I'm <b>JAI</b> — your virtual tile expert, built to help you with <b>Johnson products only</b>."
    elif "how are you" in query:
        response = "I'm all tiled up and ready to assist you! 😄 What can I help you with today?"
    elif "what can you do" in query:
        response = "I can help you choose the right Johnson tile, explain technical specs, and recommend tile types."
    elif "girlfriend" in query:
        response = "Haha 😄 I’m fully committed to tiles — no time for romance!"
    elif "sing" in query and "song" in query:
        response = random.choice(TILE_SONGS)
    else:
        try:
            response = qa.run(prompt)
        except:
            response = "⚠️ Sorry, I couldn’t understand that. Please ask something related to Johnson Tiles."

        for topic, page in topic_page_map.items():
            if topic in query:
                for file in os.listdir(IMAGE_FOLDER):
                    if file.startswith(f"page_{page}_img"):
                        st.image(os.path.join(IMAGE_FOLDER, file), caption=f"Example of {topic.title()} Tile")
                        break
                break

    st.session_state.chat_history.append({"role": "assistant", "content": response})
    st.session_state.suggestions = generate_suggestions(prompt)
    st.session_state.last_prompt = prompt

    with st.chat_message("assistant"):
        st.markdown(response, unsafe_allow_html=True)

if st.session_state.suggestions:
    st.markdown("##### 🔍 Suggestions:")
    cols = st.columns(len(st.session_state.suggestions))
    for i, q in enumerate(st.session_state.suggestions):
        if cols[i].button(q, key=f"suggestion_{i}_{q}"):
            st.session_state.chat_history.append({"role": "user", "content": q})
            try:
                res = qa.run(q)
                st.session_state.chat_history.append({"role": "assistant", "content": res})
            except:
                st.session_state.chat_history.append({"role": "assistant", "content": "⚠️ Unable to fetch answer."})
            st.rerun()
