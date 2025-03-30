import os
import random
import fitz  # PyMuPDF
import pandas as pd
import streamlit as st
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.chains.qa_with_sources import load_qa_with_sources_chain
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.schema import Document

# === CONFIGURATION ===
os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
PDF_PATH = "Johnson-Tile-Guide-2023.pdf"
IMAGE_FOLDER = "extracted_images"

# === TILE TOPIC TO IMAGE MAP ===
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
    splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs = splitter.split_documents(documents)
    embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")
    return FAISS.from_documents(docs, embeddings)

def get_suggestions_from_llm(question):
    prompt = f"Suggest 3 very relevant follow-up sales-related questions based on this user question: '{question}'. Format as a list without numbering."
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.7)
    suggestions_text = llm.invoke(prompt).content.strip()
    return [line.strip("-• ") for line in suggestions_text.split("\n") if line.strip()]

# === STREAMLIT UI ===
st.set_page_config(page_title="JAI - Johnson Tile Chatbot", page_icon="🧱")
st.markdown("""
    <h1 style='text-align: center;'>🤖 JAI — Johnson AI</h1>
    <p style='text-align: center;'>Your smart assistant for Johnson Tiles</p>
    <hr style='border:1px solid #ddd;'>
""", unsafe_allow_html=True)

extract_images_from_pdf(PDF_PATH)
vectorstore = prepare_vectorstore()
qa_chain = RetrievalQA.from_chain_type(
    llm=ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.7),
    retriever=vectorstore.as_retriever()
)

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "suggestions" not in st.session_state:
    st.session_state.suggestions = []

# Clear Chat
if st.button("🗑️ Clear Chat"):
    st.session_state.chat_history = []
    st.session_state.suggestions = []
    st.rerun()

# Display chat history
for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"], unsafe_allow_html=True)

# Get input
prompt = st.chat_input("Ask me anything about Johnson Tiles...")

# Handle suggestion button click
clicked_question = None
for i, suggestion in enumerate(st.session_state.suggestions):
    if st.button(suggestion, key=f"suggestion_{i}"):
        clicked_question = suggestion
        break

if prompt or clicked_question:
    user_query = clicked_question if clicked_question else prompt
    st.session_state.chat_history.append({"role": "user", "content": user_query})
    query = user_query.lower()

    # Handle basic fun responses
    if query in ["hi", "hello", "hi jai", "hello jai"]:
        response = "Hello! I'm JAI 😊 — happy to help you with tile advice. What would you like to know?"
    elif "your name" in query:
        response = "My name is <b>JAI — Johnson AI</b> 🤖. I'm your smart assistant for all things tiles!"
    elif "who are you" in query:
        response = "I'm <b>JAI</b> — your virtual tile expert, built to help you with <b>Johnson products only</b>."
    elif "girlfriend" in query:
        response = "Haha 😄 I’m fully committed to tiles — no time for romance!"
    elif "sing" in query and "song" in query:
        response = random.choice(TILE_SONGS)
    else:
        try:
            response = qa_chain.run(user_query)
        except Exception:
            response = "⚠️ Sorry, I couldn’t understand that. Please ask something related to Johnson Tiles."

        # Display tile images if topic matches
        for topic, page in topic_page_map.items():
            if topic in query:
                for file in os.listdir(IMAGE_FOLDER):
                    if file.startswith(f"page_{page}_img"):
                        st.image(os.path.join(IMAGE_FOLDER, file), caption=f"Example of {topic.title()} Tile")
                        break
                break

    st.session_state.chat_history.append({"role": "assistant", "content": response})
    with st.chat_message("assistant"):
        st.markdown(response, unsafe_allow_html=True)

    # Generate dynamic suggestions
    st.session_state.suggestions = get_suggestions_from_llm(user_query)

# Show suggestion buttons
if st.session_state.suggestions:
    st.markdown("##### 🔍 Suggested Questions:")
    cols = st.columns(len(st.session_state.suggestions))
    for i, suggestion in enumerate(st.session_state.suggestions):
        if cols[i].button(suggestion, key=f"suggestion_btn_{i}"):
            st.session_state.chat_history.append({"role": "user", "content": suggestion})
            st.rerun()  # ✅ Use this instead of st.experimental_rerun()
