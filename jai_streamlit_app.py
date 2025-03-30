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
from langchain.schema import Document

# === CONFIGURATION ===
os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
PDF_PATH = "Johnson-Tile-Guide-2023.pdf"
IMAGE_FOLDER = "extracted_images"

# === TILE-TOPIC TO IMAGE PAGE MAP ===
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

# === FUNCTION TO GENERATE SUGGESTIONS BASED ON USER PROMPT ===
def get_suggestions_from_llm(user_input):
    suggestion_prompt = f"""
    The user asked: '{user_input}'.
    Based on this, suggest 3 related questions that the user might be interested in next — strictly focused on tiles, their applications, usage, or styles.

    Provide your suggestions as a list, like:
    1. ...
    2. ...
    3. ...
    """
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.7)
    output = llm.invoke(suggestion_prompt)
    
    suggestions = []
    for line in output.content.split("\n"):
        line = line.strip()
        if line.startswith(("1", "2", "3")):
            question = line.split(".", 1)[1].strip()
            suggestions.append(question)
    return suggestions

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
    pdf_docs = splitter.split_documents(documents)
    embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")
    vectorstore = FAISS.from_documents(pdf_docs, embeddings)
    return vectorstore

# === STREAMLIT UI ===
st.set_page_config(page_title="JAI - (Johnson Artificial Intelligence)", page_icon="🧱")
st.markdown("""
    <h1 style='text-align: center;'>🤖 JAI — Johnson AI</h1>
    <p style='text-align: center;'>Your smart assistant for tiles</p>
    <hr style='border:1px solid #ddd;'>
""", unsafe_allow_html=True)

extract_images_from_pdf(PDF_PATH)
vectorstore = prepare_vectorstore()
qa = RetrievalQA.from_chain_type(llm=ChatOpenAI(model_name="gpt-3.5-turbo"), chain_type="stuff", retriever=vectorstore.as_retriever())

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

col1, col2 = st.columns([6, 1])
with col2:
    if st.button("🗑️ Clear Chat"):
        st.session_state.chat_history = []
        st.rerun()

for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"], unsafe_allow_html=True)

prompt = st.chat_input("Ask me anything about tiles ...")

if prompt:
    st.session_state.chat_history.append({"role": "user", "content": prompt})
    response = ""
    query = prompt.lower()

    if query in ["hi", "hello", "hi jai", "hello jai"]:
        response = "Hello! I'm JAI 😊 — happy to help you with tile advice. What would you like to know?"
    elif "your name" in query:
        response = "My name is <b>JAI — Johnson AI</b> 🤖. I'm your smart assistant for all things tiles!"
    elif "who are you" in query:
        response = "I'm <b>JAI</b> — your virtual tile expert, built to help you with <b>Johnson products only</b>."
    elif "how are you" in query:
        response = "I'm all tiled up and ready to assist you! 😄 What can I help you with today?"
    elif "what can you do" in query:
        response = "I can help you choose the right Johnson tile, explain technical specs, and guide your tile selection journey!"
    elif "girlfriend" in query:
        response = "Haha 😄 I’m fully committed to tiles — no time for romance!"
    elif "born" in query or "built" in query:
        response = "I was born in the <b>H&R Johnson office in Mumbai</b>! Built with ❤️ by <b>Arunkumar Gond</b>, under <b>Rohit Chintawar</b> in the Digital Team."
    elif "creator" in query or "who made you" in query:
        response = "I was proudly built by <b>Arunkumar Gond</b> and the amazing <b>Digital Team</b> under <b>Rohit Chintawar</b> at H&R Johnson. 👌"
    elif "sing" in query and "song" in query:
        response = random.choice(TILE_SONGS)
    else:
        try:
            response = qa.run(prompt)
        except Exception:
            response = "⚠️ Sorry, I couldn’t understand that. Please ask something related to Johnson Tiles."

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

    # === Dynamic Suggestions ===
    suggestions = get_suggestions_from_llm(prompt)
    st.markdown("##### 🔍 Suggested Questions:")
    cols = st.columns(len(suggestions))
    for i, suggestion in enumerate(suggestions):
        if cols[i].button(suggestion, key=f"suggestion_{i}"):
            st.session_state.chat_history.append({"role": "user", "content": suggestion})
            follow_up_response = qa.run(suggestion)
            st.session_state.chat_history.append({"role": "assistant", "content": follow_up_response})
            st.rerun()
