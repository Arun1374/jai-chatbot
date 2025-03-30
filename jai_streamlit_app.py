import os
import random
import base64
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

# === TILE-TOPIC TO IMAGE MAP ===
tile_image_map = {
    "bathroom": ["bathroom_1.jpg", "bathroom_2.jpg", "bathroom_3.jpg", "bathroom_4.jpg"],
    "parking": ["parking_1.jpg", "parking_2.jpg"],
    "cool roof": ["cool_roof_1.jpg"],
    "swimming pool": ["swimming_pool_1.jpg", "swimming_pool_2.jpg"],
    "living room": ["living_room_1.jpg", "living_room_2.jpg"],
    "hospital": ["hospital_1.jpg"],
    "industrial": ["industrial_1.jpg", "industrial_2.jpg"]
}

TILE_SONGS = [
    "🎵 I'm a tile and I shine so bright, step on me, your room's just right!",
    "🎶 Glossy, matte, or slip-resistant too, Johnson Tiles are made for you!",
    "🎵 Stick with me, and never fall, I grip the ground and beat them all!",
    "🎶 On the floor or on the wall, Johnson Tiles stand tall for all!",
    "🎵 From your kitchen to your bath, I pave the perfect tiled path!"
]

# === VECTOR STORE SETUP ===
@st.cache_resource
def prepare_vectorstore():
    loader = PyPDFLoader(PDF_PATH)
    documents = loader.load()
    splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    pdf_docs = splitter.split_documents(documents)
    embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")
    vectorstore = FAISS.from_documents(pdf_docs, embeddings)
    return vectorstore

# === SUGGESTION LOGIC ===
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
        return ["Which tiles are best for outdoors?", "Where can I buy Johnson tiles?", "How do I clean my tiles?"]

# === UTILITY ===
def image_to_base64(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()

# === STREAMLIT UI ===
st.set_page_config(page_title="JAI - (Johnson Artificial Intelligence)", page_icon="🧱")
st.markdown("""
    <h1 style='text-align: center;'>🤖 JAI — Johnson AI</h1>
    <p style='text-align: center;'>Your smart assistant for tiles</p>
    <hr style='border:1px solid #ddd;'>
""", unsafe_allow_html=True)

vectorstore = prepare_vectorstore()
qa = RetrievalQA.from_chain_type(llm=ChatOpenAI(model_name="gpt-3.5-turbo"), chain_type="stuff", retriever=vectorstore.as_retriever())

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "show_suggestions" not in st.session_state:
    st.session_state.show_suggestions = False

if "last_input" not in st.session_state:
    st.session_state.last_input = ""

col1, col2 = st.columns([6, 1])
with col2:
    if st.button("🗑️ Clear Chat"):
        st.session_state.chat_history = []
        st.session_state.show_suggestions = False
        st.rerun()

# Display chat history
for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"], unsafe_allow_html=True)

# Chat input box
prompt = st.chat_input("Ask me anything about tiles ...")

if prompt:
    st.session_state.chat_history.append({"role": "user", "content": prompt})
    query = prompt.lower()

    # Predefined responses
    if query in ["hi", "hello", "hi jai", "hello jai"]:
        response = "Hello! I'm JAI 😊 — happy to help you with tile advice. What would you like to know?"
    elif "your name" in query:
        response = "My name is <b>JAI — Johnson AI</b> 🤖. I'm your smart assistant for all things tiles!"
    elif "who are you" in query:
        response = "I'm <b>JAI</b> — your virtual tile expert, built to help you with <b>Johnson products only</b>."
    elif "how are you" in query:
        response = "I'm all tiled up and ready to assist you! 😄 What can I help you with today?"
    elif "what can you do" in query:
        response = "I can help you choose the right Johnson tile, explain technical specs, and suggest suitable tiles for every space!"
    elif "girlfriend" in query:
        response = "Haha 😄 I’m fully committed to tiles — no time for romance!"
    elif "born" in query or "built" in query:
        response = "I was born in the <b>H&R Johnson office in Mumbai</b>! Built with ❤️ by <b>Arunkumar Gond</b>, who works under <b>Rohit Chintawar</b> in the Digital Team."
    elif "creator" in query or "who made you" in query:
        response = "I was proudly built by <b>Arunkumar Gond</b> and the amazing <b>Digital Team</b> under <b>Rohit Chintawar</b> at H&R Johnson. 🙌"
    elif "sing" in query and "song" in query:
        response = random.choice(TILE_SONGS)
    else:
        try:
            response = qa.run(prompt)
        except Exception:
            response = "⚠️ Sorry, I couldn’t understand that. Please ask something related to Johnson Tiles."

        # === Show matching tile images (MULTIPLE + Click-to-Zoom)
        for topic, image_files in tile_image_map.items():
            if topic in query:
                st.markdown(f"#### 📸 Example of {topic.title()} Tiles")
                images_per_row = 2
                for i in range(0, len(image_files), images_per_row):
                    cols = st.columns(images_per_row)
                    for j, image_file in enumerate(image_files[i:i + images_per_row]):
                        image_path = os.path.join(IMAGE_FOLDER, image_file)
                        if os.path.exists(image_path):
                            image_base64 = image_to_base64(image_path)
                            img_html = f"""
                                <a href="data:image/jpeg;base64,{image_base64}" target="_blank">
                                    <img src="data:image/jpeg;base64,{image_base64}" style="width:100%; border-radius:10px;" />
                                </a>
                                <p style="text-align:center; font-size:14px;">{image_file.split('.')[0].replace('_', ' ').title()}</p>
                            """
                            with cols[j]:
                                st.markdown(img_html, unsafe_allow_html=True)
                break

    # Show response
    st.session_state.chat_history.append({"role": "assistant", "content": response})
    with st.chat_message("assistant"):
        st.markdown(response, unsafe_allow_html=True)

    # Enable suggestion section
    st.session_state.last_input = prompt
    st.session_state.show_suggestions = True

# === SUGGESTIONS ===
if st.session_state.show_suggestions:
    suggestions = generate_suggestions(st.session_state.last_input)
    st.markdown("##### 🔍 Suggested Questions:")
    cols = st.columns(len(suggestions))
    for i, suggestion in enumerate(suggestions):
        if cols[i].button(suggestion, key=f"suggestion_{i}"):
            st.session_state.chat_history.append({"role": "user", "content": suggestion})
            try:
                response = qa.run(suggestion)
            except Exception:
                response = "⚠️ Sorry, I couldn’t understand that. Please ask something related to Johnson Tiles."
            st.session_state.chat_history.append({"role": "assistant", "content": response})
            st.rerun()
