import os
import random
import fitz  # PyMuPDF
import pandas as pd
import streamlit as st
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.schema import Document

# === CONFIGURATION ===
os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
PDF_PATH = "Johnson-Tile-Guide-2023.pdf"
EXCEL_PATH = "HRJ DATA.xlsx"
IMAGE_FOLDER = "extracted_images"

# === TILE TOPIC PAGE MAP ===
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

# === EXTRACT PDF IMAGES ===
def extract_images_from_pdf(pdf_path):
    if not os.path.exists(IMAGE_FOLDER):
        os.makedirs(IMAGE_FOLDER)
    if os.listdir(IMAGE_FOLDER):
        return
    doc = fitz.open(pdf_path)
    for page_index in range(len(doc)):
        images = doc[page_index].get_images(full=True)
        for img_index, img in enumerate(images):
            xref = img[0]
            base_image = doc.extract_image(xref)
            ext = base_image["ext"]
            filename = f"page_{page_index+1}_img_{img_index+1}.{ext}"
            with open(os.path.join(IMAGE_FOLDER, filename), "wb") as f:
                f.write(base_image["image"])

# === LOAD EMPLOYEE DATA ===
def load_employee_data(excel_path):
    df = pd.read_excel(excel_path)
    docs = []
    for _, row in df.iterrows():
        content = f"{row['Member Name']} (Employee ID: {row['Member ID']}) works as {row['Designation']} and is based in {row['Location']}."
        docs.append(Document(page_content=content))
    return docs

# === PREPARE VECTOR DB ===
def prepare_vectorstore():
    loader = PyPDFLoader(PDF_PATH)
    docs = loader.load()
    splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    pdf_chunks = splitter.split_documents(docs)
    emp_chunks = load_employee_data(EXCEL_PATH)
    all_docs = pdf_chunks + emp_chunks
    embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")
    return FAISS.from_documents(all_docs, embeddings)

# === FORMAT RESPONSE ===
def format_response(text):
    bullets = [f"- {line.strip()}" for line in text.split("\n") if line.strip()]
    return "\n".join(bullets)

# === STREAMLIT UI ===
st.set_page_config("JAI - Johnson AI", page_icon="🧱")
st.markdown("""
    <h1 style='text-align: center;'>🤖 <b>JAI — Johnson AI</b></h1>
    <p style='text-align: center;'>Your smart assistant for tiles</p>
    <hr style='border:1px solid #ddd;'>
""", unsafe_allow_html=True)

extract_images_from_pdf(PDF_PATH)
vectorstore = prepare_vectorstore()
qa = RetrievalQA.from_chain_type(
    llm=ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.4),
    chain_type="stuff",
    retriever=vectorstore.as_retriever()
)

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

prompt = st.chat_input("Ask me anything about tiles or employees...")
if prompt:
    st.session_state.chat_history.append({"role": "user", "content": prompt})
    query = prompt.lower()

    # Friendly rules
    if query in ["hi", "hello", "hi jai", "hello jai"]:
        response = "<b>Hello!</b> I'm <b>JAI 😊</b> — happy to help you with tile advice. What would you like to know?"
    elif "your name" in query:
        response = "My name is <b>JAI — Johnson AI</b> 🤖. I'm your smart assistant for all things tiles!"
    elif "who are you" in query:
        response = "I'm <b>JAI</b> — your virtual tile expert, built to help you with <b>Johnson products only</b>."
    elif "how are you" in query:
        response = "I'm all <b>tiled up</b> and ready to assist you! 😄 What can I help you with today?"
    elif "what can you do" in query:
        response = "I can help you <b>choose tiles</b>, explain specs, and answer <b>employee</b> queries!"
    elif "girlfriend" in query:
        response = "Haha 😄 I’m fully committed to <b>tiles</b> — no time for romance!"
    elif "born" in query or "built" in query:
        response = "I was born in the <b>H&R Johnson office in Mumbai</b>! Built with ❤️ by <b>Arunkumar Gond</b> under <b>Rohit Chintawar</b>."
    elif "creator" in query or "who made you" in query:
        response = "I was built by <b>Arunkumar Gond</b> and the <b>Digital Team</b> at H&R Johnson. 🙌"
    elif "sing" in query and "song" in query:
        response = random.choice(TILE_SONGS)
    else:
        raw = qa.run(prompt)
        response = f"<div style='font-family: sans-serif;'>{format_response(raw)}</div>"

        # Image if topic detected
        for topic, page in topic_page_map.items():
            if topic in query:
                for file in os.listdir(IMAGE_FOLDER):
                    if file.startswith(f"page_{page}_img"):
                        st.image(os.path.join(IMAGE_FOLDER, file), caption=f"{topic.title()} Tile")
                        break
                break

    st.session_state.chat_history.append({"role": "assistant", "content": response})
    with st.chat_message("assistant"):
        st.markdown(response, unsafe_allow_html=True)
