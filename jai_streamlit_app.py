import os
import random
import fitz  # PyMuPDF
import pandas as pd
import streamlit as st
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.schema import Document

# === CONFIGURATION ===
os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
PDF_PATH = "Johnson-Tile-Guide-2023.pdf"
EXCEL_PATH = "HRJ DATA.xlsx"
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

def load_employee_data(excel_path):
    df = pd.read_excel(excel_path)
    employee_docs = []
    for _, row in df.iterrows():
        text = f"{row['Member Name']} (Employee ID: {row['Member ID']}) works as {row['Designation']} and is based in {row['Location']}."
        employee_docs.append(Document(page_content=text, metadata={"source": "employee_excel"}))
    return employee_docs

def prepare_vectorstore():
    loader = PyPDFLoader(PDF_PATH)
    documents = loader.load()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100,
        separators=["\n\n", "\n", ".", " "]
    )
    pdf_docs = splitter.split_documents(documents)
    for doc in pdf_docs:
        doc.metadata = {"source": "tile_pdf"}

    emp_docs = load_employee_data(EXCEL_PATH)
    all_docs = pdf_docs + emp_docs
    embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")
    vectorstore = FAISS.from_documents(all_docs, embeddings)
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
qa = RetrievalQAWithSourcesChain.from_chain_type(
    llm=ChatOpenAI(model_name="gpt-3.5-turbo"),
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
        st.markdown(msg["content"])

prompt = st.chat_input("Ask me anything about tiles ...")
if prompt:
    with st.chat_message("user"):
        st.markdown(prompt)

    st.session_state.chat_history.append({"role": "user", "content": prompt})
    response = ""
    query = prompt.lower()

    if query in ["hi", "hello", "hi jai", "hello jai"]:
        response = "Hello! I'm JAI 😊 — happy to help you with tile advice. What would you like to know?"
    elif "your name" in query:
        response = "My name is JAI — Johnson AI 🤖. I'm your smart assistant for all things tiles!"
    elif "who are you" in query:
        response = "I'm JAI — your virtual tile expert, built to help you with Johnson products."
    elif "how are you" in query:
        response = "I'm all tiled up and ready to assist you! 😄 What can I help you with today?"
    elif "what can you do" in query:
        response = "I can help you choose the right Johnson tile, explain technical specs, and answer about employees if you ask!"
    elif "girlfriend" in query:
        response = "Haha 😄 I’m fully committed to tiles — no time for romance!"
    elif "born" in query or "built" in query:
        response = "I was born in the H&R Johnson office in Mumbai! Built with ❤️ by Arunkumar Gond, who works under Rohit Chintawar in the Digital Team."
    elif "creator" in query or "who made you" in query:
        response = "I was proudly built by Arunkumar Gond and the amazing Digital Team under Rohit Chintawar at H&R Johnson. 🙌"
    elif "sing" in query and "song" in query:
        response = random.choice(TILE_SONGS)
    else:
        result = qa({"query": prompt})
        response = result.get("answer", "Sorry, I couldn't find an answer.")

        image_found = False
        for topic, page in topic_page_map.items():
            if topic in query:
                for file in os.listdir(IMAGE_FOLDER):
                    if file.startswith(f"page_{page}_img"):
                        st.image(os.path.join(IMAGE_FOLDER, file), caption=f"Example of {topic.title()} Tile")
                        image_found = True
                        break
                if not image_found:
                    st.warning(f"No tile image found for '{topic.title()}' in the PDF.")
                break

    st.session_state.chat_history.append({"role": "assistant", "content": response})
    with st.chat_message("assistant"):
        st.markdown(response)
