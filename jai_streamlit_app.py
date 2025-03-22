import os
import random
import fitz  # PyMuPDF
import pandas as pd
import streamlit as st
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import PyPDFLoader
from langchain.chains import RetrievalQA
from langchain.schema import Document

# === CONFIGURATION ===
os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
PDF_PATH = "Johnson-Tile-Guide-2023.pdf"
EXCEL_PATH = "HRJ DATA.xlsx"
IMAGE_FOLDER = "extracted_images"

# === IMAGE EXTRACTION ===
def extract_images_from_pdf(pdf_path):
    if not os.path.exists(IMAGE_FOLDER):
        os.makedirs(IMAGE_FOLDER)
    if os.listdir(IMAGE_FOLDER):
        print("📁 Images already extracted.")
        return

    print("📸 Extracting images from PDF...")
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
    print("✅ Image extraction complete.")

# === EMPLOYEE EXCEL TO TEXT ===
def load_employee_data(excel_path):
    df = pd.read_excel(excel_path)
    employee_docs = []
    for _, row in df.iterrows():
        text = f"{row['Member Name']} (Employee ID: {row['Member ID']}) works as {row['Designation']} and is based in {row['Location']}."
        employee_docs.append(Document(page_content=text))
    return employee_docs

# === LOAD ALL CONTENT ===
def prepare_vectorstore(pdf_path, excel_path):
    print("🔎 Loading PDF and employee records...")
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    pdf_docs = splitter.split_documents(documents)
    emp_docs = load_employee_data(excel_path)
    all_docs = pdf_docs + emp_docs

    print("🔐 Creating vector index...")
    embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")
    vectorstore = FAISS.from_documents(all_docs, embeddings)
    return vectorstore

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

# === TILE SONGS ===
TILE_SONGS = [
    "🎵 I'm a tile and I shine so bright, step on me, your room's just right!",
    "🎶 Glossy, matte, or slip-resistant too, Johnson Tiles are made for you!",
    "🎵 Stick with me, and never fall, I grip the ground and beat them all!",
    "🎶 On the floor or on the wall, Johnson Tiles stand tall for all!",
    "🎵 From your kitchen to your bath, I pave the perfect tiled path!"
]

# === MAIN CHAT LOOP ===
def run_chatbot():
    extract_images_from_pdf(PDF_PATH)
    vectorstore = prepare_vectorstore(PDF_PATH, EXCEL_PATH)

    qa = RetrievalQA.from_chain_type(
        llm=ChatOpenAI(model_name="gpt-3.5-turbo"),
        chain_type="stuff",
        retriever=vectorstore.as_retriever()
    )

    print("\n🤖 Welcome! I'm JAI – Johnson AI, your personal tile assistant.\n")

    while True:
        query = input("🧱 Ask me anything about tiles (or type 'exit'): ")
        if query.lower() in ["exit", "quit"]:
            print("👋 Goodbye from JAI. Stay stylish and informative! 🧱")
            break

        lower_query = query.lower().strip()
        image_path = None

        # Friendly personality (strict match to avoid false triggers)
        if lower_query in ["hi", "hello", "hi jai", "hello jai"]:
            result = "Hello! I'm JAI 😊 — happy to help you with tile advice. What would you like to know?"
        elif "your name" in lower_query:
            result = "My name is JAI — Johnson AI 🤖. I'm your smart assistant for all things tiles!"
        elif "who are you" in lower_query:
            result = "I'm JAI — your virtual tile expert, built to help you with Johnson products."
        elif "how are you" in lower_query:
            result = "I'm all tiled up and ready to assist you! 😄 What can I help you with today?"
        elif "what can you do" in lower_query:
            result = "I can help you choose the right Johnson tile, explain technical specs, and answer about employees if you ask!"
        elif "girlfriend" in lower_query:
            result = "Haha 😄 I’m fully committed to tiles — no time for romance!"
        elif "born" in lower_query or "built" in lower_query:
            result = "I was born in the H&R Johnson office in Mumbai! Built with ❤️ by Arunkumar Gond, who works under Rohit Chintawar in the Digital Team."
        elif "creator" in lower_query or "who made you" in lower_query:
            result = "I was proudly built by Arunkumar Gond and the amazing Digital Team under Rohit Chintawar at H&R Johnson. 🙌"
        elif "sing" in lower_query and "song" in lower_query:
            result = random.choice(TILE_SONGS)
        else:
            result = qa.run(query)

            # Show image if topic matches
            for topic, page_num in topic_page_map.items():
                if topic in lower_query:
                    for file in os.listdir(IMAGE_FOLDER):
                        if file.startswith(f"page_{page_num}_img"):
                            image_path = os.path.join(IMAGE_FOLDER, file)
                            break
                    break

        print("\n📘 JAI Says:", result)
        if image_path:
            print(f"🖼️ See image: {image_path}")
        print("\n")

if __name__ == "__main__":
    run_chatbot()
