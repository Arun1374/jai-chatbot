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
EXCEL_PATH = "HRJ Tiles CustomeProfileTemplate 21-01-2025.xlsx"

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

def prepare_vectorstore():
    loader = PyPDFLoader(PDF_PATH)
    documents = loader.load()
    splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    pdf_docs = splitter.split_documents(documents)
    embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")
    vectorstore = FAISS.from_documents(pdf_docs, embeddings)
    return vectorstore

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

def search_dealers(user_input, df):
    results = []
    query = user_input.lower()
    pin_codes = [str(int(pin)) for pin in df["PIN_CODE"].dropna().unique() if str(pin).isdigit()]
    for pin in pin_codes:
        if pin in query:
            matches = df[df["PIN_CODE"] == int(pin)]
            results.extend(matches.to_dict(orient="records"))
    cities = df["CITY"].dropna().unique()
    for city in cities:
        if city.lower() in query:
            matches = df[df["CITY"].str.lower() == city.lower()]
            results.extend(matches.to_dict(orient="records"))
    seen = set()
    unique_results = []
    for r in results:
        key = (r["NAME"], r["CITY"], r["PIN_CODE"])
        if key not in seen:
            seen.add(key)
            unique_results.append(r)
        if len(unique_results) >= 3:
            break
    return unique_results

# === STREAMLIT APP ===
st.set_page_config(page_title="JAI - (Johnson Artificial Intelligence)", page_icon="🧱")
st.markdown("""
    <h1 style='text-align: center;'>🤖 JAI — Johnson AI</h1>
    <p style='text-align: center;'>Your smart assistant for tiles</p>
    <hr style='border:1px solid #ddd;'>
""", unsafe_allow_html=True)

extract_images_from_pdf(PDF_PATH)
vectorstore = prepare_vectorstore()
qa = RetrievalQA.from_chain_type(llm=ChatOpenAI(model_name="gpt-3.5-turbo"), chain_type="stuff", retriever=vectorstore.as_retriever())
dealer_df = pd.read_excel(EXCEL_PATH, sheet_name="Sheet1")

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

for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"], unsafe_allow_html=True)

prompt = st.chat_input("Ask me anything about tiles ...")

if prompt:
    st.session_state.chat_history.append({"role": "user", "content": prompt})
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
        response = "I can help you choose the right Johnson tile, explain technical specs, and suggest suitable tiles for every space!"
    elif "girlfriend" in query:
        response = "Haha 😄 I’m fully committed to tiles — no time for romance!"
    elif "born" in query or "built" in query:
        response = "I was born in the <b>H&R Johnson office in Mumbai</b>! Built with ❤️ by <b>Arunkumar Gond</b>, who works under <b>Rohit Chintawar</b> in the Digital Team."
    elif "creator" in query or "who made you" in query:
        response = "I was proudly built by <b>Arunkumar Gond</b> and the amazing <b>Digital Team</b> under <b>Rohit Chintawar</b> at H&R Johnson. 🙌"
    elif "sing" in query and "song" in query:
        response = random.choice(TILE_SONGS)
    elif "buy" in query or "dealer" in query or "distributor" in query:
        dealers = search_dealers(prompt, dealer_df)
        if dealers:
            response = "<b>🏪 You can reach out to these dealers/distributors near you:</b><br><ul>"
            for d in dealers:
                response += f"<li><b>{d['NAME']}</b>, {d['ADDRESS']} - {d['CITY']} ({int(d['PIN_CODE'])})<br>📞 {d.get('MOBILE_1', 'N/A')}</li>"
            response += "</ul>"
        else:
            response = "⚠️ Sorry, I couldn't find a dealer for your location. Please try a different city or PIN code."
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
                if "buy" in suggestion or "dealer" in suggestion:
                    dealers = search_dealers(suggestion, dealer_df)
                    if dealers:
                        response = "<b>🏪 You can reach out to these dealers/distributors near you:</b><br><ul>"
                        for d in dealers:
                            response += f"<li><b>{d['NAME']}</b>, {d['ADDRESS']} - {d['CITY']} ({int(d['PIN_CODE'])})<br>📞 {d.get('MOBILE_1', 'N/A')}</li>"
                        response += "</ul>"
                    else:
                        response = "⚠️ Sorry, I couldn't find a dealer for your location."
                else:
                    response = qa.run(suggestion)
            except Exception:
                response = "⚠️ Sorry, I couldn’t understand that."
            st.session_state.chat_history.append({"role": "assistant", "content": response})
            st.rerun()
