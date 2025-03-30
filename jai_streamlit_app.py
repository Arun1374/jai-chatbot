# === IMPORTS ===
import os
import random
import base64
import pandas as pd
import re
import streamlit as st
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain.chains import RetrievalQA
from fpdf import FPDF

# === CONFIGURATION ===
os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
PDF_PATH = "Johnson-Tile-Guide-2023.pdf"
IMAGE_FOLDER = "extracted_images"
DEALER_EXCEL = "HRJ Tiles CustomeProfileTemplate 21-01-2025.xlsx"

# === TILE IMAGE MAP ===
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

@st.cache_resource
def prepare_vectorstore():
    loader = PyPDFLoader(PDF_PATH)
    documents = loader.load()
    splitter = CharacterTextSplitter(chunk_size=2000, chunk_overlap=300)
    pdf_docs = splitter.split_documents(documents)
    embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")
    vectorstore = FAISS.from_documents(pdf_docs, embeddings)
    return vectorstore

# === DEALER DATA ===
dealer_df = pd.read_excel(DEALER_EXCEL)
dealer_df.columns = [col.strip().lower().replace(" ", "_") for col in dealer_df.columns]

# === BUYING INTENT & DEALER FUNCTIONS ===
def user_intends_to_buy(text):
    buy_keywords = ["buy", "purchase", "get tiles", "dealer", "store", "shop", "distributor"]
    return any(keyword in text.lower() for keyword in buy_keywords)

def extract_city_or_pin(user_input):
    pin_match = re.findall(r"\b\d{6}\b", user_input)
    if pin_match:
        return {"pin_code": pin_match[0]}
    for city in dealer_df["city"].dropna().unique():
        if city.lower() in user_input.lower():
            return {"city": city}
    return {}

def get_dealers(location_info):
    if "pin_code" in location_info:
        dealers = dealer_df[dealer_df["pin_code"] == int(location_info["pin_code"])]
    elif "city" in location_info:
        dealers = dealer_df[dealer_df["city"].str.lower() == location_info["city"].lower()]
    else:
        return None

    if dealers.empty:
        return None

    result = f"\n📍 **Dealers in {location_info.get('city', location_info.get('pin_code'))}**\n"
    for _, row in dealers.iterrows():
        result += f"- **{row['dealer_name']}**\n  📍 {row['address']}\n  📞 {row['contact']}\n  📧 {row['email']}\n\n"
    return result

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
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 5})
qa = RetrievalQA.from_chain_type(llm=ChatOpenAI(model_name="gpt-3.5-turbo"), chain_type="stuff", retriever=retriever)

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "show_suggestions" not in st.session_state:
    st.session_state.show_suggestions = True
if "last_input" not in st.session_state:
    st.session_state.last_input = ""
if "asking_for_dealer" not in st.session_state:
    st.session_state.asking_for_dealer = False

col1, col2 = st.columns([6, 1])
with col2:
    if st.button("🗑️ Clear Chat"):
        st.session_state.chat_history = []
        st.session_state.show_suggestions = True
        st.session_state.asking_for_dealer = False
        st.rerun()

for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"], unsafe_allow_html=True)

prompt = st.chat_input("Ask me anything about tiles ...")

if prompt:
    query = prompt.strip()
    question_words = ("where", "what", "how", "who", "can", "is", "are", "does", "do", "when", "which", "should", "could", "would")
    if query.lower().startswith(question_words) and not query.endswith("?"):
        query += "?"

    st.session_state.chat_history.append({"role": "user", "content": prompt})

    if st.session_state.asking_for_dealer:
        loc = extract_city_or_pin(query)
        dealer_info = get_dealers(loc)
        if dealer_info:
            st.session_state.chat_history.append({"role": "assistant", "content": dealer_info})
            with st.chat_message("assistant"):
                st.markdown(dealer_info, unsafe_allow_html=True)
        else:
            with st.chat_message("assistant"):
                st.markdown("❌ Couldn’t find dealer info for that location. Please try another city or PIN code.")
        st.session_state.asking_for_dealer = False
        st.session_state.show_suggestions = True
        st.stop()

    response = qa.run(query)
    follow_up_msg = ""
    buying_intent_detected = user_intends_to_buy(query)

    if buying_intent_detected:
        loc = extract_city_or_pin(query)
        if loc:
            dealer_info = get_dealers(loc)
            if dealer_info:
                response += f"\n\n{dealer_info}"
            else:
                follow_up_msg = "📍 Would you like me to help you find the nearest dealer? Please share your city or PIN code."
                st.session_state.asking_for_dealer = True
                st.session_state.show_suggestions = False

    for topic, image_files in tile_image_map.items():
        if topic in query.lower():
            st.markdown(f"#### 📸 Example of {topic.title()} Tiles")
            images_per_row = 2
            for i in range(0, len(image_files), images_per_row):
                cols = st.columns(images_per_row)
                for j, image_file in enumerate(image_files[i:i + images_per_row]):
                    image_path = os.path.join(IMAGE_FOLDER, image_file)
                    if os.path.exists(image_path):
                        image_base64 = image_to_base64(image_path)
                        img_html = f"""
                            <a href=\"data:image/jpeg;base64,{image_base64}\" target=\"_blank\">
                                <img src=\"data:image/jpeg;base64,{image_base64}\" style=\"width:100%; border-radius:10px;\" />
                            </a>
                            <p style=\"text-align:center; font-size:14px;\">{image_file.split('.')[0].replace('_', ' ').title()}</p>
                        """
                        with cols[j]:
                            st.markdown(img_html, unsafe_allow_html=True)
            break

    st.session_state.chat_history.append({"role": "assistant", "content": response})
    with st.chat_message("assistant"):
        st.markdown(response, unsafe_allow_html=True)
        if follow_up_msg:
            st.markdown(follow_up_msg)
            col_yes, col_no = st.columns([1, 1])
            with col_yes:
                if st.button("✅ Yes", key="yes_followup"):
                    st.session_state.asking_for_dealer = True
                    st.session_state.show_suggestions = False
            with col_no:
                if st.button("❌ No", key="no_followup"):
                    st.session_state.asking_for_dealer = False
                    st.session_state.show_suggestions = True

    st.session_state.last_input = prompt

if st.session_state.show_suggestions:
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

    suggestions = generate_suggestions(st.session_state.last_input)
    st.markdown("##### 🔍 Suggested Questions:")
    cols = st.columns(len(suggestions))
    for i, suggestion in enumerate(suggestions):
        if cols[i].button(suggestion, key=f"suggestion_{i}"):
            st.session_state.chat_history.append({"role": "user", "content": suggestion})
            response = qa.run(suggestion)
            buying_intent_detected = user_intends_to_buy(suggestion)
            follow_up_msg = ""
            if buying_intent_detected:
                loc = extract_city_or_pin(suggestion)
                if loc:
                    dealer_info = get_dealers(loc)
                    if dealer_info:
                        response += f"\n\n{dealer_info}"
                    else:
                        follow_up_msg = "📍 Would you like me to help you find the nearest dealer? Please share your city or PIN code."
                        st.session_state.asking_for_dealer = True
                        st.session_state.show_suggestions = False

            st.session_state.chat_history.append({"role": "assistant", "content": response})
            with st.chat_message("assistant"):
                st.markdown(response, unsafe_allow_html=True)
                if follow_up_msg:
                    st.markdown(follow_up_msg)
                    col_yes, col_no = st.columns([1, 1])
                    with col_yes:
                        if st.button("✅ Yes", key=f"yes_followup_{i}"):
                            st.session_state.asking_for_dealer = True
                            st.session_state.show_suggestions = False
                    with col_no:
                        if st.button("❌ No", key=f"no_followup_{i}"):
                            st.session_state.asking_for_dealer = False
                            st.session_state.show_suggestions = True
            st.rerun()
