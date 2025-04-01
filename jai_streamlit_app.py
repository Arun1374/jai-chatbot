import os
import random
import base64
import streamlit as st
import pandas as pd
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain.chains import RetrievalQA

# === CONFIGURATION ===
os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
PDF_PATH = "Johnson-Tile-Guide-2023-Final-Complete-With-Tables.pdf"
IMAGE_FOLDER = "extracted_images"

@st.cache_resource
def prepare_vectorstore():
    loader = PyPDFLoader(PDF_PATH)
    documents = loader.load()
    splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    pdf_docs = splitter.split_documents(documents)
    embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")
    return FAISS.from_documents(pdf_docs, embeddings)

def image_to_base64(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()

# === RESTRICT TO TILE-RELATED QUERIES ONLY ===
tile_keywords = [
    "tile", "tiles", "johnson", "grout", "floor", "wall", "bathroom", "parking",
    "living room", "glossy", "matte", "anti-skid", "endura", "dealer",
    "purchase", "cool roof", "kitchen", "ceramic", "porcelain", "swimming pool"
]

other_brands = ["kajaria", "somany", "nitco", "orientbell", "asian granito", "hindware", "vora", "hrg"]

def is_tile_related(query):
    return any(keyword in query.lower() for keyword in tile_keywords)

# === DYNAMIC SUGGESTIONS ===
def generate_suggestions(user_input):
    lower = user_input.lower()

    if any(word in lower for word in ["dealer", "buy", "purchase", "distributor", "where can i buy"]):
        return ["Dealer in Mumbai", "Show me dealer by PIN code", "Where is the nearest dealer?"]
    elif any(word in lower for word in ["bathroom", "washroom", "toilet"]):
        return ["What size tiles are best for bathrooms?", "Are bathroom tiles slip-resistant?", "Glossy or matte for bathroom walls?"]
    elif "parking" in lower:
        return ["Which tiles are durable for parking areas?", "Do you have anti-skid parking tiles?", "Best color tiles for parking?"]
    elif "living room" in lower:
        return ["Best designs for living room tiles?", "Which finish suits living room flooring?", "Is glossy suitable for living rooms?"]
    elif any(word in lower for word in ["pool", "swimming"]):
        return ["Tiles suitable for pool decks?", "Are pool tiles anti-slip?", "Can Johnson tiles be used underwater?"]
    elif "industrial" in lower:
        return ["Best tiles for industrial use?", "Can tiles withstand heavy machinery?", "Are Endura tiles chemical resistant?"]
    elif "cool roof" in lower:
        return ["How do cool roof tiles work?", "Do they reduce temperature indoors?", "Which tiles for summer heat?"]
    elif "kitchen" in lower:
        return ["Are glossy tiles good for kitchens?", "Which tiles resist oil stains?", "Best tile color for modular kitchens?"]
    elif "size" in lower or "dimension" in lower:
        return ["Standard tile sizes?", "Can I cut tiles to custom sizes?", "Which tile size is best for walls?"]
    elif "cost" in lower or "price" in lower:
        return ["What is the cost of Endura tiles?", "Are Johnson tiles budget-friendly?", "Pricing of cool roof tiles?"]
    else:
        return ["Which tiles are best for outdoors?", "Where can I buy Johnson tiles?", "How do I clean my tiles?"]

# === STREAMLIT UI ===
st.set_page_config(page_title="JAI - (Johnson Artificial Intelligence)", page_icon="🧱")
st.markdown("""
    <h1 style='text-align: center;'>🤖 JAI — Johnson AI</h1>
    <p style='text-align: center;'>Your smart assistant for tiles</p>
    <hr style='border:1px solid #ddd;'>
""", unsafe_allow_html=True)

vectorstore = prepare_vectorstore()
dealer_df = pd.read_excel("Johnson_Dealer_List_Cleaned.xlsx")
qa = RetrievalQA.from_chain_type(
    llm=ChatOpenAI(model_name="gpt-4-1106-preview"),
    chain_type="stuff",
    retriever=vectorstore.as_retriever()
)

# Session setup
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
    query = prompt.strip()
    if query.isdigit() and len(query) == 6:
        query = f"Show me dealers near PIN code {query}"

    question_words = ("where", "what", "how", "who", "can", "is", "are", "does", "do", "when", "which", "should", "could", "would")
    if query.lower().startswith(question_words) and not query.endswith("?"):
        query += "?"

    st.session_state.chat_history.append({"role": "user", "content": prompt})

    # === BUY INTENT ===
    buy_intents = ["where can i buy", "buy tiles", "find dealer", "get tiles", "supplier", "purchase tiles", "distributor"]
    if any(term in query.lower() for term in buy_intents):
        user_query = query.lower()
        found = False
        match = None

        for pin in dealer_df["PIN Code"].astype(str):
            if pin in user_query:
                match = dealer_df[dealer_df["PIN Code"].astype(str) == pin]
                found = True
                break

        if not found:
            for city in dealer_df["City"].dropna().unique():
                if city.lower() in user_query:
                    match = dealer_df[dealer_df["City"].str.lower() == city.lower()]
                    found = True
                    break

        if found and match is not None and not match.empty:
            rows = match.to_dict("records")
            dealer_lines = [
                f"<b>{r['Dealer Name']}</b><br>📍 {r['Address']}, {r['City']}, {r['State']} - {r['PIN Code']}<br>📞 {r['Contact']} | ✉️ {r['E_MAIL']}"
                for r in rows[:3]
            ]
            response = "<br><br>".join(dealer_lines)
        else:
            response = (
                "You can buy <b>Johnson Tiles</b> through our nationwide dealer network.<br><br>"
                "🛒 Would you like me to find a dealer for you?<br>"
                "👉 Please provide your <b>city</b> or <b>PIN code</b> so I can help you locate the nearest dealer."
            )

        st.session_state.chat_history.append({"role": "assistant", "content": response})
        with st.chat_message("assistant"):
            st.markdown(response, unsafe_allow_html=True)
        st.session_state.show_suggestions = True
        st.session_state.last_input = "dealer"
        st.stop()

    # === GENERAL QUERY HANDLING ===
    with st.spinner("JAI is typing..."):
        try:
            if not is_tile_related(query):
                response = (
                    "🤖 I'm trained to assist only with <b>Johnson Tiles</b> related queries.<br><br>"
                    "Please ask me about tile types, uses, dealers, or anything found in our official tile guide."
                )
            elif any(brand in query.lower() for brand in other_brands):
                response = (
                    "🏆 Great question! While there are many tile brands in the market, <b>Johnson Tiles</b> stands out with over 60+ years of trust, innovation, and unmatched quality.<br><br>"
                    "From cool roof tiles to industrial-grade Endura options, Johnson Tiles delivers performance, design, and durability — all in one.<br><br>"
                    "✅ Choose Johnson — the prime name in the tile industry."
                )
            elif query.lower().startswith("show me dealers near pin code"):
                pin_code = query.split()[-1]
                matches = [doc.page_content for doc in vectorstore.docstore._dict.values() if pin_code in doc.page_content]
                if matches:
                    response = f"Here are the dealers matching PIN code {pin_code}:<br><br>" + "<br><br>".join(matches[:3])
                else:
                    response = (
                        f"⚠️ Sorry, I couldn't find any dealers for PIN code {pin_code} in the document.<br>"
                        "Please double-check the code or visit <a href='https://www.hrjohnsonindia.com' target='_blank'>www.hrjohnsonindia.com</a> for help."
                    )
            else:
                response = qa.run(query)
        except Exception:
            response = "⚠️ Sorry, I couldn’t understand that. Please ask something related to Johnson Tiles."

    st.session_state.chat_history.append({"role": "assistant", "content": response})
    with st.chat_message("assistant"):
        st.markdown(response, unsafe_allow_html=True)

    st.session_state.last_input = prompt
    st.session_state.show_suggestions = True

# === SUGGESTED FOLLOW-UPS ===
if st.session_state.show_suggestions:
    suggestions = generate_suggestions(st.session_state.last_input)
    st.markdown("##### 🔍 Suggested Questions:")
    cols = st.columns(len(suggestions))
    for i, suggestion in enumerate(suggestions):
        with cols[i]:
            if st.button(suggestion, key=f"suggestion_{i}"):
                st.session_state.chat_history.append({"role": "user", "content": suggestion})
                with st.spinner("JAI is typing..."):
                    try:
                        if not is_tile_related(suggestion):
                            response = (
                                "🤖 I'm trained to assist only with <b>Johnson Tiles</b> related queries.<br><br>"
                                "Please ask me about tile types, uses, dealers, or anything found in our official tile guide."
                            )
                        elif any(brand in suggestion.lower() for brand in other_brands):
                            response = (
                                "🏆 Among brands like Kajaria or Somany, <b>Johnson Tiles</b> remains a trusted leader in innovation, quality, and design versatility.<br><br>"
                                "Choose Johnson — the prime choice in the world of tiles."
                            )
                        else:
                            response = qa.run(suggestion)
                    except Exception:
                        response = "⚠️ Sorry, I couldn’t understand that. Please ask something related to Johnson Tiles."
                st.session_state.chat_history.append({"role": "assistant", "content": response})
                st.rerun()

# === FEEDBACK SECTION ===
with st.expander("💬 Give Feedback"):
    feedback = st.text_area("Your feedback:")
    if st.button("Submit Feedback"):
        st.success("✅ Thanks! Your feedback has been recorded.")
