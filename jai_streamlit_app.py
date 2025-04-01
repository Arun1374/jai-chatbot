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

def generate_suggestions(user_input):
    lower = user_input.lower()
    if lower == "dealer":
        return ["Dealer in Mumbai", "Show me dealer by PIN code", "Where is the nearest dealer?"]
    elif lower in ["hi", "hello", "hey"]:
        return ["Where can I buy Johnson Tiles?", "What are the latest tile trends?", "Do you have cool roof tiles?"]
    elif "bathroom" in lower:
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

    greetings = ["hi", "hello", "hey", "namaste", "good morning", "good evening"]
    if query.lower() in greetings:
        response = (
            "👋 Hello! I'm <b>JAI — Johnson AI</b>, your smart assistant for tiles.<br>"
            "Ask me anything about tile selection, design ideas, or where to find a Johnson Tiles dealer near you!"
        )
        st.session_state.chat_history.append({"role": "assistant", "content": response})
        with st.chat_message("assistant"):
            st.markdown(response, unsafe_allow_html=True)
        st.session_state.show_suggestions = True
        st.session_state.last_input = "hello"
        st.stop()

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

    with st.spinner("JAI is typing..."):
        try:
            competitor_brands = ["kajaria", "somany", "orientbell", "nitco", "asian", "hr", "jaquar"]
            allowed_keywords = [
                "tile", "tiles", "johnson", "bathroom", "floor", "wall", "dealer", "endura", "cool roof", "slip",
                "porcelain", "glazed", "granite", "marble", "ceramic", "showroom", "compare", "comparison"
            ] + competitor_brands

            if any(keyword in query.lower() for keyword in allowed_keywords):
                if any(brand in query.lower() for brand in competitor_brands):
                    preface = (
                        "🧠 Great question! While there are many tile brands in the market, "
                        "<b>Johnson Tiles</b> stands out due to its legacy, durability, and innovation. 💎<br><br>"
                        "Here’s a quick comparison based on your query:"
                    )
                    answer = qa.run(query)
                    response = preface + "<br><br>" + answer + "<br><br>" + (
                        "✅ So if you're looking for a long-lasting, stylish, and reliable tile option — "
                        "<b>Johnson Tiles</b> is the smarter choice!"
                    )
                else:
                    if query.lower().startswith("show me dealers near pin code"):
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
            else:
                response = (
                    "⚠️ I can only help with queries related to <b>Johnson Tiles</b> — design ideas, dealers, tile types, usage, and more.<br><br>"
                    "Please ask something like:<br>"
                    "• Best tiles for my bathroom?<br>"
                    "• Where can I buy Johnson Tiles near me?<br>"
                    "• Are Endura tiles suitable for parking?"
                )
        except Exception:
            response = "⚠️ Sorry, I couldn’t understand that. Please ask something related to Johnson Tiles."

    st.session_state.chat_history.append({"role": "assistant", "content": response})
    with st.chat_message("assistant"):
        st.markdown(response, unsafe_allow_html=True)

    st.session_state.last_input = prompt
    st.session_state.show_suggestions = True

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
                        response = qa.run(suggestion)
                    except Exception:
                        response = "⚠️ Sorry, I couldn’t understand that. Please ask something related to Johnson Tiles."
                st.session_state.chat_history.append({"role": "assistant", "content": response})
                st.rerun()
