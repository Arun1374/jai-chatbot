import os
import random
import base64
import streamlit as st
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain.chains import RetrievalQA

# === CONFIGURATION ===
os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
PDF_PATH = "Johnson-Tile-Guide-2023-Final-Complete-With-Tables.pdf"  # updated PDF with size tables
IMAGE_FOLDER = "extracted_images"

@st.cache_resource
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
    if lower == "dealer":
        return ["Dealer in Mumbai", "Show me dealer by PIN code", "Where is the nearest dealer?"]
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

for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"], unsafe_allow_html=True)

prompt = st.chat_input("Ask me anything about tiles ...")

if prompt:
    query = prompt.strip()
    # Auto-expand numeric-only PIN queries
    if query.isdigit() and len(query) == 6:
        query = f"Show me dealers near PIN code {query}"

    
    question_words = ("where", "what", "how", "who", "can", "is", "are", "does", "do", "when", "which", "should", "could", "would")
    if query.lower().startswith(question_words) and not query.endswith("?") and not query.endswith("?"):
        query += "?"

    st.session_state.chat_history.append({"role": "user", "content": prompt})

    # SMART BUY DETECTION
    buy_intents = ["where can i buy", "buy tiles", "find dealer", "get tiles", "supplier", "purchase tiles", "distributor"]
    if any(term in query.lower() for term in buy_intents):
        if not any(loc in query.lower() for loc in ["mumbai", "pune", "bangalore", "delhi", "pin", "code", "city"]):
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
            # === Fallback handler for direct PIN code queries ===
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

with st.expander("💬 Give Feedback"):
    feedback = st.text_area("Your feedback:")
    if st.button("Submit Feedback"):
        st.success("✅ Thanks! Your feedback has been recorded.")
