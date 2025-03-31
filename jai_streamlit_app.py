import os
import base64
import streamlit as st
import pandas as pd
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain.chains import RetrievalQA
from langchain.agents import tool, initialize_agent, AgentType
from langchain.memory import ConversationBufferMemory

# === CONFIGURATION ===
os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
PDF_PATH = "Johnson-Tile-Guide-2023-Final-Complete-With-Tables.pdf"

# === PREPARE VECTORSTORE ===
@st.cache_resource
def prepare_vectorstore():
    loader = PyPDFLoader(PDF_PATH)
    documents = loader.load()
    splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    pdf_docs = splitter.split_documents(documents)
    embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")
    return FAISS.from_documents(pdf_docs, embeddings)

vectorstore = prepare_vectorstore()
retriever = vectorstore.as_retriever()

# === DEFINE TOOLS ===
@tool
def get_tile_types() -> str:
    """Provides a list of available tile types."""
    return "Available tile types: Bathroom, Living Room, Parking, Cool Roof, Swimming Pool, Industrial."

@tool
def find_dealer(city: str) -> str:
    """Finds dealers based on the given city."""
    dealer_df = pd.read_excel("Johnson_Dealer_List_Cleaned.xlsx")
    result = dealer_df[dealer_df["City"].str.lower() == city.lower()]
    if result.empty:
        return f"No dealers found in {city}."
    top_rows = result.head(3).to_dict(orient="records")
    response = ""
    for row in top_rows:
        response += f"""
<b>{row['Dealer Name']}</b><br>
📍 {row['Address']}, {row['City']}, {row['State']} - {row['PIN Code']}<br>
📞 {row['Contact']} | ✉️ {row['E_MAIL']}<br><br>
"""
    return response

@tool
def query_tile_guide(query: str) -> str:
    """Fetches information from the Johnson Tile Guide based on the query."""
    qa = RetrievalQA.from_chain_type(llm=ChatOpenAI(model_name="gpt-4"), retriever=retriever)
    return qa.run(query)

# === MEMORY ===
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# === LLM & AGENT ===
llm = ChatOpenAI(model_name="gpt-4")
tools = [get_tile_types, find_dealer, query_tile_guide]
agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
    memory=memory,
    verbose=True
)

# === STREAMLIT UI ===
st.set_page_config(page_title="JAI - Johnson Sales Assistant", page_icon="🧱")
st.markdown("""
    <h1 style='text-align: center;'>🤖 JAI — Johnson AI</h1>
    <p style='text-align: center;'>Your smart assistant for tiles</p>
    <hr style='border:1px solid #ddd;'>
""", unsafe_allow_html=True)

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

    with st.spinner("JAI is typing..."):
        try:
            response = agent.run(query)
        except Exception as e:
            response = f"⚠️ Sorry, something went wrong: {str(e)}"

    st.session_state.chat_history.append({"role": "assistant", "content": response})
    with st.chat_message("assistant"):
        st.markdown(response, unsafe_allow_html=True)

    st.session_state.last_input = prompt
    st.session_state.show_suggestions = True

# === SUGGESTED FOLLOW-UPS ===
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
                        response = agent.run(suggestion)
                    except Exception:
                        response = "⚠️ Sorry, I couldn’t understand that. Please ask something related to Johnson Tiles."
                st.session_state.chat_history.append({"role": "assistant", "content": response})
                st.rerun()

# === FEEDBACK SECTION ===
with st.expander("💬 Give Feedback"):
    feedback = st.text_area("Your feedback:")
    if st.button("Submit Feedback"):
        st.success("✅ Thanks! Your feedback has been recorded.")
