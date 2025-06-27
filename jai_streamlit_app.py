import os
import streamlit as st
import pandas as pd
import json
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_openai import ChatOpenAI

# === CONFIGURATION ===
os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
JSON_PATH = "johnson_tiles_master_data_cleaned.json"

@st.cache_resource
def prepare_vectorstore():
    with open(JSON_PATH, "r") as f:
        json_data = json.load(f)

    def flatten_json_to_docs(json_obj, parent_key=""):
        docs = []
        if isinstance(json_obj, dict):
            for k, v in json_obj.items():
                new_key = f"{parent_key}.{k}" if parent_key else k
                docs.extend(flatten_json_to_docs(v, new_key))
        elif isinstance(json_obj, list):
            for i, item in enumerate(json_obj):
                new_key = f"{parent_key}[{i}]"
                docs.extend(flatten_json_to_docs(item, new_key))
        else:
            flat_text = f"{parent_key.replace('.', ' > ')}: {json_obj}"
            docs.append(Document(page_content=flat_text))
        return docs

    flat_docs = flatten_json_to_docs(json_data)
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    split_docs = splitter.split_documents(flat_docs)
    embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")
    return FAISS.from_documents(split_docs, embeddings)

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
        return [
            "Which Johnson tiles are best for outdoors?",
            "Where can I buy Johnson tiles?",
            "How do I clean my Johnson tiles?",
            "Do you have eco-friendly Johnson tiles?",
            "Best tiles from Johnson for a modern kitchen?"
        ]

# === STREAMLIT UI ===
st.set_page_config(page_title="JAI - (Johnson Artificial Intelligence)", page_icon="🧱")
st.markdown("""
<style>
body {
  background: linear-gradient(to bottom right, #fffaf0, #f5f5dc);
  font-family: 'Georgia', serif;
}
.chat-message {
  border: 2px solid #d4af37;
  background-color: #fff;
  padding: 1rem;
  border-radius: 12px;
  margin-bottom: 1rem;
  box-shadow: 0 2px 10px rgba(212,175,55,0.2);
}
h1, h2, h3 {
  color: #8b0000;
}
button {
  border-radius: 10px;
  background-color: #d4af37;
  color: white;
  border: none;
  padding: 0.5rem 1rem;
  font-weight: bold;
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<h1 style='text-align: center;'>🧱 JAI — Johnson AI</h1>
<p style='text-align: center;'>Your smart royal tile advisor</p>
<hr style='border:2px solid #d4af37;'>
""", unsafe_allow_html=True)

# rest of the chatbot logic continues here...
# [NOTE: The rest of your existing code remains unchanged after this block.]

# This block only beautifies the theme and branding.
# Let me know if you want it extended to message bubbles or button transitions.
