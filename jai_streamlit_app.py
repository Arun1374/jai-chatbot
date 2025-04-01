import os
import random
import base64
import streamlit as st
import pandas as pd
import requests
from bs4 import BeautifulSoup
from datetime import datetime
from langchain.docstore.document import Document
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain.chains import RetrievalQA

# === CONFIGURATION ===
os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
PDF_PATH = "Johnson-Tile-Guide-2023-Final-Complete-With-Tables.pdf"
IMAGE_FOLDER = "extracted_images"
CACHE_TILE_PATH = "johnson_tiles_data.csv"

@st.cache_data
def load_cached_tiles():
    if os.path.exists(CACHE_TILE_PATH):
        return pd.read_csv(CACHE_TILE_PATH)
    return pd.DataFrame()

def scrape_tiles():
    url = "https://www.hrjohnsonindia.com/tiles/floor-tiles"
    headers = {'User-Agent': 'Mozilla/5.0'}
    response = requests.get(url, headers=headers)
    soup = BeautifulSoup(response.text, 'html.parser')

    tiles = []
    for tile in soup.select(".product-tile"):
        name = tile.select_one(".product-title")
        img = tile.select_one("img")
        link = tile.select_one("a")

        tiles.append({
            "Name": name.text.strip() if name else "N/A",
            "Image": img['src'] if img else "N/A",
            "Link": link['href'] if link else "N/A",
            "Scraped On": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        })

    df = pd.DataFrame(tiles)
    df.to_csv(CACHE_TILE_PATH, index=False)
    return df

@st.cache_resource
def prepare_vectorstore():
    loader = PyPDFLoader(PDF_PATH)
    documents = loader.load()
    splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    pdf_docs = splitter.split_documents(documents)

    scraped_docs = []
    if os.path.exists(CACHE_TILE_PATH):
        tile_df = pd.read_csv(CACHE_TILE_PATH)
        for _, row in tile_df.iterrows():
            content = f"Tile Name: {row['Name']}\nLink: {row['Link']}\nImage: {row['Image']}"
            scraped_docs.append(Document(page_content=content))

    all_docs = pdf_docs + scraped_docs
    embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")
    return FAISS.from_documents(all_docs, embeddings)

# === TILE CATALOG SECTION ===
st.markdown("### 🧱 Johnson Tile Catalog")
cached_df = load_cached_tiles()
if not cached_df.empty:
    st.dataframe(cached_df[["Name", "Scraped On"]])
else:
    st.info("Tile data not yet scraped. Click the button below to fetch latest tiles.")

if st.button("🔄 Refresh latest tiles"):
    with st.spinner("Scraping website for latest tiles..."):
        new_df = scrape_tiles()
        st.success("✅ Latest tile data fetched!")
        st.dataframe(new_df[["Name", "Scraped On"]])

# === SUGGESTED FOLLOW-UPS (UPDATED for dynamic last message) ===
if st.session_state.show_suggestions:
    last_user_input = next((msg["content"] for msg in reversed(st.session_state.chat_history) if msg["role"] == "user"), "")
    suggestions = generate_suggestions(last_user_input)
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
