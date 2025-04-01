import os
import random
import base64
import streamlit as st
import pandas as pd
import requests
from bs4 import BeautifulSoup
from datetime import datetime
from langchain.docstore.document import Document

# === TILE SCRAPING CONFIG ===
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

# === CONFIGURATION ===
os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
PDF_PATH = "Johnson-Tile-Guide-2023-Final-Complete-With-Tables.pdf"
CACHE_TILE_PATH = "johnson_tiles_data.csv"

@st.cache_resource
def prepare_vectorstore():
    loader = PyPDFLoader(PDF_PATH)
    documents = loader.load()
    splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    pdf_docs = splitter.split_documents(documents)

    # Add scraped tile data to vectorstore
    scraped_docs = []
    if os.path.exists(CACHE_TILE_PATH):
        tile_df = pd.read_csv(CACHE_TILE_PATH)
        for _, row in tile_df.iterrows():
            content = f"Tile Name: {row['Name']}
Link: {row['Link']}
Image: {row['Image']}"
            scraped_docs.append(Document(page_content=content))

    all_docs = pdf_docs + scraped_docs
    embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")
    return FAISS.from_documents(all_docs, embeddings)

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

# === DISPLAY TILE CATALOG + REFRESH ===
st.markdown("### 🧱 Johnson Tile Catalog")
cached_df = load_cached_tiles()
st.dataframe(cached_df[["Name", "Scraped On"]] if not cached_df.empty else pd.DataFrame())

if st.button("🔄 Refresh latest tiles"):
    with st.spinner("Scraping website for latest tiles..."):
        new_df = scrape_tiles()
        st.success("✅ Latest tile data fetched!")
        st.dataframe(new_df[["Name", "Scraped On"]])
