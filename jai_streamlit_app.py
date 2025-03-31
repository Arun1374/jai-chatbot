import os
import pandas as pd
import streamlit as st
from langchain.agents import tool, initialize_agent
from langchain.agents.agent_types import AgentType
from langchain.memory import ConversationBufferMemory
from langchain.chat_models import ChatOpenAI

# === SETUP ===
os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
dealer_df = pd.read_excel("Johnson_Dealer_List_Cleaned.xlsx")

# === DEFINE TOOLS ===
@tool
def get_tile_types() -> str:
    """Provides a list of available tile types."""
    return "Available tile types: Bathroom, Living Room, Parking, Cool Roof, Swimming Pool, Industrial."

@tool
def find_dealer(city: str) -> str:
    """Finds dealers based on the given city."""
    result = dealer_df[dealer_df["City"].str.lower() == city.lower()]
    if result.empty:
        return f"No dealers found in {city}."
    top_rows = result.head(3).to_dict(orient="records")
    response = ""
    for row in top_rows:
        response += f"\n<b>{row['Dealer Name']}</b><br>📍 {row['Address']}, {row['City']}, {row['State']} - {row['PIN Code']}<br>📞 {row['Contact']} | ✉️ {row['E_MAIL']}<br><br>"
    return response

# === MEMORY ===
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# === LLM & AGENT ===
llm = ChatOpenAI(model_name="gpt-4")
tools = [get_tile_types, find_dealer]
agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
    memory=memory,
    verbose=True
)

# === STREAMLIT UI ===
st.set_page_config(page_title="JAI - Johnson Sales Agent", page_icon="🧱")
st.title("🤖 JAI — Your Johnson Tiles Sales Assistant")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

prompt = st.chat_input("Ask me anything about buying tiles ...")

for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"], unsafe_allow_html=True)

if prompt:
    st.session_state.chat_history.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.spinner("JAI is thinking..."):
        try:
            response = agent.run(prompt)
        except Exception as e:
            response = f"⚠️ Sorry, something went wrong: {str(e)}"

    st.session_state.chat_history.append({"role": "assistant", "content": response})
    with st.chat_message("assistant"):
        st.markdown(response, unsafe_allow_html=True)
