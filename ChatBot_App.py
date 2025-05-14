import streamlit as st
from sentence_transformers import SentenceTransformer
import pandas as pd
import numpy as np
import faiss
import pickle
from streamlit_chat import message
from gtts import gTTS
import base64
import time
import os
import io
# ------------------- Load Prebuilt Vector DB -------------------

@st.cache_resource
def load_model_index_data():
    model = SentenceTransformer("all-MiniLM-L6-v2")
    with open("vector_data.pkl", "rb") as f:
        vector_store = pickle.load(f)
    df = vector_store["df"]
    index = vector_store["index"]
    return model, index, df

# ------------------- Generate Audio -------------------

def generate_speech_html(text):
    # Generate audio from text and store it in memory
    tts = gTTS(text, lang='en')
    audio_buffer = io.BytesIO()
    tts.write_to_fp(audio_buffer)
    audio_buffer.seek(0)

    # Encode the audio in base64 to embed in HTML
    b64 = base64.b64encode(audio_buffer.read()).decode()
    audio_html = f"""
        <audio controls style="margin-top: 10px;">
            <source src="data:audio/mp3;base64,{b64}" type="audio/mp3">
        </audio>
    """
    return audio_html

# ------------------- Streamlit UI -------------------

st.set_page_config(page_title="⚡ DISCOM Chatbot", layout="wide")
st.markdown("""
    <style>
    .block-container {
        padding-top: 1rem;
    }
    .stChatMessage {
        font-size: 1.05rem;
    }
    </style>
""", unsafe_allow_html=True)

st.title("⚡ DISCOM Chatbot")
st.markdown("👋 Hello! I'm **Lisa**, your assistant for DISCOM queries. Type your question below.")

model, index, df = load_model_index_data()

# Initialize chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [("bot", "👋Ask me anything about electricity issues!")]

# User input
user_input = st.chat_input("Ask your DISCOM-related query here...")

# Process user input
if user_input:
    st.session_state.chat_history.append(("user", user_input))

    # Typing animation
    with st.spinner("Lisa is typing..."):
        time.sleep(1.5)
        query_vec = model.encode([user_input])
        D, I = index.search(np.array(query_vec), k=1)
        matched_answer = df.iloc[I[0][0]]["answer"]
        st.session_state.chat_history.append(("bot", matched_answer))

# Show messages
for role, msg in st.session_state.chat_history:
    message(msg, is_user=(role == "user"))
    if role == "bot" and msg != st.session_state.chat_history[0][1]:
        st.markdown("🔊 **Click to hear:**", unsafe_allow_html=True)
        st.markdown(generate_speech_html(msg), unsafe_allow_html=True)
