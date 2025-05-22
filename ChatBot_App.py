import streamlit as st
from sentence_transformers import SentenceTransformer
import numpy as np
import pickle
from gtts import gTTS
import base64
import io
import time

# ==================== Custom CSS for Attractive UI ====================
st.set_page_config(page_title="⚡ DISCOM Chatbot", layout="centered")
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Nunito:wght@400;700&family=Montserrat:wght@700&display=swap');

html, body, .main, .block-container {
    background: linear-gradient(120deg, #e0eafc 0%, #cfdef3 100%) !important;
    font-family: 'Nunito', 'Montserrat', Arial, sans-serif !important;
}
.header-bar {
    display: flex;
    align-items: center;
    background: linear-gradient(90deg, #1f3c88 0%, #43cea2 100%);
    padding: 20px 24px 16px 24px;
    border-radius: 0 0 24px 24px;
    margin-bottom: 24px;
    box-shadow: 0 3px 16px #0088aa22;
}
.header-bar img {
    border-radius: 50%;
    height: 54px;
    margin-right: 18px;
    border: 3px solid #fff;
    background: #fff;
    box-shadow: 0 2px 8px #1f3c8822;
}
.header-bar .chatbot-title {
    font-size: 2.1rem;
    color: #fff;
    font-weight: 700;
    font-family: 'Montserrat', Arial, sans-serif;
    letter-spacing: 1px;
    margin-bottom: 3px;
}
.header-bar .chatbot-desc {
    font-size: 1.04rem;
    color: #eaf8fd;
    font-family: 'Nunito', Arial, sans-serif;
}
.chat-window {
    max-width: 600px;
    margin: auto;
    background: #ffffffbb;
    border-radius: 24px;
    box-shadow: 0 8px 32px #1f3c8835;
    padding: 12px 0 14px 0;
    min-height: 56vh;
}
.message-row {
    display: flex;
    margin: 0 0 4px 0;
}
.message-row.user {
    flex-direction: row-reverse;
    justify-content: flex-end;
}
.message-row.bot {
    justify-content: flex-start;
}
.bubble {
    display: flex;
    flex-direction: column;
    max-width: 75%;
    padding: 14px 22px;
    border-radius: 2em;
    margin: 9px 10px 4px 10px;
    font-size: 1.13rem;
    font-family: 'Nunito', Arial, sans-serif;
    box-shadow: 0 1px 9px #005bea14;
    word-break: break-word;
    transition: background 0.2s;
}
.bubble.user {
    background: linear-gradient(120deg, #43cea2 0%, #185a9d 100%);
    color: #fff;
    align-items: flex-end;
    border-bottom-right-radius: 10px 35px;
    border-top-right-radius: 30px 75px;
}
.bubble.bot {
    background: linear-gradient(120deg, #ece9e6 0%, #ffffff 100%);
    color: #34495e;
    align-items: flex-start;
    border-bottom-left-radius: 10px 35px;
    border-top-left-radius: 30px 75px;
}
.avatar {
    width: 44px;
    height: 44px;
    border-radius: 50%;
    margin: 5px 12px;
    border: 2px solid #1f3c88;
    box-shadow: 0 2px 8px #36d1c422;
}
.avatar.user {
    margin-left: 10px;
    background: #185a9d;
}
.avatar.bot {
    margin-right: 10px;
    background: #43cea2;
}
.timestamp {
    font-size: 0.82em;
    color: #a1a7bb;
    margin-top: 2px;
    text-align: right;
}
audio {
    margin-left: 10px;
    margin-bottom: -5px;
}
/* Typing dots animation */
.dot-flashing {
    position: relative;
    width: 30px;
    height: 12px;
}
.dot-flashing span, .dot-flashing:before, .dot-flashing:after {
    content: '';
    display: inline-block;
    position: absolute;
    top: 0;
    width: 8px;
    height: 8px;
    border-radius: 4px;
    background: #43cea2;
    animation: dotFlashing 1s infinite linear alternate;
}
.dot-flashing:before {
    left: 0;
    animation-delay: 0s;
}
.dot-flashing span {
    left: 10px;
    animation-delay: 0.3s;
}
.dot-flashing:after {
    left: 20px;
    animation-delay: 0.6s;
}
@keyframes dotFlashing {
    0% { opacity: 0.2; }
    50%,100% { opacity: 1; }
}
@media (max-width: 700px) {
    .chat-window {
        max-width: 98vw;
        padding: 4px 0 10px 0;
    }
    .header-bar {
        padding: 14px 9px 8px 9px;
    }
    .bubble {
        padding: 11px 11px;
        font-size: 0.99rem;
    }
    .avatar, .avatar.user, .avatar.bot {
        width: 32px;
        height: 32px;
        margin: 4px 4px;
    }
}
.stChatInput {
    border-radius: 18px !important;
    border: 2px solid #43cea2 !important;
}
</style>
""", unsafe_allow_html=True)

# ==================== Header ====================
st.markdown("""
<div class="header-bar">
    <img src="https://cdn-icons-png.flaticon.com/512/4712/4712035.png" alt="Chatbot" />
    <div>
        <div class="chatbot-title">DISCOM Chatbot</div>
        <div class="chatbot-desc">Lisa – Your AI Power Assistant</div>
    </div>
</div>
""", unsafe_allow_html=True)

# ==================== Load Model & Vector DB ====================
@st.cache_resource
def load_model_index_data():
    model = SentenceTransformer("all-MiniLM-L6-v2")
    with open("vector_data.pkl", "rb") as f:
        vector_store = pickle.load(f)
    df = vector_store["df"]
    index = vector_store["index"]
    return model, index, df

model, index, df = load_model_index_data()

# ==================== Generate Audio ====================
def generate_speech_html(text):
    tts = gTTS(text, lang='en')
    audio_buffer = io.BytesIO()
    tts.write_to_fp(audio_buffer)
    audio_buffer.seek(0)
    b64 = base64.b64encode(audio_buffer.read()).decode()
    audio_html = f"""
        <audio controls style="margin-top: 10px;">
            <source src="data:audio/mp3;base64,{b64}" type="audio/mp3">
        </audio>
    """
    return audio_html

# ==================== Session State ====================
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [("bot", "👋 Ask me anything about electricity issues!")]
if "bot_typing" not in st.session_state:
    st.session_state.bot_typing = False

# ==================== User Input ====================
user_input = st.chat_input("Ask your DISCOM-related query here...")

if user_input:
    st.session_state.chat_history.append(("user", user_input))
    st.session_state.bot_typing = True
    st.rerun()

# ==================== Show Typing Animation if Needed ====================
typing_idx = None
if st.session_state.bot_typing:
    # WhatsApp style typing indicator bubble
    typing_msg = """
    <span style='display: flex; align-items: center;'>
      <span class="dot-flashing"><span></span></span>
      <span style='margin-left:12px;color:#43cea2;font-weight:bold;'>Lisa is typing...</span>
    </span>
    """
    st.session_state.chat_history.append(("bot_typing", typing_msg))
    typing_idx = len(st.session_state.chat_history) - 1

# ==================== Display Chat ====================
st.markdown('<div class="chat-window">', unsafe_allow_html=True)

for idx, (role, msg) in enumerate(st.session_state.chat_history):
    is_user = (role == "user")
    if role == "bot_typing":
        avatar_url = "https://cdn-icons-png.flaticon.com/512/4712/4712035.png"
        row_class = "bot"
        bubble_class = "bubble bot"
        avatar_class = "avatar bot"
        st.markdown(f"""
        <div class="message-row {row_class}">
            <div class="{avatar_class}"><img src="{avatar_url}" width="44"/></div>
            <div class="{bubble_class}">
                {msg}
            </div>
        </div>
        """, unsafe_allow_html=True)
        continue

    avatar_url = (
        "https://cdn-icons-png.flaticon.com/512/4712/4712035.png" if not is_user
        else "https://cdn-icons-png.flaticon.com/512/9131/9131546.png"
    )
    row_class = "user" if is_user else "bot"
    bubble_class = "bubble user" if is_user else "bubble bot"
    avatar_class = "avatar user" if is_user else "avatar bot"

    st.markdown(f"""
    <div class="message-row {row_class}">
        <div class="{avatar_class}"><img src="{avatar_url}" width="44"/></div>
        <div class="{bubble_class}">
            {msg}
        </div>
    </div>
    """, unsafe_allow_html=True)

    if role == "bot" and idx != 0:
        st.markdown("🔊 <b>Click to hear:</b>", unsafe_allow_html=True)
        st.markdown(generate_speech_html(msg), unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)

# ==================== Generate Actual Bot Response and Remove Typing ====================
if st.session_state.bot_typing:
    if typing_idx is not None:
        st.session_state.chat_history.pop(typing_idx)
    with st.spinner("Lisa is typing..."):
        time.sleep(1.2)
        last_user_msg = [msg for role, msg in reversed(st.session_state.chat_history) if role == "user"][0]
        query_vec = model.encode([last_user_msg])
        D, I = index.search(np.array(query_vec), k=1)
        matched_answer = df.iloc[I[0][0]]["answer"]
        st.session_state.chat_history.append(("bot", matched_answer))
        st.session_state.bot_typing = False
    st.rerun()
