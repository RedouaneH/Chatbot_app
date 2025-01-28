from openai import OpenAI
import streamlit as st
from dotenv import load_dotenv
import os
from PyPDF2 import PdfReader

from db import *
from generation import *
from data import *


# --- Init ---

load_dotenv()

st.title("RawBot")

USER_AVATAR = "👤"
BOT_AVATAR = "🤖"
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

if "openai_model" not in st.session_state:
    st.session_state["openai_model"] = "gpt-3.5-turbo"

if "messages" not in st.session_state:
    st.session_state.messages = []

if "search_web" not in st.session_state:
    st.session_state.search_web = False

if "file_uploaded" not in st.session_state:
    st.session_state.file_uploaded = False

db = FAISSDb()
top_k_url = 5
top_k_similar = 5
score_threshold = 0.5
chunk_size = 500
overlap_ratio = 0.15

# --- Side Bar ---


with st.sidebar:

    # Delete Chat Button
    if st.button("Delete Chat History"):
        st.session_state.messages = []
        
    # PDF uploader
    uploaded_file = st.sidebar.file_uploader("Upload a PDF", type="pdf")
    if uploaded_file is not None:
        db = ChromaDb()
        st.session_state.file_uploaded = True
        pdf_reader = PdfReader(uploaded_file)
        pdf_text = ""
        for page in pdf_reader.pages:
            pdf_text += page.extract_text()
        db.add_from_doc(pdf_text, chunk_size=1000, overlap=100)

    # Search Web button
    if uploaded_file:
        st.session_state.search_web = False
        st.sidebar.toggle("Search the web", value=False, disabled=True) 
    else:
        st.session_state.search_web = False
        st.session_state.search_web = st.sidebar.toggle("Search the web", value=st.session_state.search_web)

    if st.session_state.file_uploaded and uploaded_file is None:
        st.session_state.file_uploaded = False
        db.reset()

# Display The chat
for message in st.session_state.messages:
    if message["role"] != "system":
        avatar = USER_AVATAR if message["role"] == "user" else BOT_AVATAR
        with st.chat_message(message["role"], avatar=avatar):
            st.markdown(message["content"])

# Main chat Interface
if prompt := st.chat_input("How can I help?"):

    if uploaded_file:
        # RAG 

        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user", avatar=USER_AVATAR):
            st.markdown(prompt)

        rag_prompt = generate_rag_prompt(db, prompt, top_k_similar, score_threshold)

        st.session_state.messages.append({"role": "system", "content": rag_prompt})
        

    elif st.session_state.search_web:
        # Web Search
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user", avatar=USER_AVATAR):
            st.markdown(prompt)

        st.session_state.messages.append({"role": "assistant", "content": "Searching the web ... "})
        with st.chat_message("assistant", avatar=BOT_AVATAR):
            st.markdown("Searching the web ...")

        chat_history = [message["content"] for message in st.session_state.messages if message["role"] in ["assistant", "user"]]

        web_search_prompt = generate_web_search_prompt(
            db, prompt, top_k_url, chunk_size, overlap_ratio, top_k_similar, score_threshold, chat_history
            )

        st.session_state.messages.append({"role": "system", "content": web_search_prompt})

        db.reset()

    else:
        # Basic AI interaction
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user", avatar=USER_AVATAR):
            st.markdown(prompt)


    st.session_state["messages"] = truncate_messages(
        st.session_state["messages"],
        max_tokens=16000
    )


    with st.chat_message("assistant", avatar=BOT_AVATAR):
        message_placeholder = st.empty()
        full_response = ""
        for response in client.chat.completions.create(
            model=st.session_state["openai_model"],
            messages=st.session_state["messages"],
            stream=True,
        ):
            full_response += response.choices[0].delta.content or ""
            message_placeholder.markdown(full_response + "|")
        message_placeholder.markdown(full_response)
    st.session_state.messages.append({"role": "assistant", "content": full_response})


