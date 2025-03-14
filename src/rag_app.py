from openai import OpenAI
import streamlit as st
from dotenv import load_dotenv
import os
from PyPDF2 import PdfReader
from datetime import date
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
    st.session_state["openai_model"] = "gpt-4o-mini-2024-07-18"

if "messages" not in st.session_state:
    st.session_state.messages = []

if "search_web" not in st.session_state:
    st.session_state.search_web = False

if "file_uploaded" not in st.session_state:
    st.session_state.file_uploaded = False

today_date_given = False

db = FAISSDb()
top_k_url = 5
top_k_similar = 10
score_threshold = 0.2
chunk_size = 500

# --- Side Bar ---


with st.sidebar:

    st.info("ℹ️ Note: This chatbot is optimized for English. Responses in other languages may be inaccurate or highly unreliable.")

    # Delete Chat Button
    if st.button("Delete Chat History"):
        st.session_state.messages = []
        
    # PDF uploader
    uploaded_file = st.sidebar.file_uploader("Upload a PDF", type="pdf")
    if uploaded_file is not None:
        db = FAISSDb()
        st.session_state.file_uploaded = True
        pdf_reader = PdfReader(uploaded_file)
        pdf_text = ""
        for page in pdf_reader.pages:
            pdf_text += page.extract_text()
        db.add_from_doc(pdf_text, chunk_size=1000)

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

    # Give the date to the chatbot
    if st.session_state.search_web and not today_date_given:
        
        today_date = date.today()
        date_prompt = f"Today data is : {today_date}"

        st.session_state.messages.append({"role": "system", "content": date_prompt})

        today_date_given = True


# Display The chat
for message in st.session_state.messages:
    if message["role"] != "system":
        avatar = USER_AVATAR if message["role"] == "user" else BOT_AVATAR
        with st.chat_message(message["role"], avatar=avatar):
            st.markdown(message["content"])



# Main chat Interface
if prompt := st.chat_input("How can I help?"):

    print(f"\nUser prompt : {prompt}")

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
            db, prompt, top_k_url, chunk_size , top_k_similar, score_threshold, chat_history
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


