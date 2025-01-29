

import os
from typing import List
from dotenv import load_dotenv
from googleapiclient.discovery import build
from langchain_openai import ChatOpenAI
import time
import requests
from bs4 import BeautifulSoup
import re
import requests
import time
from tenacity import retry, stop_after_attempt, wait_fixed


def split_into_chunks(txt, chunk_size = 1000):
    """Divise le texte en morceaux tout en respectant la structure des phrases et des paragraphes."""
    sentences = re.split(r'(\.|\?|!|\n\n)', txt)
    chunks = []
    current_chunk = ""
    
    for sentence in sentences:
        if len(current_chunk) + len(sentence) < chunk_size:
            current_chunk += sentence
        else:
            chunks.append(current_chunk.strip())
            current_chunk = sentence
    
    if current_chunk:
        chunks.append(current_chunk.strip())
    
    return chunks

def search_google(query, top_k):
    load_dotenv()
    google_api_key = os.getenv("GOOGLE_API_KEY")
    google_cse_id = os.getenv("GOOGLE_CSE_ID")

    service = build("customsearch", "v1", developerKey=google_api_key)
    results = []

    for start in range(1, top_k + 1, 10):
        res = service.cse().list(
            q=query,
            cx=google_cse_id,
            start=start,
            num=min(10, top_k - len(results))
        ).execute()

        if "items" in res:
            for item in res["items"]:
                results.append(item["link"])
                if len(results) >= top_k:
                    break
        else:
            print(f"No items found in the response: {res}")
            break
    return results

@retry(stop=stop_after_attempt(3), wait=wait_fixed(2))
def extract_text_from_url(url):
    start = time.time()
    with requests.Session() as session:
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/91.0.4472.124 Safari/537.36"
            )
        }
        response = session.get(url, headers=headers, timeout=10)
        response.raise_for_status()

    soup = BeautifulSoup(response.content, "lxml")
    for script_or_style in soup(["script", "style"]):
        script_or_style.decompose()
    text = soup.get_text(separator=" ").strip()
    text = " ".join([chunk.strip() for chunk in text.split() if chunk.strip()])
    print(f"The function extract_text_from_url takes: {time.time() - start:.4f} seconds")
    return text

def build_google_search_query(query, chat_history):

    chat_history = truncate_chat_history(chat_history, threshold=10000)

    load_dotenv()
    model = ChatOpenAI(model="gpt-3.5-turbo-16k")
    prompt = get_google_search_prompt()
    prompt = prompt + '\n' + "\n\n".join(chat_history) + "\n" + f"The query is : {query}"
    result = model.invoke(prompt)
    return result.content

def read_txt_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()
        return content

def get_google_search_prompt():
    file_path = os.path.join(os.path.dirname(__file__), '../prompts/google_search_query_prompt.txt')
    return read_txt_file(file_path)

def get_rag_prompt():
    file_path = os.path.join(os.path.dirname(__file__), '../prompts/rag_prompt.txt')
    return read_txt_file(file_path)

def get_web_search_prompt():
    file_path = os.path.join(os.path.dirname(__file__), '../prompts/web_search_prompt.txt')
    return read_txt_file(file_path)


import tiktoken

def truncate_chat_history(chat_history, threshold, model="gpt-3.5-turbo"):

    # Initialize the tokenizer for the specified model
    tokenizer = tiktoken.encoding_for_model(model)
    
    # Calculate the number of tokens for each message
    token_counts = [len(tokenizer.encode(message)) for message in chat_history]
    
    # Calculate total tokens in the chat history
    total_tokens = sum(token_counts)
    
    # Remove messages from the beginning until total_tokens is below the threshold
    while total_tokens > threshold and chat_history:
        total_tokens -= token_counts.pop(0)
        chat_history.pop(0)
    
    return chat_history

import tiktoken

def truncate_messages(messages, max_tokens, model="gpt-3.5-turbo"):
    """
    Truncates the messages list to ensure the total token count is under max_tokens.
    
    Parameters:
        messages (list): List of messages with 'role' and 'content'.
        max_tokens (int): Maximum token count allowed.
        model (str): The OpenAI model to use for tokenization.
        
    Returns:
        list: The truncated list of messages.
    """
    tokenizer = tiktoken.encoding_for_model(model)
    
    def count_tokens(msg):
        """Calculate the number of tokens in a message."""
        return len(tokenizer.encode(msg["role"])) + len(tokenizer.encode(msg["content"]))

    total_tokens = sum(count_tokens(msg) for msg in messages)
    while total_tokens > max_tokens and messages:
        # Remove the oldest message
        total_tokens -= count_tokens(messages.pop(0))
    
    return messages


