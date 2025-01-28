
from data import *
from db import *


def generate_web_search_prompt(db, query, top_k_url, chunk_size, overlap_ratio, top_k_similar, score_threshold, chat_history):

    google_search_query = build_google_search_query(query, chat_history)
    print(f"\ngoogle_search_query : {google_search_query}\n")
    urls = search_google(google_search_query, top_k=top_k_url)
    for u in urls:
        print(u)
    if(len(urls)==0):
        print("Error Zero Urls Retrieved")
    db.load_from_urls(urls, chunk_size, int(overlap_ratio*chunk_size))

    retrieved_chunks = db.retrieve(query, top_k_similar, score_threshold)

    web_search_prompt = get_web_search_prompt() + "\n\n".join(retrieved_chunks)

    return web_search_prompt

def generate_rag_prompt(db, query, top_k_similar, score_threshold):

    retrieved_chunks = db.retrieve(query, top_k_similar, score_threshold)

    rag_prompt = get_rag_prompt() + "\n\n".join(retrieved_chunks)

    return rag_prompt
