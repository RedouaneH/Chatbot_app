from data import *
from db import *
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage, AIMessage


def generate_up_to_date_answer(query, top_k_url=5, top_k_similarity=5, score_threshold=0.7, chunk_size=500, overlap_ratio=0.2):

    load_dotenv()

    google_search_query = build_google_search_query(query)

    db = ChromaDb()
    urls = search_google(google_search_query, top_k_url)

    overlap= int(overlap_ratio * chunk_size)
    db.load_from_urls(urls, chunk_size, overlap)

    retriever = db.vector_store.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={"k": top_k_similarity, "score_threshold": score_threshold}
    )

    retrieved_chunks = [value.page_content for value in retriever.invoke(query)]

    system_prompt = get_web_search_prompt() + "\n\n".join(retrieved_chunks)

    messages = [
        SystemMessage(content = system_prompt),
        HumanMessage(content = "The question is : "+ query)
    ]

    llm = ChatOpenAI(model="gpt-4o")

    llm_answer = llm.invoke(messages).content

    return llm_answer

def generate_web_search_prompt(db, query, top_k_url, chunk_size, overlap_ratio, top_k_similar, score_threshold, chat_history):

    google_search_query = build_google_search_query(query, chat_history)
    print(f"\ngoogle_search_query : {google_search_query}\n")
    urls = search_google(google_search_query, top_k=top_k_url)
    db.load_from_urls(urls, chunk_size, int(overlap_ratio*chunk_size))

    retriever = db.vector_store.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={"k": top_k_similar, "score_threshold": score_threshold}
    )

    retrieved_chunks = [value.page_content for value in retriever.invoke(query)]

    web_search_prompt = get_web_search_prompt() + "\n\n".join(retrieved_chunks)

    return web_search_prompt

def generate_rag_prompt(db, query, top_k_similar, score_threshold):

    retriever = db.vector_store.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={"k": top_k_similar, "score_threshold": score_threshold}
    )

    retrieved_chunks = [value.page_content for value in retriever.invoke(query)]

    rag_prompt = get_rag_prompt() + "\n\n".join(retrieved_chunks)

    return rag_prompt