

from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_community.docstore.in_memory import InMemoryDocstore
from data import *
import faiss
import uuid

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


class FAISSDb:
    def __init__(self):
        self.embeddings = OpenAIEmbeddings()
        # Determine the embedding dimension dynamically
        dummy_text = "test"
        embedding_dim = len(self.embeddings.embed_query(dummy_text))

        # Initialize FAISS index, docstore, and index-to-docstore mapping
        self.index = faiss.IndexFlatL2(embedding_dim)
        self.docstore = InMemoryDocstore({})
        self.index_to_docstore_id = {}

        # Wrap the FAISS index in the LangChain FAISS implementation
        self.vector_store = FAISS(
            index=self.index,
            docstore=self.docstore,
            index_to_docstore_id=self.index_to_docstore_id,
            embedding_function=self.embeddings,
        )

    def add(self, chunks, metadatas=None):
        # Generate unique IDs for each chunk
        ids = [str(uuid.uuid4()) for _ in chunks]
        self.vector_store.add_texts(
            texts=chunks,
            metadatas=metadatas,
            ids=ids,
        )

    def add_from_doc(self, doc, metadatas=None, chunk_size=1000):
        chunks = split_into_chunks(doc, chunk_size)
        self.add(chunks, metadatas)

    def load_from_urls(self, urls, chunk_size=1000):
        for url in urls:
            try:
                txt = extract_text_from_url(url)
                chunks = split_into_chunks(txt, chunk_size)
                metadatas = [{"url": url} for _ in chunks]
                self.add(chunks, metadatas)
            except Exception as e:
                print(f"Error accessing {url}: {e}")
        return

    def retrieve(self, query, k, score_threshold):
        retriever = self.vector_store.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={"k": k, "score_threshold": score_threshold},
        )

        retrieved_chunks = []
        for value in retriever.invoke(query):
            page_content = value.page_content
            # Check if metadata contains the "url" key and only add it if it exists
            if "url" in value.metadata:
                retrieved_chunks.append(f"{page_content} - from the following url: {value.metadata['url']}")
            else:
                retrieved_chunks.append(page_content)

        return retrieved_chunks


    def reset(self):
        # Clear and reinitialize the FAISS vector store
        self.index = faiss.IndexFlatL2(self.index.d)
        self.docstore = InMemoryDocstore({})
        self.index_to_docstore_id = {}

        self.vector_store = FAISS(
            index=self.index,
            docstore=self.docstore,
            index_to_docstore_id=self.index_to_docstore_id,
            embedding_function=self.embeddings,
        )

    def erase(self):
        """Clears the vector store index."""
        self.index.reset()




