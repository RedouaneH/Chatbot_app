

from langchain.vectorstores import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings
from data import *

__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

class ChromaDb:

    def __init__(self):

        self.embeddings = OpenAIEmbeddings()
        self.vector_store = Chroma(
        embedding_function=self.embeddings
        )
        self.vector_store.clear_system_cache()

    def add(self, chunks, metadatas=None):
        self.vector_store.add_texts(
            texts=chunks,
            metadatas=metadatas
        )

    def add_from_doc(self, doc, metadatas=None, chunk_size=1000, overlap=100):
        chunks = split_into_chunks(doc, chunk_size, overlap)
        self.add(chunks)
    
    def load_from_urls(self, urls, chunk_size=1000, overlap=100):
        for url in urls:
            try:
                txt = extract_text_from_url(url)
                chunks = split_into_chunks(txt, chunk_size, overlap)
                self.add(chunks)
            except Exception as e:
                print(f"Error accessing {url}: {e}")
        return


    def reset(self):
        if self.vector_store:
            print(self.vector_store._collection.name)
            self.vector_store.delete_collection()

        self.vector_store = Chroma(
            embedding_function=self.embeddings
        )
