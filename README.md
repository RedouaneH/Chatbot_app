
# Chatbot App

Chatbot App is a web application that uses a Retrieval-Augmented Generation (RAG) system to provide up-to-date information. It utilizes Google Search API, OpenAI, and the frameworks LangChain, ChromaDB, and Streamlit to deliver real-time responses to user queries.

## Usage

To run the project locally, follow these steps:

1. **add a `.env` file** at the root of the project with your own API keys

```bash
OPEN_AI_KEY=your_openai_api_key
GOOGLE_API_KEY=your_google_api_key
GOOGLE_CSE_ID=your_google_cse_id
```

2. **Install dependencies** by running:

```bash
pip install -r requirements.txt
```
3. **Run the application locally** with Streamlit:


```bash
streamlit run src/rag_app.py
```
