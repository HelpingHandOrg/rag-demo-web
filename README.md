# RAG Demo System

A Retrieval-Augmented Generation (RAG) system that answers questions about organizations and their learning paths. The system combines semantic search with LLM-based answer generation to provide accurate, context-aware responses.

## Features

- Semantic search using sentence-transformers
- Context-aware answer generation using OpenAI's GPT model
- Interactive web interface built with Streamlit
- Real-time processing and response generation
- Transparent display of retrieved context and sources

## Setup

1. Clone this repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Create a `.env` file with your credentials:
   ```
   OPENAI_API_KEY=your-openai-api-key
   MONGODB_URI=your-mongodb-uri
   ```
4. Run the application:
   ```bash
   streamlit run rag_ui.py
   ```

## Deployment

This application can be deployed on Streamlit Cloud:

1. Push your code to a GitHub repository
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub account
4. Select your repository
5. Add your environment variables in the Streamlit Cloud settings
6. Deploy!

## Environment Variables

- `OPENAI_API_KEY`: Your OpenAI API key
- `MONGODB_URI`: Your MongoDB connection string

## Project Structure

- `rag_ui.py`: Streamlit web interface
- `rag_system.py`: RAG system implementation
- `semantic_search.py`: Semantic search engine
- `requirements.txt`: Project dependencies
- `.env`: Environment variables (not included in repository)

## License

MIT License
