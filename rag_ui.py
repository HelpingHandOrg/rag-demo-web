import streamlit as st
from rag_system import RAGSystem
import time
import traceback

# Set page configuration
st.set_page_config(
    page_title="RAG Demo System",
    page_icon="🔍",
    layout="wide"
)

# Add custom CSS
st.markdown("""
    <style>
    .stAlert {
        padding: 10px;
        margin-bottom: 20px;
    }
    .source-box {
        padding: 10px;
        border-radius: 5px;
        border: 1px solid #ddd;
        margin: 5px 0;
    }
    </style>
""", unsafe_allow_html=True)

# Debug information
st.sidebar.title("Debug Information")
debug_mode = st.sidebar.checkbox("Enable Debug Mode", value=True)

# Initialize RAG system
@st.cache_resource
def get_rag_system():
    try:
        if debug_mode:
            st.sidebar.info("Initializing RAG system...")
        rag = RAGSystem()
        if debug_mode:
            st.sidebar.success("RAG system initialized successfully!")
        return rag
    except Exception as e:
        st.sidebar.error(f"Error initializing RAG system: {str(e)}")
        st.sidebar.error(traceback.format_exc())
        raise

# Title and description
st.title("🔍 RAG Demo System")
st.markdown("""
This demo showcases a Retrieval-Augmented Generation (RAG) system that answers questions about organizations
and their learning paths. The system:
1. Retrieves relevant information using semantic search
2. Generates context-aware answers using OpenAI's GPT model
3. Provides transparency by showing the retrieved context and sources
""")

# Sidebar configuration
st.sidebar.title("Configuration")
top_k = st.sidebar.slider("Number of documents to retrieve", min_value=1, max_value=5, value=3)

# Example questions
st.sidebar.title("Example Questions")
example_questions = [
    "Which organizations have the highest learning path completion rates?",
    "What are the characteristics of organizations using the Enterprise platform?",
    "Compare the performance of organizations with Basic vs Enterprise subscriptions.",
    "Which organizations might need intervention due to low engagement?",
    "What patterns do you see in CNE usage across different platforms?"
]

# Create columns for the main content
col1, col2 = st.columns([2, 1])

with col1:
    # Question input with submit button
    question_container = st.container()
    with question_container:
        col_input, col_button = st.columns([4, 1])
        with col_input:
            question = st.text_input(
                "Enter your question",
                placeholder="Type your question here or select an example from the sidebar...",
                key="question_input"
            )
        with col_button:
            submit_button = st.button("Ask", key="submit_button", use_container_width=True)

    # Example question buttons
    st.markdown("### Or select an example question:")
    for i in range(0, len(example_questions), 2):
        col_a, col_b = st.columns(2)
        with col_a:
            if st.button(f"Example {i+1}", key=f"ex_{i}"):
                st.session_state.question_input = example_questions[i]
                st.experimental_rerun()
        with col_b:
            if i+1 < len(example_questions):
                if st.button(f"Example {i+2}", key=f"ex_{i+1}"):
                    st.session_state.question_input = example_questions[i+1]
                    st.experimental_rerun()

with col2:
    st.markdown("### System Status")
    status_placeholder = st.empty()
    status_placeholder.info("Ready to answer questions!")

# Process question when submitted
if submit_button and question:
    try:
        if debug_mode:
            st.sidebar.info(f"Processing question: {question}")
        
        # Initialize RAG system
        rag = get_rag_system()
        
        # Update status
        status_placeholder.info("Processing question...")
        
        # Get response
        with st.spinner("Retrieving relevant information and generating answer..."):
            start_time = time.time()
            result = rag.query(question, top_k=top_k)
            end_time = time.time()
            
        # Update status
        status_placeholder.success(f"Answer generated in {end_time - start_time:.2f} seconds!")
        
        # Display results
        st.markdown("### Retrieved Context and Answer")
        
        # Create tabs for different views
        context_tab, answer_tab, sources_tab = st.tabs(["Context", "Answer", "Sources"])
        
        with context_tab:
            st.markdown("#### Context Retrieved from Database")
            st.text_area("Retrieved Context", result.get('context', 'No context available'), height=300)
        
        with answer_tab:
            st.markdown("#### Generated Answer")
            st.markdown(result['answer'])
        
        with sources_tab:
            st.markdown("#### Sources Used")
            for source in result['sources']:
                with st.container():
                    st.markdown(f"""
                    <div class="source-box">
                        <strong>Organization:</strong> {source['organization']}<br>
                        <strong>Similarity Score:</strong> {source['similarity']:.3f}<br>
                        <strong>Description:</strong> {source['description']}
                    </div>
                    """, unsafe_allow_html=True)
        
    except Exception as e:
        error_msg = f"Error: {str(e)}\n\n{traceback.format_exc()}"
        status_placeholder.error(error_msg)
        st.error("An error occurred while processing your question. Check the debug information in the sidebar.")
        if debug_mode:
            st.sidebar.error(error_msg) 