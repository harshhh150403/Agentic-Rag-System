import streamlit as st
import os
import time
from dotenv import load_dotenv

# Imports
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI

from langchain.storage import InMemoryStore

from langchain.retrievers import ParentDocumentRetriever
from langchain.tools import Tool
from sentence_transformers import CrossEncoder
from langchain import hub
from langchain.agents import create_react_agent, AgentExecutor
from langchain.callbacks.base import BaseCallbackHandler
from langchain_core.messages import HumanMessage, AIMessage


# HELPER CLASS FOR STREAMLIT UI ---
class StreamlitCallbackHandler(BaseCallbackHandler):
    """Callback handler to display agent's thought process in Streamlit."""
    def __init__(self, container, initial_text="", display_method='markdown'):
        super().__init__()
        self.container = container
        self.text = initial_text
        self.display_method = display_method

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.text += token
        self.container.markdown(self.text)

    def on_tool_end(self, output: str, **kwargs) -> None:
        self.container.markdown(f"**Tool Output:**\n```\n{output}\n```")
        self.text = ""

# DATA PROCESSING AND AGENT SETUP (WITH PARENT DOCUMENT RETRIEVER) ---
@st.cache_resource(show_spinner="Setting up the RAG agent...")
def setup_agent():
    """
    This function sets up an advanced RAG pipeline using a Parent Document Retriever
    and a Cross-Encoder for re-ranking.
    """
    persist_directory = "chroma_db_parent_retriever"
    
    # 1. Load Models (Embedding and Re-ranking) ---
    st.info("Loading local embedding and re-ranking models...")
    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2", 
        model_kwargs={'device': 'cpu'}
    )
    cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
    st.success("Models loaded.")

    # --- 2. Load and Split Documents ---
    docs_folder = "docs"
    if not os.path.exists(docs_folder):
        st.error("The 'docs' folder is missing. Please create it and add your PDF files.")
        st.stop()
    loaders = [PyPDFLoader(os.path.join(docs_folder, fn)) for fn in os.listdir(docs_folder) if fn.endswith('.pdf')]
    docs = [doc for loader in loaders for doc in loader.load()]

    # Set up the Parent Document Retriever ---
    # Use an in-memory store for parent documents.
    store = InMemoryStore()
    
    vectorstore = Chroma(
        collection_name="parent_document_retriever", 
        embedding_function=embedding_model,
        persist_directory=persist_directory
    )

    parent_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
    child_splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=100)

    parent_document_retriever = ParentDocumentRetriever(
        vectorstore=vectorstore,
        docstore=store,
        child_splitter=child_splitter,
        parent_splitter=parent_splitter,
        search_kwargs={"k": 10}
    )

    # Simplified the check. Caching handles session persistence.
    # This will run once per session on the first run.
    try:
        if vectorstore._collection.count() == 0:
            st.info("Populating stores... (one-time process per session)")
            time.sleep(3)
            parent_document_retriever.add_documents(docs, ids=None)
            st.success("Stores populated.")
        else:
            st.info("Loading existing vector store from disk.")
    except Exception as e:
        st.error(f"An error occurred while setting up the store: {e}")
        st.info("Attempting to populate stores...")
        parent_document_retriever.add_documents(docs, ids=None)
        st.success("Stores populated.")


    # --- 4. Create the Final Tool with Re-ranking ---
    def retrieve_and_rerank(query: str) -> str:
        """
        Retrieves parent documents and then re-ranks them for higher relevance.
        """
        st.info(f"1. Retrieving parent documents for query: '{query}'")
        initial_docs = parent_document_retriever.get_relevant_documents(query)
        
        st.info("2. Re-ranking retrieved documents with Cross-Encoder...")
        pairs = [[query, doc.page_content] for doc in initial_docs]
        scores = cross_encoder.predict(pairs)
        
        scored_docs = list(zip(scores, initial_docs))
        scored_docs.sort(key=lambda x: x[0], reverse=True)
        
        reranked_docs = [doc for score, doc in scored_docs[:3]]
        st.success("3. Re-ranking complete. Top 3 documents selected.")
        
        return "\n\n".join([f"Source: {doc.metadata.get('source', 'N/A')}, Page: {doc.metadata.get('page', 'N/A')}\nContent: {doc.page_content}" for doc in reranked_docs])

    retriever_tool = Tool(
        name="ai_research_paper_retriever",
        func=retrieve_and_rerank,
        description="Searches and returns the most relevant excerpts from the AI research papers on Transformers, BERT, and GPT-3. This is the primary tool for answering questions."
    )
    tools = [retriever_tool]

    # --- 5. Create the Agent ---
    prompt = hub.pull("hwchase17/react-chat")
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-preview-05-20", temperature=0, convert_system_message_to_human=True)
    agent = create_react_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, handle_parsing_errors=True)
    
    return agent_executor

# --- STREAMLIT UI (No changes needed below this line) ---
def main():
    """The main function that runs the Streamlit application."""
    load_dotenv()
    st.set_page_config(page_title="AI Research Paper Q&A", page_icon="🤖")
    st.title("🤖 AI Research Paper Q&A")
    st.markdown("Ask questions about 'Attention Is All You Need', 'BERT', and 'GPT-3'.")

    if not os.getenv("GOOGLE_API_KEY"):
        st.error("Google API key not found. Please create a .env file with GOOGLE_API_KEY='your_key'.")
        st.stop()
        
    agent_executor = setup_agent()

    if 'messages' not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if user_question := st.chat_input("Ask your question:", key="user_question"):
        st.session_state.messages.append({"role": "user", "content": user_question})
        with st.chat_message("user"):
            st.markdown(user_question)

        with st.chat_message("assistant"):
            with st.spinner("Agent is thinking..."):
                chat_history = []
                for msg in st.session_state.messages[:-1]:
                    if msg["role"] == "user":
                        chat_history.append(HumanMessage(content=msg["content"]))
                    elif msg["role"] == "assistant":
                        chat_history.append(AIMessage(content=msg["content"]))

                thought_process_container = st.expander("Show Agent's Thought Process", expanded=False)
                with thought_process_container:
                    st_callback = StreamlitCallbackHandler(st.container(), display_method='markdown')
                    response = agent_executor.invoke(
                        {"input": user_question, "chat_history": chat_history},
                        {"callbacks": [st_callback]}
                    )
                final_answer = response.get("output", "Sorry, I couldn't find an answer.")
                st.markdown(final_answer)
                st.session_state.messages.append({"role": "assistant", "content": final_answer})

if __name__ == "__main__":
    main()

