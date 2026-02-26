import streamlit as st
import os
from PyPDF2 import PdfReader

# ----------------------------------------------------------------------
# Robust imports for LangChain components (handles older & newer versions)
# ----------------------------------------------------------------------
# Text splitter
try:
    from langchain.text_splitter import RecursiveCharacterTextSplitter
except ImportError:
    try:
        from langchain_text_splitters import RecursiveCharacterTextSplitter
    except ImportError:
        st.error("Missing required package: langchain-text-splitters. Please install it.")
        st.stop()

# Vector store (FAISS)
try:
    from langchain.vectorstores import FAISS
except ImportError:
    try:
        from langchain_community.vectorstores import FAISS
    except ImportError:
        st.error("Missing required package: langchain-community. Please install it.")
        st.stop()

# QA chain and prompts
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate

# DeepSeek via OpenAI-compatible ChatOpenAI
from langchain_openai import ChatOpenAI

# Local HuggingFace embeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
# ----------------------------------------------------------------------

# Set up the Streamlit interface
st.set_page_config(page_title="SS 638 Q&A Bot", layout="wide")
st.title("SS 638 (2018) Code of Practice Query Bot")
st.write(
    "Ask a question about the Singapore Standard for Electrical Installations, "
    "and get instant answers with clause references!"
)

# Securely load the DeepSeek API Key from Streamlit secrets
if "DEEPSEEK_API_KEY" in st.secrets:
    os.environ["DEEPSEEK_API_KEY"] = st.secrets["DEEPSEEK_API_KEY"]
else:
    st.error("Please set your DEEPSEEK_API_KEY in the Streamlit Secrets.")
    st.stop()  # Stop execution if no API key

@st.cache_resource
def process_pdf():
    """
    Load the PDF, split into chunks, and create a FAISS vector store
    using local HuggingFace embeddings.
    """
    # 1. Read the PDF
    pdf_reader = PdfReader("SS638_document.pdf")
    text = ""
    for page in pdf_reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text

    # 2. Split the text into manageable chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    chunks = text_splitter.split_text(text)

    # 3. Create a vector store using local HuggingFace embeddings
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_store = FAISS.from_texts(chunks, embedding=embeddings)
    return vector_store

# Try to load and process the document
try:
    vector_store = process_pdf()
    st.success("✅ SS 638 Document loaded successfully!")
except FileNotFoundError:
    st.error("Please ensure 'SS638_document.pdf' is in the same directory as this script.")
    st.stop()
except Exception as e:
    st.error(f"An error occurred while processing the PDF: {e}")
    st.stop()

# Set up the Prompt Template
prompt_template = """
You are an expert in the Singapore Standard SS 638: 2018 Code of practice for electrical installations.
Answer the user's question based ONLY on the provided context. 
Always include the specific Clause number or Section reference in your answer.
If the answer is not in the context, just say "I cannot find the answer in the provided SS 638 document."

Context:\n {context}\n
Question: \n{question}\n

Answer:
"""
prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

# DeepSeek chat model via OpenAI-compatible API
model = ChatOpenAI(
    model="deepseek-chat",
    temperature=0.2,
    openai_api_key=os.environ["DEEPSEEK_API_KEY"],
    openai_api_base="https://api.deepseek.com/v1"
)

# Create the QA chain
chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)

# User Input
user_question = st.text_input(
    "Enter your question here (e.g., 'What are the requirements for locations containing a bath or shower?'):"
)

if user_question:
    with st.spinner("🔍 Searching the standard..."):
        # Retrieve relevant chunks
        docs = vector_store.similarity_search(user_question)
        # Get answer from the LLM
        response = chain(
            {"input_documents": docs, "question": user_question},
            return_only_outputs=True
        )
        st.write("### Answer:")
        st.write(response["output_text"])
