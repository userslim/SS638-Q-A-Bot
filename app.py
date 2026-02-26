import streamlit as st
import os
from PyPDF2 import PdfReader

# ----------------------------------------------------------------------
# Robust imports with fallbacks and user-friendly errors
# ----------------------------------------------------------------------
# First, ensure langchain itself is installed
try:
    import langchain
except ImportError:
    st.error("Missing required package: langchain. Please check your requirements.txt.")
    st.stop()

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

# QA chain
try:
    from langchain.chains.question_answering import load_qa_chain
except ImportError:
    st.error("Failed to import load_qa_chain from langchain. Ensure langchain version is >=0.1.0.")
    st.stop()

# Prompts
try:
    from langchain.prompts import PromptTemplate
except ImportError:
    st.error("Failed to import PromptTemplate from langchain.")
    st.stop()

# DeepSeek via ChatOpenAI
try:
    from langchain_openai import ChatOpenAI
except ImportError:
    st.error("Missing required package: langchain-openai. Please install it.")
    st.stop()

# Local HuggingFace embeddings
try:
    from langchain_community.embeddings import HuggingFaceEmbeddings
except ImportError:
    st.error("Missing required package: langchain-community. Please install it.")
    st.stop()
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
    st.stop()

@st.cache_resource
def process_pdf():
    """Load PDF, split into chunks, and create FAISS vector store."""
    try:
        pdf_reader = PdfReader("SS638_document.pdf")
    except FileNotFoundError:
        st.error("SS638_document.pdf not found in the current directory.")
        st.stop()

    text = ""
    for page in pdf_reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_text(text)

    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_store = FAISS.from_texts(chunks, embedding=embeddings)
    return vector_store

# Load document
with st.spinner("Loading SS 638 document and creating embeddings..."):
    vector_store = process_pdf()
st.success("✅ SS 638 Document loaded successfully!")

# Set up prompt template
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

# DeepSeek chat model
model = ChatOpenAI(
    model="deepseek-chat",
    temperature=0.2,
    openai_api_key=os.environ["DEEPSEEK_API_KEY"],
    openai_api_base="https://api.deepseek.com/v1"
)

chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)

# User input
user_question = st.text_input(
    "Enter your question here (e.g., 'What are the requirements for locations containing a bath or shower?'):"
)

if user_question:
    with st.spinner("🔍 Searching the standard..."):
        docs = vector_store.similarity_search(user_question)
        response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
        st.write("### Answer:")
        st.write(response["output_text"])
