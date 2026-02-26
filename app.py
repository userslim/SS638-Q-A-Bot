import streamlit as st
import os
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate

# Set up the Streamlit interface
st.set_page_config(page_title="SS 638 Q&A Bot", layout="wide")
st.title("SS 638 (2018) Code of Practice Query Bot")
st.write("Ask a question about the Singapore Standard for Electrical Installations, and get instant answers with clause references!")

# Securely load the API Key from Streamlit secrets
if "GOOGLE_API_KEY" in st.secrets:
    os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]
else:
    st.error("Please set your GOOGLE_API_KEY in the Streamlit Secrets.")

@st.cache_resource
def process_pdf():
    # 1. Read the PDF
    pdf_reader = PdfReader("SS638_document.pdf")
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
        
    # 2. Split the text into manageable chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_text(text)
    
    # 3. Create a vector store (database) of the text
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(chunks, embedding=embeddings)
    return vector_store

# Try to load and process the document
try:
    vector_store = process_pdf()
    st.success("SS 638 Document loaded successfully!")
except FileNotFoundError:
    st.error("Please ensure 'SS638_document.pdf' is in the same directory as this script.")
    st.stop()

# Set up the Prompt Template
prompt_template = """
You are an expert in the Singapore Standard SS 638: 2018 Code of practice for electrical installations.
Answer the user's question based ONLY on the provided context. 
Always include the specific Clause number or Section reference in your answer.
If the answer is not in the context, just say "I cannot find the answer in the provided SS 638 document."

Context:\n {context}?\n
Question: \n{question}\n

Answer:
"""
prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.2)
chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)

# User Input
user_question = st.text_input("Enter your question here (e.g., 'What are the requirements for locations containing a bath or shower?'):")

if user_question:
    with st.spinner("Searching the standard..."):
        docs = vector_store.similarity_search(user_question)
        response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
        st.write("### Answer:")
        st.write(response["output_text"])