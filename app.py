import streamlit as st 
from PyPDF2 import PdfReader 
from langchain.text_splitter import RecursiveCharacterTextSplitter 
import os 
 
from langchain_google_genai import GoogleGenerativeAIEmbeddings 
import google.generativeai as genai  
from langchain_community.vectorstores import FAISS 
from langchain_google_genai import ChatGoogleGenerativeAI 
from langchain.chains.question_answering import load_qa_chain 
from langchain.prompts import PromptTemplate 
from dotenv import load_dotenv 
load_dotenv() 
 
GOOGLE_GENAI_API_KEY='AIzaSyDAOXIGyVz8pYoda7Frc0WjHn3PNGHidTQ' 
genai.configure(api_key=os.getenv("GOOGLE_GENAI_API_KEY"))  
 
 
def get_pdf_text(pdf_docs): 
    text = "" 
    for pdf in pdf_docs: 
        pdf_reader = PdfReader(pdf) 
        for page in pdf_reader.pages: 
            text += page.extract_text() 
    return text 
 
def get_text_chunks(text): 
    text_splitter = RecursiveCharacterTextSplitter( 
        chunk_size=1000, 
        chunk_overlap=1000) 
    chunks = text_splitter.split_text(text) 
    return chunks 
   
def get_vector_store(text_chunks): 
    embeddings = GoogleGenerativeAIEmbeddings(model="model/emedding-001") 
    vector_store = FAISS.from_texts(texts=text_chunks, embedding=embeddings) 
    vector_store.save_local("faiss_index") 
 
def get_conversation_chain(vector_store): 
    prompt_templates="Answer the question based on the context below. If the question cannot be answered using the information provided answer" 
    model = ChatGoogleGenerativeAI(model="gemini-pro",temperature=0.3) 
    prompt = PromptTemplate( 
        input_variables=["context", "question"], 
        template=prompt_templates 
    ) 
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt) 
    return chain 
 
def user_input(user_question): 
    embeddings = GoogleGenerativeAIEmbeddings(model="model/emedding-001") 
    new_db = FAISS.load_local("faiss_index", embeddings) 
    docs=new_db.similarity_search(user_question) 
    chain=get_conversation_chain() 
    reponse=chain({"input_documents": docs, "question": user_question},return_only_outputs=True) 
    print(reponse) 
    st.write("Reply:",reponse["output_text"]) 
 
def main(): 
  st.set_page_config("Chat With Multiple PDF") 
  st.header("Chat With Multiple PDF using Gemini") 
  user_question=st.text_input("Ask a question about your documents:")  
 
  if user_question: 
    user_input(user_question) 
 
  with st.sidebar: 
    st.title("Menu:") 
    pdf_docs=st.file_uploader("Upload your PDFs here and click on 'Process'",accept_multiple_files=True) 
    if st.button("Submit & Process"): 
      with st.spinner("Processing"): 
        raw_text=get_pdf_text(pdf_docs) 
        text_chunks=get_text_chunks(raw_text) 
        vector_store=get_vector_store(text_chunks) 
        st.sucess("Done") 
 
if os.name == "__main__": 
  main()