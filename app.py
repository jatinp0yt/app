import streamlit as st
# from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS

def hf_token():
    HUGGINGFACEHUB_API_TOKEN=st.text_input('enter your hf token :')
    return HUGGINGFACEHUB_API_TOKEN

def get_pdf_text(pdf_docs):
    text = ''
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunk(raw_text):
    text_splitter = CharacterTextSplitter(
        separator='\n',
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(raw_text)
    return chunks

def get_vectorstore(text_chunks):
    embeddings = HuggingFaceInstructEmbeddings(model_name='hkunlp/instructor-xl')
    vectorstore = FAISS.from_texts(texts = text_chunks, embedding = embeddings)
    return vectorstore

def main():
    # load_dotenv()
    st.set_page_config(page_title='MultiPDFchat',page_icon=':books:')
    hf_token()
#  sidebar
    with st.sidebar:
        st.header('Your Docs :books:')
        pdf_docs = st.file_uploader('Upload here', accept_multiple_files=True)
        if st.button('Process'):
            with st.spinner('processing on it..'):
                #get pdf
                raw_text = get_pdf_text(pdf_docs)
                
                #get text chunk
                text_chunks = get_text_chunk(raw_text)
                st.write(text_chunks)

                #create vectore store
                vectorstore = get_vectorstore(text_chunks)


    st.header('Multiple PDF chat Here :books:')
    st.text_input('Ask Here...',)


if __name__ == '__main__':
    main()