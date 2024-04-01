import streamlit as st
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA, ConversationalRetrievalChain
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter


#Function to create document chunks
def create_document_chunks(documents):
    # st.write('Inside Chunks')
    # Split
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 500,
        chunk_overlap = 50
    )
    data_chunks = text_splitter.split_documents(documents)
    # st.write('No of data chunks: ', len(data_chunks))
    return data_chunks

#Function to create vector store
def create_vector_store(chunks):
    # st.write(chunks)
    openai_embeddings = OpenAIEmbeddings()
    
    vectordb = Chroma.from_documents(
    documents=chunks,
    embedding=openai_embeddings)
    # st.write('vectordb collection.count', vectordb._collection.count())
    return vectordb

#Function to generate response
def get_response(question, vectorDB):
    # Build prompt
    template = """You are helpful AI assistant. Answer the question in detail from the provided context, make sure to provide all the details.
    If you don't know the answer, just say that you don't know, don't try to make up an answer. 
    Context: {context}
    Question: {question}
    Helpful Answer:"""
    QA_CHAIN_PROMPT = PromptTemplate.from_template(template=template)

    openai_llm = ChatOpenAI(model_name='gpt-3.5-turbo', temperature=0)

    qa_chain = RetrievalQA.from_chain_type(
    openai_llm,
    chain_type='stuff',
    retriever=vectorDB.as_retriever(),
    return_source_documents=True,
    chain_type_kwargs={'prompt': QA_CHAIN_PROMPT}
    )

    result = qa_chain({'query': question})
    # st.write('result: ', result)

    return result