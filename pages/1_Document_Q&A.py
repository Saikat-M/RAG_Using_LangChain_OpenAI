import os 
import streamlit as st
import ops

from dotenv import load_dotenv

from langchain_google_genai import GoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings

from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.summarize import load_summarize_chain
from langchain.memory import ConversationBufferMemory
from langchain.chains import RetrievalQA, ConversationalRetrievalChain


load_dotenv()
GEMINI_API_KEY = os.environ['GEMINI_API_KEY']
OPENAI_API_KEY = os.environ['OPENAI_API_KEY']

st.title('PDF Q&A')

# #Function to create document loaders
def create_document(file_path):
    # Get a list of filenames
    document_filenames = os.listdir(file_path)
    # st.write(document_filenames)
    document_loaders = []
    documents = []
    # Load PDF
    for filename in document_filenames:
        document_path = os.path.join(file_path, filename)
        document_loaders.append(PyPDFLoader(document_path))

    for loader in document_loaders:
        documents.extend(loader.load())

    # st.write('No of documents: ', len(documents))
    # st.write('Documents: ', documents)
    return documents

memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True
)


uploaded_files = st.file_uploader("Upload a file", type = ['pdf'], accept_multiple_files=True)

if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

for message in st.session_state.chat_history:
    with st.chat_message(message['role']):
        st.markdown(message['text'])

# def translate_text(text, language):
    # # https://huggingface.co/google/flan-t5-xl
    # llm = HuggingFaceHub(repo_id="google/flan-t5-xl", model_kwargs={"temperature":0, "max_length":100})

    # response = llm("translate English to {language}: {text}")
    # response2 = llm("translate English to German: How old are you?")
    # return response2

if uploaded_files:
    with st.spinner("Processing..."):
        # Create the directory if it doesn't exist
        os.makedirs('files', exist_ok=True)
        # Save the uploaded file to the specified directory
        for docs in uploaded_files:
            with open(os.path.join('files', docs.name), 'wb') as f:
                f.write(docs.getbuffer())

        file_path = 'files'
        documents = create_document(file_path)
        document_chunks = ops.create_document_chunks(documents)
        vectorDB = ops.create_vector_store(document_chunks)

        user_query = st.chat_input('Enter your questions here...')

        if user_query:
            with st.chat_message('user'):
                st.markdown(user_query)
                st.session_state.chat_history.append({"role":'user', "text":user_query})

            response = ops.get_response(user_query, vectorDB)

            with st.chat_message('assistant'):
                st.markdown(response['result'])
                st.session_state.chat_history.append({"role":'assistant', "text": response['result']})

            # st.write(response)


    # if q_and_a_prompt: 
    #     #Sending the Query to ChatVector to get answers
    #     result = pdf_qa({"question": q_and_a_prompt, "chat_history": ''})
    #     st.write(result["answer"])
    #     # st.write(translate_text(result["answer"], 'Bengali'))


