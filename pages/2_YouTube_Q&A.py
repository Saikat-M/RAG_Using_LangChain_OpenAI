import os
import random 
import streamlit as st
import ops

from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory

from langchain_community.document_loaders import YoutubeLoader
from langchain_core.output_parsers import StrOutputParser

load_dotenv()
GEMINI_API_KEY = os.environ['GEMINI_API_KEY']
OPENAI_API_KEY = os.environ['OPENAI_API_KEY']
processed_documents = False

model = ChatOpenAI(openai_api_key = OPENAI_API_KEY, model='gpt-3.5-turbo', temperature=0)
embedding = OpenAIEmbeddings()
parser = StrOutputParser()
memory = ConversationBufferMemory(
    memory_key="video_chat_history",
    return_messages=True
)

st.title('YouTube Q&A')
url = st.text_input('Enter the YouTube vidoe url here...')

if 'video_chat_history' not in st.session_state:
    st.session_state.video_chat_history = []

for message in st.session_state.video_chat_history:
    with st.chat_message(message['role']):
        st.markdown(message['text'])


# #Function to create document loaders
def create_document(url):
    loader = YoutubeLoader.from_youtube_url(url, add_video_info = True)
    documents = loader.load()

    # st.write('No of documents: ', len(documents))
    # st.write('Documents: ', documents)
    return documents

if url:
    if not processed_documents:
        with st.spinner("Processing..."):
            # st.write('Form URL: ', random.random())
            documents = create_document(url)
            document_chunks = ops.create_document_chunks(documents)
            vectorDB = ops.create_vector_store(document_chunks)
            processed_documents = True

if processed_documents:
    user_query = st.chat_input('Enter your questions here...')

    if user_query:
        # st.write('From Query: ', random.random())
        with st.chat_message('user'):
            st.markdown(user_query)
            st.session_state.video_chat_history.append({"role":'user', "text":user_query})

        response = ops.get_response(user_query, vectorDB)

        with st.chat_message('assistant'):
            st.markdown(response['result'])
            st.session_state.video_chat_history.append({"role":'assistant', "text": response['result']})

