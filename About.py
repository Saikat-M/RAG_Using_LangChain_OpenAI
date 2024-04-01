import os 
from dotenv import load_dotenv

import streamlit as st 
from streamlit_extras.switch_page_button import switch_page


load_dotenv()
OPENAI_API_KEY = os.environ['OPENAI_API_KEY']

st.title('RAG Q&A with Youtube Videos & PDF Documents')
st.divider()

st.caption('In this App an user can upload a PDF Document & provide link for YouTube Videos to retrieve information though Q&A')

chosenOption = st.selectbox(
    'What do You want to chat with?',
    ('','Youtube Video', 'PDF Document'))

if chosenOption == 'Youtube Video':
  switch_page('YouTube_Q&A')
elif chosenOption == 'PDF Document':
  switch_page('Document_Q&A')


