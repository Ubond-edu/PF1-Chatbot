import streamlit as st
from streamlit_chat import message
from streamlit_extras.colored_header import colored_header
from streamlit_extras.add_vertical_space import add_vertical_space
from hugchat import hugchat

from document_processing import index, chain

st.set_page_config(page_title="HugChat - An LLM-powered Streamlit app")
st.title('🎈 PF1-Chatbot')
st.write('Hello student!')

with st.sidebar:
    st.title('🤗💬 HugChat App')
    st.markdown('''
    ''')
    add_vertical_space(5)
    st.write('Made by Matt')

    if 'generated' not in st.session_state:
        st.session_state['generated'] = ["I'm HugChat, How may I help you?"]

    if 'past' not in st.session_state:
        st.session_state['past'] = ['Hi!']

input_container = st.container()
colored_header(label='', description='', color_name='blue-30')
response_container = st.container()

# User input
## Function for taking user provided prompt as input
def get_text():
    input_text = st.text_input("You: ", "", key="input")
    return input_text

## Applying the user input box
with input_container:
    user_input = get_text()

# Response output
## Function for taking user prompt as input followed by producing AI generated responses
def generate_response(prompt):
    chatbot = hugchat.ChatBot()
    response = chatbot.chat(prompt)
    return response

def get_similar_docs(query, k=2, score=False):
    if score:
        similar_docs = get_similar_docs_with_score(query, k=k)
    else:
        similar_docs = get_similar_docs_without_score(query, k=k)
    return similar_docs

def get_similar_docs_with_score(query, k=3):
    similar_docs = index.similarity_search_with_score(query, k=k)
    return similar_docs

def get_similar_docs_without_score(query, k=3):
    similar_docs = index.similarity_search(query, k=k)
    return similar_docs

def get_answer(query):
    similar_docs = get_similar_docs(query)
    answer = chain.run(input_documents=similar_docs, question=query)
    return answer

if user_input:
    answer = get_answer(user_input)
    with response_container:
        st.write(answer)  # Use st.write() to display the message
else:
    st.write("Please enter a prompt.")  # Use st.write() for the prompt message
