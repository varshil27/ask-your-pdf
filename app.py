import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from htmltemp import css, bot_template, user_template
import os
from dotenv import load_dotenv, dotenv_values

def txt2text(file):
    text = file.read().decode("utf-8")
    return text


def pdf2text(pdf):
    text = ""
    pdf_reader = PdfReader(pdf)
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text


def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks


def get_vectorstore(text_chunks):
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore


def get_conversation_chain(vectorstore):
    llm = ChatOpenAI(model='gpt-3.5-turbo')

    memory = ConversationBufferMemory(
        memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain


def userinput_llmoutput(user_question):
    response = st.session_state.conversation({'question': user_question})
    st.session_state.chat_history = st.session_state.chat_history or []
    if 'chat_history' in response:
        st.session_state.chat_history += response['chat_history']
    if st.button("Clear"):
        st.session_state.chat_history = []

    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(user_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)
            with st.expander('Document Similarity Search'):
                search = st.session_state.vectorstore.similarity_search_with_score(user_question)
                st.write(search[0][0].page_content)


def main():
    load_dotenv()
    os.environ['OPENAI_API_KEY'] = os.getenv('key')
    st.set_page_config(page_title="Chat with multiple documents",
                       page_icon=":books:")
    st.write(css, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None
    if "vectorstore" not in st.session_state:
        st.session_state.vectorstore = None

    st.header("Chat with multiple documents :robot_face:")
    if "raw_text" not in st.session_state:
        st.session_state.raw_text = ""
    if "tex_chunks" not in st.session_state:
        st.session_state.text_chunks = ""
    with st.sidebar:
        st.subheader("Your documents :file_folder:")
        files = st.file_uploader(
            "Upload your documents here and click on 'Process'",type=["txt", "pdf"], accept_multiple_files=True)
        if st.button("Process :envelope_with_arrow:"):
            with st.spinner("Processing"):
                for file in files:
                    file_extension = file.name.split(".")[-1].lower()

                    if file_extension == "txt":
                        st.session_state.raw_text += txt2text(file)

                    elif file_extension == "pdf":
                        st.session_state.raw_text += pdf2text(file)

                
                
    st.session_state.text_chunks = get_text_chunks(st.session_state.raw_text)
    vectorstore = None
    if len(st.session_state.text_chunks):
        vectorstore = get_vectorstore(st.session_state.text_chunks)
        st.session_state.conversation = get_conversation_chain(vectorstore)
        st.session_state.vectorstore = vectorstore 

    user_question = st.text_input("Ask a question to your documents:")
    if user_question:
        userinput_llmoutput(user_question)


if __name__ == '__main__':
    main()
