import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain.schema import Document
from langchain.document_loaders.pdf import PyPDFLoader
# from langchain_community.vectorstores import Pinecone
from langchain_community.vectorstores import Pinecone as PineconeStore
# from langchain.vectorstores import Pinecone
from dotenv import load_dotenv
from pinecone import Pinecone
import asyncio
import os

import json
load_dotenv()
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
OPENAI_API_KEY=os.environ.get('OPENAI_API_KEY')

embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

# # Function to push embedded data to Vector Store - Pinecone
# def push_to_pinecone(pinecone_apikey,pinecone_environment,pinecone_index_name,embeddings,docs):
#     text_splitter = RecursiveCharacterTextSplitter()
#     document_chunks = text_splitter.split_documents(docs)
#     pinecone = Pinecone(
#         api_key=pinecone_apikey,environment=pinecone_environment
#         )
#     # create a vectorstore from the chunks
#     vector_store=PineconeStore.from_documents(document_chunks,embeddings,index_name=pinecone_index_name)

def get_vectorstore():
    vector_store = PineconeStore.from_existing_index(index_name="qa",embedding=embeddings)
    return vector_store

def get_context_retriever_chain(vector_store):
    llm = ChatOpenAI()
    retriever = vector_store.as_retriever()  
    prompt = ChatPromptTemplate.from_messages([
      MessagesPlaceholder(variable_name="chat_history"),
      ("user", "{input}"),
      ("user", "Given the above conversation, generate a search query to look up in order to get information relevant to the conversation")
    ])
    # If there is no chat_history, then the input is just passed directly to the retriever. 
    retriever_chain = create_history_aware_retriever(llm, retriever, prompt)
    return retriever_chain
    
def get_conversational_rag_chain(retriever_chain): 
    llm = ChatOpenAI()
    prompt = ChatPromptTemplate.from_messages([
      ("system", "you are the chatmodel of the given document, your name is 'PdChat', you can answerr in the very familier and good manner,Answer the user's questions based on the below context:\n\n{context}"),
      MessagesPlaceholder(variable_name="chat_history"),
      ("user", "{input}"),
    ])
    # for passing a list of Documents to a model.
    stuff_documents_chain = create_stuff_documents_chain(llm,prompt)
    return create_retrieval_chain(retriever_chain, stuff_documents_chain)

def get_response(user_input):
    retriever_chain = get_context_retriever_chain(st.session_state.vector_store)
    conversation_rag_chain = get_conversational_rag_chain(retriever_chain)
    response = conversation_rag_chain.invoke({
        "chat_history": st.session_state.chat_history,
        "input": user_input
    })
    return response['answer']


# loader = PyPDFLoader('G:\VKAPS Internship Projects AI\chat with link\HR Policy Summary.pdf')
# docs = loader.load()
# push_to_pinecone(PINECONE_API_KEY,"gcp-starter","qa",embeddings,docs)


# app config
st.set_page_config(page_title="Chat with Your Websites", page_icon="🤖")

# session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        AIMessage(content="Hello This is PdChat here how can I assist you today..?"),
    ]
if "vector_store" not in st.session_state:
    st.session_state.vector_store = get_vectorstore()  

# Render selected section
st.header('PdChat Your PDF ChatBot ')
st.text('Powerd by OpenAI')

# conversation
user_query = st.chat_input("Ask your query here About the Given PDF...")
for message in st.session_state.chat_history :
    if isinstance(message, HumanMessage)  :
        with st.chat_message("You")   :
            st.markdown(message.content)
    else  :
        with st.chat_message("AI"):
            st.markdown(message.content)


if user_query:
    response = get_response(user_query)
    # Display user's question
    with st.chat_message("You"):
        st.markdown(user_query)
    # Display AI's answer
    with st.chat_message("AI"):
        st.markdown(response)

    # Update chat history
    st.session_state.chat_history.append(HumanMessage(content=user_query))
    st.session_state.chat_history.append(AIMessage(content=response))
