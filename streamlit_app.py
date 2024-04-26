import streamlit as st 
import fitz
import shelve
import transformers
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import os
import warnings
warnings.simplefilter("ignore")

from transformers import DistilBertForQuestionAnswering
from transformers import AutoTokenizer
trained_checkpoint = "distilbert-base-uncased"

#load the model and tokenizer 
tokenizer = AutoTokenizer.from_pretrained(trained_checkpoint)
model = DistilBertForQuestionAnswering.from_pretrained('my_new_model')

st.set_page_config(layout="wide")
st.title("Question Answer Bot")

#extract text from the PDF data 
def extract_text_from_pdf(pdf_file):
    text = ""
    with fitz.open(pdf_file) as doc:
        for page in doc:
            text += page.get_text()
    return text

#function to return answer from taking context and question as a input of the model 
def output_answer( question, context):
  model = DistilBertForQuestionAnswering.from_pretrained('my_new_model')
  inputs = tokenizer(question, context, return_tensors="pt")
  outputs = model(**inputs)

  start_logits = outputs.start_logits.argmax().item()
  end_logits = outputs.end_logits.argmax().item()

  answer = tokenizer.decode(inputs["input_ids"][0][start_logits : end_logits + 1])

  return answer
col1 , col2 = st.columns(2)
context_text = None
with col1:
    # st.header("Enter Context")
    # Text input section
    context_text = None  # Initialize context variable
    with st.expander("Enter Context"):
        context_container = st.empty()
        context_text = context_container.text_area("Write here")
with col2:
    # st.header("Upload PDF")
    # Upload PDF section
    uploaded_file = st.file_uploader("Upload PDF", type=["pdf"])
    if uploaded_file is not None:
        context_text = extract_text_from_pdf(uploaded_file)

if context_text:
    st.subheader("context:")
    st.info(context_text)
    context_container = st.empty()
    
# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

#load chat history from shelve file    
def load_chat_history():
    with shelve.open("chat_history") as db:
        return db.get("messages",[])
    
#save the history file to shelve file
def save_chat_history(messages):
    with shelve.open("chat_history") as db:
        db["messages"] = messages

#initialize or load chat history        
if "messages" not in st.session_state:
    st.session_state.messages = load_chat_history()   
    
        
#display chat messages from history on app rerun 
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"]) 
        
#Sidebar with a button to delete chat history
with st.sidebar:
    if st.button("Delete Chat History"):
        st.session_state.messages = []
        save_chat_history([])
        context_text = None
        context_container=st.empty()
 #User input
question = st.chat_input("Ask question") 
if question:
    #Display user message in chat message container
    with st.chat_message("user"):
        question_responce = f"User: {question}"
        st.markdown(question_responce)
        
    #add user message to chat history
    st.session_state.messages.append({"role":"user","content":question})
    
    #generate the answer from the model 
    answer = output_answer(question= question, context= context_text)
    response = f"Bot: {answer}"
    
    #display assistant responce in chat message container
    with st.chat_message("assistant"):
        st.markdown(response)
        
    #add assistant responce to chat history
    st.session_state.messages.append({"role":"assistant","content":response})
    
    
    
    
