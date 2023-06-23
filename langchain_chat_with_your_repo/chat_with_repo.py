import os
from functools import partial

from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv()) # read local .env file for our API Keys
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import GitLoader
from langchain.text_splitter import (
    RecursiveCharacterTextSplitter,
    Language,
)
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.memory import ConversationTokenBufferMemory
import gradio as gr
import markdown
from langchain.prompts import ChatPromptTemplate

repo_url = "https://github.com/ganlanyuan/tiny-slider"
local_path = os.getcwd() + '/repo'
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

def get_loader(local_path, repo_url, branch = 'master', has_file_ext = ['.md', '.js', '.html'], ignore_paths = ['dist/']):
    """
    Helper function to create a Loader to load the repo

    Args:
        local_path (str): Path to the local repo
        repo_url (str): URL of the repo to clone from
        branch (str): Branch of the repo to checkout
        has_file_ext (list): ist of file extensions to load
        ignore_paths (list): List of paths to ignore
    
    Returns:
        GitLoader Object
    """
    
    # If the path exists, the GitLoader will throw an Error when trying to clone
    if os.path.exists(local_path):
        repo_url = None
    
    file_filter_functions = []

    def not_in_ignore_paths(file_path, ignore_paths):
        return all(file_path.find(path) == -1 for path in ignore_paths)

    def has_allowed_extension(file_path, extensions):
        return any(file_path.endswith(ext) for ext in extensions)

    if len(ignore_paths):
        file_filter_functions.append(partial(not_in_ignore_paths, ignore_paths=ignore_paths))

    if len(has_file_ext):
        file_filter_functions.append(partial(has_allowed_extension, extensions=has_file_ext))

    def file_filter_function(file_path):
        return all(func(file_path) for func in file_filter_functions)
    
    return GitLoader(repo_path=local_path, clone_url=repo_url, branch=branch, file_filter=file_filter_function)

def split_docs(docs):
    """
    Helper function to split the docs into chunks by supported languages.

    Args:
        docs (list): List of documents to split
    
    Returns:
        list of documents
    """
    
    js_splitter = RecursiveCharacterTextSplitter.from_language(
        language=Language.JS, chunk_size=1024, chunk_overlap=0
    )
    html_splitter = RecursiveCharacterTextSplitter.from_language(
        language=Language.HTML, chunk_size=1024, chunk_overlap=0
    )
    markdown_splitter = RecursiveCharacterTextSplitter.from_language(
        language=Language.MARKDOWN, chunk_size=1024, chunk_overlap=0
    )

    # Only retrieve the text from the documents
    text_docs = [doc.page_content for doc in docs]
    
    js, html, markdown = js_splitter.create_documents(texts=text_docs), html_splitter.create_documents(texts=text_docs), markdown_splitter.create_documents(texts=text_docs)

    # merge the docs to List
    return js + html + markdown

loader = get_loader(local_path, repo_url)
docs = loader.load()
splitted_docs = split_docs(docs)

embeddings = OpenAIEmbeddings()
vectorstore = Chroma.from_documents(documents=splitted_docs, embedding=embeddings)

vector_retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

memory = ConversationTokenBufferMemory(llm=llm, max_token_limit=512)

prompt_template = ChatPromptTemplate.from_template(
    """
    Your task is to assist with questions from a code repository. \
    The Repository is a library called tiny slider. \
    Use the following pieces of context to answer the question at the end. \
    The context are snippets of files from the repository \
    When answering with code snippets, make sure to wrap them in the correct syntax using markdown backticks. \
    If you don't know the answer, just say that you don't know, don't try to make up an answer.

    {context}

    Question: {question}
    Helpful Answer:
    """
)

chain = RetrievalQA.from_llm(
    llm=llm,
    memory=memory, 
    retriever=vector_retriever, 
    prompt=prompt_template, 
)

# This list will keep track of the conversation history for the ui
conversation_history = []

def format_input(user, user_input):
    return f"## {user}:\n\n {user_input}\n\n"

def handle_submit(input_text):
    # Add user input to conversation history
    conversation_history.append({'type': 'User', 'text': input_text})
    
    # Get our result from the chain and append to history:
    result = chain.run(input_text)
    conversation_history.append({'type': 'AI', 'text': f"{result}\n\n"})
    
    # Construct markdown text from conversation_history
    md_text = ""
    for entry in conversation_history:
        md_text += format_input(entry['type'], entry['text'])

    # Convert markdown to HTML
    html_conversation = markdown.markdown(md_text)
    
    # Return the conversation history as HTML
    return html_conversation

# Define the Gradio interface
iface = gr.Interface(
    fn=handle_submit,                                         # Function to be called on user input
    inputs=gr.components.Textbox(lines=2),                    # Text input
    outputs=gr.components.HTML(),                             # Output type set to HTML
    title="Chat with your Repo",                              # Title of the web page
    description="Enter your question and get a response.",    # Description
    allow_flagging="never"                                    # Disable flagging
)

# Launch the Gradio app
iface.launch()
