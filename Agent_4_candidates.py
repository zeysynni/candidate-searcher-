import gradio as gr
import os
from dotenv import load_dotenv
import glob
from bs4 import BeautifulSoup

from openai import OpenAI
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.schema import SystemMessage

from langchain.chat_models import ChatOpenAI
from langchain.agents import initialize_agent, AgentType
from langchain.prompts import MessagesPlaceholder
from langchain.schema.runnable import RunnableMap
from langchain.tools import tool
from langchain.agents.openai_functions_agent.base import OpenAIFunctionsAgent

# Use selenium to scrap the LinkedIN page.
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from bs4 import BeautifulSoup
import json, time

import warnings
warnings.filterwarnings("ignore")


# Set the Model and db name
MODEL = "gpt-4o-mini"
db_name = "cv_db"

# Load all the necesssary keys
load_dotenv(override=True)
api_key = os.getenv('OPENAI_API_KEY')
LinkedIN_Username = os.getenv('LinkedIN_Username')
LinkedIN_Password = os.getenv('LinkedIN_Password')

openai = OpenAI()

# Set root for CV folder
cvs = glob.glob("cv_base/*")

# Set URL to job page on LinkedIN (for testing)
#url = ""

# Create a system prompt for deciding wheather there are appropritate candidate for a given job. 
system_prompt = "You are an expert in searching for ideal candidate for a job advertisement. "
system_prompt += "If you give suggestions for suitable candidates, always make sure the following: \n "
system_prompt += "1. Give the full name of appropriate candidates. "
system_prompt += "2. Also give brief reasoning for you decision. " 
system_prompt += "\n Always answer in English. "

# User prompt for retrieving name from a CV.
def user_prompt_for_name_retrievement(cv):
    """
    Input:
        cv: str of pdf content
    """
    user_prompt = "You are looking at a CV. \n"
    user_prompt += "\n The contents of this CV is as follows; \
    Please give the name of the applicant. \n\n"
    user_prompt += cv
    return user_prompt

# Retrieve name from a CV.
def retrieve_name(cv_content: str) -> str:
    """
    Given the content of an applicant CV, return the name of the applicant.
    Args:
        cv_content (str): A str of CV.
    Returns:
        str: The name of the applicant.
    """
    system_prompt_name_retrievement = "You are an Assistant analyzes the contents of a CV \
    and provides the name of the CV holder. Ignoring text that might be part of a symbol. \
    Output only the name of the CV holder, nothing else."

    user_prompt = user_prompt_for_name_retrievement(cv_content)

    messages = [
        
        {"role": "system", "content": system_prompt_name_retrievement},
        {"role": "user", "content": user_prompt}
    ]

    response = openai.chat.completions.create(model="gpt-4o-mini", messages=messages)

    return response.choices[0].message.content

@tool
def get_jd(url: str) -> str:
    """
    Input:
        url: str of job content.
    Output:
        str of scrapped job description.
    """
    class_name = "application-outlet"
    class_name = "jobs-description__content"
    
    driver = webdriver.Chrome()
    driver.get("https://www.linkedin.com/login")
    
    driver.find_element(By.ID, "username").send_keys(LinkedIN_Username)
    driver.find_element(By.ID, "password").send_keys(LinkedIN_Password)
    driver.find_element(By.XPATH, "//button[@type='submit']").click()
    driver.get(url)
    
    try:
        mehr_anzeigen = WebDriverWait(driver, 5).until(
            EC.element_to_be_clickable((By.XPATH, "//button[.//span[text()='Mehr anzeigen']]"))
        )
    
        #driver.execute_script("arguments[0].scrollIntoView(true);", mehr_anzeigen)
        driver.execute_script("arguments[0].click();", mehr_anzeigen)
    
    
        WebDriverWait(driver, 5).until(
            EC.element_to_be_clickable((By.XPATH, "//button[.//span[text()='Mehr anzeigen']]"))
        )
        
        mehr_anzeigen.click()
        print("Clicked 'Mehr anzeigen'")
    except Exception:
        print("Mehr anzeigen not found")

    html = driver.page_source
    soup = BeautifulSoup(html, "html.parser")

    jd = soup.body.get_text(separator="/n", strip=True).split("}")[-1]
    parts = jd.split("Details zum Jobangebot")
    jd = parts[-1]
    jd = jd.split("Mehr anzeigen")
    return jd[0]

@tool
def rag_tool_fn(query: str) -> str:
    """Use this to retrieve candidates information from internal documents."""
    return conversation_chain.invoke({"question": query})["answer"]

@tool
def rag_tool_fn(query: str) -> str:
    """Use this to retrieve candidates information from internal documents."""
    return conversation_chain.invoke({"question": query})["answer"]

# Read in the content of CVs in the folder.
cv_contents = []  # contents from all the cvs in the folder
candidate_names = []
for cv in cvs:
    loader = PyPDFLoader(cv)
    pages = loader.load() # 2 objects for 2 pages
    applicant_name = retrieve_name(pages[0].page_content)
    for page in pages:
        page.metadata["applicant_name"] = applicant_name
        cv_contents.append(page)
    candidate_names.append(applicant_name)

# Divide texts into chunks.
text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = text_splitter.split_documents(cv_contents)

# Convert chunks into embeddings.
embeddings = OpenAIEmbeddings()
if os.path.exists(db_name):
    Chroma(persist_directory=db_name, embedding_function=embeddings).delete_collection()

vectorstore = Chroma.from_documents(documents=chunks, embedding=embeddings, persist_directory=db_name)

# Initialization of pipeline
llm = ChatOpenAI(temperature=0, model_name=MODEL)

memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
memory.chat_memory.messages.insert(0, SystemMessage(
    content=system_prompt
))

tools = [rag_tool_fn, get_jd, candidate_list_fn]

retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

conversation_chain = ConversationalRetrievalChain.from_llm(llm=llm, retriever=retriever, memory=memory)

agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.OPENAI_FUNCTIONS,
    verbose=True,
    memory=memory
)

# Gradio UI
def chat(question, history):
    response = agent.run(question)
    return response

view = gr.ChatInterface(chat, type="messages").launch(inbrowser=True)