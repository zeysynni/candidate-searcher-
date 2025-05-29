# AI Candidate Assistant with LinkedIn Scraping, RAG & Gradio 

This project is a conversational AI Agent built using **Retrieval-Augmented Generation (RAG)**, a **custom web scraping tool** and Gradio to help users to answer candidate-related queries and interact with LinkedIn job pages.

This project is an intelligent assistant that uses **Retrieval-Augmented Generation (RAG)** and a **custom web scraping tool** to help users interact with job data from LinkedIn. It leverages `LangChain`, `Gradio`, and `OpenAI GPT-4` to provide a conversational interface for exploring job offers and candidate information.

It is able to answer questions about candidates that are stored in its database. 

For that, all the CVs as PDFs should be stored in the folder cv_base. They will be read as the knowledge base for RAG

One possible usage of the Agent will be the user provides the URL links to a job on LinkedIN. Then ask questions like: "Give me all the suitable candidates for this position.". 

- When parse the LinkedIN page, notice that LinkedIN might change tag name in html file regullarly to prevent scraping. In this case, check the corresponding tag name manuelly.

## Features

- **LinkedIn Scraping**
  Scrapes job description from public LinkedIn job URL.

- **RAG**
  Uses candidates CVs to build a knowledge database for user queries.

- **LangChain Agent with Tool Intergration**
  Combines multiple tools in a single intelligent agent using LangChain and OpenAI Functions Agent and maintains a memory of past conversations using LangChain memory.

- **Gradio UI**
  User-friendly web interface for interactive and handy dialogues with the assistant.

## Tech Stack

- `Python 3.10+`
- `LangChain`
- `OpenAI GPT-4o`
- `Gradio`
- `BeautifuldSoup`, `Selenium`
- `ChromaDB`

## How to Run

1. **Clone the repo**

   ```bash
   git clone https://github.com/your-username/job-assistant-rag.git
   cd job-assistant-rag

3. **Install dependencies**

   ```bash
   pip install -r requirements.txt

5. **Set environment variables**

   Create a .env file or export these in your shell:
   ```bash
   OPENAI_API_KEY=your-openai-api-key
   LINKEDIN_USERNAME=your-linkedin-email
   LINKEDIN_PASSWORD=your-linkedin-password
   
7. **Run the app**
   ```bash
   python main.py

   This will start the Gradio UI in your browser. 
   