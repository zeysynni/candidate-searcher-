## Candidate Search Agent with Tools and RAG

This project is a conversational AI system built using LangChain, OpenAI's GPT, and Gradio. It combines Retrieval-Augmented Generation (RAG) with custom tools to help answer job-related queries and interact with LinkedIn job pages.

It is able to answer questions about candidates that are stored in its database. 

For that, all the CVs as PDFs should be stored in the folder cv_base. They will be read as the knowledge base for RAG

One possible usage of the Agent will be the user provides the URL links to a job on LinkedIN. Then ask questions like: "Give me all the suitable candidates for this position.". 

- When parse the LinkedIN page, notice that LinkedIN might change tag name in html file regullarly to prevent scraping. In this case, check the corresponding tag name manuelly.

