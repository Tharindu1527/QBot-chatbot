# QBot - PDF Query Answering Chatbot


**QBot** is an intelligent chatbot that answers user queries by extracting relevant information from uploaded PDF documents. The application uses **LangChain**, **Hugging Face**, and **vector databases** to deliver accurate and context-aware responses.

---

## **Features**

- **PDF Upload & Processing**: Users can upload a PDF file, and the document is parsed into smaller chunks for efficient querying.
- **Semantic Search**: The document content is embedded using **Hugging Face's** pre-trained models and stored in a vector database (**Chroma**).
- **Retrieval-Based Question Answering**: Answers are generated by retrieving the most relevant sections of the document using a retrieval-based QA pipeline powered by **LangChain**.
- **Gradio Interface**: A simple and intuitive interface powered by **Gradio**, where users can upload PDFs and ask questions interactively.

---

## **Technologies Used**

- **LangChain**: For document processing, embeddings, and building the question-answering pipeline.
- **Hugging Face**: Deployed the GPT model (**EleutherAI/gpt-neo-2.7B, google/flan-t5-large, google/flan-t5-xl**) for natural language understanding and question answering.
- **Chroma**: Used as a vector store for semantic search.
- **Gradio**: For creating an easy-to-use interactive web interface.

---

## **QBot Interface**

<p align="center">
  <img src="https://github.com/Tharindu1527/QBot-chatbot/blob/main/QBot%20Interface%20Images/1.png" alt="Interface 1" width="500">
  <img src="https://github.com/Tharindu1527/QBot-chatbot/blob/main/QBot%20Interface%20Images/2.png" alt="Interface 2" width="500">
  <img src="https://github.com/Tharindu1527/QBot-chatbot/blob/main/QBot%20Interface%20Images/3.png" alt="Interface 3" width="500">
</p>

---

## **How to Run the Project Locally**

### Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/Tharindu1527/QBot-chatbot.git
    cd QBot-chatbot
    ```

2. Create and activate a virtual environment (optional but recommended):
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use venv\Scripts\activate
    ```

3. Install the dependencies:
    ```bash
    pip install -r requirements.txt
    ```

### Running the Application

1. Launch the app by running the following command:
    ```bash
    python QBot.py
    ```

2. Visit the provided URL (usually `http://127.0.0.1:7860/`) to access the web interface.

---

## **Usage**

1. **Upload PDF File**: Click on the "Upload PDF" button and select the PDF document you wish to query.
2. **Ask Questions**: Enter your query in the "Input Query" field, and **QBot** will return relevant information from the PDF.

---
