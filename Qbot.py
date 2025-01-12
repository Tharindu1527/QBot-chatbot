from langchain_community.llms import HuggingFaceHub
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain.chains import RetrievalQA
import gradio as gr

#ignoring Unnecessary warnings
def warn(*arg, **kwargs):
    pass
import warnings
warnings.warn = warn
warnings.filterwarnings("ignore")

#Initialize LLM

def get_llm():
    model_id = "EleutherAI/gpt-neo-1.3B"

    huggingface_api_key = "" 

    #initialize the pipeline
    Qbot_llm = HuggingFaceHub(
        repo_id=model_id,
        model_kwargs={"max_new_tokens": 128, "temperature": 0.5},
        huggingfacehub_api_token=huggingface_api_key
    )
    return Qbot_llm

#Document Loader
def document_loader(file):
    loader = PyPDFLoader(file.name)
    loaded_document = loader.load()
    return loaded_document

#Define Text Splitter
def text_splitter(data):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 1000,
        chunk_overlap = 50,
        length_function = len,
    )
    chunks = text_splitter.split_documents(data)
    return chunks

#Define Vector store
def vector_database(chunks):
    embedding_model = HuggingFaceEmbeddings()
    vectordb = Chroma.from_documents(chunks, embedding_model)
    return vectordb

#Define Embedding Model
def huggingface_embeddings():
    model_name = "sentence-transformers/all-mpnet-base-v2"
    huggingface_embedding = HuggingFaceEmbeddings(model_name=model_name)
    
    return huggingface_embedding

#Define Retriever
def retriever(file):
    splits = document_loader(file)
    chunks = text_splitter(splits)
    vectordb = vector_database(chunks)
    retriever = vectordb.as_retriever()
    return retriever

#Define a Question Answering Chain
#QA chain
def retriever_qa(file, query):
    llm = get_llm()
    retriever_obj = retriever(file)
    qa = RetrievalQA.from_chain_type(llm = llm,
                                     chain_type = "stuff",
                                     retriever = retriever_obj,
                                     return_source_documents = False
                                     )
    response = qa.invoke(query)
    return response["result"]

#Gradio Interface
rag_application = gr.Interface(
    fn = retriever_qa,
    allow_flagging = "never",
    inputs=[
        gr.File(label="Upload PDF File", file_count="single", file_types=[".pdf"],type="filepath"), # Drag and drop file upload
        gr.Textbox(label = "Input Query", lines=2, placeholder="Type your question here..."),
        gr.Slider(label="Chunk Size", minimum=100, maximum=2000, value=1000, step=50),
        gr.Slider(label="Chunk Overlap", minimum=0, maximum=500, value=50, step=10),
        gr.Slider(label="Max Tokens", minimum=32, maximum=512, value=128, step=32),
        gr.Slider(label="Temperature", minimum=0.0, maximum=1.0, value=0.5, step=0.1)
    ],
    outputs=gr.Textbox(label="Output"),
    title = "QBot",
    description="Simply upload your PDF, ask any question, and let the chatbot provide answers straight from your document"
)

#Launch app
rag_application.launch(share=True)
