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

def get_llm(model_id, temperature, max_new_tokens):

    huggingface_api_key = "" 

    #initialize the pipeline
    Qbot_llm = HuggingFaceHub(
        repo_id=model_id,
        model_kwargs={"max_new_tokens": max_new_tokens, "temperature": temperature},
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
def vector_database(chunks, embedding_model_name):
    embedding_model = HuggingFaceEmbeddings(model_name = embedding_model_name)
    vectordb = Chroma.from_documents(chunks, embedding_model)
    return vectordb

#Define Embedding Model
def huggingface_embeddings(model_name):
    huggingface_embedding = HuggingFaceEmbeddings(model_name=model_name)
    
    return huggingface_embedding

#Define Retriever
def retriever(file, embedding_model_name):
    splits = document_loader(file)
    chunks = text_splitter(splits)
    vectordb = vector_database(chunks, embedding_model_name)
    retriever = vectordb.as_retriever()
    return retriever

#Define a Question Answering Chain
#QA chain
def retriever_qa(file, query, llm_model, temperature, max_new_tokens, embedding_model):
    llm = get_llm(llm_model, temperature, max_new_tokens)
    retriever_obj = retriever(file, embedding_model)
    qa = RetrievalQA.from_chain_type(llm = llm,
                                     chain_type = "stuff",
                                     retriever = retriever_obj,
                                     return_source_documents = False
                                     )
    response = qa.invoke(query)
    return response["result"]

llm_models = [
    "EleutherAI/gpt-neo-2.7B",
    "google/flan-t5-large",
    "google/flan-t5-xl",
]

embedding_models = [
    "sentence-transformers/all-distilroberta-v1",
    "sentence-transformers/all-mpnet-base-v2",
    "sentence-transformers/all-MiniLM-L6-v2",
]

# CSS for custom styling
custom_css = """
#component-0 {
    max-width: 800px;
    margin: auto;
    padding: 20px;
}
.gradio-container {
    font-family: 'Arial', sans-serif;
}
.gr-button {
    background: linear-gradient(90deg, #4CAF50 0%, #45a049 100%);
    border: none;
    color: white;
}
.gr-button:hover {
    background: linear-gradient(90deg, #45a049 0%, #4CAF50 100%);
    transform: translateY(-2px);
    box-shadow: 0 5px 15px rgba(0,0,0,0.1);
}
.gr-form {
    background-color: #ffffff;
    border-radius: 12px;
    padding: 20px;
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
}
.gr-box {
    border-radius: 8px;
    border: 1px solid #e0e0e0;
}
"""

#Gradio Interface
rag_application = gr.Interface(
    fn = retriever_qa,
    allow_flagging = "never",
    inputs=[
        gr.File(label="Upload PDF File", file_count="single", file_types=[".pdf"],type="filepath", elem_classes="gr-box"), # Drag and drop file upload
        gr.Textbox(label = "Input Query", lines=2, placeholder="Type your question here...", elem_classes="gr-box"),
        gr.Dropdown(choices=llm_models, value="EleutherAI/gpt-neo-2.7B", label="LLM Model", elem_classes="gr-box"),
        gr.Slider(minimum=0.0, maximum=1.0, value=0.5, step=0.1, label="Temperature", elem_classes="gr-box"),
        gr.Slider(minimum=64, maximum=512, value=128, step=32, label="Max Tokens", elem_classes="gr-box"),
        gr.Dropdown(choices=embedding_models, value="sentence-transformers/all-distilroberta-v1", label="Embedding Model", elem_classes="gr-box")
    ],
    outputs=gr.Textbox(label="Output"),
    title = "📚 QBot - Your PDF Assistant",
    description="""
    ### Welcome to QBot - Your Intelligent PDF Analysis Companion! 
    
    Transform any PDF document into an interactive knowledge base. Ask questions naturally and get precise answers powered by advanced language models.
    
    #### Features:
    🔍 Intelligent PDF Processing
    💡 Multiple Language Models
    🎯 Customizable Response Settings
    🔄 Various Embedding Options
    
    #### How to Use:
    1. **Upload PDF**: Drop your document in the file uploader
    2. **Ask Questions**: Type any question about your document
    3. **Customize Settings**:
       - Choose your preferred Language Model
       - Adjust Temperature (0-1) for response creativity
       - Set Max Tokens for response length
       - Select Embedding Model for document processing
    4. **Get Answers**: Receive AI-powered responses from your document
    """,
    article="""
    #### Advanced Tips:
    📊 **Model Selection**:
    - GPT-Neo 2.7B: Best for general-purpose queries
    - FLAN-T5 Large: Efficient for straightforward questions
    - FLAN-T5 XL: Ideal for complex analysis
    
    🎛️ **Parameter Guide**:
    - Temperature: Lower (0.1-0.4) for factual, Higher (0.6-0.9) for creative
    - Max Tokens: 128 for brief answers, 256+ for detailed explanations
    - Embedding Models: Choose based on document complexity and language
    
    💫 Powered by LangChain and Hugging Face
    Made with 🤖 for seamless document interaction
    """,
    theme=gr.themes.Soft(
        primary_hue="green",
        secondary_hue="gray",
        neutral_hue="gray",
        radius_size=gr.themes.sizes.radius_sm,
        font=[gr.themes.GoogleFont("Inter"), "system-ui", "sans-serif"]
    ),
    css=custom_css
)

#Launch app
rag_application.launch(share=True)
