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
    ### Welcome to QBot! 
    
    Upload your PDF and ask any question about its contents. QBot will analyze the document and provide relevant answers.
    
    #### How to use:
    1. Upload your PDF file using the file uploader
    2. Type your question in the query box
    3. Adjust Temperature (controls creativity) and Max Tokens (controls response length) if needed
    4. Get your answer instantly!
    """,
    article="""
    #### Tips for best results:
    - Keep questions clear and specific
    - Adjust temperature higher for more creative answers
    - Increase max tokens for longer responses
    
    Made with ❤️ using LangChain and Hugging Face
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
