import os
import subprocess
from glob import glob
import langchain
from langchain.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Pinecone
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
import pinecone

def convert_office_doc(input_filename, output_filename, target_format):
    try:
        subprocess.call([
            "/Applications/LibreOffice.app/Contents/MacOS/soffice",
            "--headless",
            "--convert-to",
            target_format,
            "--outdir",
            os.path.dirname(output_filename),
            input_filename
        ])
        print(f"Converted {input_filename} to {target_format}.")
    except FileNotFoundError:
        print("LibreOffice is not installed or the 'soffice' executable cannot be found.")
        print("Please make sure LibreOffice is installed on your system and try again.")

def process_documents():
    # Specify the directory containing the documents
    directory = '/Users/matthewalanfarmer/Documents/knowledge'

    # Specify the target format (PDF)
    target_format = "pdf"

    # Create a directory to store the converted PDF documents
    pdf_directory = os.path.join(directory, "pdf")
    os.makedirs(pdf_directory, exist_ok=True)

    # Convert each document in the directory to PDF using LibreOffice
    for input_file in glob(os.path.join(directory, "*.doc")):
        filename = os.path.basename(input_file)
        output_file = os.path.join(pdf_directory, f"{os.path.splitext(filename)[0]}.pdf")
        convert_office_doc(input_file, output_file, target_format)

    # Count the number of converted PDF documents
    num_documents = len(glob(os.path.join(pdf_directory, "*.pdf")))
    print(f"Number of converted PDF documents: {num_documents}")

    # Specify the directory containing the PDF documents
    directory = '/Users/matthewalanfarmer/Documents/knowledge/pdf'

    # Create a DirectoryLoader instance to load the documents
    loader = DirectoryLoader(directory)

    # Load the PDF documents
    documents = loader.load()

    # Define the chunk size and overlap for text splitting
    chunk_size = 1000
    chunk_overlap = 300

    # Create a RecursiveCharacterTextSplitter instance
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    # Split the documents into smaller chunks
    docs = text_splitter.split_documents(documents)

    # Set your Pinecone API key and environment
    pinecone_api_key = "b8a6e51f-fdea-4480-a245-021c2449babd"
    pinecone_environment = "us-west4-gcp-free"

    # Initialize Pinecone
    pinecone.init(api_key=pinecone_api_key, environment=pinecone_environment)

    # Set your index name
    index_name = "pf-chatbot"

    # Set your OpenAI API key
    openai_api_key = "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"

    # Initialize the OpenAIEmbeddings
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)

    # Continue with the rest of the code
    index = Pinecone.from_documents(docs, embeddings, index_name=index_name)

    # Initialize OpenAI LLM
    model_name = "gpt-3.5-turbo"  # Replace with a valid model name
    openai_api_key = "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"  # Replace with your actual OpenAI API key
    llm = OpenAI(model_name=model_name, openai_api_key=openai_api_key, temperature=0.7)  # Adjust the temperature value

    # Load the question-answering chain
    chain = load_qa_chain(llm, chain_type="stuff")

    return index, chain

index, chain = process_documents()
