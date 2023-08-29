
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores.faiss import FAISS
from langchain import OpenAI, Cohere
from langchain.chains.qa_with_sources import load_qa_with_sources_chain
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.docstore.document import Document
from langchain.chat_models import ChatOpenAI
from langchain.vectorstores import FAISS, VectorStore
import docx2txt
from typing import List, Dict, Any
import re
import numpy as np
from io import StringIO
from io import BytesIO
from pypdf import PdfReader
from openai.error import AuthenticationError

openai_key = "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"


def parse_docx(file: BytesIO) -> str:
    text = docx2txt.process(file)
    # Remove multiple newlines
    text = re.sub(r"\n\s*\n", "\n\n", text)
    return text

def parse_pdf(file: BytesIO) -> List[str]:
    pdf = PdfReader(file)
    output = []
    for page in pdf.pages:
        text = page.extract_text()
        # Merge hyphenated words
        text = re.sub(r"(\w+)-\n(\w+)", r"\1\2", text)
        # Fix newlines in the middle of sentences
        text = re.sub(r"(?<!\n\s)\n(?!\s\n)", " ", text.strip())
        # Remove multiple newlines
        text = re.sub(r"\n\s*\n", "\n\n", text)

        output.append(text)

    return output

def text_to_docs(text: str) -> List[Document]:
    """Converts a string or list of strings to a list of Documents
    with metadata."""
    if isinstance(text, str):
        # Take a single string as one page
        text = [text]
    page_docs = [Document(page_content=page) for page in text]

    # Add page numbers as metadata
    for i, doc in enumerate(page_docs):
        doc.metadata["page"] = i + 1

    # Split pages into chunks
    doc_chunks = []

    for doc in page_docs:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""],
            chunk_overlap=0,
        )
        chunks = text_splitter.split_text(doc.page_content)
        for i, chunk in enumerate(chunks):
            doc = Document(
                page_content=chunk, metadata={"page": doc.metadata["page"], "chunk": i}
            )
            # Add sources a metadata
            doc.metadata["source"] = f"{doc.metadata['page']}-{doc.metadata['chunk']}"
            doc_chunks.append(doc)
    return doc_chunks


def embed_docs(docs: List[Document]) -> VectorStore:
        embeddings = OpenAIEmbeddings(openai_api_key= openai_key, disallowed_special=())  # type: ignore
        index = FAISS.from_documents(docs, embeddings)

        return index


def search_docs(index: VectorStore, query: str) -> List[Document]:
    """Searches a FAISS index for similar chunks to the query
    and returns a list of Documents."""
    # Search for similar chunks
    docs = index.similarity_search(query, k=15)
    return docs

def get_answer(docs: List[Document], query: str, check) -> Dict[str, Any]:
    """Gets an answer to a question from a list of Documents."""

    # Get the answer

    chain = load_qa_with_sources_chain(ChatOpenAI(temperature=0, openai_api_key=openai_key), chain_type="map_reduce") 
    if check == 0:
        chain = load_qa_with_sources_chain(OpenAI(temperature=0, openai_api_key=openai_key), chain_type="map_reduce") 
    answer = chain(
        {"input_documents": docs, "question": query}, return_only_outputs=True
    )
    return answer



def get_sources(answer: Dict[str, Any], docs: List[Document]) -> List[Document]:
    """Gets the source documents for an answer."""

    # Get sources for the answer
    source_keys = [s for s in answer["output_text"].split("SOURCES: ")[-1].split(", ")]

    source_docs = []
    for doc in docs:
        if doc.metadata["source"] in source_keys:
            source_docs.append(doc)

    return source_docs


def wrap_text_in_html(text: str | List[str]) -> str:
    """Wraps each text block separated by newlines in <p> tags"""
    if isinstance(text, list):
        # Add horizontal rules between pages
        text = "\n<hr/>\n".join(text)
    return "".join([f"<p>{line}</p>" for line in text.split("\n")])