import re
from langchain.docstore.document import Document
from langchain_community.vectorstores import Chroma
import shutil
import os
# os.environ["OPENAI_API_KEY"] = os.environ.get("OPENAI_API_KEY", "")


def langchain_doc(doc_dict):
    """
    Processes a dictionary of document sections, organizing content by headers, page numbers, and metadata.

    Args:
        doc_dict (list): List of dictionaries, each representing a document section with titles in hierarchy levels.

    Returns:
        tuple: A list of langchain Document objects with organized content and metadata, and a list of unique headers processed.
    """
    docs = []
    process_header = []
    page_number = 1
    id = 0
    for doc in doc_dict:
        current_doc = ""
        current_index = ""
        new_pg = 0
        for key in [
            "parent_title",
            "child_title",
            "grand_child_title",
            "great_grand_child_title",
        ]:
            if doc[key]:
                title = doc[key].split("\n")[0]
                current_index = current_index + "\n\n" + title
                if title in process_header:
                    current_doc = current_doc + "\n" + title
                else:
                    process_header.append(title)
                    current_doc = current_doc + "\n" + doc[key]

            start_page = page_number
            count = current_doc.count("\n-----\n")
            if re.search(r"\n-----\n\s*[^a-zA-Z]*$", current_doc):
                count -= 1
                new_pg = 1
            end_page = start_page + count

        page_number = end_page + new_pg

        id = id + 1
        doc_ = Document(
            page_content=current_doc,
            metadata={
                "start_page": start_page,
                "end_page": end_page,
                "index": current_index,
                "id": id,
            },
        )
        docs.append(doc_)

    return docs, process_header


def create_vectorstore(data, embedding_model):
    """
    Creates a vector store from document data using a specified embedding model,
    useful for efficient similarity search and retrieval.

    Args:
        data (list): List of dictionaries representing document sections.
        embedding_model: Model used to generate embeddings for each document.

    Returns:
        tuple: The vector store object, list of Document objects, and a list of processed headers.
    """

    docs, process_header = langchain_doc(data)
    # folder_name = "chroma_langchain_db"
    # folder_path = os.path.join(os.getcwd(), folder_name)
    # if os.path.exists(folder_path) and os.path.isdir(folder_path):
    #     # Delete the folder and all its contents
    #     shutil.rmtree(folder_path)

    vectorstore = Chroma.from_documents(
        documents=docs,
        embedding=embedding_model,  # persist_directory=folder_name
    )
    # vectorstore_retreiver = vectorstore.as_retriever()

    return vectorstore, docs, process_header
