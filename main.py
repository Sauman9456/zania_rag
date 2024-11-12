from fastapi import FastAPI, UploadFile, File, Body, HTTPException, Form
from typing import List
from pydantic import BaseModel
import os
from openai import OpenAI
from langchain_openai import OpenAIEmbeddings
from process_doc import parse_doc
from vector_db_setup import create_vectorstore
from rag import get_retriver, rag_execution
import json
import tempfile
import ast


os.environ["OPENAI_API_KEY"] = os.environ.get("OPENAI_API_KEY", "")
os.environ["COHERE_API_KEY"] = os.environ.get("COHERE_API_KEY", "")


client = OpenAI()
embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")


app = FastAPI()


@app.post("/get_answer")
async def get_answer(questions: str, pdf_file: UploadFile = File(...)):
    """
    Endpoint to get answer for given questions.

    Args:
        questions: List of questions.
        pdf_file: uploaded pdf file

    Returns:
        answers: JSON blob that pairs each question with its corresponding answer.
    """
    try:
        if isinstance(questions, str):
            questions = ast.literal_eval(questions)
        if not isinstance(questions, list) or not all(
            isinstance(s, str) for s in questions
        ):
            return {
                "error": "The provided list is not valid. It should be a list of strings."
            }
    except:
        return {
            "error": """Invalid format for the list of strings, expected ["Q1", "Q2"]"""
        }

    if pdf_file.content_type != "application/pdf":
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_pdf:
        temp_pdf.write(await pdf_file.read())
        pdf_path = temp_pdf.name

    vectorstore, docs, process_header = create_vectorstore(
        parse_doc(pdf_path), embedding_model
    )

    compress_retriever = get_retriver(vectorstore, docs, embedding_model, 8)
    index = "\n".join(process_header)

    answers = {}

    for question in questions:
        answer = rag_execution(question, index, compress_retriever, client)
        answers[question] = answer

    return json.dumps(answers)


# ["What is the name of the company?", "Who is the CEO of the company?", "What is their vacation policy?", "What is the termination policy?"]
