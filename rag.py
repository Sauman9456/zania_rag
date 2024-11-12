from langchain.retrievers import (
    EnsembleRetriever,
    BM25Retriever,
    ContextualCompressionRetriever,
)
from langchain_cohere import CohereRerank
from pydantic import BaseModel, Field


class Alternate_Questions(BaseModel):
    question: str = Field(description="One alternate question")


def get_alternate_questions(question, index, client):
    """
    Generate alternate questions based on the given question and document index.

    Args:
        question (str): Original question.
        titles (str): index of documents.
        client: OpenAI client

    Returns:
        str: Original and one alternate question.
    """

    system_prompt = f"""You are an AI language model assistant. Your task is to generate one alternate versions of the given user question to retrieve relevant documents from a vector database.
By generating multiple perspectives on the user question, your goal is to help the user overcome some of the limitations of the distance-based similarity search and keyword search.
Provide these alternative questions separated by newlines and number.

Note: Refer to the provided document index from the vector database to generate alternate questions. These index contain domain-specific jargon, terminology, acronyms, and synonyms that will assist you in creating contextually accurate questions.
My very few findings:

# Document index
----------------------------------------------------------------------------------------------------------------
{index}
----------------------------------------------------------------------------------------------------------------

Original question: {question}

Alternate Questions:
1.
"""
    completion = client.beta.chat.completions.parse(
        model="gpt-4o-mini-2024-07-18",
        messages=[
            {"role": "system", "content": system_prompt},
        ],
        temperature=0.075,
        response_format=Alternate_Questions,
    )

    return [question] + [completion.choices[0].message.parsed.question]


def get_retriver(vectorstore, docs, embedding_model, top=5):
    """
    Initialize and return retrievers, including a compression retriever.

    Args:
        persist_directory (str): Directory where vector store is stored.
        embedding_model: Embedding model for vectorization.
        top (int, optional): Number of top documents to retrieve. Default is 20.

    Returns:
        tuple: Compression retriever and vector store retriever.
    """
    vectorstore_retreiver = vectorstore.as_retriever(search_kwargs={"k": int(top)})

    keyword_retriever = BM25Retriever.from_documents(docs, k=int(top), top_n=int(top))
    ensemble_retriever = EnsembleRetriever(
        retrievers=[vectorstore_retreiver, keyword_retriever],
        weights=[0.5, 0.5],
        k=int(top),
        top_n=int(top),
    )

    compressor = CohereRerank(model="rerank-english-v3.0", top_n=top)
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor, base_retriever=ensemble_retriever
    )

    return compression_retriever


def get_retrive_doc(query, index, compress_retriever, client, top=5):
    """
    Retrieve documents based on query, returning relevant documents and queries.

    Args:
        query (str): User query.
        index: document indexes.
        compress_retriever: Retriever for Advance rag and re-ranking compression.

    Returns:
        tuple: Retrieved documents and all queries.
    """

    retrive_doc = []
    check_unique_id = []

    all_query = get_alternate_questions(query, index, client)

    for q in all_query:
        compressed_docs = compress_retriever.invoke(q)
        for doc in compressed_docs:
            if doc.metadata["id"] not in check_unique_id:
                check_unique_id.append(doc.metadata["id"])
                retrive_doc.append(doc)
            else:
                for i in range(len(retrive_doc)):
                    if retrive_doc[i].metadata["id"] == doc.metadata["id"]:
                        if (
                            retrive_doc[i].metadata["relevance_score"]
                            < doc.metadata["relevance_score"]
                        ):
                            retrive_doc[i].metadata["relevance_score"] = doc.metadata[
                                "relevance_score"
                            ]

    retrive_doc = sorted(
        retrive_doc, key=lambda doc: doc.metadata["relevance_score"], reverse=True
    )
    retrive_doc = retrive_doc[:top]  # selecting top 5 based on score
    return retrive_doc, all_query


def get_ans(queries, docs, client):
    """
    Generate answer from retrieved documents.

    Args:
        query (list): List of alternate queries.
        docs (list): Retrieved documents.
        client: OpenAi client

    Returns:
        Str: Answer.
    """

    context_str = "\n\n".join(
        f"{context.page_content}" for i, context in enumerate(docs)
    )
    user_query = f"""
  
Documents
------------------------------------------
{context_str}
------------------------------------------


**Query:**
{queries[0]}
"""

    system_prompt = f"""
INSTRUCTIONS:
1. You are an assistant who helps users answer their queries.
2. Always Answer the user's query from the given documents. The user will provide documents.
3. Give answer in step by step format if required.
4. Keep your answer concise with all required and requested details and solely on the information given in the document.
5. Do not create or derive your own answer. If the answer is not directly available in the documents, just reply stating, 'Data Not Available'.
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0.05,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_query},
        ],
    )
    # print(system_prompt)

    return response.choices[0].message.content


def rag_execution(query, index, compress_retriever, client):
    """
    Execute the retrieval and answer generation process.

    Args:
        query (str): User query.
        index: document Index.
        compress_retriever: Compression retriever instance.

    Returns:
        tuple: Answer text.
    """

    retrive_docs, all_query = get_retrive_doc(query, index, compress_retriever, client)
    ans = get_ans(all_query, retrive_docs, client)

    return ans
