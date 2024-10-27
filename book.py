import os
from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
import openpyxl
from PyPDF2 import PdfReader
import docx
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain.vectorstores import FAISS  # Corrected module import
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
import json

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Replace with your frontend origin
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class Question(BaseModel):
    question: str

# Hard code the API key
GOOGLE_API_KEY = "AIzaSyAtkAbHM189Q566Ezh4MyDpdGmEO1Gw_pc"
genai.api_key = GOOGLE_API_KEY

def get_pdf_text(path):
    text = ""
    pr = PdfReader(path)
    for pg in pr.pages:
        text += pg.extract_text()
    return text

def get_docx_text(path):
    text = ""
    doc = docx.Document(path)
    for para in doc.paragraphs:
        text += para.text
    return text

def get_excel_text(path):
    text = ""
    wb = openpyxl.load_workbook(path)
    for sheet in wb:
        for row in sheet.iter_rows(values_only=True):
            text += '\n'.join(map(str, row))
    return text

def get_text_chunks(text):
    ts = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = ts.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    emb = GoogleGenerativeAIEmbeddings(model='models/embedding-001', google_api_key=GOOGLE_API_KEY)
    vs = FAISS.from_texts(text_chunks, embedding=emb)
    vs.save_local('vectors_storage')

def get_conversational_chain():
    prmt_temp = """
    Consider synonyms and related terms when answering. For example, treat 'submission date', 'due date', and 'deadline' as potentially referring to the same concept, unless the context clearly distinguishes them.
Answer the question as detailed and thoroughly as possible using the provided context. Never answer with just the answer; always frame sentences. If the answer is not found in the context, say "Unable to find the answer", but do not provide incorrect information.
Be comprehensive in your response and provide as much detail as possible.
context:\n{context}\n
Question:\n{question}\n
Answer:
"""
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.7, google_api_key=GOOGLE_API_KEY)
    prompt = PromptTemplate(template=prmt_temp, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

def user_input(ques):
    emb = GoogleGenerativeAIEmbeddings(model='models/embedding-001', google_api_key=GOOGLE_API_KEY)
    new_db = FAISS.load_local('vectors_storage', emb, allow_dangerous_deserialization=True)
    dox = new_db.similarity_search(ques)
    chain = get_conversational_chain()
    res = chain.invoke(
        {"input_documents": dox, "question": ques},
    )
    return res["output_text"]

# Modified upload to accept JSON and handle Blob data
@app.post("/upload/")
async def upload_file(request: Request):
    try:
        body = await request.json()  # Receive the incoming JSON
        file_data = body.get('files', [])
        
        if not file_data:
            return JSONResponse(status_code=400, content={"error": "No text data uploaded."})

        # Assuming the 'files' contains the raw text (not as a blob)
        text_blob = file_data[0]  # Directly treat this as plain text

        # Process the text blob as plain text
        chunks = get_text_chunks(text_blob)  # Use the get_text_chunks method to split the text
        get_vector_store(chunks)  # Store the chunks in the vector store

        return {"status": "success", "message": "Text uploaded and processed successfully."}

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.post("/ask/")
async def ask_question(question: Question):
    answer = user_input(question.question)
    return JSONResponse(content={"question": question.question, "answer": answer})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
