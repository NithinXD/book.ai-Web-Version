import streamlit as st
import requests

# FastAPI endpoints
UPLOAD_URL = "http://localhost:8000/upload/"
ASK_URL = "http://localhost:8000/ask/"

st.title("book.ai microservice:1 doc parser")

# File Upload Section
st.header("Upload Documents")
uploaded_files = st.file_uploader("Choose files", type=["pdf", "docx", "xlsx"], accept_multiple_files=True)

if st.button("Upload"):
    if uploaded_files:
        files = [("files", (file.name, file, file.type)) for file in uploaded_files]
        response = requests.post(UPLOAD_URL, files=files)
        if response.status_code == 200:
            st.success("Files uploaded successfully!")
        else:
            st.error("Failed to upload files.")
    else:
        st.error("Please upload at least one file.")

# Ask Question Section
st.header("Ask a Question")
question = st.text_input("Enter your question:")

if st.button("Ask"):
    if question:
        response = requests.post(ASK_URL, json={"question": question})
        if response.status_code == 200:
            answer = response.json().get("answer")
            st.write(f"**Answer:** {answer}")
        else:
            st.error("Failed to retrieve the answer.")
    else:
        st.error("Please enter a question.")

# Run the app with: streamlit run app.py
