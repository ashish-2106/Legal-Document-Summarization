import streamlit as st
from helper_functions import (
    extract_text_from_pdf,
    split_text_into_chunks,
    create_pinecone_index,
    retrieve_chunks,
)
from groq import summarize_text  # Importing the Groq summarizer

# App title
st.title("Smart Legal Assistant")
st.sidebar.title("Document Upload & Interaction")

# Sidebar file upload
uploaded_file = st.sidebar.file_uploader("Upload your legal PDF document:", type=["pdf"])

# Initialize variables
index_name = "smart-legal-index"

if uploaded_file:
    # Extract text from the uploaded PDF
    document_text = extract_text_from_pdf(uploaded_file)
    st.write("### Extracted Document Preview:")
    st.write(document_text[:1500] + "..." if len(document_text) > 1500 else document_text)  # Show up to 1500 characters

    # Summarize document using Groq
    if st.button("Generate Summary"):
        with st.spinner("Summarizing the document... Please wait."):
            summary = summarize_text(document_text)  # Summarizing using Groq API
        if summary:
            st.write("### Summary:")
            st.write(summary)
        else:
            st.error("An error occurred while summarizing the document.")

    # Ask questions using RAG
    st.write("### Interactive Q&A:")
    if "vector_store" not in st.session_state:
        chunks = split_text_into_chunks(document_text)
        st.session_state.vector_store = create_pinecone_index(index_name, chunks)

    user_query = st.text_input("Have a question about the document? Ask here:", placeholder="e.g., What is the key point in clause 5?")
    if user_query:
        with st.spinner("Finding the most relevant information..."):
            relevant_text = retrieve_chunks(user_query, st.session_state.vector_store)
            response = summarize_text(relevant_text + f"\n\nAnswer the question: {user_query}")  # Summarizing relevant text for the question
        if response:
            st.write("### Response:")
            st.write(response)
        else:
            st.error("An error occurred while generating the response.")

# Add a reset button
if st.sidebar.button("Reset Session"):
    st.session_state.clear()
    st.success("Session reset successfully!")
