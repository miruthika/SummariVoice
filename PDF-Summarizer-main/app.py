import os
from PyPDF2 import PdfReader
import streamlit as st
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import HuggingFacePipeline
from transformers import pipeline
from gtts import gTTS
import tempfile
import base64
st.set_page_config(
    page_title="SummariVoice",
    page_icon="🔊",  # You can change this emoji or use a custom image icon
    layout="centered"
)
st.markdown("""
    <style>
    .css-1n76uvr, .stFileUploader {
        color: white !important; /* Text color */
    }
    .stFileUploader > label {
        color: white !important; /* Label text */
    }
    .stFileUploader .uploadedFileName {
        color: white !important; /* Uploaded filename */
    }
    </style>
""", unsafe_allow_html=True)
st.markdown("""
    <style>
    body {
        color: white;
    }
    .stApp {
        color: white;
    }
    .css-10trblm, .css-1v0mbdj, .css-qbe2hs, .css-2trqyj {
        color: white !important;
    }
    .css-1cpxqw2, .css-1vencpc {
        background-color: rgba(0, 0, 0, 0.6) !important;
    }
    </style>
""", unsafe_allow_html=True)
st.markdown("""
    <style>
    footer {visibility: hidden;}
    footer:after {
        content:''; 
        visibility: hidden; 
        display: block; 
        position: relative; 
        height: 0; 
    }
    </style>
""", unsafe_allow_html=True)
def set_bg_from_local(image_file):
    with open(image_file, "rb") as img_file:
        encoded = base64.b64encode(img_file.read()).decode()
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/jpg;base64,{encoded}");
            background-size: cover;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

set_bg_from_local("background.jpg") 
def process_text(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    knowledge_base = FAISS.from_texts(chunks, embeddings)
    return knowledge_base
def main():
    st.title("📄 PDF Summarizer & Audio Conversion")
    st.divider()
    pdf = st.file_uploader('Upload your PDF Document', type='pdf')
    if pdf is not None:
        pdf_reader = PdfReader(pdf)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
        knowledge_base = process_text(text)
        query = "Summarize the content of the uploaded PDF file in approximately 3-5 sentences. Focus on capturing the main ideas and key points discussed in the document. Use your own words and ensure clarity and coherence in the summary."
        if query:
            docs = knowledge_base.similarity_search(query)
            combined_text = " ".join([doc.page_content for doc in docs])
            summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")
            max_chunk = 1024 
            summary = summarizer(combined_text[:max_chunk], max_length=130, min_length=30, do_sample=False)
            response = summary[0]['summary_text']
            st.subheader('Summary Results:')
            st.write(response)
            if st.button("🔊 Play Summary Audio"):
                tts = gTTS(text=response, lang='en')
                with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as fp:
                    tts.save(fp.name)
                    audio_file = open(fp.name, 'rb')
                    audio_bytes = audio_file.read()
                    st.audio(audio_bytes, format='audio/mp3')
if __name__ == '__main__':
    main()
