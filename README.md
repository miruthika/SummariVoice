# 📄 SummariVoice

**SummariVoice** is a smart PDF summarizer that extracts key information and converts the conclusion into speech. Designed with a dark-themed user interface and intuitive design, it helps users quickly understand and listen to PDF documents.

---

## 🔥 Features

- 📤 Upload any PDF file
- 📚 Extracts purpose and conclusion of the document
- 🧠 Uses AI to summarize the key insights
- 🔊 Converts the **conclusion** into speech using `gTTS`
- 🎧 Audio playback of summary directly in browser
- 🎨 Modern UI with black-themed layout and open book image

---

## 🚀 Tech Stack

- **Frontend:** Streamlit
- **Backend:** Python
- **Libraries Used:**
  - `PyPDF2` – for PDF parsing
  - `LangChain`, `sentence-transformers` – for text embedding and summarization
  - `gTTS` – for text-to-speech
  - `FAISS` – for vector storage
  - `Transformers` – for loading language models

---

## 🛠️ Installation

1. **Clone the repo**
   ```bash
   git clone https://github.com/yourusername/SummariVoice.git
   cd SummariVoice
