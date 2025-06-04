# 📈 EquiVerse AI : Your AI-powered edge in equity analysis

**EquiVerse AI** empowers investors, analysts, and enthusiasts to explore, understand, and question equity market news through advanced AI-driven natural language capabilities. Just enter URLs of news articles, and get precise answers with reliable source references.

---

## 🎯 Features

- 🔗 **Multi-URL News Ingestion**: Input up to 3 article URLs from reliable sources.
- 🧠 **AI-Powered Q&A**: Ask questions based on the content of the articles and receive summarized, accurate answers.
- 🗂️ **Source-Linked Answers**: All insights are backed by clickable source URLs for verification.
- 📄 **PDF Export**: Download any Q&A interaction in a well-formatted PDF document.
- 🔄 **Session History**: View all your previous questions and answers during a session.
- 🧩 **Progress Indicators**: Real-time updates during article loading, chunking, and vector creation.
- 💬 **Interactive Chat UI**: Clean, responsive interface with Lottie animations and Streamlit Cards.

---

## 🏗️ How It's Built

- **Frontend**: [Streamlit](https://streamlit.io/) for rapid, interactive UI development.
- **Document Loading**: `WebBaseLoader` from LangChain to scrape and structure article content.
- **Text Processing**: `RecursiveCharacterTextSplitter` breaks articles into meaningful chunks.
- **Embeddings**: Google Generative AI Embeddings (`embedding-001`) to convert text to vector space.
- **Vector Store**: FAISS (Facebook AI Similarity Search) for fast similarity-based document retrieval.
- **Q&A Chain**: `RetrievalQAWithSourcesChain` from LangChain that connects the LLM to the vector store.
- **PDF Output**: `fpdf2` for generating downloadable, readable Q&A documents.
- **Styling & UX**: Streamlit extras like `streamlit-lottie`, `streamlit-card` for polished UI and feedback animations.

---

## 🤖 Used LLM

- **Model**: `gemini-1.5-flash-latest`  
- **Provider**: [Google Generative AI](https://ai.google/discover/gemini/)
- **Capabilities**: Efficient, accurate long-context question answering and summarization.
- **Integration**: `ChatGoogleGenerativeAI` from LangChain's Google Generative AI wrapper.

---

# 🚀 Getting Started

## Installation

### 1. Clone the repository:

```bash
git clone https://github.com/AkshadaMane26/EquiVerseAI-.git
```

### 2. Create and activate a virtual environment:

```bash
python -m venv myvenv
source venv/bin/activate  # On Windows use `myvenv\Scripts\activate`
```

### 3. Install the required packages:

```bash
   pip install -r requirements.txt
```

## Setup

1. First, you need to set up the proper API keys and environment variables. To set it up, create the GOOGLE_API_KEY in the Google Cloud credential console (https://console.cloud.google.com/apis/credentials) then create a .env file and paste that API Key in it as shown below.
```bash
   GOOGLE_API_KEY=YOUR_API_KEY
```

## Running the Application

```bash
streamlit run main.py
```

## Usage

1.  Open the Streamlit application in your browser.
2.  For URL :
    - Provide the URLs for the news articles.
    - Click on "Process URLs" to fetch and analyze the articles.
3.  Enter a query in the text input box and click "Submit" to get answers based on the processed data.

## Example:

1.  Enter 3 as number of urls
2.  Provide following urls:
    1. https://www.moneycontrol.com/news/business/tata-motors-to-use-new-1-billion-plant-to-make-jaguar-land-rover-cars-report-12666941.html
    2. https://www.moneycontrol.com/news/business/stocks/tata-motors-stock-jumps-x-after-robust-jlr-sales-brokerages-bullish-12603201.html
    3. https://www.moneycontrol.com/news/business/stocks/buy-tata-motors-target-of-rs-1188-sharekhan-12411611.html
3.  Click "Process URLs" to start processing.
4.  Enter a query like `what is the target price of tata motors ?` and click `Submit` to get the answer.

## Author

👤 **Akshada Mane**

- Github: [@AkshadaMane](https://github.com)
- LinkedIn: [@AkshadaMane](https://linkedin.com)

## Show your support
Give a ⭐️ if this project helped you!
