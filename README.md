# Car Manual RAG Assistant 🚗🤖

A Retrieval-Augmented Generation (RAG) chatbot that answers car warning and maintenance questions using an MG ZS car manual.

## Features

- RAG pipeline using LangChain
- OpenAI GPT-4o-mini
- Chroma Vector Database
- HTML manual ingestion
- Context-aware responses

## Setup

### 1. Clone repo

```bash
git clone https://github.com/YOUR_USERNAME/car-manual-rag.git
cd car-manual-rag
```

### 2. Create virtual environment

```bash
python -m venv venv
```

Activate:

#### Windows

```bash
venv\Scripts\activate
```

#### Linux/Mac

```bash
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Set OpenAI API key

Create `.env`

```env
OPENAI_API_KEY=your_api_key_here
```

### 5. Run project

```bash
python app.py
```

## Example

```text
Question:
The Gasoline Particular Filter Full warning has appeared.

Answer:
The gasoline particulate filter is full and requires regeneration...
```