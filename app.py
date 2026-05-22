from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import UnstructuredHTMLLoader
from langchain_core.runnables import RunnablePassthrough
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
import os

# Load HTML manual
loader = UnstructuredHTMLLoader(
    file_path="data/mg-zs-warning-messages.html"
)

car_docs = loader.load()

# Load models
llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0,
    openai_api_key=os.environ["OPENAI_API_KEY"]
)

embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",
    openai_api_key=os.environ["OPENAI_API_KEY"]
)

# Split documents
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

split_docs = text_splitter.split_documents(car_docs)

# Create vector DB
vectorstore = Chroma.from_documents(
    documents=split_docs,
    embedding=embeddings
)

# Retriever
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# Prompt
prompt = ChatPromptTemplate.from_template("""
You are an MG ZS vehicle assistant.

Use the provided context to answer the user's question.
If the answer is not found in the context, say you do not know.

Context:
{context}

Question:
{question}
""")

# Format docs
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# Build chain
rag_chain = (
    {
        "context": retriever | format_docs,
        "question": RunnablePassthrough()
    }
    | prompt
    | llm
)

# User query loop
print("MG ZS Manual Assistant")
print("Type 'exit' to quit\n")

while True:
    query = input("Ask a question: ")

    if query.lower() == "exit":
        break

    response = rag_chain.invoke(query)

    print("\nAnswer:")
    print(response.content)
    print("\n" + "-" * 50 + "\n")