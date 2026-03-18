from langchain_community.document_loaders import PyPDFLoader
from langchain_classic.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFaceEndpoint
from langchain_groq import ChatGroq
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda,RunnableSequence
from langchain_core.output_parsers import StrOutputParser
from flask import Flask, render_template, request, redirect, url_for, jsonify
import os
from dotenv import load_dotenv

app = Flask(__name__)


load_dotenv()
FAISS_FOLDER = "faiss_indexes"
os.makedirs(FAISS_FOLDER, exist_ok=True)
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

embeding = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-mpnet-base-v2")
parser = StrOutputParser()
spliter = RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200)
chat_history = []

vector_store = None
retriver = None

@app.route("/")
def home():
    global chat_history 
    chat_history = []
    return render_template("index.html")


@app.route("/process_pdf", methods=["POST"])
def process_pdf():

    file = request.files["pdf_file"]
    if not file.filename:
        return redirect(url_for("home"))
    file_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(file_path)

    pdf_name = os.path.splitext(file.filename)[0]
    index_path = os.path.join(FAISS_FOLDER, pdf_name)

    global vector_store, retriver

    if os.path.exists(index_path):
        vector_store = FAISS.load_local(index_path, embeding, allow_dangerous_deserialization=True)

    else:
        # Load PDF using PyPDFLoader
        loader = PyPDFLoader(file_path)
        docs = loader.load()
        chunks = spliter.split_documents(docs)
        vector_store = FAISS.from_documents(chunks, embeding)
        vector_store.save_local(index_path)
    retriver = vector_store.as_retriever(search_type='mmr', search_kwargs={'k':8})
    return redirect(url_for("chat_page"))


def format_docs(docs):
    context = ""
    for doc in docs:
        # Extract exactly the fields you requested safely
        title = doc.metadata.get("title", "N/A")
        author = doc.metadata.get("author", "N/A")
        producer = doc.metadata.get("producer", "N/A")
        creator = doc.metadata.get("creator", "N/A")
        creationdate = doc.metadata.get("creationdate", "N/A")
        total_pages = doc.metadata.get("total_pages", "N/A")
        
        # Format them as a clean list for the LLM to read
        context += "--- Document Metadata ---\n"
        context += f"Title: {title}\n"
        context += f"Author: {author}\n"
        context += f"Producer: {producer}\n"
        context += f"Creator: {creator}\n"
        context += f"Creation Date: {creationdate}\n"
        context += f"Total Pages: {total_pages}\n"
        
        # Add the actual text content below the metadata
        context += "\n--- Content ---\n"
        context += doc.page_content + "\n\n"
        
    return context



llm = ChatGroq(model="meta-llama/llama-4-scout-17b-16e-instruct",temperature=0.81)
# Create Prompt 
from langchain_core.prompts import PromptTemplate

template = """You are a helpful, intelligent, and friendly AI assistant. Your goal is to help the user understand their uploaded document while maintaining a natural conversation.

Follow these guidelines:
1. Document Questions: When the user asks about the document, base your answer STRICTLY on the "Retrieved Context" provided below. Do not invent information or use outside knowledge.
2. Missing Information: If the "Retrieved Context" does not contain the answer to the user's question, politely state that you cannot find that information in the document.
3. Casual Conversation: If the user is greeting you or making general small talk, respond warmly and naturally. You can use the "Chat History" to understand the flow of the conversation.
4. Stop Generating: Provide exactly ONE answer to the current User Question. Do NOT generate hypothetical follow-up questions or simulate a back-and-forth conversation. Stop writing immediately after you have answered the prompt.
Chat History:
{chat_history}

Retrieved Context:
{retrieved_context}

User Question: 
{user_question}

Helpful Answer:"""

prompt = PromptTemplate(
    template=template,
    input_variables=["chat_history", "retrieved_context", "user_question"]
)

def format_history():
    history_text = ""
    for chat in chat_history:
        history_text += f"User: {chat['question']}\nAssistant: {chat['answer']}\n"
    return history_text

def que_ans(question):
    global retriver
    if retriver is None:
        return render_template("index.html", error="Please upload a PDF first.")

    parallel_chain = RunnableParallel({
    'retrieved_context': retriver | RunnableLambda(format_docs),
    'user_question' : RunnablePassthrough(),
    "chat_history": RunnableLambda(lambda _: format_history())
    })
    main_chain = parallel_chain | prompt | llm | parser

    return main_chain.invoke(question)

@app.route("/chat")
def chat_page():
    return render_template("chat.html", history=chat_history)

@app.route("/chat", methods=["POST"])
def chat():
    question = request.form["question"]

    # call RAG system
    answer = que_ans(question)

    # Save History
    chat_history.append({
    "question": question,
    "answer": answer
    })

    return jsonify({"answer": answer})


if __name__ == "__main__":
    app.run(debug=True)