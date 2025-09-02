# inference_server.py
import logging
from transformers import pipeline, AutoTokenizer, AutoModelForTokenClassification
from mcp.server.fastmcp import FastMCP
from starlette.responses import JSONResponse
from starlette.requests import Request

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

PORT = 8002
mcp_server = FastMCP("InferenceServer", host="0.0.0.0", port=PORT)

logger.info("Loading multilingual NER models from Hugging Face...")
try:
    ENG_HF_MODEL_REPO = "kuzyshyn/eng_ner"
    english_tokenizer = AutoTokenizer.from_pretrained(ENG_HF_MODEL_REPO)
    english_model = AutoModelForTokenClassification.from_pretrained(
        ENG_HF_MODEL_REPO,
        id2label={0: "O", 1: "B-per", 2: "I-per", 3: "B-org", 4: "I-org"},
        label2id={"O": 0, "B-per": 1, "I-per": 2, "B-org": 3, "I-org": 4}
    )
    english_ner = pipeline("ner", model=english_model, tokenizer=english_tokenizer, aggregation_strategy="simple")

    UKR_HF_MODEL_REPO = "kuzyshyn/ukr_ner"
    ukrainian_tokenizer = AutoTokenizer.from_pretrained(UKR_HF_MODEL_REPO)
    ukrainian_model = AutoModelForTokenClassification.from_pretrained(
        UKR_HF_MODEL_REPO,
        id2label={0: "O", 1: "B-per", 2: "I-per", 3: "B-org", 4: "I-org"},
        label2id={"O": 0, "B-per": 1, "I-per": 2, "B-org": 3, "I-org": 4}
    )
    ukrainian_ner = pipeline("ner", model=ukrainian_model, tokenizer=ukrainian_tokenizer, aggregation_strategy="simple")

    logger.info("NER models loaded successfully ✅")
except Exception as e:
    logger.error(f"Failed to load NER models: {e}")
    english_ner, ukrainian_ner = None, None

@mcp_server.tool(description="Performs Named Entity Recognition (NER) on a given text.")
def ner_inference(text: str, language: str = 'en'):
    if language == 'en' and english_ner:
        return english_ner(text)
    elif language == 'uk' and ukrainian_ner:
        return ukrainian_ner(text)
    else:
        return []

logger.info("Initializing RAG system...")
PROFILE_PATH = "student_profile.txt"
if os.path.exists(PROFILE_PATH):
    with open(PROFILE_PATH, "r", encoding="utf-8") as f:
        raw_text = f.read()
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = splitter.create_documents([raw_text])
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectordb = FAISS.from_documents(docs, embeddings)
    retriever = vectordb.as_retriever(search_kwargs={"k": 3})
else:
    retriever = None
    logger.warning("student_profile.txt not found. RAG will be disabled.")

rag_prompt_template_uk = PromptTemplate(
    template="""Ви — корисний помічник, який відповідає виключно на основі наданого КОНТЕКСТУ.
Якщо відповіді немає, скажіть, що ви не знаєте.
КОНТЕКСТ:
{context}
ПИТАННЯ: {question}
ВІДПОВІДЬ:""",
    input_variables=["context", "question"]
)

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.1, api_key=api_key)

@mcp_server.tool(description="Answers questions about a student based on student_profile.txt using RAG.")
def student_rag(question: str) -> str:
    if not retriever:
        return "RAG is not initialized (student_profile.txt missing)."
    docs_list = retriever.invoke(question)
    context = "\n\n".join([doc.page_content for doc in docs_list])
    if not context.strip():
        return "Я не знаю відповіді на це питання."
    rag_chain = LLMChain(llm=llm, prompt=rag_prompt_template_uk)
    return rag_chain.invoke({"context": context, "question": question})['text']

@mcp_server.custom_route("/", methods=["GET"])
async def health_check(request):
    return JSONResponse({"status": "ok", "message": "Inference MCP server is running"})

if __name__ == "__main__":
    mcp_server.run(transport="streamable-http")
