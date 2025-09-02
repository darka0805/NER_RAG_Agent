# agent_api.py

import os
import logging
import asyncio
import langid
from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import httpx
from dotenv import load_dotenv

from langchain.agents import Tool, initialize_agent, AgentType
from langchain.chains import LLMChain
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

from transformers import pipeline, AutoTokenizer, AutoModelForTokenClassification

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()


@app.on_event("startup")
async def startup():
    global agent, ukr_ner_pipe, eng_ner_pipe, retriever, rag_prompt_template_en, rag_prompt_template_uk, llm

    LABELS = ["O", "B-per", "I-per", "B-org", "I-org"]
    label2id = {l: i for i, l in enumerate(LABELS)}
    id2label = {i: l for i, l in enumerate(LABELS)}

    # Load English NER model
    ENG_HF_MODEL_REPO = "kuzyshyn/eng_ner"
    eng_tokenizer = AutoTokenizer.from_pretrained(ENG_HF_MODEL_REPO)
    eng_model = AutoModelForTokenClassification.from_pretrained(
        ENG_HF_MODEL_REPO, id2label=id2label, label2id=label2id
    )
    eng_ner_pipe = pipeline(
        "ner",
        model=eng_model,
        tokenizer=eng_tokenizer,
        aggregation_strategy="simple",
    )
    logger.info("English NER model loaded from Hugging Face ✅")

    # Load Ukrainian NER model
    UKR_HF_MODEL_REPO = "kuzyshyn/ukr_ner"
    ukr_tokenizer = AutoTokenizer.from_pretrained(UKR_HF_MODEL_REPO)
    ukr_model = AutoModelForTokenClassification.from_pretrained(
        UKR_HF_MODEL_REPO, id2label=id2label, label2id=label2id
    )
    ukr_ner_pipe = pipeline(
        "ner",
        model=ukr_model,
        tokenizer=ukr_tokenizer,
        aggregation_strategy="simple",
    )
    logger.info("Ukrainian NER model loaded from Hugging Face ✅")

    # Load student profile for RAG
    PROFILE_PATH = "student_profile.txt"
    with open(PROFILE_PATH, "r", encoding="utf-8") as f:
        raw_text = f.read()

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = splitter.create_documents([raw_text])

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectordb = FAISS.from_documents(docs, embeddings)
    retriever = vectordb.as_retriever(search_kwargs={"k": 3})

    # Prompt templates
    rag_prompt_template_en = PromptTemplate(
        template="""You are a helpful assistant that answers strictly from the provided CONTEXT. 
If the answer is not present, say you don't know. 
CONTEXT: {context} 
QUESTION: {question} 
ANSWER:""",
        input_variables=["context", "question"],
    )

    rag_prompt_template_uk = PromptTemplate(
        template="""Ви — корисний помічник, який відповідає виключно на основі наданого КОНТЕКСТУ. 
Якщо відповіді немає, скажіть, що ви не знаєте. 
КОНТЕКСТ: {context} 
ПИТАННЯ: {question} 
ВІДПОВІДЬ:""",
        input_variables=["context", "question"],
    )

    # Define tools
    def get_entities_from_pipe(text: str, ner_pipe):
        if not ner_pipe:
            return []
        return ner_pipe(text)

    def ner_anonymize(text: str, ner_pipe, per_token="[PERSON]", org_token="[ORG]"):
        ents = get_entities_from_pipe(text, ner_pipe)
        spans = []
        for e in ents:
            label = e.get("entity_group", "").upper()
            if label.startswith("PER") or label == "ORG":
                replacement = per_token if label.startswith("PER") else org_token
                spans.append((e["start"], e["end"], replacement))
        spans.sort(key=lambda x: x[0])

        merged = []
        for s in spans:
            if not merged or s[0] > merged[-1][1]:
                merged.append(list(s))
            else:
                merged[-1][1] = max(merged[-1][1], s[1])

        out = text
        for start, end, repl in reversed(merged):
            out = out[:start] + repl + out[end:]
        return out

    def anonymize_text_fn(text: str):
        try:
            lang, _ = langid.classify(text)
        except Exception:
            lang = "en"

        ner_pipe_to_use = None
        if lang == "uk" and ukr_ner_pipe:
            ner_pipe_to_use = ukr_ner_pipe
        elif lang == "en" and eng_ner_pipe:
            ner_pipe_to_use = eng_ner_pipe
        else:
            return f"Error: No NER model available for language '{lang}'."

        return ner_anonymize(text, ner_pipe_to_use)

    anonymize_tool = Tool(
        name="AnonymizeText",
        func=anonymize_text_fn,
        description=(
            "Useful for anonymizing or hiding person and organization names from a sentence. "
            "Use this tool when the user's request is to process a sentence that contains sensitive data, "
            "but is not a direct question. For example, 'Сьогодні Тарас відвідав садок'."
        ),
    )

    def student_rag_tool_fn(question: str):
        lang, _ = langid.classify(question)
        prompt_template = rag_prompt_template_uk if lang == "uk" else rag_prompt_template_en
        docs_list = retriever.invoke(question)
        context = "\n\n".join([doc.page_content for doc in docs_list])

        if not context.strip():
            return "Я не знаю відповіді на це питання." if lang == "uk" else "I don't know the answer."

        rag_chain = LLMChain(llm=llm, prompt=prompt_template)
        return rag_chain.invoke({"context": context, "question": question})["text"]

    student_rag_tool = Tool(
        name="StudentRAG",
        func=student_rag_tool_fn,
        description="Answers questions about the student from student_profile.txt.",
    )

    # LLM
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.1, api_key=OPENAI_API_KEY)

    # Agent
    tools = [anonymize_tool, student_rag_tool]
    agent = initialize_agent(
        tools=tools,
        llm=llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True,
        handle_parsing_errors=True,
    )


class TextQuery(BaseModel):
    text: str


@app.post("/process")
def process_endpoint(query: TextQuery):
    if agent is None:
        raise HTTPException(status_code=500, detail="Agent not initialized.")
    try:
        response = agent.run(query.text)
        return {"response": response}
    except Exception as e:
        logger.error("Agent processing error: %s", e)
        raise HTTPException(status_code=500, detail="Processing error.")


@app.get("/")
def read_root():
    return {"message": "LangChain Agent API is running."}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("agent_api:app", host="0.0.0.0", port=8001, reload=True)
