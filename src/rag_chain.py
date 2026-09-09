import os
from dotenv import load_dotenv
from groq import Groq

from src.retriever import retrieve_chunks

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

PROMPT_TEMPLATE = """Tu es un assistant pédagogique qui répond aux questions en te basant UNIQUEMENT sur les extraits de cours fournis ci-dessous.
Si la réponse ne se trouve pas dans les extraits, dis clairement que l'information n'est pas présente dans le cours.

Extraits de cours :
{context}

Question : {question}

Réponse :"""

NO_CONTEXT_MESSAGE = "Cette information ne se trouve pas dans les cours indexés."


def build_context(chunks):
    parts = []
    for chunk in chunks:
        source = chunk.metadata.get("source", "inconnu")
        page = chunk.metadata.get("page", "?")
        parts.append(f"[Source : {source}, page {page}]\n{chunk.page_content}")
    return "\n\n".join(parts)


def contextualize_question(question, history):
    """Réécrit la question en tenant compte des derniers échanges, pour gérer les questions de suivi."""
    if not history:
        return question

    recent = history[-4:]  # les 2 derniers échanges (question + réponse)
    context_lines = "\n".join(f"{m['role']}: {m['content']}" for m in recent)
    return f"Contexte de la conversation précédente :\n{context_lines}\n\nNouvelle question : {question}"


def answer_question(question, history=None):
    history = history or []
    search_query = contextualize_question(question, history)

    chunks = retrieve_chunks(search_query)
    context = build_context(chunks)
    prompt = PROMPT_TEMPLATE.format(context=context, question=question)

    response = client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
    )

    answer = response.choices[0].message.content
    return answer, chunks