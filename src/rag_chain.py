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

def build_context(chunks):
    "Fonction qui assemble les chunks récupérés en un seul bloc de texte, avec la source de chaque chunk."
    parts = []
    for chunk in chunks:
        source = chunk.metadata.get("source", "inconnu")
        page = chunk.metadata.get("page", "?")
        parts.append(f"[Source : {source}, page {page}]\n{chunk.page_content}")
    return"\n\n".join(parts)

def answer_question(question):
    """Pipeline complet : recherche des chunks pertinents, puis génération de la réponse."""
    chunks = retrieve_chunks(question)
    context = build_context(chunks)

    prompt = PROMPT_TEMPLATE.format(context=context, question=question)

    response = client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
    )

    answer = response.choices[0].message.content
    return answer, chunks