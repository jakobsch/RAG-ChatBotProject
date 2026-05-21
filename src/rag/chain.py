from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough


def build_rag_chain(retriever, llm):
    prompt = ChatPromptTemplate.from_template("""
Du bist ein hilfreicher Assistent für Sustainability Reports.

Beantworte die Frage ausschließlich auf Basis des folgenden Kontexts.
Wenn die Antwort nicht im Kontext steht, sage ehrlich:
"Das geht aus dem bereitgestellten Dokument nicht hervor."

Kontext:
{context}

Frage:
{question}

Antwort:
""")

    chain = (
        {
            "context": retriever,
            "question": RunnablePassthrough()
        }
        | prompt
        | llm
    )

    return chain