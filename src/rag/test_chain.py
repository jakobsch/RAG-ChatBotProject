from langchain_core.runnables import RunnableLambda

from src.rag.chain import build_rag_chain


def fake_retriever(question):
    return """
    Walmart berichtet über Emissionen, erneuerbare Energien,
    Lieferketten und soziale Verantwortung.
    """


def fake_llm(prompt):
    print("PROMPT AN DAS LLM:")
    print(prompt.to_string())

    return "Fake-Antwort: Walmart berichtet über Emissionen, erneuerbare Energien, Lieferketten und soziale Verantwortung."


def main():
    retriever = RunnableLambda(fake_retriever)
    llm = RunnableLambda(fake_llm)

    chain = build_rag_chain(retriever, llm)

    response = chain.invoke("Worüber berichtet Walmart im ESG Report?")

    print("\nANTWORT:")
    print(response)


if __name__ == "__main__":
    main()