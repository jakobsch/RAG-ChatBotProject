from langchain_openai import ChatOpenAI

from src.rag.chain import build_rag_chain
from src.vectorstore.chromadb_storage import VectorStore


def main():
    vector_store = VectorStore()
    retriever = vector_store.as_retriever({"k": 3})

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    chain = build_rag_chain(retriever, llm)

    question = "Welche Nachhaltigkeitsthemen werden im Bericht genannt?"

    response = chain.invoke(question)

    print(response.content)


if __name__ == "__main__":
    main()