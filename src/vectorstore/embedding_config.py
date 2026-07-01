import os
from dataclasses import dataclass
from typing import Literal, Optional

from dotenv import load_dotenv

load_dotenv(override=True)


@dataclass
class EmbeddingConfig:
    provider: Literal["huggingface", "gwdg"]
    model_name: str
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    query_prefix: Optional[str] = None

    @property
    def cache_key(self) -> str:
        """Eindeutiger Bezeichner, u.a. für den Chroma-Persist-Pfad je Modell."""
        return f"{self.provider}__{self.model_name}".replace("/", "_")

    @classmethod
    def from_env(cls) -> "EmbeddingConfig":
        """Liest die Embedding-Konfiguration ausschließlich aus Umgebungsvariablen."""
        provider = os.getenv("EMBEDDING_PROVIDER", "huggingface")

        if provider == "gwdg":
            api_key = os.getenv("GWDG_API_KEY")
            if not api_key:
                raise ValueError("GWDG_API_KEY fehlt in der .env.")
            return cls(
                provider="gwdg",
                model_name=os.getenv("GWDG_EMBEDDING_MODEL", "e5-mistral-7b-instruct"),
                api_key=api_key,
                base_url=os.getenv("GWDG_BASE_URL", "https://chat-ai.academiccloud.de/v1"),
                query_prefix=(
                    "Instruct: Given a query, retrieve relevant passages\nQuery: "
                ),
            )

        return cls(
            provider="huggingface",
            model_name=os.getenv("HF_EMBEDDING_MODEL", "all-MiniLM-L6-v2"),
        )