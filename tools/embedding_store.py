import chromadb
from sentence_transformers import SentenceTransformer
import json


class TalentVectorStore:

    def __init__(self, db_path="data/chroma_store"):
        self.client = chromadb.PersistentClient(path=db_path)
        self.collection = self.client.get_or_create_collection(
            name="employees"
        )
        self.model = SentenceTransformer("all-MiniLM-L6-v2")

    def _clean_metadata(self, metadata: dict) -> dict:
        """
        Make metadata safe for Chroma:
        - Remove empty lists
        - Remove empty dicts
        - Convert dicts to JSON strings
        - Remove None values
        """

        clean_meta = {}

        for key, value in metadata.items():

            # Skip None
            if value is None:
                continue

            # Skip empty list
            if isinstance(value, list) and len(value) == 0:
                continue

            # Convert dict to JSON string (if not empty)
            if isinstance(value, dict):
                if len(value) == 0:
                    continue
                clean_meta[key] = json.dumps(value)
                continue

            # Keep non-empty list
            if isinstance(value, list):
                clean_meta[key] = value
                continue

            # Keep primitive types
            if isinstance(value, (str, int, float, bool)):
                if value == "":
                    continue
                clean_meta[key] = value
                continue

            # Fallback
            clean_meta[key] = str(value)

        return clean_meta

    def add_employee(self, emp_id: str, text: str, metadata=None):

        embedding = self.model.encode(text).tolist()

        safe_metadata = self._clean_metadata(metadata or {})

        self.collection.upsert(
            ids=[emp_id],
            documents=[text],
            embeddings=[embedding],
            metadatas=[safe_metadata]
        )

    def query(self, query_text: str, top_k=5):

        query_embedding = self.model.encode(query_text).tolist()

        return self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k
        )
    

