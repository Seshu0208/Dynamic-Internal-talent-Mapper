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
        Strict metadata sanitizer for Chroma.
        Ensures:
        - Only primitive values
        - Lists contain only same-type primitives
        - Dicts converted to string
        """

        clean_meta = {}

        for key, value in metadata.items():

            if value is None:
                continue

            # -------------------------
            # Primitive values
            # -------------------------
            if isinstance(value, (str, int, float, bool)):
                if value == "":
                    continue
                clean_meta[key] = value
                continue

            # -------------------------
            # Dictionary → convert to string
            # -------------------------
            if isinstance(value, dict):
                if len(value) == 0:
                    continue
                clean_meta[key] = json.dumps(value)
                continue

            # -------------------------
            # List handling (STRICT)
            # -------------------------
            if isinstance(value, list):

                if len(value) == 0:
                    continue

                # Case 1: list of primitives → OK
                if all(isinstance(v, (str, int, float, bool)) for v in value):
                    clean_meta[key] = value
                    continue

                # Case 2: list of dicts → extract "name" if exists
                if all(isinstance(v, dict) for v in value):
                    extracted = []
                    for item in value:
                        if "name" in item:
                            extracted.append(str(item["name"]))
                        else:
                            extracted.append(json.dumps(item))
                    clean_meta[key] = extracted
                    continue

                # Case 3: mixed types → stringify everything
                clean_meta[key] = [str(v) for v in value]
                continue

            # -------------------------
            # Fallback
            # -------------------------
            clean_meta[key] = str(value)

        return clean_meta

    def add_employee(self, emp_id: str, text: str, metadata=None):

        embedding = self.model.encode(text, normalize_embeddings=True).tolist()

        safe_metadata = self._clean_metadata(metadata or {})

        # 🔥 FIX: ensure metadata is never empty
        if not safe_metadata:
            safe_metadata = {"placeholder": "none"}

        self.collection.upsert(
            ids=[emp_id],
            documents=[text],
            embeddings=[embedding],
            metadatas=[safe_metadata]
        )

    def query(self, query_text: str, top_k=5):

        query_embedding = self.model.encode(query_text, normalize_embeddings=True).tolist()

        return self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k
        )
    

