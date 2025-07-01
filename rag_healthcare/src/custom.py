import os
import uuid
from typing import List, Dict, Optional, Any, Iterable

from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.models import Distance, VectorParams, PointStruct

from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.documents import Document

from langchain_core.vectorstores import VectorStore
from langchain_core.embeddings import Embeddings

from langchain.storage import InMemoryStore
from langchain.retrievers import ParentDocumentRetriever
from langchain.text_splitter import RecursiveCharacterTextSplitter

from dotenv import load_dotenv

load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY environment variable is not set.")

EMBEDDING_MODEL_NAME = os.getenv("embedding_model","models/embedding-001")
try:
    VECTOR_SIZE = int(os.getenv("vector_size", 768))
except ValueError:
    print("Warning: vector_size is not set or invalid, defaulting to 768.")
    VECTOR_SIZE = 768

QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
class QdrantVectorStore(VectorStore):
    client: QdrantClient
    embeddings: Embeddings 
    collection_name: str
    
    def create_collection(self) -> bool:
        """
        Create a new collection if it doesn't exist.
        
        Returns:
            bool: True if collection was created or already exists
        """
        try:
            collections = self.client.get_collections().collections
            collection_names = [collection.name for collection in collections]
            
            if self.collection_name not in collection_names:
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(
                        size=self.vector_size,
                        distance=Distance.COSINE
                    )
                )
                print(f"Created collection: {self.collection_name}")
            else:
                print(f"Collection {self.collection_name} already exists")
            
            return True
        except Exception as e:
            print(f"Error creating collection: {str(e)}")
            return False
 
    def delete_collection(self) -> bool: 
        try:
            self.client.delete_collection(collection_name=self.collection_name)
            print(f"Deleted collection: '{self.collection_name}'")
            return True
        except Exception as e:
            print(f"Error deleting collection '{self.collection_name}' (it might not exist): {str(e)}")
            return False

    def add_texts(
        self,
        texts: Iterable[str],
        metadatas: Optional[List[dict]] = None,
        ids: Optional[List[str]] = None, 
        **kwargs: Any,
    ) -> List[str]:
        texts_list = list(texts) 
        if ids is None:
            actual_ids = [str(uuid.uuid4()) for _ in texts_list]
        else:
            if len(ids) != len(texts_list):
                raise ValueError("If ids are provided, their length must match the number of texts.")
            actual_ids = ids
        
        if metadatas is None:
            actual_metadatas = [{} for _ in texts_list]
        else:
            if len(metadatas) != len(texts_list):
                raise ValueError("If metadatas are provided, their length must match the number of texts.")
            actual_metadatas = metadatas

        embeddings = self.embeddings.embed_documents(texts_list)

        points = []
        for i, text_content in enumerate(texts_list):
            point = PointStruct(
                id=actual_ids[i], 
                vector=embeddings[i],
                payload={
                    "text": text_content,
                    "metadata": actual_metadatas[i] or {} 
                }
            )
            points.append(point)

        if points:
            self.client.upsert(
                collection_name=self.collection_name,
                points=points
            )
        return actual_ids 


    def add_documents(self, documents: List[Document], *, ids: Optional[List[str]] = None, **kwargs: Any) -> List[str]:
        texts = [doc.page_content for doc in documents]
        metadatas = [doc.metadata for doc in documents] 

        return self.add_texts(texts, metadatas=metadatas, ids=None, **kwargs)

    def similarity_search(
        self, query: str, k: int = 4, filter: Optional[Dict] = None, **kwargs: Any
    ) -> List[Document]:
        """Run similarity search with score."""
        results_with_scores = self.similarity_search_with_score(query, k, filter=filter, **kwargs)
        return [doc for doc, _score in results_with_scores]

    def similarity_search_with_score(
        self, query: str, k: int = 4, filter: Optional[Dict] = None, **kwargs: Any
    ) -> List[tuple[Document, float]]:
        """Run similarity search with Qdrant and return documents with scores."""
        query_embedding = self.embeddings.embed_query(query)

        qdrant_filter_model = None
        if filter:
            conditions = []
            for key, value in filter.items():
                actual_key = key.replace("metadata.", "")
                conditions.append(
                    models.FieldCondition(
                        key=actual_key,
                        match=models.MatchValue(value=value)
                    )
                )
            if conditions:
                qdrant_filter_model = models.Filter(must=conditions)


        search_results = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_embedding,
            query_filter=qdrant_filter_model,
            limit=k,
            with_payload=True,
            with_vectors=False 
        )

        docs_with_scores = []
        for hit in search_results:
            metadata = hit.payload.get("metadata", {})
            page_content = hit.payload.get("text", "")
            doc = Document(page_content=page_content, metadata=metadata)
            docs_with_scores.append((doc, hit.score))
        return docs_with_scores

    @classmethod
    def from_texts(
        cls,
        texts: List[str],
        embedding: Embeddings,
        metadatas: Optional[List[dict]] = None,
        ids: Optional[List[str]] = None,
        collection_name: Optional[str] = None,
        client: Optional[QdrantClient] = None,
        **kwargs: Any,
    ) -> "QdrantVectorStore":
        """Construct QdrantLangchainVectorStore from raw texts."""
        if collection_name is None:
            collection_name = "langchain-" + str(uuid.uuid4())
        
        qdrant_client = client if client is not None else QdrantClient(url=os.getenv("QDRANT_URL", "http://localhost:6333"))

        instance = cls(collection_name, embedding, qdrant_client, **kwargs)
        instance.add_texts(texts, metadatas=metadatas, ids=ids)
        return instance

def main():
    # Example usage
    embedding_function = GoogleGenerativeAIEmbeddings(
        model=EMBEDDING_MODEL_NAME,
        google_api_key=GOOGLE_API_KEY
    )
    qdrant_global_client = QdrantClient(url=QDRANT_URL) 
    child_vector_store = QdrantVectorStore(
        client=qdrant_global_client,
        embeddings=embedding_function,
        collection_name="test_pydantic_fix_v2" 
    )
    # vector_store.delete_collection()
    child_vector_store.create_collection()

    store = InMemoryStore() 
    parent_retriever = ParentDocumentRetriever(
        vectorstore=child_vector_store, # Pass the vector store instance directly
        docstore=store,                 
        child_splitter=RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=20)

    )

    parent_documents = [
        Document(
            page_content="Các phương pháp phục hồi chức năng vận động và ngôn ngữ sau đột quỵ",
            metadata={"source": "suc_khoe_doi_thuong", "doc_id": "doc_suckhoe"} # doc_id quan trọng cho docstore
        ),
        Document(
            page_content="Bộ Khoa học và Công nghệ (MOST) chịu trách nhiệm quản lý về các hoạt động nghiên cứu khoa học, phát triển công nghệ và đổi mới sáng tạo. MOST cũng thúc đẩy hợp tác quốc tế trong lĩnh vực khoa học và công nghệ, nhằm nâng cao năng lực quốc gia.",
            metadata={"source": "website_most", "doc_id": "doc_khcn"}
        ),
         Document(
            page_content="An toàn thực phẩm là vấn đề quan trọng. Bộ Y Tế và Bộ Nông nghiệp cùng phối hợp quản lý. Các quy chuẩn về vệ sinh an toàn thực phẩm được cập nhật thường xuyên.",
            metadata={"source": "bao_suc_khoe", "doc_id": "doc_attp"}
        )
    ]

    parent_retriever.add_documents(parent_documents, ids=None)
    query = "Các phương pháp nào để phục hồi chức năng sau vận động?"
    

    results = parent_retriever.invoke(query, k=2)
    # print("results",results)
    print(f"\n--- Search Results for query: '{query}' ---")
    for i, doc in enumerate(results, 1):
        print(f"\n{i}. Score: {doc['score']:.4f}")
        print(f"   Text: {doc['text']}")
        print(f"   Metadata: {doc['metadata']}")


if __name__ == "__main__":
    main()