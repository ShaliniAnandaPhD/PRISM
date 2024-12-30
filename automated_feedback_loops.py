import json
from typing import List, Dict, Any
from langchain.chains import RetrievalQA
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import OpenAI
from pydantic import BaseModel

class FeedbackConfig(BaseModel):
    """
    Configuration for the automated feedback loop system.
    """
    vector_store_path: str  # Path to the FAISS vector store
    embedding_model_name: str  # Name of the embedding model for vector retrieval
    llm_model_name: str  # Name of the language model for generation
    feedback_file_path: str  # Path to save user feedback
    top_k: int = 5  # Number of documents to retrieve

class AutomatedFeedbackLoop:
    """
    Implements automated feedback loops to refine and improve retrieval and generation results dynamically.
    """

    def __init__(self, config: FeedbackConfig):
        """
        Initialize the feedback loop system.

        Args:
            config (FeedbackConfig): Configuration for the feedback loop.
        """
        self.config = config

        # Load embedding model
        self.embeddings = HuggingFaceEmbeddings(model_name=config.embedding_model_name)
        print(f"Embedding model {config.embedding_model_name} loaded successfully.")

        # Load vector store
        self.vector_store = FAISS.load_local(config.vector_store_path, self.embeddings)
        print("FAISS vector store loaded successfully.")

        # Load language model
        self.llm = OpenAI(model=config.llm_model_name)
        print(f"Language model {config.llm_model_name} loaded successfully.")

        # Initialize feedback storage
        self.feedback = self.load_feedback()

    def load_feedback(self) -> List[Dict[str, Any]]:
        """
        Load feedback data from a JSON file.

        Returns:
            List[Dict[str, Any]]: List of feedback entries.
        """
        try:
            with open(self.config.feedback_file_path, "r") as file:
                feedback_data = json.load(file)
                print(f"Loaded feedback from {self.config.feedback_file_path}")
                return feedback_data
        except FileNotFoundError:
            print("No existing feedback file found. Starting fresh.")
            return []

    def save_feedback(self):
        """
        Save feedback data to a JSON file.
        """
        with open(self.config.feedback_file_path, "w") as file:
            json.dump(self.feedback, file, indent=4)
        print(f"Feedback saved to {self.config.feedback_file_path}")

    def collect_feedback(self, query: str, generated_response: str, user_feedback: str):
        """
        Collect and store user feedback for a given query and response.

        Args:
            query (str): The original query.
            generated_response (str): The response generated by the system.
            user_feedback (str): User-provided feedback.
        """
        feedback_entry = {
            "query": query,
            "generated_response": generated_response,
            "user_feedback": user_feedback
        }
        self.feedback.append(feedback_entry)
        print("Feedback collected for query:", query)
        self.save_feedback()

    def refine_vector_store(self):
        """
        Refine the vector store using positive feedback to improve retrieval.
        """
        positive_feedback = [f for f in self.feedback if "positive" in f["user_feedback"].lower()]
        if not positive_feedback:
            print("No positive feedback to process.")
            return

        # Add positive examples to the vector store
        for feedback in positive_feedback:
            self.vector_store.add_texts([feedback["query"]], embedding=self.embeddings)
            print(f"Refined vector store with query: {feedback['query']}")

        # Save the updated vector store
        self.vector_store.save_local(self.config.vector_store_path)
        print("Vector store refined and saved.")

    def execute_pipeline(self, query: str) -> str:
        """
        Execute the RAG pipeline and collect user feedback.

        Args:
            query (str): User query.

        Returns:
            str: Final generated response.
        """
        retriever = self.vector_store.as_retriever(search_kwargs={"k": self.config.top_k})
        qa_chain = RetrievalQA.from_chain_type(
            retriever=retriever,
            chain_type="stuff",
            llm=self.llm,
            return_source_documents=True
        )

        # Generate response
        result = qa_chain({"query": query})
        generated_response = result["result"]
        print("Generated Response:", generated_response)

        # Simulate user feedback collection (replace with real feedback in production)
        user_feedback = input(f"Feedback for query '{query}': Positive/Negative? ")
        self.collect_feedback(query, generated_response, user_feedback)

        return generated_response

if __name__ == "__main__":
    """
    Entry point for the automated feedback loop system.

    What We Did:
    - Collected user feedback dynamically for generated responses.
    - Refined the vector store using positive feedback examples.
    - Improved the pipeline through continuous feedback integration.

    What's Next:
    - Automate the feedback collection process using a front-end interface.
    - Visualize feedback trends for better system understanding.
    - Integrate with real-time APIs for live improvement.
    """
    # Configuration
    config = FeedbackConfig(
        vector_store_path="./vector_store_faiss",
        embedding_model_name="sentence-transformers/all-MiniLM-L6-v2",
        llm_model_name="gpt-4",
        feedback_file_path="./user_feedback.json",
        top_k=5
    )

    # Initialize feedback loop
    feedback_loop = AutomatedFeedbackLoop(config)

    # Test the pipeline with feedback loop
    query = "What are the confidentiality clauses in a standard NDA?"
    feedback_loop.execute_pipeline(query)

    # Refine the vector store based on collected feedback
    feedback_loop.refine_vector_store()
