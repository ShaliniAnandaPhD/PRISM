(The project is ongoing and updates will be made going into January 2025 )


# pRISM  
**Procedural Intelligence & RAG-based Semantic Multi-Model**

pRISM is a research-focused repository demonstrating how **Retrieval-Augmented Generation (RAG)** workflows can integrate outputs from multiple language models (e.g., OpenAI) to create procedural documents in the legal field and conduct advanced AI-driven tasks. It includes consensus techniques, **LoRA fine-tuning**, and **Explainable AI** for transparency and optimization.

> **Note**: This repository is for research and demonstration purposes only. No real-world or proprietary data is included.


## Features  

- **Multi-Model Integration**: Aggregates results from OpenAI and retrieval models for robust, high-quality outputs.  
- **Consensus Algorithms**: Implements methods like Weighted Majority Voting and Bayesian Model Averaging.  
- **RAG Workflow**: Uses vector-based retrieval for enhanced AI-generated outputs.  
- **LoRA Fine-Tuning**: Enables task-specific model adaptation with minimal resources.  
- **Explainable AI**: Visualizes how retrieved context and model contributions shape decisions.

## Installation  

1. Clone the repository:  
   
   ```bash
   git clone https://github.com/your-repo/pRISM.git
   cd pRISM
   ```

2. Install dependencies:  
   
   ```bash
   pip install -r requirements.txt
   ```

---

## Usage  

### Key Scripts  

#### **1. LoRA Fine-Tuning**
**Script**: `scripts/lora_fine_tuning.py`  
This script demonstrates the application of **Low-Rank Adaptation (LoRA)** by:
- Fine-tuning pre-trained models for specific procedural tasks without retraining the entire model.
- Adapting models for domain-specific applications, such as legal or technical document drafting, using minimal computational resources.

#### **2. RAG Pipeline**
**Script**: `scripts/rag_pipeline.py`  
This script implements the complete **Retrieval-Augmented Generation (RAG)** pipeline by:
- Integrating vector retrieval with FAISS and embedding models.
- Using language models (e.g., OpenAI) to generate responses based on retrieved context.
- Supporting efficient query-based document retrieval and response generation.

#### **3. Real-Time Inference API**
**Script**: `scripts/real_time_inference_api.py`  
This script creates a FastAPI-based service for real-time inference by:
- Serving real-time Retrieval-Augmented Generation (RAG) queries via RESTful endpoints.
- Integrating LangChain with FAISS for vector-based retrieval.
- Providing robust query handling and response generation capabilities.

#### **4. Performance Benchmarking**
**Script**: `scripts/performance_benchmarking.py`  
This script benchmarks the RAG pipeline by:
- Measuring latency for query processing.
- Evaluating accuracy against predefined expected outputs.
- Assessing consensus reliability across retrieved documents.

#### **5. Evaluation Report Generator**
**Script**: `scripts/evaluation_report_generator.py`  
This script automates report generation by:
- Summarizing latency, accuracy, and consensus reliability results.
- Generating PDF reports with key insights and metrics.

#### **6. Automated Feedback Loops**
**Script**: `scripts/automated_feedback_loops.py`  
This script implements dynamic feedback collection and improvement by:
- Allowing users to provide feedback on generated responses.
- Refining vector stores based on positive feedback to improve retrieval.
- Dynamically updating pipeline outputs based on iterative feedback.

These scripts collectively implement the core functionality of pRISM, ensuring efficient data preparation, robust model evaluation, dynamic real-time inference, and domain-specific optimization.  

## License  

This project is licensed under the [MIT License](./LICENSE).  

**Disclaimer**: This repository is for research and demonstration purposes only. Always review AI-generated outputs with domain experts before applying them in real-world scenarios.
