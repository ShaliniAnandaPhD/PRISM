# pRISM  
**Procedural Intelligence & RAG-based Semantic Multi-Model**

pRISM is a research-focused repository demonstrating how **Retrieval-Augmented Generation (RAG)** workflows can integrate outputs from multiple language models (Mistral, Claude 3.5, and OpenAI) to create procedural documents and conduct advanced AI-driven tasks. It includes consensus techniques, **LoRA fine-tuning**, and **Explainable AI** for transparency and optimization.

> **Note**: This repository is for research and demonstration purposes only. No real-world or proprietary data is included.

---

## Features  

- **Multi-Model Integration**: Aggregates results from Mistral, Claude 3.5, and OpenAI for robust, high-quality outputs.  
- **Consensus Algorithms**: Implements methods like Weighted Majority Voting, Bayesian Model Averaging, and referee LLMs.  
- **RAG Workflow**: Uses vector-based retrieval for enhanced AI-generated outputs.  
- **LoRA Fine-Tuning**: Enables task-specific model adaptation with minimal resources.  
- **Explainable AI**: Visualizes how retrieved context and model contributions shape decisions.

---

## Repository Structure  


pRISM/
├── README.md
├── LICENSE
├── requirements.txt
├── scripts/
│   ├── 01_data_preprocessing.py
│   ├── 02_build_vector_index.py
│   ├── 03_test_llms_individually.py
│   ├── 04_vote_based_approach.py
│   ├── 05_generate_procedural_claims.py
│   ├── 06_weighted_majority_voting.py
│   ├── 07_bayesian_model_averaging.py
│   ├── 08_referee_llm_approach.py
│   └── 09_lora_fine_tuning.py
├── src/
│   ├── embeddings/
│   │   ├── chunking.py
│   │   ├── embedding.py
│   │   └── vector_store.py
│   ├── llm_integration/
│   │   ├── mistral_model.py
│   │   ├── claude35_model.py
│   │   ├── openai_model.py
│   │   └── voting.py
│   ├── rag_pipeline/
│   │   ├── retrieval.py
│   │   └── generation.py
│   └── procedural_drafting/
│       ├── procedural_format.py
│       └── claims_constructor.py
├── notebooks/
│   └── data_exploration.ipynb
└── docs/
    ├── architecture_diagram.png
    └── usage_guide.md


---

## Installation  

1. Clone the repository:  
   
   git clone https://github.com/your-repo/pRISM.git
   cd pRISM
   

2. Install dependencies:  
   
   pip install -r requirements.txt


---

## Usage  

### Key Scripts  

- **Data Preprocessing**:  
  `scripts/01_data_preprocessing.py` – Cleans and prepares raw text data.  

- **Vector Indexing**:  
  `scripts/02_build_vector_index.py` – Splits data, generates embeddings, and stores them in a vector database.  

- **Model Evaluation**:  
  `scripts/03_test_llms_individually.py` – Evaluates Mistral, Claude 3.5, and OpenAI models.  

- **Consensus Algorithms**:  
  - `scripts/06_weighted_majority_voting.py` – Implements Weighted Majority Voting.  
  - `scripts/07_bayesian_model_averaging.py` – Combines outputs using Bayesian averaging.  

- **Referee LLM**:  
  `scripts/08_referee_llm_approach.py` – Resolves conflicts in outputs using a specialized LLM.  

- **LoRA Fine-Tuning**:  
  `scripts/09_lora_fine_tuning.py` – Demonstrates LoRA fine-tuning for domain-specific optimization.  

---

## License  

This project is licensed under the [MIT License](./LICENSE).  

**Disclaimer**: This repository is for research and demonstration purposes only. Always review AI-generated outputs with domain experts before applying them in real-world scenarios.

