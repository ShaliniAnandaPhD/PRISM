# pRISM  
**Procedural Intelligence & RAG-based Semantic Multi-Model**

pRISM is a research-focused repository demonstrating how **Retrieval-Augmented Generation (RAG)** workflows can integrate outputs from multiple language models (Mistral, Claude 3.5, and OpenAI) to create procedural documents and conduct advanced AI-driven tasks. It includes consensus techniques, **LoRA fine-tuning**, and **Explainable AI** for transparency and optimization.

> **Note**: This repository is for research and demonstration purposes only. No real-world or proprietary data is included.


## Features  

- **Multi-Model Integration**: Aggregates results from Mistral, Claude 3.5, and OpenAI for robust, high-quality outputs.  
- **Consensus Algorithms**: Implements methods like Weighted Majority Voting, Bayesian Model Averaging, and referee LLMs.  
- **RAG Workflow**: Uses vector-based retrieval for enhanced AI-generated outputs.  
- **LoRA Fine-Tuning**: Enables task-specific model adaptation with minimal resources.  
- **Explainable AI**: Visualizes how retrieved context and model contributions shape decisions.

## Installation  

1. Clone the repository:  
   
   git clone https://github.com/your-repo/pRISM.git
   cd pRISM
   

2. Install dependencies:  
   
   pip install -r requirements.txt


---

## Usage  

### Key Scripts  

#### **1. Data Preprocessing**
**Script**: `scripts/01_data_preprocessing.py`  
This script cleans and prepares raw text data for downstream tasks. It ensures text consistency by:
- Removing special characters, stopwords, and irrelevant content.
- Tokenizing the input for efficient embedding generation.
- Formatting text into manageable chunks to optimize embedding and retrieval processes.

#### **2. Vector Indexing**
**Script**: `scripts/02_build_vector_index.py`  
This script processes the pre-cleaned data by:
- Splitting text into contextually meaningful chunks.
- Generating embeddings for each chunk using pre-trained models.
- Storing the embeddings in a vector database for fast and accurate retrieval during query generation.


#### **3. Model Evaluation**
**Script**: `scripts/03_test_llms_individually.py`  
This script evaluates the performance of multiple language models, including Mistral, Claude 3.5, and OpenAI. It:
- Sends test prompts to each model.
- Collects and analyzes responses to determine their relevance, accuracy, and coherence.
- Establishes a baseline for comparing model outputs in multi-model workflows.


#### **4. Consensus Algorithms**

**a. Weighted Majority Voting**  
**Script**: `scripts/06_weighted_majority_voting.py`  
Implements a voting mechanism where:
- Each model's output is assigned a weight based on its performance or reliability.
- The final decision is made by aggregating these weighted votes, ensuring a balanced and optimized consensus.

**b. Bayesian Model Averaging**  
**Script**: `scripts/07_bayesian_model_averaging.py`  
This script uses Bayesian averaging to:
- Combine outputs probabilistically, taking into account prior knowledge of each model's strengths.
- Produce a refined result that integrates insights from all participating models.

#### **5. Referee LLM**
**Script**: `scripts/08_referee_llm_approach.py`  
This script leverages a specialized referee LLM to:
- Resolve conflicts in multi-model outputs by analyzing and selecting the most contextually accurate or coherent response.
- Serve as an arbiter when model outputs are ambiguous or contradictory.

#### **6. LoRA Fine-Tuning**
**Script**: `scripts/09_lora_fine_tuning.py`  
This script demonstrates the application of **Low-Rank Adaptation (LoRA)** by:
- Fine-tuning pre-trained models for specific procedural tasks without retraining the entire model.
- Adapting models for domain-specific applications, such as legal or technical document drafting, using minimal computational resources.


These scripts collectively implement the core functionality of pRISM, ensuring efficient data preparation, robust model evaluation, advanced consensus-building, and domain-specific optimization.  


## License  

This project is licensed under the [MIT License](./LICENSE).  

**Disclaimer**: This repository is for research and demonstration purposes only. Always review AI-generated outputs with domain experts before applying them in real-world scenarios.

