(The project is ongoing and updates will be made going into January 2025 )

# pRISM
*Procedural Intelligence & RAG-based Semantic Multi-Model*

pRISM combines RAG (Retrieval-Augmented Generation) with multi-model consensus to create more reliable AI outputs, specifically designed for legal document processing. By integrating multiple language models and incorporating human expert feedback, we aim to make AI-generated legal content more transparent and trustworthy.

> **Note**: This is a research project - please don't use it for actual legal work without expert review!

## What Makes pRISM Different?

- **Multi-Model Consensus**: Instead of relying on a single LLM, we aggregate outputs from multiple models using techniques like weighted majority voting. In our testing, this has significantly reduced hallucination rates (full benchmarks coming soon).

- **Domain-Specific Fine-Tuning**: We use LoRA to adapt models for legal tasks without breaking the bank on compute costs. Our initial tests show promising improvements in legal terminology accuracy.

- **Human-in-the-Loop**: Legal experts review outputs and provide feedback, which gets incorporated back into the system to improve future results.

## Quick Start

1. Clone and install:
```bash
git clone https://github.com/your-repo/pRISM.git
cd pRISM
pip install -r requirements.txt
```

## Core Components

### RAG Pipeline (`rag_pipeline.py`)
The heart of pRISM. Handles document retrieval and generation using FAISS for vector search and LangChain for orchestration. 

### Real-Time API (`real_time_inference_api.py`)
FastAPI service for running inferences. Currently supports:
- Basic document queries
- Multi-model consensus outputs
- Confidence scoring
- Explanation generation

### Evaluation Tools
- `performance_benchmarking.py`: Measures latency and accuracy
- `evaluation_report_generator.py`: Generates detailed PDF reports
- `automated_feedback_loops.py`: Incorporates user feedback for continuous improvement

## Demos

CLI Tool : https://www.linkedin.com/posts/shalinianandaphd_cli-demo-for-prism-continuation-from-my-activity-7305376243787382784-rdm3?utm_source=share&utm_medium=member_desktop&rcm=ACoAAATH3cgBLB3ZhNKdiK83PyAA1KPddyaaY2I

Evaluations: https://www.linkedin.com/posts/shalinianandaphd_legaltech-accesstojustice-ai-activity-7305289571565518848-xSOX?utm_source=share&utm_medium=member_desktop&rcm=ACoAAATH3cgBLB3ZhNKdiK83PyAA1KPddyaaY2I

Notion page with architecture: https://www.linkedin.com/posts/shalinianandaphd_prism-proactive-retrieval-intelligence-activity-7303428785339543552-UIzc?utm_source=share&utm_medium=member_desktop&rcm=ACoAAATH3cgBLB3ZhNKdiK83PyAA1KPddyaaY2I



## Contributing
This is a research project and we'd love your input! Feel free to open issues or PRs, especially if you have experience with:
- Legal document processing
- RAG implementations
- Model fine-tuning
- Evaluation methodologies

## License
MIT License - see LICENSE.md

---
Built with 💡 for exploring better ways to handle legal AI

