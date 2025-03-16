# Phi-4 + PRISM Fusion Model for Legal Analysis

A system that combines Microsoft's Phi-4 general-purpose model with PRISM (Precise Retrieval and Inference System for Legal Materials) using LoRA fusion for enhanced legal document analysis.

##  Features

- **Model Fusion**: Combines Phi-4's general knowledge with PRISM's legal expertise using LoRA adapters
- **Multimodal Support**: Handles text, documents, and images
- **Domain Expertise**: Enhanced performance on legal tasks including contract analysis and regulatory compliance
- **Configurable Fusion Ratio**: Tune the balance between general capabilities and legal specialization
- **Comprehensive Benchmarking**: Tools for evaluating performance, accuracy, and efficiency
- **Citation Tracking**: Provides source citations for legal references
- **Document Processing**: Support for PDF, Word, and HTML documents

##  Requirements

- Python 3.9+
- PyTorch 2.0+
- Transformers 4.30+
- CUDA-capable GPU (recommended)

##  Installation

```bash
# Clone the repository
git clone https://github.com/ailegaltech/phi4-prism-fusion.git
cd phi4-prism-fusion

# Install dependencies
pip install -r requirements.txt

# Install the package
pip install -e .
```

##  Quick Start

```python
from fusion import FusionModel
from models.phi4_model import Phi4Model
from models.prism_model import PRISMModel

# Initialize models
phi4 = Phi4Model(model_path="models/phi-4-mini")
prism = PRISMModel(model_path="models/prism-legal")

# Create fusion model
fusion = FusionModel(
    phi4_model=phi4,
    prism_model=prism,
    fusion_ratio=[0.7, 0.3]  # 70% Phi-4, 30% PRISM
)

# Process a query
result = fusion.generate(
    query="Analyze this contract for potential risks",
    document="Your contract text here..."
)

# Print the response
print(result["response"])

# Print citations
if "citations" in result:
    print("\nCitations:")
    for citation in result["citations"]:
        print(f"- {citation}")
```

##  Command-Line Usage

```bash
# Basic query
phi4-prism-cli --query "Analyze this contract for potential risks" --file contract.pdf

# Interactive mode
phi4-prism-cli --interactive

# Run benchmarks
phi4-prism-cli --benchmark
```

## Benchmarking

The system includes comprehensive benchmarking tools:

```bash
# Run performance benchmarks
phi4-prism-cli --benchmark

# Test different fusion ratios
phi4-prism-cli --benchmark --fusion-ratio 0.5 0.5
```

##  Architecture

The system architecture includes:

1. **Phi-4 Model**: Provides general knowledge and multimodal capabilities
2. **PRISM Model**: Specialized for legal document retrieval and reasoning
3. **LoRA Fusion Layer**: Combines model outputs with configurable weights
4. **Context Processor**: Enhances fused representations for better reasoning
5. **Response Generator**: Produces final responses with citations

##  License

This project is licensed under the MIT License - see the LICENSE file for details.

##  Acknowledgements

- Microsoft Research for the Phi-4 model
- The open-source community for various libraries and tools
