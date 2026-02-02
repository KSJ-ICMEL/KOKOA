# KOKOA: Knowledge-Oriented kMC Optimizing Agent

**Knowledge-driven Optimization of Kinetic Monte Carlo with Organized AI Agents**

A submission to the 2026 AI Co-Scientist Challenge Korea for autonomous scientific research in computational materials science.

## Overview

KOKOA is an AI-powered research framework designed to autonomously improve kinetic Monte Carlo (kMC) simulations for predicting lithium-ion conductivity in LLZO (Li₇La₃Zr₂O₁₂) garnet-type solid electrolytes. The system systematically analyzes and relaxes common kMC assumptions to improve prediction accuracy through physics-informed modeling approaches.

### Key Features

- **Autonomous Scientific Research**: AI agents that can independently formulate hypotheses, design experiments, and analyze results
- **Physics-Informed Modeling**: Systematic relaxation of kMC assumptions including phonon-assisted hopping, site energy distinctions, and Haven ratio corrections
- **Multi-Agent Architecture**: Coordinated AI agents for literature review, experimental design, and result analysis
- **Comprehensive Documentation**: Automatic generation of technical reports and research documentation

## Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager
- Git

### Dependencies Installation

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/KOKOA.git
cd KOKOA
```

2. **Install Python dependencies**

Most dependencies can be installed via requirements.txt:

```bash
pip install -r requirements.txt
```

⚠️ **Important Note on PyTorch Installation**

For `torch`, it's highly recommended to install it separately according to your system's CUDA version to avoid dependency conflicts. Visit the [PyTorch official website](https://pytorch.org/get-started/locally/) and select the appropriate installation command for your system (CPU or GPU with specific CUDA version).

Example for CUDA 12.6:
```bash
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu126
```

Example for CPU only:
```bash
pip3 install torch torchvision
```

### Environment Setup

Before running `main.py`, you must create a `.env` file in the project root directory with the required API keys.

**Create `.env` file:**

```bash
# KOKOA Environment Variables
# 필수
# Tavily Search API (AI-optimized web search)
TAVILY_API_KEY=

# OpenAI
OPENAI_API_KEY=

# Gemini
GEMINI_API_KEY=

# 모니터링 시 권장
# LangChain
LANGCHAIN_TRACING_V2=
LANGCHAIN_ENDPOINT=
LANGCHAIN_API_KEY=
LANGCHAIN_PROJECT=
```

**Required API Keys:**
- **TAVILY_API_KEY**: For AI-optimized web search functionality
- **OPENAI_API_KEY**: For OpenAI models (e.g., gpt-5.1-2025-11-13)
- **GEMINI_API_KEY**: For Google Gemini models

**Optional (Recommended for Monitoring):**
- **LANGCHAIN_TRACING_V2**: Enable LangChain tracing (set to `true`)
- **LANGCHAIN_ENDPOINT**: LangChain API endpoint
- **LANGCHAIN_API_KEY**: LangChain API key
- **LANGCHAIN_PROJECT**: Project name for LangChain monitoring

**Additional Configuration:**
- **Ollama**: Ensure Ollama is running locally (default: http://localhost:11434)

## Quick Start

### Basic Usage

Run KOKOA with the recommended model:

```bash
python main.py -m gpt-5.1-2025-11-13
```

or

```bash
python main.py --model gpt-5.1-2025-11-13
```

### Supported Models

KOKOA currently supports the following models:

- `gpt-oss:120b` (Ollama)
- `gemini-2.5-pro` (Google Gemini)
- `gemini-3-flash-preview` (Google Gemini)
- `gpt-5.1-2025-11-13` (OpenAI)

To add support for additional models, edit `config.py` and add the model name to the `SUPPORTED_MODELS` list:

```python
SUPPORTED_MODELS = [
    "gpt-oss:120b",
    "gemini-2.5-pro",
    "gemini-3-flash-preview",
    "gpt-5.1-2025-11-13",
    "your-new-model-name"  # Add your model here
]
```

### Command Line Options

```bash
python main.py --help
```

Common options:
- `-m, --model MODEL`: Specify the LLM model to use
- `--endpoint ENDPOINT`: API endpoint (ollama/gemini/openai)
- `--temperature TEMP`: Sampling temperature (default: 0.3)
- `--max-iterations N`: Maximum research iterations

## Tools and Utilities

KOKOA includes several utility scripts in the `tools/` directory for managing the research database and document stores.

### Paper Store Initialization

Initialize or reset the vector store for scientific papers:

```bash
python tools/init_paper_store.py --force
```

This will:
1. Clear the existing paper vector store
2. Parse all PDF files in `initial_state/pdf/` using PyMuPDF4LLM
3. Split documents using Recursive text splitting
4. Create embeddings using the BAAI/bge-m3 model
5. Store in the vector database for semantic search

### Technical Report Management

**Initialize/Reset Technical Report Store:**

```bash
python tools/init_technical_report_store.py --force
```

**Selectively Delete Technical Reports:**

```bash
python tools/init_technical_report_store.py --select
```

This opens an interactive menu to select and delete specific technical reports.

### Viewing Memory and Reports

**View Technical Reports in HTML Format:**

```bash
python tools/view_memory.py
```

This generates a well-formatted HTML view of all stored technical reports for easy monitoring and review of the AI agent's research progress.

## Project Structure

```
KOKOA/
├── main.py                      # Main entry point
├── config.py                    # Configuration and model settings
├── requirements.txt             # Python dependencies
├── agents/                      # AI agent implementations
│   ├── researcher.py           # Research coordination agent
│   ├── literature_reviewer.py  # Literature search agent
│   └── experimenter.py         # Experiment design agent
├── tools/                       # Utility scripts
│   ├── init_paper_store.py    # Paper database initialization
│   ├── init_technical_report_store.py  # Report database management
│   └── view_memory.py          # Report viewer
├── initial_state/              # Initial research materials
│   └── pdf/                    # Scientific papers for indexing
├── simulation/                 # kMC simulation code
│   ├── lattice.py             # Crystal lattice definitions
│   ├── hopping.py             # Hopping mechanisms
│   └── kmc_engine.py          # Monte Carlo engine
├── data/                       # Experimental data and results
│   ├── vector_stores/         # Embeddings databases
│   └── technical_reports/     # Generated reports
└── docs/                       # Documentation
    └── template_2026.tex      # Paper template
```

## Research Workflow

1. **Initialization**: The AI agent loads existing knowledge from the paper vector store
2. **Hypothesis Generation**: Based on literature review, the agent identifies research gaps
3. **Experimental Design**: Systematic experiments are designed to test hypotheses
4. **Simulation**: kMC simulations are executed with modified assumptions
5. **Analysis**: Results are analyzed and compared with experimental data
6. **Documentation**: Technical reports are automatically generated
7. **Iteration**: The cycle repeats with refined hypotheses

## Key Research Components

### kMC Assumption Relaxations

The project systematically tests various physics corrections:

- **A2**: Phonon-assisted hopping mechanisms
- **A4**: Site energy distinctions
- **A10**: Haven ratio corrections for correlated motion
- **Combined approaches**: Multi-mechanism models (e.g., A2+A10)

### Target Material

- **System**: Li₇La₃Zr₂O₁₂ (LLZO) cubic garnet structure
- **Property**: Lithium-ion conductivity at 298K
- **Validation**: Awaka et al. (2009) experimental data

## Output Files

### Generated Artifacts

- **Simulation Results**: Stored in `runs/[timestamp]/simulation/`
- **Technical Reports**: JSON and Markdown format in `data/technical_reports/`
- **Manuscripts**: LaTeX format following template_2026.sty

### Reproducibility

All simulations include:
- Complete parameter documentation
- Version-controlled code
- Execution timestamps
- Statistical significance metrics (R², RMSE, standard deviations)

## Advanced Usage

### Custom Experiments

Define custom experiments by modifying the experiment configuration:

```python
# In your custom script
from agents.experimenter import ExperimentDesigner

designer = ExperimentDesigner()
experiment = designer.design_experiment(
    hypothesis="Custom hypothesis",
    parameters={
        "temperature": 300,
        "assumption_to_test": "A2",
        # ... additional parameters
    }
)
```

### Batch Processing

Run multiple experiments in sequence:

```bash
python main.py --batch experiments.json
```

## Performance Notes

- **Embedding Generation**: Initial paper store creation may take 10-30 minutes depending on the number of PDFs
- **Simulation Runtime**: Each kMC simulation takes approximately 5-15 minutes
- **Memory Requirements**: Minimum 8GB RAM recommended, 16GB for large document sets
- **GPU Acceleration**: Optional but recommended for faster simulations and embeddings

## Troubleshooting

### Common Issues

1. **PyTorch Installation Errors**
   - Solution: Install PyTorch separately as described in Dependencies Installation

2. **API Rate Limits**
   - Solution: Implement rate limiting or use local models via Ollama

3. **Vector Store Corruption**
   - Solution: Re-run `python tools/init_paper_store.py --force`

4. **Memory Errors During Embedding**
   - Solution: Reduce batch size in configuration or process PDFs in smaller batches

### Debug Mode

Enable verbose logging:

```bash
python main.py --debug --log-level DEBUG
```

## Contributing

This is a research project for the 2026 AI Co-Scientist Challenge Korea. For questions or collaboration inquiries, please contact the development team.

## Citation

If you use KOKOA in your research, please cite:

```bibtex
@article{kokoa2026,
  title={KOKOA: Knowledge-driven Optimization of Kinetic Monte Carlo with Organized AI Agents},
  author={[Authors]},
  journal={2026 AI Co-Scientist Challenge Korea},
  year={2026}
}
```

## License

[Specify your license here]

## Acknowledgments

- POSTECH Department of Materials Science and Engineering
- 2026 AI Co-Scientist Challenge Korea organizing committee
- Contributors to the scientific literature on LLZO solid electrolytes

## References

Key scientific references:
- Awaka et al. (2009) - Experimental LLZO conductivity data
- [Additional references as appropriate]

---

**Project Status**: Active Development for 2026 AI Co-Scientist Challenge Korea

**Last Updated**: February 2026
