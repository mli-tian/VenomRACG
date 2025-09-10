# Exploring the Security Threats of Retriever Backdoors in Retrieval-Augmented Code Generation

This project investigates security vulnerabilities in **retriever backdoors** for **Retrieval-Augmented Code Generation ** systems. The goal is to study how data poisoning and knowledge base poisoning can influence the output of RACG models.

## Project Structure
```
├── data_process
│   ├── knowledge_base_process
│   │   ├── attack_util.py
│   │   ├── generate_poisoned_knowledge_base.py
│   │   └── importance_metadata
│   │       ├── type_idf_values.png
│   │       ├── type_softmax_weights.png
│   │       ├── type_statistics.csv
│   │       ├── type_total_counts.png
│   │       ├── TYPE_WEIGHTS.json
│   │       ├── type_weights.py
│   │       └── utils.py
│   └── training_data_process
│       ├── attack_util.py
│       ├── poison_data.py
│       ├── preprocess_data.py
│       └── target_trigger_selection
│           ├── find_tok10_target.py
│           └── trigger_score.py
└── RACG
    ├── Generator
    │   └── run_rag_pattern_api.py
    └── Retriever
        ├── dense_retriever.py
        ├── dpr/
        ├── generate_dense_embeddings.py
        ├── setup.py
        └── train_dense_encoder.py

```

The codebase is organized into two main parts:

### 1. `data_process/`

- Contains all data processing scripts.
- Supports **fine-tuning data poisoning** and **knowledge base poisoning**.
- Prepares both **poisoned** and **clean** datasets for downstream RACG experiments.

Subfolders:

#### (1) `training_data_process/`
- Scripts for generating and processing **training/fine-tuning datasets**.
- Includes functions for injecting backdoor triggers into the training code.

#### (2) `knowledge_base_process/`
- Scripts for processing and poisoning the **knowledge base** used by the RACG system.
- Prepares knowledge snippets for retrieval experiments.

### 2. `RACG/`

- Contains the complete RACG execution pipeline.
- Handles retrieval, generation, and evaluation of code snippets using both clean and poisoned data.

Subfolders:

#### (1) `Retriever/`
- Implements the **code retriever** modules.
- Supports dense retrieval using models like CodeBERT.
- Generates retrieval results for given queries.

#### (2) `Generator/`
- Implements the **code generation** modules.
- Takes retrieval results as input and generates final code snippets using LLMs.
- Supports experiments on backdoor-triggered code generation.



