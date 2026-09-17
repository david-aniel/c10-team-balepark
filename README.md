# c10-balepark
AgriAdviser-NG

### 🌾 Project Overview

AgriAdviser-NG is a domain-specific small language model (SLM) project that provides climate-aware agronomic advisory to smallholder rice and maize farmers in Nigeria. Farmers interact with the system via short, informal text questions (Nigerian English) sent over low-bandwidth channels, and receive concise, source-grounded advice, or an explicit "I'm not confident, contact your extension officer" response when no reliable match exists in the retrieval layer.

The project fine-tunes a small open-weight model (Gemma-2-2b-it) using LoRA on a curated corpus of agronomic Q&A, paired with a custom retrieval layer that matches incoming questions to the most relevant source document before generation.

### 🎯 Objectives
Build a domain Q&A dataset from verified agricultural extension sources, tagged by topic, crop, and agro-ecological zone.
Fine-tune a small language model to generate concise, accurate advisory answers grounded in retrieved context.
Evaluate generation quality against held-out reference answers using character-level similarity.
Document the dataset's provenance, limitations, and ethical considerations transparently (see docs/).

### 📊 Dataset

The dataset consists of short-form agronomic Q&A pairs and reference source documents, organized by topic (crop_diseases, pests, soil_health, climate_adaptation, water_management, fertiliser, livestock, post_harvest), crop, and agro_zone.

Two complementary sources make up the corpus:

Real extraction: Passages paraphrased and attributed from official extension publications - IITA field guides (cassava pest/disease control, cowpea pests and diseases), NAERLS farmer manuals, and related agronomic bulletins, each row tagged with a source citation.
Synthetic generation: Where authentic farmer-voice question data was unavailable, question paraphrases were generated from the same verified source content, varying phrasing and formality, to help the model recognize informal, symptom-first questions rather than only formally worded ones.

Every row's origin field distinguishes synthetic from extracted content, so provenance is never ambiguous. The dataset was expanded iteratively, starting from an initial 24 documents / 45 QA pairs and growing to 76 documents and 280 QA pairs - deliberately targeting crop and topic gaps (notably beans/cowpea, cassava, sorghum, and groundnut, which were underrepresented in the initial set) rather than expanding uniformly. Full sourcing rationale and known coverage gaps are documented in docs/data_card.pdf.

### 🛠️ Training Pipeline

Preprocessing: Source documents and QA pairs are validated for referential integrity (every QA row must reference an existing document; every document must be used by at least one QA row) before training.

Retrieval layer: A custom scoring function matches each question to its most relevant source document using a weighted combination of categorical metadata (topic, crop, agro-ecological zone) and keyword overlap between the question and candidate document text, rather than relying on topic-matching alone.

Prompt construction: Each training example follows the format Crop | Zone | Topic / Question / Context / Answer, where Context is the full matched source document (not truncated), and Answer is the verified reference answer.

Fine-tuning: LoRA adapters (rank and target-module configuration tuned via sweep) are applied to Gemma-2-2b-it using Hugging Face trl's SFTTrainer, with a fixed random seed across model initialization and batch shuffling to ensure reproducible comparisons between configurations.

Hyperparameter search: Epoch count, LoRA rank/alpha, and target-module scope (attention-only vs. attention + MLP projections) were swept systematically, with each configuration trained from a fresh base model to avoid adapter-stacking artifacts, and freed from GPU memory between runs.

### 📈 Evaluation

Model outputs are evaluated on a held-out split of the training set (never seen during fine-tuning) using character-level Levenshtein distance and similarity between generated and reference answers, the same metric used for final scoring. Generation is deterministic (greedy decoding) with repetition penalties and explicit stop-sequence truncation to prevent the model from generating hallucinated follow-up questions or context beyond the intended answer. Hyperparameter configurations are compared strictly by holdout performance, not training loss, to avoid selecting an overfit configuration.

### ▶️ Reproduction
Install dependencies: pip install -q "trl<0.12.0" "transformers<4.46.0" peft accelerate datasets

Run scripts/balepark-s-agriculture-climate-slm.ipynb — handles both local and Kaggle-style input mounting and training pipeline.

To run the training sweep, uncomment the training sweep code cell in scripts/balepark-s-agriculture-climate-slm.ipynb to compare hyperparameter configurations via holdout Levenshtein score.

Train the final model with the best configuration and generate predictions commenting the training sweep cell again.

Outputs are validated (column names, row count, ID alignment, no missing answers) before being written as the final result.

This project is currently notebook-based (developed and run as a Kaggle-style notebook); a standalone script/app deployment has not yet been built.

📁 Repository Structure

C10-team-balepark/

├── README.md

├── docs/

    └── problem_statement.pdf

    └── data_card.pdf

    └── impact_statement_card.pdf

    └── stakeholder_engagement.pdf

├── scripts/

    └── balepark-s-agriculture-climate-slm.ipynb

├── data/

    └── new_documents_4.csv
    
    └── new_train_qa_4.csv
    
### 👥 Appendix — Contributors

Team: balepark

Team Members:

Femi Ajanaku


Mentors:

Samuel Taiwo,
John Evan

### 🔗 References


IITA (International Institute of Tropical Agriculture) — extension field guides and manuals

NAERLS (National Agricultural Extension and Research Liaison Services)

AfricaRice, NiMet, state Agricultural Development Programmes
