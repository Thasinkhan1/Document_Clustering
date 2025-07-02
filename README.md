
# 🧠 LDA-Based Document Clustering using Gensim & SpaCy

This project performs **topic modeling** using **Latent Dirichlet Allocation (LDA)** on a collection of documents. It includes a modular pipeline for loading, cleaning, transforming, and modeling large text datasets.

---

## 📂 Project Structure

```
Document-Clustring/
├── project_root/
│   ├── config/
│   │   └── config.py
│   ├── data_cleaning/
│   │   └── cleaned_data.py
│   ├── data_loading/
│   │   └── data_loadings.py
│   ├── data_transformation/
│   │   └── transforming_data.py
│   ├── training/
│   │   └── train.py
│   ├── models/
│   │   └── lda_model.gensim  # Saved model files and all dict,lemmatized.txt as well
│   ├── inference.py
│   └── training_pipeline.py
├── requirements.txt
└── README.md
```

---

## 🚀 Features

- Load and clean custom text data
- Tokenization and lemmatization using **SpaCy**
- Train an **LDA model using Gensim**
- Save and reload trained models
- Perform inference to get topic distributions for new text

---

## 🛠️ Setup Instructions

1. **Clone this repository**
   ```bash
   git clone https://github.com/Thasinkhan1/Document-Clustring.git
   cd Document-Clustring
   ```

2. **Create and activate virtual environment**
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate   # Linux/macOS
   ```

3. **Install required packages**
   ```bash
   pip install -r requirements.txt
   ```

4. **Download and install SpaCy model**
   ```bash
   make sure you dowload english language processing model  before testing the model using below cmd 
   python -m spacy download en_core_web_sm
   ```

---

## 📊 Running the Project

### 🏋️‍♂️ To Train the LDA Model
```bash
python3 project_root/training_pipeline.py
```

### 🔍 To Perform Inference
```bash
python3 project_root/inference.py
```

---

## 📁 Configuration

All configurable paths (data, models, outputs) are stored in:
```python
project_root/config/config.py
```

Update the following variables:
```python
NEWS_PAPER_DATA_PATH = "/path/to/full/dataset"
NEWS_MINI_DATA_PATH = "/path/to/mini/sample"
SAVED_MODEL = "project_root/models/lda_model.gensim"
SAVED_DICT = "project_root/models/dictionary.dict"
```

---

## ✅ Requirements

- Python 3.8+
- Gensim
- SpaCy
- NLTK
- Pandas etc

All dependencies are listed in `requirements.txt`.

---

## 📌 Example Inference Output

```
Tokenized: [['crop', 'leaves', 'yellow', 'fungal']]
Lemmatized: ['crop', 'leave', 'yellow', 'fungal']
BoW: [(288, 1), (739, 1), (9043, 1)]
Predicted Topics:
Topic 5: 0.6303
Topic 3: 0.2693
...
```

---

## 🙋‍♂️ Author

**Thasin Khan**

---

> ⭐ If you like this project, give it a star!  
