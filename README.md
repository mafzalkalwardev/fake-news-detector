# 🔍 FakeScope — AI Fake News Detection System

> A complete machine-learning pipeline to detect fake news using NLP and scikit-learn, served via a sleek Flask web app.

---

## 📸 Screenshots

| Home / Input | Analysis Result |
|---|---|
| *(screenshot)* | *(screenshot)* |

---

## 🚀 Quick Start

```bash
# 1. Clone / extract project
cd fake-news-detector

# 2. Install dependencies
pip install -r requirements.txt

# 3. Train models (auto-generates dataset if not present)
python train.py

# 4. Launch web app
python app.py
# → open http://127.0.0.1:5000
```

---

## 🗂️ Project Structure

```
fake-news-detector/
├── dataset/
│   ├── generate_dataset.py   # Synthetic dataset generator
│   └── news_dataset.csv      # Generated after first run
├── models/
│   ├── best_model.pkl         # Best trained model
│   ├── vectorizer.pkl         # TF-IDF vectorizer
│   ├── logistic_regression.pkl
│   ├── naive_bayes.pkl
│   ├── random_forest.pkl
│   └── metrics_summary.pkl
├── static/
│   ├── comparison_chart.png   # Model comparison bar chart
│   ├── confusion_matrices.png # 3-panel confusion matrices
│   └── performance_overview.png
├── preprocess.py              # Text cleaning pipeline
├── train.py                   # Training + evaluation pipeline
├── app.py                     # Flask web application
├── requirements.txt
└── README.md
```

---

## 🧠 How It Works

### 1 · Dataset
- **1,000 samples** (500 REAL / 500 FAKE) generated from curated real-world patterns
- Each sample has `title`, `text`, and `label` columns
- **Production dataset**: [Kaggle Fake News Dataset](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset) (replace `dataset/news_dataset.csv`)

### 2 · Preprocessing
| Step | Detail |
|---|---|
| Lowercase | Normalize case |
| URL removal | Strip `http://…` links |
| Punctuation removal | Keep only `[a-z\s]` |
| Tokenization | Whitespace split |
| Stopword removal | 170+ English stopwords |
| Stemming | Suffix-based fast stemmer |

### 3 · Feature Extraction
- **TF-IDF Vectorizer** with 15,000 features
- **Bigrams** (1,2) for phrase-level patterns
- `sublinear_tf=True` to dampen high-frequency terms

### 4 · Models Trained

| Model | Typical Accuracy | Notes |
|---|---|---|
| Logistic Regression | ~94-97% | Fast, great baseline |
| Naive Bayes (Multinomial) | ~91-95% | Excellent for text |
| Random Forest | ~93-96% | Ensemble, most robust |

### 5 · Evaluation Metrics
- Accuracy, Precision, Recall, F1-Score
- Confusion Matrix per model
- Side-by-side bar charts + heatmap

### 6 · Prediction
- Input: raw text (headline or full article)
- Output: `FAKE` / `REAL` + confidence % + probability bar

---

## 🌐 API

```bash
POST /predict
Content-Type: application/json
{ "text": "News text here…" }

Response:
{
  "label": "FAKE",
  "confidence": 96.3,
  "fake_prob": 96.3,
  "real_prob": 3.7,
  "cleaned_tokens": 14
}
```

```bash
GET /health   →  { "status": "ok", "model": "Logistic Regression" }
```

---

## 🔬 Key Fake News Signals (learned by models)

Fake news tends to include:
- ALL-CAPS words, excessive punctuation (`!!!`)
- Phrases: *"they don't want you to know"*, *"share before deleted"*
- Appeals to anonymous sources / whistleblowers
- Conspiracy-adjacent terms: *globalist*, *deep state*, *cover-up*

Real news tends to include:
- Named institutions (MIT, FDA, CDC, IPCC)
- Specific statistics and dates
- Attributed quotes to named individuals

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| Language | Python 3.10+ |
| ML | scikit-learn |
| Data | pandas, numpy |
| NLP | Custom preprocessor (NLTK-compatible) |
| Visualization | matplotlib, seaborn |
| Web | Flask |
| Persistence | joblib |

---

## 🔮 Future Improvements

- [ ] Deep learning: LSTM / BERT fine-tuning for higher accuracy
- [ ] Real-time news scraping for live fact-checking
- [ ] Explainability: SHAP values showing which words triggered verdict
- [ ] Multi-language support
- [ ] Browser extension for in-line news checking
- [ ] User feedback loop to continuously improve model

---

## 📄 License
MIT — free to use, modify, and distribute.

---

## About the Developer

**Muhammad Afzal Kalwar** — Full-Stack Developer & Automation Engineer  
GitHub: [@mafzalkalwardev](https://github.com/mafzalkalwardev) · Portfolio: [mafzalkalwardev.github.io](https://mafzalkalwardev.github.io)

<details>
<summary>SEO Keywords</summary>
Muhammad Afzal Kalwar, mafzalkalwardev, fake news detection Python, Flask ML NLP, scikit-learn text classification Pakistan
</details>

---

<div align="center"><sub>Built by Muhammad Afzal Kalwar · FT Solutions</sub></div>
