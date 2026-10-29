<div align="center">

# Fake News Detector

<img src="https://readme-typing-svg.demolab.com?font=Inter&weight=700&size=28&duration=2800&pause=700&color=EF4444&center=true&vCenter=true&width=900&lines=AI+Fake+News+Detection;Flask+%2B+scikit-learn+NLP;TF-IDF+%2B+Model+Comparison;Interactive+dashboard" alt="Typing SVG" />

<p>
  <img src="https://komarev.com/ghpvc/?username=mafzalkalwardev&label=Project%20Views&color=ef4444&style=for-the-badge" alt="Project views" />
  <img src="https://img.shields.io/github/stars/mafzalkalwardev/fake-news-detector?style=for-the-badge&color=22c55e" alt="GitHub stars" />
  <img src="https://img.shields.io/github/forks/mafzalkalwardev/fake-news-detector?style=for-the-badge&color=f59e0b" alt="GitHub forks" />
  <img src="https://img.shields.io/github/license/mafzalkalwardev/fake-news-detector?style=for-the-badge&color=64748b" alt="License" />
</p>

<p>
  <img src="https://skillicons.dev/icons?i=python,flask" alt="Skill icons" />
</p>

<p>
  <strong>AI fake news detection with Flask, scikit-learn NLP, TF-IDF, and a model comparison dashboard.</strong>
</p>

</div>

---

## Trophy Cards

<div align="center">
  <img src="https://github-profile-trophy.vercel.app/?username=mafzalkalwardev&theme=algolia&no-frame=true&no-bg=true&margin-w=12&margin-h=12&column=4" alt="GitHub trophy cards" />
</div>

---

## Stats Cards

<div align="center">
  <img height="165" src="https://github-readme-stats-sigma-five.vercel.app/api?username=mafzalkalwardev&show_icons=true&theme=tokyonight&hide_border=true&rank_icon=github" alt="GitHub stats" />
  <img height="165" src="https://github-readme-stats-sigma-five.vercel.app/api/top-langs/?username=mafzalkalwardev&layout=compact&theme=tokyonight&hide_border=true" alt="Top languages" />
</div>

---

## Activity Graph

<div align="center">
  <img src="https://github-readme-activity-graph.vercel.app/graph?username=mafzalkalwardev&theme=react-dark&hide_border=true&area=true" alt="GitHub activity graph" />
</div>

---

## Contribution Snake

<div align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/mafzalkalwardev/google-voice-dispatch-agent/output/github-contribution-grid-snake-dark.svg">
    <source media="(prefers-color-scheme: light)" srcset="https://raw.githubusercontent.com/mafzalkalwardev/google-voice-dispatch-agent/output/github-contribution-grid-snake.svg">
    <img alt="Contribution snake" src="https://raw.githubusercontent.com/mafzalkalwardev/google-voice-dispatch-agent/output/github-contribution-grid-snake.svg">
  </picture>
</div>

---

## SEO Keywords

`fake news` `NLP` `Flask` `TF-IDF`

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
