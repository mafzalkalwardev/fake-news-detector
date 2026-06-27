"""
app.py — Flask Web Application for Fake News Detection
Serves the prediction UI and handles API requests.
"""

import os, sys, json
import joblib
import numpy as np
from flask import Flask, request, jsonify, render_template_string

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR  = os.path.join(BASE_DIR, "models")
STATIC_DIR = os.path.join(BASE_DIR, "static")

app = Flask(__name__, static_folder=STATIC_DIR)

# ── Load model + vectorizer ────────────────────────────────────────────────
def load_artifacts():
    model_path = os.path.join(MODEL_DIR, "best_model.pkl")
    vec_path   = os.path.join(MODEL_DIR, "vectorizer.pkl")
    if not os.path.exists(model_path) or not os.path.exists(vec_path):
        print("Models not found. Running training pipeline…")
        sys.path.insert(0, BASE_DIR)
        import train
        train.main()
    model      = joblib.load(model_path)
    vectorizer = joblib.load(vec_path)
    model_name = joblib.load(os.path.join(MODEL_DIR, "best_model_name.pkl"))
    try:
        metrics = joblib.load(os.path.join(MODEL_DIR, "metrics_summary.pkl"))
    except Exception:
        metrics = {}
    return model, vectorizer, model_name, metrics

MODEL, VECTORIZER, MODEL_NAME, METRICS = load_artifacts()

# ── Preprocessing (inline — avoids import issues in some envs) ────────────
import re

STOPWORDS = {
    "i","me","my","myself","we","our","ours","ourselves","you","your","yours",
    "yourself","yourselves","he","him","his","himself","she","her","hers",
    "herself","it","its","itself","they","them","their","theirs","themselves",
    "what","which","who","whom","this","that","these","those","am","is","are",
    "was","were","be","been","being","have","has","had","having","do","does",
    "did","doing","a","an","the","and","but","if","or","because","as","until",
    "while","of","at","by","for","with","about","against","between","into",
    "through","during","before","after","above","below","to","from","up","down",
    "in","out","on","off","over","under","again","further","then","once","here",
    "there","when","where","why","how","all","both","each","few","more","most",
    "other","some","such","no","nor","not","only","own","same","so","than",
    "too","very","s","t","can","will","just","don","should","now","d","ll",
    "m","o","re","ve","y","ain","aren","couldn","didn","doesn","hadn","hasn",
    "haven","isn","ma","mightn","mustn","needn","shan","shouldn","wasn","weren",
    "won","wouldn","said","say","says","also","would","could","one","like",
    "get","got","make","made","know","think","time","year","new","may","even",
    "well","way","back","see","go","come","people","take","use","good","give",
    "look","want","seem","help","show","put","keep","last","let","large",
    "end","need","long","hand","place","big","right","high","something","tell",
    "every","found","still","us","set","mr","mrs","ms","dr",
}

def simple_stem(w):
    for sfx in ["ing","tion","tions","ness","ment","ments","ful","ous","ious",
                "er","ers","est","ed","ly","ies","es","s"]:
        if w.endswith(sfx) and len(w) - len(sfx) > 3:
            return w[:-len(sfx)]
    return w

def clean(text):
    if not text: return ""
    text = text.lower()
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    tokens = [simple_stem(t) for t in text.split()
              if t not in STOPWORDS and len(t) > 2]
    return " ".join(tokens)


def predict(text: str):
    cleaned = clean(text)
    vec     = VECTORIZER.transform([cleaned])
    pred    = MODEL.predict(vec)[0]
    proba   = MODEL.predict_proba(vec)[0] if hasattr(MODEL, "predict_proba") else None
    label   = "FAKE" if pred == 1 else "REAL"
    conf    = float(max(proba)) * 100 if proba is not None else 0.0
    fake_p  = float(proba[1]) * 100 if proba is not None else (100 if pred==1 else 0)
    real_p  = float(proba[0]) * 100 if proba is not None else (0 if pred==1 else 100)
    return {"label": label, "confidence": round(conf, 1),
            "fake_prob": round(fake_p, 1), "real_prob": round(real_p, 1),
            "cleaned_tokens": len(cleaned.split())}


# ── HTML Template ──────────────────────────────────────────────────────────
HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>FakeScope — AI Fake News Detector</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Space+Mono:ital,wght@0,400;0,700;1,400&family=Syne:wght@400;600;700;800&display=swap" rel="stylesheet">
<style>
  :root {
    --bg:       #070b0f;
    --surface:  #0d1117;
    --card:     #161b22;
    --border:   #21262d;
    --accent:   #58a6ff;
    --green:    #3fb950;
    --red:      #f85149;
    --amber:    #e3b341;
    --muted:    #8b949e;
    --text:     #e6edf3;
    --radius:   10px;
  }
  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
  html { scroll-behavior: smooth; }
  body {
    background: var(--bg);
    color: var(--text);
    font-family: 'Syne', sans-serif;
    min-height: 100vh;
    line-height: 1.6;
  }

  /* ── grid noise texture overlay ── */
  body::before {
    content: "";
    position: fixed; inset: 0; z-index: 0; pointer-events: none;
    background-image:
      repeating-linear-gradient(0deg, transparent, transparent 40px, rgba(88,166,255,.02) 40px, rgba(88,166,255,.02) 41px),
      repeating-linear-gradient(90deg, transparent, transparent 40px, rgba(88,166,255,.02) 40px, rgba(88,166,255,.02) 41px);
  }

  .container { max-width: 900px; margin: 0 auto; padding: 0 1.5rem; position: relative; z-index: 1; }

  /* ── header ── */
  header {
    padding: 2.5rem 0 1.5rem;
    border-bottom: 1px solid var(--border);
    display: flex; align-items: center; gap: 1rem;
  }
  .logo-icon {
    width: 44px; height: 44px;
    background: linear-gradient(135deg, var(--accent), #a5d6ff);
    border-radius: 10px;
    display: flex; align-items: center; justify-content: center;
    font-size: 1.3rem; flex-shrink: 0;
    box-shadow: 0 0 20px rgba(88,166,255,.35);
  }
  header h1 {
    font-size: 1.6rem; font-weight: 800; letter-spacing: -0.5px;
  }
  header h1 span { color: var(--accent); }
  header p { color: var(--muted); font-size: .85rem; margin-top: 2px; font-family: 'Space Mono', monospace; }

  /* ── model badge ── */
  .model-badge {
    margin-left: auto; background: var(--card); border: 1px solid var(--border);
    border-radius: 6px; padding: .3rem .75rem; font-family: 'Space Mono', monospace;
    font-size: .72rem; color: var(--muted);
  }
  .model-badge span { color: var(--accent); }

  /* ── main card ── */
  .card {
    background: var(--card); border: 1px solid var(--border);
    border-radius: var(--radius); padding: 2rem; margin: 2rem 0 1.5rem;
  }

  .input-label {
    font-size: .8rem; font-weight: 700; letter-spacing: .08em; text-transform: uppercase;
    color: var(--muted); margin-bottom: .6rem; display: flex; align-items: center; gap: .4rem;
  }
  .input-label::before { content: "//"; color: var(--accent); font-family: 'Space Mono', monospace; }

  textarea {
    width: 100%; min-height: 160px; padding: 1rem;
    background: var(--surface); border: 1px solid var(--border);
    border-radius: var(--radius); color: var(--text);
    font-family: 'Syne', sans-serif; font-size: .95rem;
    resize: vertical; transition: border-color .2s, box-shadow .2s;
    outline: none;
  }
  textarea:focus { border-color: var(--accent); box-shadow: 0 0 0 3px rgba(88,166,255,.12); }
  textarea::placeholder { color: var(--muted); }

  .btn-row { display: flex; gap: .75rem; margin-top: 1.2rem; align-items: center; }
  button {
    padding: .7rem 1.8rem; border: none; border-radius: var(--radius);
    font-family: 'Space Mono', monospace; font-size: .85rem; font-weight: 700;
    cursor: pointer; transition: all .18s; letter-spacing: .04em;
  }
  .btn-primary {
    background: var(--accent); color: #000;
    box-shadow: 0 0 18px rgba(88,166,255,.35);
  }
  .btn-primary:hover { background: #79b8ff; transform: translateY(-1px); box-shadow: 0 4px 20px rgba(88,166,255,.45); }
  .btn-primary:active { transform: translateY(0); }
  .btn-secondary {
    background: transparent; color: var(--muted);
    border: 1px solid var(--border);
  }
  .btn-secondary:hover { color: var(--text); border-color: var(--muted); }
  .char-counter { margin-left: auto; font-family: 'Space Mono', monospace; font-size: .75rem; color: var(--muted); }

  /* ── result ── */
  #result { display: none; }
  .result-card {
    background: var(--card); border-radius: var(--radius);
    border: 1px solid var(--border); overflow: hidden;
    animation: slideIn .35s ease;
  }
  @keyframes slideIn { from { opacity:0; transform:translateY(12px); } to { opacity:1; transform:none; } }

  .result-header {
    padding: 1.5rem 2rem;
    display: flex; align-items: center; gap: 1.2rem;
  }
  .result-header.fake { border-left: 4px solid var(--red);   background: rgba(248,81,73,.06); }
  .result-header.real { border-left: 4px solid var(--green); background: rgba(63,185,80,.06); }

  .verdict-icon { font-size: 2.4rem; }
  .verdict-label {
    font-size: 1.8rem; font-weight: 800; letter-spacing: -1px;
  }
  .verdict-label.fake { color: var(--red); }
  .verdict-label.real { color: var(--green); }
  .verdict-sub { color: var(--muted); font-size: .82rem; margin-top: 2px; font-family: 'Space Mono', monospace; }

  .result-body { padding: 1.5rem 2rem; border-top: 1px solid var(--border); }
  .metrics-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 1rem; }

  .metric-box {
    background: var(--surface); border-radius: 8px;
    padding: .9rem 1rem; border: 1px solid var(--border);
  }
  .metric-title { font-size: .7rem; text-transform: uppercase; letter-spacing: .1em; color: var(--muted); margin-bottom: .4rem; }
  .metric-value { font-size: 1.6rem; font-weight: 800; font-family: 'Space Mono', monospace; }
  .metric-value.fake { color: var(--red); }
  .metric-value.real { color: var(--green); }
  .metric-value.neutral { color: var(--accent); }

  /* probability bar */
  .prob-section { margin-top: 1.2rem; }
  .prob-label { font-size: .75rem; text-transform: uppercase; letter-spacing: .08em; color: var(--muted); margin-bottom: .6rem; }
  .prob-bar-outer {
    background: var(--surface); border-radius: 4px; height: 24px; overflow: hidden;
    border: 1px solid var(--border); display: flex;
  }
  .prob-bar-fake { background: linear-gradient(90deg, var(--red), #ff6961); height: 100%; transition: width .6s ease; display: flex; align-items: center; justify-content: flex-end; padding-right: .4rem; }
  .prob-bar-real { background: linear-gradient(90deg, var(--green), #56d364); height: 100%; flex: 1; display: flex; align-items: center; padding-left: .4rem; }
  .prob-bar-text { font-family: 'Space Mono', monospace; font-size: .7rem; font-weight: 700; color: #000; }

  /* ── stats grid ── */
  .stats-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 1rem; margin-bottom: 1.5rem; }
  .stat-card {
    background: var(--card); border: 1px solid var(--border); border-radius: var(--radius);
    padding: 1.2rem; text-align: center;
  }
  .stat-value { font-size: 1.5rem; font-weight: 800; color: var(--accent); font-family: 'Space Mono', monospace; }
  .stat-label { font-size: .72rem; color: var(--muted); text-transform: uppercase; letter-spacing: .07em; margin-top: .25rem; }

  /* ── charts ── */
  .charts-section { margin-top: 1.5rem; }
  .charts-section h3 { font-size: .9rem; font-weight: 700; color: var(--muted); text-transform: uppercase; letter-spacing: .08em; margin-bottom: 1rem; }
  .chart-img { width: 100%; border-radius: 8px; border: 1px solid var(--border); margin-bottom: 1rem; }

  /* ── examples ── */
  .examples { margin-top: 1rem; }
  .examples h3 { font-size: .75rem; font-weight: 700; color: var(--muted); text-transform: uppercase; letter-spacing: .1em; margin-bottom: .6rem; }
  .example-chips { display: flex; flex-wrap: wrap; gap: .5rem; }
  .chip {
    padding: .35rem .75rem; border-radius: 20px; font-size: .78rem;
    cursor: pointer; border: 1px solid; transition: all .15s; font-family: 'Space Mono', monospace;
  }
  .chip.real { border-color: var(--green); color: var(--green); background: rgba(63,185,80,.08); }
  .chip.fake { border-color: var(--red);   color: var(--red);   background: rgba(248,81,73,.08); }
  .chip:hover { opacity: .75; transform: scale(.97); }

  /* ── spinner ── */
  .spinner {
    width: 20px; height: 20px; border: 2px solid rgba(88,166,255,.2);
    border-top-color: var(--accent); border-radius: 50%;
    animation: spin .6s linear infinite; display: none;
  }
  @keyframes spin { to { transform: rotate(360deg); } }

  footer { text-align: center; padding: 2rem 0; color: var(--muted); font-size: .75rem; font-family: 'Space Mono', monospace; border-top: 1px solid var(--border); margin-top: 2rem; }
  footer a { color: var(--accent); text-decoration: none; }

  @media (max-width: 600px) {
    .metrics-grid, .stats-grid { grid-template-columns: 1fr 1fr; }
    .model-badge { display: none; }
    .verdict-label { font-size: 1.4rem; }
  }
</style>
</head>
<body>
<div class="container">

  <header>
    <div class="logo-icon">🔍</div>
    <div>
      <h1>Fake<span>Scope</span></h1>
      <p>AI-powered news authenticity analysis</p>
    </div>
    <div class="model-badge">model: <span>{{ model_name }}</span></div>
  </header>

  <!-- Stats bar -->
  {% if metrics %}
  <div class="stats-grid" style="margin-top:2rem">
    {% for name, m in metrics.items() %}
    <div class="stat-card">
      <div class="stat-value">{{ "%.1f"|format(m.acc * 100) }}%</div>
      <div class="stat-label">{{ name }}</div>
    </div>
    {% endfor %}
  </div>
  {% endif %}

  <!-- Main input card -->
  <div class="card">
    <div class="input-label">Input News Text or Headline</div>
    <textarea id="newsInput" placeholder="Paste a news headline or article here…&#10;&#10;Example: 'Government secretly putting mind-control chemicals in tap water, whistleblower reveals'"></textarea>
    <div class="btn-row">
      <button class="btn-primary" onclick="analyze()">⚡ Analyze</button>
      <button class="btn-secondary" onclick="clearInput()">Clear</button>
      <div class="spinner" id="spinner"></div>
      <div class="char-counter" id="charCount">0 chars</div>
    </div>
    <!-- Examples -->
    <div class="examples" style="margin-top:1.4rem">
      <h3>Try an example</h3>
      <div class="example-chips">
        <div class="chip real" onclick="setExample(this.textContent)">Scientists discover new antibiotic at MIT</div>
        <div class="chip real" onclick="setExample(this.textContent)">Federal Reserve raises interest rates amid inflation</div>
        <div class="chip fake" onclick="setExample(this.textContent)">BREAKING: vaccines contain mind-control microchips activated by 5G</div>
        <div class="chip fake" onclick="setExample(this.textContent)">Secret cure for cancer suppressed by Big Pharma for 50 years</div>
      </div>
    </div>
  </div>

  <!-- Result area -->
  <div id="result">
    <div class="result-card">
      <div class="result-header" id="resultHeader">
        <div class="verdict-icon" id="verdictIcon"></div>
        <div>
          <div class="verdict-label" id="verdictLabel"></div>
          <div class="verdict-sub" id="verdictSub"></div>
        </div>
      </div>
      <div class="result-body">
        <div class="metrics-grid">
          <div class="metric-box">
            <div class="metric-title">Confidence</div>
            <div class="metric-value neutral" id="confValue">—</div>
          </div>
          <div class="metric-box">
            <div class="metric-title">Fake Score</div>
            <div class="metric-value fake" id="fakeValue">—</div>
          </div>
          <div class="metric-box">
            <div class="metric-title">Real Score</div>
            <div class="metric-value real" id="realValue">—</div>
          </div>
        </div>
        <div class="prob-section">
          <div class="prob-label">Probability Distribution</div>
          <div class="prob-bar-outer">
            <div class="prob-bar-fake" id="fakeBar" style="width:50%">
              <span class="prob-bar-text" id="fakePct">50%</span>
            </div>
            <div class="prob-bar-real" id="realBar">
              <span class="prob-bar-text" id="realPct">50%</span>
            </div>
          </div>
          <div style="display:flex;justify-content:space-between;margin-top:.3rem">
            <span style="font-size:.7rem;color:var(--red);font-family:'Space Mono',monospace">FAKE</span>
            <span style="font-size:.7rem;color:var(--green);font-family:'Space Mono',monospace">REAL</span>
          </div>
        </div>
      </div>
    </div>
  </div>

  <!-- Charts -->
  {% if has_charts %}
  <div class="charts-section">
    <h3>Model Evaluation Charts</h3>
    <img src="/static/comparison_chart.png"   class="chart-img" alt="Model Comparison">
    <img src="/static/confusion_matrices.png" class="chart-img" alt="Confusion Matrices">
    <img src="/static/performance_overview.png" class="chart-img" alt="Performance Overview">
  </div>
  {% endif %}

  <footer>
    FakeScope v1.0 — built with Python · scikit-learn · Flask ·
    <a href="https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset" target="_blank">Dataset</a>
  </footer>
</div>

<script>
  const textarea = document.getElementById("newsInput");
  const charCount = document.getElementById("charCount");
  textarea.addEventListener("input", () => { charCount.textContent = textarea.value.length + " chars"; });

  function setExample(text) {
    textarea.value = text;
    charCount.textContent = text.length + " chars";
    textarea.focus();
  }
  function clearInput() {
    textarea.value = "";
    charCount.textContent = "0 chars";
    document.getElementById("result").style.display = "none";
  }

  async function analyze() {
    const text = textarea.value.trim();
    if (!text) { textarea.focus(); return; }

    document.getElementById("spinner").style.display = "block";
    document.getElementById("result").style.display = "none";

    try {
      const res = await fetch("/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text })
      });
      const data = await res.json();
      if (data.error) { alert("Error: " + data.error); return; }
      renderResult(data);
    } catch(e) {
      alert("Request failed: " + e.message);
    } finally {
      document.getElementById("spinner").style.display = "none";
    }
  }

  function renderResult(d) {
    const isFake = d.label === "FAKE";
    const hdr = document.getElementById("resultHeader");
    hdr.className = "result-header " + (isFake ? "fake" : "real");

    document.getElementById("verdictIcon").textContent  = isFake ? "🚨" : "✅";
    const vl = document.getElementById("verdictLabel");
    vl.textContent  = isFake ? "LIKELY FAKE" : "LIKELY REAL";
    vl.className    = "verdict-label " + (isFake ? "fake" : "real");

    document.getElementById("verdictSub").textContent = "Analyzed by " + {{ model_name | tojson }};
    document.getElementById("confValue").textContent  = d.confidence + "%";
    document.getElementById("fakeValue").textContent  = d.fake_prob + "%";
    document.getElementById("realValue").textContent  = d.real_prob + "%";

    const fp = d.fake_prob;
    document.getElementById("fakeBar").style.width = fp + "%";
    document.getElementById("fakePct").textContent  = fp + "%";
    document.getElementById("realPct").textContent  = d.real_prob + "%";

    document.getElementById("result").style.display = "block";
    document.getElementById("result").scrollIntoView({ behavior: "smooth", block: "nearest" });
  }

  document.addEventListener("keydown", e => {
    if ((e.ctrlKey || e.metaKey) && e.key === "Enter") analyze();
  });
</script>
</body>
</html>"""


@app.route("/")
def index():
    has_charts = all(os.path.exists(os.path.join(STATIC_DIR, f))
                     for f in ["comparison_chart.png", "confusion_matrices.png"])
    # Build metrics display: round values
    display_metrics = {}
    for name, m in METRICS.items():
        display_metrics[name] = {k: round(v, 4) for k, v in m.items()
                                  if isinstance(v, float)}
    return render_template_string(HTML, model_name=MODEL_NAME,
                                  metrics=display_metrics, has_charts=has_charts)


@app.route("/predict", methods=["POST"])
def predict_route():
    try:
        data = request.get_json(force=True)
        text = data.get("text", "").strip()
        if not text:
            return jsonify({"error": "No text provided"}), 400
        result = predict(text)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/health")
def health():
    return jsonify({"status": "ok", "model": MODEL_NAME})


if __name__ == "__main__":
    print(f"\n🔍 FakeScope running — http://127.0.0.1:5000\n")
    app.run(debug=False, host="0.0.0.0", port=5000)
