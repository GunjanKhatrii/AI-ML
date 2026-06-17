💎 Aufgabe 2 — Spotify Genre Classification with SVM
Course: Künstliche Intelligenz & Maschinelles Lernen (KI & ML)
Task Type: Classification Problem
Final Score: MCC = 0.709 🏆

---

## 📌 Task Description

In this assignment, I acted as a Junior Machine Learning Engineer hired by **Spotify**.
The goal was to train a Support Vector Machine (SVM) that automatically classifies newly uploaded tracks into one of two genres based on their audio properties — replacing the need for manual playlist curation.

Since we are predicting a **discrete category** (Acoustic/Chill or High-Energy Party), this is a **classification problem**.

> ⚠️ **Note:** Some files in this project contain German language content, including variable names, comments, and task descriptions. This is because the course is taught in German at HAW Hamburg. Key translations are provided below. You can also use [DeepL](https://www.deepl.com) or [Google Translate](https://translate.google.com) for full translations.

---

## 🗂️ Project Structure

```
Aufgabe2/
    ├── vorlage.py            # Main Python script (template provided by course)
    ├── train.csv             # Training dataset with labels (Trainingsdaten)
    ├── test.csv              # Test dataset without labels (Testdaten)
    ├── result.csv            # Generated predictions — submitted for grading
    └── README.md             # This file
```

---

## 📊 Dataset Description

The dataset is a cleaned subset of the [Spotify Tracks dataset](https://www.kaggle.com/datasets/darrylljk/spotify-tracks).

| Column (German) | Column (English) | Description |
|-----------------|-----------------|-------------|
| `danceability` | Danceability | How suitable the track is for dancing |
| `energy` | Energy | Perceived intensity and activity level |
| `speechiness` | Speechiness | Proportion of spoken words in the track |
| `liveness` | Liveness | Probability the track was recorded live |
| `valence` | Valence | Musical positivity (happy vs. sad/angry) |
| `tempo` | Tempo | Speed of the track in Beats Per Minute (BPM) |
| `label` | Label | 🎯 Target variable — `-1` = Acoustic/Chill, `1` = High-Energy Party |

The training file (`train.csv`) contains the `label` column.
The test file (`test.csv`) does not — the model must predict it.

> ⚠️ The training set is **imbalanced**: 75% Acoustic/Chill and 25% High-Energy Party. This is why MCC is used as the evaluation metric instead of plain accuracy.

---

## 🧠 My SVM Configuration

```python
# === BLOCK 3 — HYPERPARAMETERS ===

KERNEL       = 'rbf'
C            = 2.2
GAMMA        = 0.09
CLASS_WEIGHT = {-1: 1, 1: 2.8}
```

### Hyperparameters (Hyperparameter)

| Parameter | Value | Reason |
|-----------|-------|--------|
| `KERNEL` | `'rbf'` | Draws curved, flexible boundaries — best for non-linear audio feature relationships |
| `C` (Kostenfunktion) | `2.2` | Balances margin width vs. misclassification penalty — avoids overfitting |
| `GAMMA` | `0.09` | Medium influence radius per point — smooth but flexible boundary |
| `CLASS_WEIGHT` (Klassengewichtung) | `{-1: 1, 1: 2.8}` | Compensates for class imbalance — missing a Party track is penalized 2.8× more |

---

## 🔄 How It Works

```
train.csv
    ↓
Data Preparation (Block 2)
• Feature scaling with StandardScaler
• Labels separated from features
    ↓
SVM Training (Block 3)
• RBF kernel projects data into higher-dimensional space
• Maximum margin hyperplane found between classes
• Class weights applied to handle imbalance
    ↓
Predictions on test.csv
    ↓
result.csv → submitted for grading
```

---

## 📈 Model Performance

The model was evaluated using 10-fold cross-validation on the training set:

- Party recall stayed consistently around **89%** — the model reliably catches High-Energy tracks
- Acoustic recall around **82%** — correctly identifies most Chill tracks
- Cross-validation MCC stayed stable across folds → no overfitting ✅

---

## 🏆 Result

| Metric | Value |
|--------|-------|
| MCC Score (server) | **0.709** |
| Required minimum | 0.70 |
| Passed | ✅ Yes |

An MCC of 0.709 means the model is genuinely learning the audio signature of each genre — not just exploiting the class imbalance. It significantly exceeds the required threshold of 0.70.

---

## ⚙️ How to Run

### 1. Clone the repository
```bash
git clone <your-repo-url>
cd Aufgabe2
```

### 2. Create a virtual environment
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Mac / Linux
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install scikit-learn pandas numpy
```

### 4. Run the script
```bash
python vorlage.py
```

### 5. Submit result
Upload the generated `result.csv` to the course platform for grading.

---

## 🔑 Key German Terms Used in Code / Task

| German | English |
|--------|---------|
| Kostenfunktion | Cost / Loss Function |
| Klassengewichtung | Class Weight |
| Hyperparameter | Hyperparameters |
| Trennlinie / Hyperebene | Decision Boundary / Hyperplane |
| Trainingsdaten | Training Data |
| Testdaten | Test Data |
| Vorhersage | Prediction |
| Entscheidungsgrenze | Decision Boundary |
| Kernel-Trick | Kernel Trick |
| Stützvektor | Support Vector |
| Genauigkeit | Accuracy |
| Fehler | Error |

---

## 📚 Concepts Used

- **SVM (Support Vector Machine)** — Finds the decision boundary that maximizes the margin between classes
- **RBF Kernel (Radial Basis Function)** — Projects data into higher dimensions where a flat separating plane can be found
- **Maximum Margin Hyperplane** — The boundary equidistant from the nearest points of each class
- **Support Vectors (Stützvektoren)** — The data points closest to the boundary that define its position
- **Soft Margin (C parameter)** — Allows some misclassifications for a wider, more robust boundary
- **Class Weighting (Klassengewichtung)** — Compensates for imbalanced training data
- **StandardScaler** — Feature normalization for stable SVM training
- **MCC (Matthews Correlation Coefficient)** — Evaluation metric that accounts for class imbalance
- **Cross-Validation** — Used to estimate real-world performance before final submission

---

*Assignment completed as part of the KI & ML elective course — HAW Hamburg 🎓*
