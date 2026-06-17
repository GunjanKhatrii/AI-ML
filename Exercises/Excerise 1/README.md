# 💎 Aufgabe 1 — Diamond Price Prediction with Neural Networks

> **Course:** Künstliche Intelligenz & Maschinelles Lernen (KI & ML)  
> **Task Type:** Regression Problem  
> **Final Score:** R² = 0.95 🏆

---

## 📌 Task Description

In this assignment, I acted as an **AI consultant** hired by a diamond wholesaler.  
The goal was to train a **neural network** that automatically predicts the price of a diamond based on its properties — instead of relying on human appraisers.

Since we are predicting a **continuous numerical value** (the price in USD), this is a **regression problem**.

> ⚠️ **Note:** Some files in this project contain **German language** content, including variable names, comments, and column headers. This is because the course is taught in German at OTH Regensburg. Key translations are provided below.

---

## 🗂️ Project Structure

```
Aufgabe1/
    ├── vorlage.py            # Main Python script (template provided by course)
    ├── diamonds_train.csv    # Training dataset with prices (Trainingsdaten)
    ├── diamonds_test.csv     # Test dataset without prices (Testdaten)
    ├── result.csv            # Generated predictions — submitted for grading
    └── README.md             # This file
```

---

## 📊 Dataset Description

The dataset is a cleaned version of the classic **Diamonds dataset** from Kaggle.

| Column (German) | Column (English) | Description |
|---|---|---|
| `carat` | Carat | Weight of the diamond |
| `cut` | Cut quality | Fair → Good → Very Good → Premium → Ideal |
| `color` | Color grade | J (worst) → D (best) |
| `clarity` | Clarity grade | I1 (worst) → IF (best) |
| `price` | Price (USD) | 💰 Target variable — what the model predicts |

> The training file (`diamonds_train.csv`) contains the `price` column.  
> The test file (`diamonds_test.csv`) does **not** — the model must predict it.

---

## 🧠 My Neural Network Architecture

```python
model = Sequential([
    Dense(256, activation='relu'),
    Dense(128, activation='relu'),
    Dense(64,  activation='relu'),
    Dense(32,  activation='relu'),
    Dense(1,   activation='linear')   # Output layer — single price prediction
])
```

### Hyperparameters (Hyperparameter)

| Parameter | Value | Reason |
|---|---|---|
| **Lernrate** (Learning Rate) | 0.001 | Safe standard value for Adam optimizer |
| **Epochen** (Epochs) | 200 | Enough for full convergence without overfitting |
| **Kostenfunktion** (Cost Function) | MSE | Penalizes large errors — good for price outliers |
| **Batch Size** | 128 | Set by the template, not changed |
| **Optimizer** | Adam | Set by the template, not changed |

---

## 🔄 How It Works

```
diamonds_train.csv
        ↓
  Data Preparation (Block 2)
  • Categorical encoding (cut, color, clarity → numbers)
  • Feature scaling with StandardScaler
        ↓
  Neural Network Training (Block 3)
  • Forward Propagation → prediction
  • Cost calculated with MSE
  • Gradient Descent → weights updated
  • Repeated for 200 epochs
        ↓
  Predictions on diamonds_test.csv
        ↓
  result.csv → submitted for grading
```

---

## 📈 Training Progress

The model showed a **healthy learning curve**:

- Loss dropped steeply in the first **~15 epochs**
- Gradually flattened and **fully converged** by epoch ~150
- Training loss and Validation loss stayed **close together** → no overfitting ✅

---

## 🏆 Result

| Metric | Value |
|---|---|
| **R² Score** | **0.95** |
| Required minimum | 0.85 |
| Passed | ✅ Yes |

> An R² of 0.95 means the model can explain **95% of the variance** in diamond prices — significantly above the required threshold of 0.85.

---

## ⚙️ How to Run

### 1. Clone the repository
```bash
git clone <your-repo-url>
cd Aufgabe1
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
pip install tensorflow scikit-learn pandas numpy matplotlib
```

### 4. Run the script
```bash
python vorlage.py
```

### 5. Submit result
Upload the generated `result.csv` to the course platform for grading.

---

## 🔑 Key German Terms Used in Code

| German | English |
|---|---|
| Lernrate | Learning Rate |
| Epochen | Epochs |
| Kostenfunktion | Cost / Loss Function |
| Trainingsdaten | Training Data |
| Testdaten | Test Data |
| Schichten | Layers |
| Gewichte | Weights |
| Fehler | Error |
| Vorhersage | Prediction |
| Neuronales Netzwerk | Neural Network |

---

## 📚 Concepts Used

- **Neural Networks (Neuronale Netzwerke)** — Multi-layer perceptron for regression
- **ReLU Activation** — Non-linear activation for hidden layers
- **Linear Activation** — Output layer for continuous value prediction
- **Gradient Descent** — Optimization algorithm to minimize cost
- **MSE (Mean Squared Error)** — Cost function used for training
- **StandardScaler** — Feature normalization for stable training
- **OrdinalEncoder** — Converts categorical features to numbers
- **R² Score** — Evaluation metric (coefficient of determination)

---

*Assignment completed as part of the KI & ML elective course — HAW Hamburg* 🎓
