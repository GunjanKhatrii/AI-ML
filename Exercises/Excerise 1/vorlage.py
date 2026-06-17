# ==========================================
# Aufgabe 1: Neuronale Netzwerke
# ==========================================

# ==========================================
# BLOCK 1: Bibliotheken importieren
# (nicht bearbeiten)
# ==========================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

import warnings
warnings.filterwarnings('ignore')

# ==========================================
# BLOCK 2: Datensatz laden und vorbereiten
# (nicht bearbeiten)
# ==========================================
train_csv = 'diamonds_train.csv'
test_csv = 'diamonds_test.csv'

train_df = pd.read_csv(train_csv)
test_df = pd.read_csv(test_csv)

print("Trainingsdatensatz (Ausschnitt):")
print(train_df.head())
print("\nTestdatensatz (Ausschnitt):")
print(test_df.head())

X_train_raw = train_df.drop(columns=['price'])
y_train = train_df['price'].values
X_test_raw = test_df.copy()

cut_order = ['Fair', 'Good', 'Very Good', 'Premium', 'Ideal']
color_order = ['J', 'I', 'H', 'G', 'F', 'E', 'D']
clarity_order = ['I1', 'SI2', 'SI1', 'VS2', 'VS1', 'VVS2', 'VVS1', 'IF']

encoder = OrdinalEncoder(categories=[cut_order, color_order, clarity_order])
cat_cols = ['cut', 'color', 'clarity']
num_cols = ['carat']

X_train_cat = encoder.fit_transform(X_train_raw[cat_cols])
X_test_cat = encoder.transform(X_test_raw[cat_cols])

X_train_processed = np.hstack((X_train_raw[num_cols].values, X_train_cat))
X_test_processed = np.hstack((X_test_raw[num_cols].values, X_test_cat))

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_processed)
X_test_scaled = scaler.transform(X_test_processed)

# ==========================================
# BLOCK 3: Neuronales Netzwerk & Hyperparameter
# (HIER BEARBEITEN)
# ==========================================
# Bauen Sie hier Ihr neuronales Netzwerk und legen Sie die Hyperparameter fest!
# ❗ Sie müssen das Skript bei jeder Änderung neu ausführen.

# --- 1. HYPERPARAMETER ---
LERNRATE = 0.001
EPOCHEN = 200
KOSTENFUNKTION = 'mse'

# --- 2. NEURONALES NETZWERK ---
model = Sequential([
    Dense(256, activation='relu'),
    Dense(128, activation='relu'),
    Dense(64, activation='relu'),
    Dense(32, activation='relu'),
    Dense(1, activation='linear')  # NICHT löschen - finaler Output-Layer!
])

# --- 3. MODELL KOMPILIEREN ---
# NICHT bearbeiten - das baut das Netzwerk für das Training zusammen.
model.compile(
    optimizer=Adam(learning_rate=LERNRATE), 
    loss=KOSTENFUNKTION
)

print("\nModell erfolgreich erstellt! Training wird gestartet...")

# ==========================================
# BLOCK 4: Training und Vorhersage
# (nicht bearbeiten)
# ==========================================
# Nun lassen wir das Modell anhand der Trainingsdaten lernen. 
# Die generierte Grafik zeigt Ihnen, wie schnell Ihr Modell lernt. 
# Sobald das Training abgeschlossen ist, wird das Modell die Preise 
# für den Testsatz vorhersagen und zur Bewertung in einer Datei 
# namens `result.csv` speichern.

BATCH_SIZE = 128

history = model.fit(
    X_train_scaled,
    y_train,
    epochs=EPOCHEN,
    batch_size=BATCH_SIZE,
    validation_split=0.1,
    verbose=1
)

# Lernfortschritt anzeigen
plt.figure(figsize=(10, 6))
plt.plot(history.history['loss'], label='Training-Loss (Fehler)')
plt.plot(history.history['val_loss'], label='Validation-Loss (Fehler)')
plt.title('Modell-Lernfortschritt')
plt.xlabel('Epoche')
# Note: KOSTENFUNKTION needs to be a string (e.g., 'mse') for this print to work natively
plt.ylabel(f'Kosten') 
plt.legend()
plt.grid(True)
plt.show()

print("\nTraining abgeschlossen! Vorhersagen für den Testsatz werden generiert...")
predictions = model.predict(X_test_scaled)

submission_df = pd.DataFrame({'price': predictions.flatten()})

submission_filename = 'result.csv'
submission_df.to_csv(submission_filename, index=False)

print(f"\nVorhersage abgeschlossen! Die Datei wurde lokal als '{submission_filename}' in Ihrem Ausführungsverzeichnis gespeichert.")