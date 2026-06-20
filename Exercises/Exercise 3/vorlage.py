"""
Aufgabe 3: Clustering
Bibliotheken importieren (nicht bearbeiten):
"""

import pandas as pd
import sys
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
import warnings

warnings.filterwarnings('ignore')

"""Datensatz laden und vorbereiten (nicht bearbeiten):"""

df = pd.read_csv('student_data.csv')
print(f"Anzahl der Bohnen: {len(df)}")
print(f"Datensatz (Ausschnitt):")
print(df.head())

def prepare_dataset(data, scaling_method, feature_weights):
    processed_data = data.copy()

    if scaling_method == "standard":
        scaled = StandardScaler().fit_transform(processed_data)
    elif scaling_method == "minmax":
        scaled = MinMaxScaler().fit_transform(processed_data)
    else:
        scaled = processed_data.values

    scaled_df = pd.DataFrame(scaled, columns=data.columns)

    for feature, weight in feature_weights.items():
        if feature in scaled_df.columns:
            scaled_df[feature] = scaled_df[feature] * weight

    return scaled_df

"""
Beim Ausführen wird zuerst die WCSS-Kurve bei variierender Anzahl von Clustern in K-means (K von 1 bis 12) gezeichnet.

Nutzen Sie das Diagramm, um den optimalen Wert für K zu bestimmen!
Standardmäßig ist K auf 'None' gesetzt. Der "tatsächliche" Durchlauf von K-means (welcher die bewerteten Vorhersagen produziert) wird erst durchgeführt, sobald Sie diesen Standardwert mit Ihrer vermuteten Clusteranzahl ersetzen. 
Achtung: In realen Datensätzen ist der Knick oft nicht eindeutig... Probieren Sie daher mehrere Werte aus, die Sie für möglich halten!
"""

print("Berechne Modelle für den Ellenbogen-Plot. Bitte warten...")

test_data = prepare_dataset(df, "standard", {col: 1.0 for col in df.columns})
wcss = []
ks = range(2, 13)

for k in ks:
    # Wir testen verschiedene Werte für k
    kmeans_test = KMeans(n_clusters=k, init='k-means++', n_init=10, random_state=42)
    kmeans_test.fit(test_data)
    wcss.append(kmeans_test.inertia_)

# Das Ergebnis plotten
plt.figure(figsize=(10, 6))
plt.plot(ks, wcss, marker='o', linestyle='--', color='b')
plt.title('Die Ellenbogen-Methode (Elbow Method)')
plt.xlabel('Anzahl der Cluster (K)')
plt.ylabel('WCSS (Summe der quadratischen Abweichungen innerhalb der Cluster)')
plt.xticks(ks)
plt.grid(True, linestyle=':', alpha=0.7)
plt.show()

"""
Konfigurieren Sie hier Ihre Hyperparameter für das Clustering via K-means (HIER BEARBEITEN)!
"""

# === ZU BEARBEITENDER BLOCK ===

# 1. Wie viele Bohnensorten gibt es?
K = 7

# 2. Wie sollen die Daten skaliert werden?
SCALING_METHOD = "standard"  # Optionen: "standard", "minmax", None (keine Skalierung)

# 3. Wie sollen die Schwerpunkte initialisiert werden?
INIT_METHOD = "k-means++"

# 4. Wie oft soll der Algorithmus neu starten?
N_ATTEMPTS = 50

# 5. Merkmalsgewichte
FEATURE_WEIGHTS = {
    "Area":            1.0,
    "Perimeter":       1.0,
    "MajorAxisLength": 1.0,
    "MinorAxisLength": 1.0,
    "AspectRation":    1.5,
    "Eccentricity":    1.0,
    "ConvexArea":      1.0,
    "EquivDiameter":   1.0,
    "Extent":          1.0,
    "Solidity":        1.0,
    "roundness":       1.5,
    "Compactness":     1.0,
}

"""
Nun führen wir das Clustering durch (sollten Sie eine Clusteranzahl bestimmt haben). Sobald abgeschlossen, wird eine Unterteilung der Bohnen in Sorten vorhergesagt und zur Bewertung in einer CSV mit dem Namen `result.csv` gespeichert.
"""

if not K:
    print("\nParameter K wurde nicht gesetzt! Beenden...")
    sys.exit(1)

print("Bereite Daten vor...")
final_data = prepare_dataset(df, SCALING_METHOD, FEATURE_WEIGHTS)

print(f"Trainiere K-Means Modell mit {K} Clustern...")
kmeans = KMeans(
    n_clusters=K,
    init=INIT_METHOD,
    n_init=N_ATTEMPTS
)

# Vorhersagen treffen
predictions = kmeans.fit_predict(final_data)

# In Datei speichern
result_df = pd.DataFrame({'class': predictions})
filename = 'result.csv'
result_df.to_csv(filename, index=False)

print(f"Fertig! Ergebnisse wurden in '{filename}' gespeichert.")

"""## Optional: Visualisierung Ihrer Cluster
Da wir 12 Merkmale haben, können wir die Cluster nicht direkt zeichnen. Stattdessen kann man Dimensionsreduktiontechniken wie z.B. PCA (Principal Component Analysis) nutzen, um die Daten auf 2 Dimensionen "zusammenzudrücken" für einen "visuellen Eindruck", wie Ihre Cluster aufgebaut sind.
"""

pca = PCA(n_components=2)
data_2d = pca.fit_transform(final_data)
centroids = pca.transform(kmeans.cluster_centers_)

plt.figure(figsize=(12, 8))
scatter = plt.scatter(data_2d[:, 0], data_2d[:, 1], c=predictions, cmap='viridis', alpha=0.5, s=20)

# Die Zentren der Cluster mit einem roten X markieren
plt.scatter(centroids[:, 0], centroids[:, 1], c='red', marker='X', s=200, edgecolors='black', label='Centroids (Zentren)')

plt.title(f'Ihre K-Means Cluster in 2D (K={K})')
plt.xlabel('Hauptkomponente 1 (PCA 1)')
plt.ylabel('Hauptkomponente 2 (PCA 2)')
plt.legend()
plt.grid(alpha=0.3)
plt.show()