import pandas as pd
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


# Charger le dataset
data = pd.read_csv("data.csv")

# Colonnes utilisées pour apprendre
X = data[
    [
        "gap_total",
        "skill_match_count",
        "user_avg_level",
        "required_avg_level",
        "already_followed",
    ]
]

# Résultat attendu
y = data["result"]

# Séparer les données : entraînement / test
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Créer le modèle IA
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

# Entraîner le modèle
model.fit(X_train, y_train)

# Tester le modèle
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print("Accuracy du modèle :", accuracy)

# Sauvegarder le modèle entraîné
joblib.dump(model, "model.pkl")

print("Modèle sauvegardé dans model.pkl")