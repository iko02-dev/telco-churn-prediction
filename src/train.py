import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report

# 1. Load dan bersihkan data
df = pd.read_csv("data/telco_churn.csv")
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df = df.drop(columns="customerID")
df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

X = df.drop(columns="Churn")
y = df["Churn"]

# 2. Pisahkan kolom angka dan kategori
num_cols = X.select_dtypes(include="number").columns.tolist()
cat_cols = X.select_dtypes(exclude="number").columns.tolist()

# 3. Preprocessing
preprocess = ColumnTransformer([
    ("num", Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ]), num_cols),
    ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
])

# 4. Split data (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 5. Dua model yang dibandingkan
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced"),
    "Random Forest": RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=42),
}

best_name, best_f1, best_pipe = None, 0, None
for name, model in models.items():
    pipe = Pipeline([("prep", preprocess), ("model", model)])
    pipe.fit(X_train, y_train)
    pred = pipe.predict(X_test)
    acc = accuracy_score(y_test, pred)
    f1 = f1_score(y_test, pred)
    print(f"\n=== {name} ===")
    print(f"Accuracy: {acc:.3f} | F1-score: {f1:.3f}")
    print(classification_report(y_test, pred))
    if f1 > best_f1:
        best_name, best_f1, best_pipe = name, f1, pipe

# 6. Simpan model terbaik
print(f"\nModel terbaik: {best_name} (F1 = {best_f1:.3f})")
joblib.dump(best_pipe, "model/churn_model.joblib")
print("Model tersimpan di model/churn_model.joblib")