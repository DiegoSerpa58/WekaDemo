from fastapi import FastAPI, UploadFile, File, Form
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_validate, cross_val_predict
from sklearn.dummy import DummyClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, cohen_kappa_score, mean_absolute_error
import io

app = FastAPI()

# Mapeo de algoritmos tipo Weka a Scikit-Learn
ALGORITHMS = {
    "zeroR": {
        "name": "ZeroR", 
        "model": DummyClassifier(strategy="most_frequent"), 
        "desc": "Predice siempre la clase mayoritaria."
    },
    "oneR": {
        "name": "OneR", 
        "model": DecisionTreeClassifier(max_depth=1), 
        "desc": "Clasificador basado en un solo atributo."
    },
    "j48": {
        "name": "J48 (Decision Tree)", 
        "model": DecisionTreeClassifier(), 
        "desc": "Algoritmo de árbol de decisión estándar."
    },
    "naiveBayes": {
        "name": "Naive Bayes", 
        "model": GaussianNB(), 
        "desc": "Clasificador probabilístico Bayesiano."
    }
}

@app.get("/api/weka/algorithms")
async def get_algorithms():
    return [{"id": k, "name": v["name"], "description": v["desc"]} for k, v in ALGORITHMS.items()]

@app.post("/api/weka/classify")
async def classify(
    file: UploadFile = File(...),
    algorithm: str = Form(...),
    evaluationMethod: str = Form(...),
    folds: int = Form(10),
    trainPercent: float = Form(66),
    seed: int = Form(1)
):
    # 1. Cargar datos
    contents = await file.read()
    # Soporte básico para CSV
    df = pd.read_csv(io.BytesIO(contents))
    
    # Asumimos que la última columna es el target
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]
    
    # Convertir a numérico si es necesario para sklearn
    X = pd.get_dummies(X)
    
    class_names = [str(c) for c in np.unique(y)]
    model = ALGORITHMS[algorithm]["model"]
    
    if evaluationMethod == "percentagesplit":
        test_size = (100 - trainPercent) / 100
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=seed)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        accuracy = model.score(X_test, y_test) * 100
        kappa = cohen_kappa_score(y_test, y_pred)
        # MAE requiere targets numéricos o probabilidades, simplificamos:
        mae = 0.0 
        cm = confusion_matrix(y_test, y_pred)
        p, r, f, _ = precision_recall_fscore_support(y_test, y_pred, zero_division=0)
    else:
        # Cross Validation
        y_pred = cross_val_predict(model, X, y, cv=folds)
        accuracy = np.mean(y_pred == y) * 100
        kappa = cohen_kappa_score(y, y_pred)
        mae = 0.0
        cm = confusion_matrix(y, y_pred)
        p, r, f, _ = precision_recall_fscore_support(y, y_pred, zero_division=0)

    return {
        "algorithm": ALGORITHMS[algorithm]["name"],
        "datasetName": file.filename,
        "numInstances": len(df),
        "numAttributes": len(df.columns),
        "accuracy": float(accuracy),
        "kappa": float(kappa),
        "meanAbsoluteError": float(mae),
        "confusionMatrix": cm.tolist(),
        "classNames": class_names,
        "precision": p.tolist(),
        "recall": r.tolist(),
        "fMeasure": f.tolist(),
        "evaluationMethod": f"{evaluationMethod} ({folds} folds)" if evaluationMethod == "crossvalidation" else f"Split {trainPercent}%"
    }

# Servir estáticos
app.mount("/", StaticFiles(directory="static", html=True), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
