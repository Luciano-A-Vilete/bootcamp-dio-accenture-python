"""
Detecção de Anomalias em Transações de Cartão de Crédito
=========================================================

Dataset: Credit Card Fraud Detection (Kaggle / OpenML id=1597)
- 284.807 transações realizadas por portadores europeus em setembro de 2013
- 492 fraudes (~0,172% — dataset altamente desbalanceado)
- Features V1..V28 já transformadas via PCA (anonimização)
- Features originais disponíveis: 'Time' e 'Amount'
- Variável-alvo: 'Class' (0 = legítima, 1 = fraude)

Modelos comparados:
1. Isolation Forest      - rápido, escalável, baseado em árvores
2. Local Outlier Factor  - baseado em densidade local
3. One-Class SVM         - opcional (lento; ative com USE_OCSVM=True)

Autor: gerado para Vilete (Data Analyst @ IFC Benefit Solutions)
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import fetch_openml
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    precision_recall_curve,
    average_precision_score,
)

# ---------------------------------------------------------------------------
# Configurações globais
# ---------------------------------------------------------------------------
RANDOM_STATE = 42
USE_OCSVM = False           # True para incluir One-Class SVM (lento)
SAMPLE_FOR_LOF_OCSVM = 50_000  # subamostra para algoritmos O(n²)

sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (10, 5)


# ---------------------------------------------------------------------------
# 1. Carregar dataset
# ---------------------------------------------------------------------------
def load_data() -> pd.DataFrame:
    """
    Baixa o dataset 'creditcard' do OpenML (id=1597) na primeira execução
    e armazena em cache local (~/scikit_learn_data).
    """
    print("[1/6] Carregando dataset Credit Card Fraud (OpenML id=1597)...")
    bunch = fetch_openml(
        name="creditcard",
        version=1,
        as_frame=True,
        parser="auto",
    )
    df = bunch.frame
    # A coluna alvo vem como string ('0'/'1'); convertemos para int
    df["Class"] = df["Class"].astype(int)
    print(f"    -> shape: {df.shape}")
    print(f"    -> fraudes: {df['Class'].sum()} "
          f"({df['Class'].mean() * 100:.4f}%)")
    return df


# ---------------------------------------------------------------------------
# 2. Análise exploratória rápida
# ---------------------------------------------------------------------------
def quick_eda(df: pd.DataFrame) -> None:
    print("\n[2/6] Análise exploratória...")
    print("    Distribuição da classe:")
    print(df["Class"].value_counts().to_string())

    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    sns.histplot(df["Amount"], bins=60, ax=ax[0], color="steelblue")
    ax[0].set_title("Distribuição de Amount")
    ax[0].set_yscale("log")

    sns.histplot(df["Time"], bins=60, ax=ax[1], color="darkorange")
    ax[1].set_title("Distribuição de Time (segundos)")
    plt.tight_layout()
    plt.savefig("eda_distribuicoes.png", dpi=120)
    plt.close()
    print("    -> salvo: eda_distribuicoes.png")


# ---------------------------------------------------------------------------
# 3. Pré-processamento
# ---------------------------------------------------------------------------
def preprocess(df: pd.DataFrame):
    """
    - Escala 'Amount' e 'Time' com RobustScaler (resistente a outliers)
    - As features V1..V28 já são produto de PCA, então não precisam reescalar
    - Separa X (features) e y (rótulo)
    """
    print("\n[3/6] Pré-processando...")
    df = df.copy()
    scaler = RobustScaler()
    df[["Amount", "Time"]] = scaler.fit_transform(df[["Amount", "Time"]])

    X = df.drop(columns=["Class"]).values
    y = df["Class"].values

    # Split estratificado para preservar a proporção de fraudes
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.30,
        stratify=y,
        random_state=RANDOM_STATE,
    )
    print(f"    -> treino: {X_train.shape}  | fraudes: {y_train.sum()}")
    print(f"    -> teste : {X_test.shape}   | fraudes: {y_test.sum()}")
    return X_train, X_test, y_train, y_test


# ---------------------------------------------------------------------------
# 4. Treinamento dos modelos
# ---------------------------------------------------------------------------
def to_binary(pred):
    """Converte saída sklearn (-1 anomalia, 1 normal) para (1 fraude, 0 normal)."""
    return np.where(pred == -1, 1, 0)


def train_isolation_forest(X_train, X_test, contamination):
    print("\n[4/6] Treinando Isolation Forest...")
    model = IsolationForest(
        n_estimators=200,
        contamination=contamination,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    model.fit(X_train)
    y_pred = to_binary(model.predict(X_test))
    # score_samples: maior = mais normal. Invertemos para "score de anomalia".
    y_score = -model.score_samples(X_test)
    return y_pred, y_score


def train_lof(X_train, X_test, contamination):
    print("    Treinando Local Outlier Factor...")
    # LOF em modo 'novelty=True' permite usar predict() em dados novos.
    # Subamostramos o treino para manter custo viável (LOF é O(n²)).
    if len(X_train) > SAMPLE_FOR_LOF_OCSVM:
        rng = np.random.default_rng(RANDOM_STATE)
        idx = rng.choice(len(X_train), SAMPLE_FOR_LOF_OCSVM, replace=False)
        X_train_s = X_train[idx]
    else:
        X_train_s = X_train

    model = LocalOutlierFactor(
        n_neighbors=20,
        contamination=contamination,
        novelty=True,
        n_jobs=-1,
    )
    model.fit(X_train_s)
    y_pred = to_binary(model.predict(X_test))
    y_score = -model.decision_function(X_test)
    return y_pred, y_score


def train_ocsvm(X_train, X_test, contamination):
    print("    Treinando One-Class SVM (pode demorar)...")
    if len(X_train) > SAMPLE_FOR_LOF_OCSVM:
        rng = np.random.default_rng(RANDOM_STATE)
        idx = rng.choice(len(X_train), SAMPLE_FOR_LOF_OCSVM, replace=False)
        X_train_s = X_train[idx]
    else:
        X_train_s = X_train

    model = OneClassSVM(kernel="rbf", gamma="scale", nu=contamination)
    model.fit(X_train_s)
    y_pred = to_binary(model.predict(X_test))
    y_score = -model.decision_function(X_test)
    return y_pred, y_score


# ---------------------------------------------------------------------------
# 5. Avaliação
# ---------------------------------------------------------------------------
def evaluate(name: str, y_true, y_pred, y_score) -> dict:
    print(f"\n--- Resultados: {name} ---")
    print(classification_report(y_true, y_pred,
                                target_names=["Legítima", "Fraude"],
                                digits=4))
    cm = confusion_matrix(y_true, y_pred)
    print("Matriz de confusão:")
    print(pd.DataFrame(
        cm,
        index=["real_Legítima", "real_Fraude"],
        columns=["pred_Legítima", "pred_Fraude"],
    ))

    roc = roc_auc_score(y_true, y_score)
    ap = average_precision_score(y_true, y_score)
    print(f"ROC-AUC                : {roc:.4f}")
    print(f"Average Precision (PR) : {ap:.4f}")

    return {"model": name, "roc_auc": roc, "avg_precision": ap, "cm": cm}


def plot_pr_curves(results, y_true, scores_dict):
    plt.figure(figsize=(8, 6))
    for name, scores in scores_dict.items():
        precision, recall, _ = precision_recall_curve(y_true, scores)
        ap = average_precision_score(y_true, scores)
        plt.plot(recall, precision, label=f"{name} (AP={ap:.3f})")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Curvas Precision-Recall")
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig("precision_recall_curves.png", dpi=120)
    plt.close()
    print("\n    -> salvo: precision_recall_curves.png")


# ---------------------------------------------------------------------------
# 6. Pipeline principal
# ---------------------------------------------------------------------------
def main():
    df = load_data()
    quick_eda(df)
    X_train, X_test, y_train, y_test = preprocess(df)

    # contamination = taxa esperada de anomalias no conjunto
    contamination = float(y_train.mean())

    # --- Isolation Forest ---
    iso_pred, iso_score = train_isolation_forest(X_train, X_test, contamination)

    # --- LOF ---
    lof_pred, lof_score = train_lof(X_train, X_test, contamination)

    scores_dict = {
        "Isolation Forest": iso_score,
        "Local Outlier Factor": lof_score,
    }

    print("\n[5/6] Avaliando modelos...")
    results = [
        evaluate("Isolation Forest", y_test, iso_pred, iso_score),
        evaluate("Local Outlier Factor", y_test, lof_pred, lof_score),
    ]

    if USE_OCSVM:
        oc_pred, oc_score = train_ocsvm(X_train, X_test, contamination)
        scores_dict["One-Class SVM"] = oc_score
        results.append(evaluate("One-Class SVM", y_test, oc_pred, oc_score))

    print("\n[6/6] Gerando curva Precision-Recall...")
    plot_pr_curves(results, y_test, scores_dict)

    # Resumo final
    print("\n=== Resumo ===")
    summary = pd.DataFrame([
        {"Modelo": r["model"],
         "ROC-AUC": round(r["roc_auc"], 4),
         "Avg Precision": round(r["avg_precision"], 4)}
        for r in results
    ])
    print(summary.to_string(index=False))
    summary.to_csv("resumo_modelos.csv", index=False)
    print("\n-> salvo: resumo_modelos.csv")
    print("\nOK. Concluído.")


if __name__ == "__main__":
    main()
