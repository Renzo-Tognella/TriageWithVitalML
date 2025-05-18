import os
import re
import pandas as pd
import numpy as np
from sklearn.model_selection import (
    train_test_split, GridSearchCV, learning_curve, StratifiedKFold
)
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.metrics import (
    mean_squared_error, accuracy_score,
    classification_report, confusion_matrix,
    make_scorer
)
import matplotlib.pyplot as plt

# Define a pasta base (funciona tanto em .py quanto em notebook)
try:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    BASE_DIR = os.getcwd()


def load_labeled():
    path = os.path.join(BASE_DIR, "Vital Signs Training with Label.txt")
    df = pd.read_csv(
        path, header=None,
        names=["i", "si1", "si2", "si3", "si4", "si5", "gi", "yi"]
    )
    # CORREÇÃO: Selecionar apenas as features si3, si4, si5
    feats = ["si3", "si4", "si5"]
    return df[feats], df["gi"], df["yi"]


def load_blind():
    path = os.path.join(BASE_DIR, "Vital Signs Training No Label.txt")
    # Ajustar nomes das colunas se o arquivo "No Label" não contiver si1 e si2.
    # Assumindo que o arquivo "Vital Signs Training No Label.txt" tem o mesmo número
    # de colunas que o "Vital Signs Training with Label.txt" antes da remoção de gi e yi,
    # mas só usaremos si3, si4, si5.
    # Se o arquivo "No Label" SÓ tiver i, si3, si4, si5, gi, os nomes precisam ser ajustados.
    # Para este exemplo, vamos supor que a estrutura de colunas é similar ao load_labeled
    # e selecionaremos as features corretas.
    try:
        df = pd.read_csv(
            path, header=None,
            names=["i", "si1", "si2", "si3", "si4", "si5", "gi"] # Nomes originais para leitura
        )
    except pd.errors.ParserError:
        # Tentativa alternativa se o número de colunas for diferente
        # (ex: se si1 e si2 realmente não estiverem no arquivo No Label)
        df = pd.read_csv(
            path, header=None,
            names=["i", "si3", "si4", "si5", "gi"] # Nomes se si1, si2 não presentes
        )


    df["gi"] = (
        df["gi"].astype(str)
        .str.replace(r"[^0-9\.\-]", "", regex=True)
        .astype(float)
    )
    # CORREÇÃO: Selecionar apenas as features si3, si4, si5
    feats = ["si3", "si4", "si5"]
    return df[feats], df["gi"], df["i"]


def fit_regressor(X_train, y_train):
    # Ajuste de poda com parâmetros para reduzir overfitting
    params = {
        "max_depth": [3, 5, 7],
        "min_samples_leaf": [5, 10, 15, 20],
        "max_leaf_nodes": [10, 20, 30, 50],
        "min_impurity_decrease": [1e-3, 1e-2, 5e-2],
        "ccp_alpha": [0.001, 0.005, 0.01, 0.02, 0.05, 0.1]
    }
    gs = GridSearchCV(
        DecisionTreeRegressor(random_state=0), params,
        cv=5, scoring="neg_mean_squared_error", n_jobs=-1
    )
    gs.fit(X_train, y_train)
    print(f"Melhores parâmetros para Regressor: {gs.best_params_}")
    return gs.best_estimator_, gs.best_score_


def fit_classifier(X_train, y_train, cv_obj):
    # Ajuste de poda com parâmetros para reduzir overfitting, incluindo ccp_alpha
    params = {
        "max_depth": [3, 5, 7],
        "min_samples_leaf": [5, 10, 15, 20],
        "criterion": ["gini", "entropy"],
        "class_weight": [None, "balanced"],
        "ccp_alpha": [0.001, 0.005, 0.01, 0.015, 0.02, 0.05]
    }
    gs = GridSearchCV(
        DecisionTreeClassifier(random_state=0), params,
        cv=cv_obj, scoring="accuracy", n_jobs=-1
    )
    gs.fit(X_train, y_train)
    print(f"Melhores parâmetros para Classificador: {gs.best_params_}")
    return gs.best_estimator_, gs.best_score_


def show_feature_importances(model, feature_names, title):
    imp = model.feature_importances_
    # Assegurar que o número de feature_names corresponde às importâncias
    if len(feature_names) != len(imp):
        print(f"Aviso: Discrepância no número de nomes de features ({len(feature_names)}) e importâncias ({len(imp)}). Usando nomes genéricos.")
        feature_names = [f"feature_{i}" for i in range(len(imp))]

    df = pd.DataFrame({"feature": feature_names, "importance": imp})
    df = df.sort_values(by="importance", ascending=False)
    print(f"\n{title} - Importância das Features:")
    print(df.to_string(index=False))


def plot_confusion_heatmap(cm, classes, title):
    plt.figure()
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.viridis)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes)
    fmt = 'd'
    thresh = cm.max() / 2.
    for i, j in np.ndindex(cm.shape):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.xlabel('Predito')
    plt.ylabel('Verdadeiro')
    plt.tight_layout()
    plt.show()


def plot_learning_curves(best_regressor, best_classifier, X, y_reg, y_clf, cv_clf):
    train_sizes = np.linspace(0.1, 1.0, 10)

    rmse_scorer = make_scorer(
        lambda y, y_pred: np.sqrt(mean_squared_error(y, y_pred)),
        greater_is_better=False
    )
    ts_r, tr_r, val_r = learning_curve(
        best_regressor,
        X, y_reg, cv=5,
        scoring=rmse_scorer,
        train_sizes=train_sizes,
        shuffle=True, random_state=0,
        n_jobs=-1
    )
    train_rmse = -tr_r.mean(axis=1)
    val_rmse   = -val_r.mean(axis=1)
    plt.figure()
    plt.plot(ts_r, train_rmse, 'o-', label="RMSE Treino")
    plt.plot(ts_r, val_rmse, 'o-',  label="RMSE Validação")
    plt.xlabel("Tamanho do Treino")
    plt.ylabel("RMSE")
    plt.title("Curva de Aprendizado - Regressão (Modelo Otimizado)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    ts_c, tr_c, val_c = learning_curve(
        best_classifier,
        X, y_clf, cv=cv_clf,
        scoring="accuracy",
        train_sizes=train_sizes,
        shuffle=True, random_state=0,
        n_jobs=-1
    )
    train_acc = tr_c.mean(axis=1)
    val_acc   = val_c.mean(axis=1)
    plt.figure()
    plt.plot(ts_c, train_acc, 'o-', label="Acurácia Treino")
    plt.plot(ts_c, val_acc, 'o-',  label="Acurácia Validação")
    plt.xlabel("Tamanho do Treino")
    plt.ylabel("Acurácia")
    plt.title("Curva de Aprendizado - Classificação (Modelo Otimizado)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def main():
    # 1) Carrega dados com rótulo
    X, y_reg, y_clf = load_labeled()

    # 2) Split treino/validação
    X_tr_r, X_va_r, y_tr_r, y_va_r = train_test_split(
        X, y_reg, test_size=0.3, random_state=42, shuffle=True
    )
    X_tr_c, X_va_c, y_tr_c, y_va_c = train_test_split(
        X, y_clf, test_size=0.3, random_state=42, shuffle=True, stratify=y_clf
    )

    # 3) EstratifiedKFold para classificação
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)

    # 4) Treina DecisionTree
    reg_dt, cv_dt_r_score = fit_regressor(X_tr_r, y_tr_r)
    clf_dt, cv_dt_c_score = fit_classifier(X_tr_c, y_tr_c, cv_obj=skf)

    # 5) Avaliação no conjunto de validação
    preds_reg_val = reg_dt.predict(X_va_r)
    rmse_val = np.sqrt(mean_squared_error(y_va_r, preds_reg_val))
    
    preds_clf_val = clf_dt.predict(X_va_c)
    acc_val  = accuracy_score(y_va_c, preds_clf_val)
    
    print("\n— Regressão (val) —")
    print(f" RMSE val   : {rmse_val:.3f}")
    print(f" CV best RMSE (sqrt(-neg_mse)): {np.sqrt(-cv_dt_r_score):.3f}\n")
    
    print("— Classificação (val) —")
    print(f" Acurácia val: {acc_val:.3f}")
    print(f" CV best Acurácia : {cv_dt_c_score:.3f}\n")
    
    print(classification_report(y_va_c, preds_clf_val))
    cm_val = confusion_matrix(y_va_c, preds_clf_val)
    plot_confusion_heatmap(cm_val, classes=np.unique(y_va_c), title="Matriz de Confusão - Validação (Modelo Otimizado)")

    # 6) Importâncias
    # CORREÇÃO: Atualizar a lista de nomes de features
    feats_names = ["si3", "si4", "si5"]
    show_feature_importances(reg_dt, feats_names, "DecisionTreeRegressor (Otimizado)")
    show_feature_importances(clf_dt, feats_names, "DecisionTreeClassifier (Otimizado)")

    # 7) Teste cego de regressão
    X_test, y_test_gi, ids = load_blind()
    preds_gi = reg_dt.predict(X_test)
    rmse_test = np.sqrt(mean_squared_error(y_test_gi, preds_gi))
    print(f"\n— Regressão (teste cego) — RMSE cego: {rmse_test:.3f}\n")

    # 8) Previsões cegas de classe
    preds_yi = clf_dt.predict(X_test)
    pd.DataFrame({"i": ids, "pred_gi": preds_gi, "pred_yi": preds_yi})\
      .to_csv(os.path.join(BASE_DIR, "predicoes_blind_otimizado_feats_corrigidas.csv"), index=False)

    # 9) Curvas de Aprendizado para os modelos otimizados
    # Ao plotar as curvas de aprendizado, X deve ter o mesmo número de features
    # que os modelos (reg_dt, clf_dt) foram treinados.
    # Como X já foi carregado com as features corretas por load_labeled(), está ok.
    plot_learning_curves(
        reg_dt, clf_dt,
        X.values if hasattr(X, 'values') else X, 
        y_reg, y_clf, 
        cv_clf=skf
    )

if __name__ == "__main__":
    main()