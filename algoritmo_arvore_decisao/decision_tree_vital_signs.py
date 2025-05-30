import os
import re
import pandas as pd
import numpy as np
from sklearn.model_selection import (
    train_test_split, learning_curve, StratifiedKFold
)
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier, plot_tree
from sklearn.metrics import (
    mean_squared_error, accuracy_score,
    classification_report, confusion_matrix,
    make_scorer
)
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict, Any, Union, Optional
from dataclasses import dataclass, field

@dataclass
class Config:
    REGRESSOR_PARAMS: Dict[str, Any]
    CLASSIFIER_PARAMS: Dict[str, Any]
    BASE_DIR: str = field(default_factory=lambda: os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd())
    LABELED_DATA_FILENAME: str = "Vital Signs Training with Label.txt"
    BLIND_DATA_FILENAME: str = "Vital Signs Training No Label.txt"
    OUTPUT_CSV_FILENAME: str = "predicoes_blind_feats_corrigidas.csv"
    DECISION_TREE_IMAGE_FILENAME: str = "decision_tree_classifier.png"

    FEATURE_COLUMNS: List[str] = field(default_factory=lambda: ["si3", "si4", "si5"])
    REGRESSION_TARGET_COLUMN: str = "gi"
    CLASSIFICATION_TARGET_COLUMN: str = "yi"
    ID_COLUMN: str = "i"

    LABELED_COLUMN_NAMES: List[str] = field(init=False)
    BLIND_COLUMN_NAMES_FULL: List[str] = field(init=False)
    BLIND_COLUMN_NAMES_REDUCED: List[str] = field(init=False)

    RANDOM_STATE_SPLIT: int = 42
    RANDOM_STATE_MODEL: int = 0
    TEST_SPLIT_SIZE: float = 0.3
    CV_SPLITS: int = 5
    LEARNING_CURVE_TRAIN_SIZES: np.ndarray = field(default_factory=lambda: np.linspace(0.1, 1.0, 10))

    PLOT_FIG_SIZE_TREE: Tuple[int, int] = (60, 30)
    PLOT_FONT_SIZE_TREE: int = 7

    def __post_init__(self):
        self.LABELED_DATA_FILE = os.path.join(self.BASE_DIR, self.LABELED_DATA_FILENAME)
        self.BLIND_DATA_FILE = os.path.join(self.BASE_DIR, self.BLIND_DATA_FILENAME)
        self.OUTPUT_CSV_FILE = os.path.join(self.BASE_DIR, self.OUTPUT_CSV_FILENAME)
        self.DECISION_TREE_IMAGE_FILE = os.path.join(self.BASE_DIR, self.DECISION_TREE_IMAGE_FILENAME)

        self.LABELED_COLUMN_NAMES = [self.ID_COLUMN, "si1", "si2"] + self.FEATURE_COLUMNS + [self.REGRESSION_TARGET_COLUMN, self.CLASSIFICATION_TARGET_COLUMN]
        self.BLIND_COLUMN_NAMES_FULL = [self.ID_COLUMN, "si1", "si2"] + self.FEATURE_COLUMNS + [self.REGRESSION_TARGET_COLUMN]
        self.BLIND_COLUMN_NAMES_REDUCED = [self.ID_COLUMN] + self.FEATURE_COLUMNS + [self.REGRESSION_TARGET_COLUMN]


class DataLoader:
    def __init__(self, config: Config):
        self.config = config

    def _load_raw_data(self, file_path: str, column_names: List[str]) -> pd.DataFrame:
        try:
            return pd.read_csv(file_path, header=None, names=column_names)
        except FileNotFoundError:
            print(f"Erro: Arquivo não encontrado em {file_path}")
            raise
        except Exception as e:
            print(f"Erro ao carregar o arquivo {file_path}: {e}")
            raise

    def get_labeled_data(self) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
        df = self._load_raw_data(self.config.LABELED_DATA_FILE, self.config.LABELED_COLUMN_NAMES)
        return df[self.config.FEATURE_COLUMNS], df[self.config.REGRESSION_TARGET_COLUMN], df[self.config.CLASSIFICATION_TARGET_COLUMN]

    def get_blind_data(self) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
        try:
            df = self._load_raw_data(self.config.BLIND_DATA_FILE, self.config.BLIND_COLUMN_NAMES_FULL)
        except pd.errors.ParserError:
            print(f"ParserError ao ler {self.config.BLIND_DATA_FILE} com todas as colunas, tentando com colunas reduzidas.")
            df = self._load_raw_data(self.config.BLIND_DATA_FILE, self.config.BLIND_COLUMN_NAMES_REDUCED)
        except Exception as e:
            print(f"Erro inesperado ao carregar dados cegos: {e}")
            raise

        df[self.config.REGRESSION_TARGET_COLUMN] = (
            df[self.config.REGRESSION_TARGET_COLUMN].astype(str)
            .str.replace(r"[^0-9\\\\.\\\\-]", "", regex=True)
            .replace('', np.nan)
            .astype(float)
        )
        return df[self.config.FEATURE_COLUMNS], df[self.config.REGRESSION_TARGET_COLUMN], df[self.config.ID_COLUMN]

class ModelTrainer:
    def __init__(self, config: Config):
        self.config = config

    def train_regressor(self, X_train: pd.DataFrame, y_train: pd.Series) -> DecisionTreeRegressor:
        regressor = DecisionTreeRegressor(
            random_state=self.config.RANDOM_STATE_MODEL,
            **self.config.REGRESSOR_PARAMS
        )
        regressor.fit(X_train, y_train)
        return regressor

    def train_classifier(self, X_train: pd.DataFrame, y_train: pd.Series) -> DecisionTreeClassifier:
        classifier = DecisionTreeClassifier(
            random_state=self.config.RANDOM_STATE_MODEL,
            **self.config.CLASSIFIER_PARAMS
        )
        classifier.fit(X_train, y_train)
        return classifier

class ModelEvaluator:
    def __init__(self, config: Config):
        self.config = config

    def evaluate_regressor(self, model: DecisionTreeRegressor, X_val: pd.DataFrame, y_val: pd.Series) -> float:
        preds_reg_val = model.predict(X_val)
        rmse_val = np.sqrt(mean_squared_error(y_val, preds_reg_val))
        print("\\n\\n— Avaliação da Regressão (conjunto de validação) —")
        print(f" RMSE (validação) : {rmse_val:.3f}\\n")
        return rmse_val

    def evaluate_classifier(self, model: DecisionTreeClassifier, X_val: pd.DataFrame, y_val: pd.Series) -> Tuple[float, str, np.ndarray]:
        preds_clf_val = model.predict(X_val)
        acc_val = accuracy_score(y_val, preds_clf_val)
        report = classification_report(y_val, preds_clf_val, zero_division=0)
        cm = confusion_matrix(y_val, preds_clf_val)

        print("\\n— Avaliação da Classificação (conjunto de validação) —")
        print(f" Acurácia (validação): {acc_val:.3f}\\n")
        print("\\nRelatório de Classificação (validação):")
        print(report)
        print()
        return acc_val, report, cm

    def calculate_blind_rmse(self, model: DecisionTreeRegressor, X_blind: pd.DataFrame, y_blind_true: pd.Series) -> float:
        preds_gi_blind = model.predict(X_blind)
        rmse_blind_test = np.sqrt(mean_squared_error(y_blind_true, preds_gi_blind))
        print(f"\\n— Regressão (teste cego) — RMSE cego: {rmse_blind_test:.3f}\\n")
        return rmse_blind_test

class Visualizer:
    def __init__(self, config: Config):
        self.config = config

    def display_feature_importances(self, model: Any, feature_names: List[str], title: str) -> None:
        if not hasattr(model, 'feature_importances_'):
            print(f"Modelo {title} não suporta feature_importances_.")
            return

        importances = model.feature_importances_
        if len(feature_names) != len(importances):
            print(f"Aviso: Discrepância no número de nomes de features ({len(feature_names)}) e importâncias ({len(importances)}). Usando nomes genéricos.")
            effective_feature_names = [f"feature_{i}" for i in range(len(importances))]
        else:
            effective_feature_names = feature_names

        df_importances = pd.DataFrame({"feature": effective_feature_names, "importance": importances})
        df_importances = df_importances.sort_values(by="importance", ascending=False)
        
        print(f"\n{title} - Importância das Features:")
        print(df_importances.to_string(index=False))
        print("\n")
        
        plt.figure(figsize=(10, max(6, len(effective_feature_names) * 0.5)))
        plt.title(f"Importância das Features - {title}")
        plt.barh(df_importances["feature"], df_importances["importance"])
        plt.gca().invert_yaxis()
        plt.xlabel("Importância")
        plt.tight_layout()
        plt.show()

    def plot_confusion_matrix_heatmap(self, data_matrix: np.ndarray, classes: List[str], title: str) -> None:
        plt.figure(figsize=(8, 6))
        plt.imshow(data_matrix, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title(title)
        plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)
        fmt = 'd'
        thresh = data_matrix.max() / 2.
        for i, j in np.ndindex(data_matrix.shape):
            plt.text(j, i, format(data_matrix[i, j], fmt),
                     horizontalalignment="center",
                     color="white" if data_matrix[i, j] > thresh else "black")
        plt.xlabel("Predito")
        plt.ylabel("Verdadeiro")
        plt.tight_layout()
        plt.show()

    def plot_learning_curves(
        self, estimator: Any, title_suffix: str, X: pd.DataFrame, y: pd.Series,
        cv: Union[int, StratifiedKFold], scoring: Union[str, callable], y_label: str, higher_is_better: bool = True
    ) -> None:
        train_sizes, train_scores, val_scores = learning_curve(
            estimator, X, y, cv=cv, scoring=scoring,
            train_sizes=self.config.LEARNING_CURVE_TRAIN_SIZES,
            shuffle=True, random_state=self.config.RANDOM_STATE_MODEL, n_jobs=-1
        )

        train_scores_mean = train_scores.mean(axis=1)
        val_scores_mean = val_scores.mean(axis=1)

        if not higher_is_better:
            train_scores_mean = -train_scores_mean
            val_scores_mean = -val_scores_mean

        plt.figure(figsize=(10, 6))
        plt.plot(train_sizes, train_scores_mean, 'o-', label=f"{y_label} Treino")
        plt.plot(train_sizes, val_scores_mean, 'o-',  label=f"{y_label} Validação")
        plt.xlabel("Tamanho do Treino")
        plt.ylabel(y_label)
        plt.title(f"Curva de Aprendizado - {title_suffix}")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    def plot_and_save_decision_tree(
        self, model: DecisionTreeClassifier, feature_names: List[str], class_names: List[str]
    ) -> None:
        plt.figure(figsize=self.config.PLOT_FIG_SIZE_TREE)
        plot_tree(model,
                  feature_names=feature_names,
                  class_names=class_names,
                  filled=True,
                  rounded=True,
                  fontsize=self.config.PLOT_FONT_SIZE_TREE,
                  impurity=True,
                  proportion=False,
                  max_depth=15
                  )
        plt.title("Árvore de Decisão - Classificador ()")
        try:
            plt.savefig(self.config.DECISION_TREE_IMAGE_FILE, dpi=300, bbox_inches='tight')
            print(f"Árvore de decisão do classificador salva em: {self.config.DECISION_TREE_IMAGE_FILE}\n")
        except Exception as e:
            print(f"Erro ao salvar a imagem da árvore de decisão: {e}\n")
        plt.show()

class PredictionSaver:
    def __init__(self, config: Config):
        self.config = config

    def save_blind_predictions(self, ids_blind: pd.Series, preds_gi_blind: np.ndarray, preds_yi_blind: np.ndarray) -> None:
        df_predictions_blind = pd.DataFrame({
            self.config.ID_COLUMN: ids_blind,
            f"pred_{self.config.REGRESSION_TARGET_COLUMN}": preds_gi_blind,
            f"pred_{self.config.CLASSIFICATION_TARGET_COLUMN}": preds_yi_blind
        })
        try:
            df_predictions_blind.to_csv(self.config.OUTPUT_CSV_FILE, index=False)
            print(f"Predições do teste cego salvas em: {self.config.OUTPUT_CSV_FILE}\n")
        except Exception as e:
            print(f"Erro ao salvar as predições: {e}\n")

class ResultsSummary:
    def __init__(self, config: Config):
        self.config = config
        self.results = {}
    
    def add_result(self, key: str, value: Any) -> None:
        """Add a result to the summary"""
        self.results[key] = value
    
    def display_complete_summary(self) -> None:
        """Display comprehensive summary of all parameters and results"""
        print("\n" + "="*80)
        print("RESUMO COMPLETO DOS RESULTADOS E PARÂMETROS")
        print("="*80)
        
        print("\n📋 PARÂMETROS DE ENTRADA:")
        print("-" * 50)
        
        print("\n🔹 Configurações Gerais:")
        print(f"  • Colunas de Features: {self.config.FEATURE_COLUMNS}")
        print(f"  • Coluna Alvo (Regressão): {self.config.REGRESSION_TARGET_COLUMN}")
        print(f"  • Coluna Alvo (Classificação): {self.config.CLASSIFICATION_TARGET_COLUMN}")
        print(f"  • Coluna ID: {self.config.ID_COLUMN}")
        print(f"  • Tamanho do Conjunto de Teste: {self.config.TEST_SPLIT_SIZE}")
        print(f"  • Número de Folds CV: {self.config.CV_SPLITS}")
        print(f"  • Random State (Split): {self.config.RANDOM_STATE_SPLIT}")
        print(f"  • Random State: {self.config.RANDOM_STATE_MODEL}")
        
        print("\n🔹 Parâmetros do Regressor (DecisionTreeRegressor):")
        for param, value in self.config.REGRESSOR_PARAMS.items():
            print(f"  • {param}: {value}")
        
        print("\n🔹 Parâmetros do Classificador (DecisionTreeClassifier):")
        for param, value in self.config.CLASSIFIER_PARAMS.items():
            print(f"  • {param}: {value}")
        
        print("\n📊 RESULTADOS OBTIDOS:")
        print("-" * 50)
        
        if 'validation_rmse' in self.results:
            print(f"\n🔸 Regressão - Conjunto de Validação:")
            print(f"  • RMSE: {self.results['validation_rmse']:.4f}")
        
        if 'blind_rmse' in self.results:
            print(f"\n🔸 Regressão - Conjunto Cego:")
            print(f"  • RMSE: {self.results['blind_rmse']:.4f}")
        
        if 'validation_accuracy' in self.results:
            print(f"\n🔸 Classificação - Conjunto de Validação:")
            print(f"  • Acurácia: {self.results['validation_accuracy']:.4f}")
        
        if 'regressor_importances' in self.results:
            print(f"\n🔸 Importância das Features - Regressor:")
            for feature, importance in self.results['regressor_importances'].items():
                print(f"  • {feature}: {importance:.4f}")
        
        if 'classifier_importances' in self.results:
            print(f"\n🔸 Importância das Features - Classificador:")
            for feature, importance in self.results['classifier_importances'].items():
                print(f"  • {feature}: {importance:.4f}")
        
        if 'dataset_info' in self.results:
            print(f"\n🔸 Informações do Dataset:")
            info = self.results['dataset_info']
            print(f"  • Total de amostras rotuladas: {info['total_labeled']}")
            print(f"  • Amostras de treino (regressão): {info['train_reg']}")
            print(f"  • Amostras de validação (regressão): {info['val_reg']}")
            print(f"  • Amostras de treino (classificação): {info['train_clf']}")
            print(f"  • Amostras de validação (classificação): {info['val_clf']}")
            print(f"  • Amostras do conjunto cego: {info['blind_samples']}")
        
        print(f"\n📁 ARQUIVOS GERADOS:")
        print("-" * 50)
        print(f"  • Predições CSV: {self.config.OUTPUT_CSV_FILENAME}")
        print(f"  • Árvore de Decisão: {self.config.DECISION_TREE_IMAGE_FILENAME}")
        
        print("\n" + "="*80)
        print("FIM DO RESUMO")
        print("="*80 + "\n")

class DecisionTreeAnalysisPipeline:
    def __init__(self, regressor_params: Dict[str, Any], classifier_params: Dict[str, Any]):
        self.config = Config(
            REGRESSOR_PARAMS=regressor_params,
            CLASSIFIER_PARAMS=classifier_params
        )
        self.data_loader = DataLoader(self.config)
        self.model_trainer = ModelTrainer(self.config)
        self.model_evaluator = ModelEvaluator(self.config)
        self.visualizer = Visualizer(self.config)
        self.prediction_saver = PredictionSaver(self.config)
        self.results_summary = ResultsSummary(self.config)

    def run(self) -> None:
        print("\\n--- Carregando Dados Rotulados ---")
        X_labeled, y_reg_labeled, y_clf_labeled = self.data_loader.get_labeled_data()

        print("\\n\\n--- Dividindo Dados para Treino e Validação ---")
        X_train_reg, X_val_reg, y_train_reg, y_val_reg = train_test_split(
            X_labeled, y_reg_labeled, test_size=self.config.TEST_SPLIT_SIZE,
            random_state=self.config.RANDOM_STATE_SPLIT, shuffle=True
        )
        
        skf_splitter = StratifiedKFold(n_splits=self.config.CV_SPLITS, shuffle=True, random_state=self.config.RANDOM_STATE_SPLIT)
        X_train_clf, X_val_clf, y_train_clf, y_val_clf = train_test_split(
            X_labeled, y_clf_labeled, test_size=self.config.TEST_SPLIT_SIZE,
            random_state=self.config.RANDOM_STATE_SPLIT, shuffle=True, stratify=y_clf_labeled
        )

        print("\\n\\n--- Treinamento do Regressor ---")
        best_regressor = self.model_trainer.train_regressor(X_train_reg, y_train_reg)

        print("\\n\\n--- Treinamento do Classificador ---")
        best_classifier = self.model_trainer.train_classifier(X_train_clf, y_train_clf)

        print("\\n\\n--- Avaliação dos Modelos no Conjunto de Validação ---")
        validation_rmse = self.model_evaluator.evaluate_regressor(best_regressor, X_val_reg, y_val_reg)
        validation_accuracy, _, cm_val = self.model_evaluator.evaluate_classifier(best_classifier, X_val_clf, y_val_clf)
        self.visualizer.plot_confusion_matrix_heatmap(
            cm_val, classes=[str(c) for c in np.unique(y_val_clf)],
            title="Matriz de Confusão - Validação"
        )

        print("\n\n--- Importância das Features ---")
        self.visualizer.display_feature_importances(best_regressor, self.config.FEATURE_COLUMNS, "DecisionTreeRegressor ()")
        self.visualizer.display_feature_importances(best_classifier, self.config.FEATURE_COLUMNS, "DecisionTreeClassifier ()")

        print("\n\n--- Carregando Dados Cegos e Realizando Predições ---")
        X_blind, gi_blind_true, ids_blind = self.data_loader.get_blind_data()

        blind_rmse = self.model_evaluator.calculate_blind_rmse(best_regressor, X_blind, gi_blind_true)
        preds_yi_blind = best_classifier.predict(X_blind)
        preds_gi_blind = best_regressor.predict(X_blind)

        self.prediction_saver.save_blind_predictions(ids_blind, preds_gi_blind, preds_yi_blind)

        print("\n\n--- Gerando Curvas de Aprendizado ---")
        rmse_scorer = make_scorer(lambda y, y_pred: np.sqrt(mean_squared_error(y, y_pred)), greater_is_better=False)
        self.visualizer.plot_learning_curves(
            best_regressor, "Regressão", X_labeled, y_reg_labeled,
            cv=self.config.CV_SPLITS, scoring=rmse_scorer, y_label="RMSE", higher_is_better=False
        )
        self.visualizer.plot_learning_curves(
            best_classifier, "Classificação", X_labeled, y_clf_labeled,
            cv=skf_splitter, scoring="accuracy", y_label="Acurácia", higher_is_better=True
        )

        print("\n\n--- Gerando Árvore de Decisão do Classificador ---")
        self.visualizer.plot_and_save_decision_tree(
            best_classifier,
            feature_names=self.config.FEATURE_COLUMNS,
            class_names=[str(c) for c in best_classifier.classes_]
        )

        self.results_summary.add_result("validation_rmse", validation_rmse)
        self.results_summary.add_result("blind_rmse", blind_rmse)
        self.results_summary.add_result("validation_accuracy", validation_accuracy)
        self.results_summary.add_result("regressor_importances", dict(zip(self.config.FEATURE_COLUMNS, best_regressor.feature_importances_)))
        self.results_summary.add_result("classifier_importances", dict(zip(self.config.FEATURE_COLUMNS, best_classifier.feature_importances_)))
        self.results_summary.add_result("dataset_info", {
            "total_labeled": len(X_labeled),
            "train_reg": len(X_train_reg),
            "val_reg": len(X_val_reg),
            "train_clf": len(X_train_clf),
            "val_clf": len(X_val_clf),
            "blind_samples": len(X_blind)
        })

        print("\n\n--- Análise Concluída ---\n")
        self.results_summary.display_complete_summary()

if __name__ == "__main__":
    custom_regressor_params = {
        "max_depth": 7,
        "min_samples_leaf": 10,
        "max_leaf_nodes": 30,
        "min_impurity_decrease": 0.001,
        "ccp_alpha": 0.01
    }

    custom_classifier_params = {
        "max_depth": 7,
        "min_samples_leaf": 5,
        "criterion": "gini",
        "class_weight": "balanced",
        "ccp_alpha": 0.001
    }

    pipeline = DecisionTreeAnalysisPipeline(
        regressor_params=custom_regressor_params,
        classifier_params=custom_classifier_params
    )
    pipeline.run()