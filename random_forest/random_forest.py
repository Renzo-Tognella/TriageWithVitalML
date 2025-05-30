import numpy as np
import pandas as pd
from collections import Counter
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score, confusion_matrix, classification_report
import time
from datetime import datetime
from typing import Dict, Any, List

# Fixed hyperparameters
FIXED_HYPERPARAMS_REGRESSOR = {
    "n_trees": 200,
    "max_depth": None,
    "min_samples_to_split": 3,
    "min_samples_leaf": 1,
    "n_features_per_tree": "all",
    "random_state": 77,
    "ccp_alpha": 0.0050
}

FIXED_HYPERPARAMS_CLASSIFIER = {
    "n_trees": 20,
    "max_depth": 8,
    "min_samples_to_split": 3,
    "min_samples_leaf": 2,
    "n_features_per_tree": "sqrt",
    "random_state": 123,
    "ccp_alpha": 0.0030,
    "criterion": "gini"
}

RANDOM_SEED = 77

class ResultsSummary:
    def __init__(self):
        self.parameters = {}
        self.results = {}
        self.timing_info = {}
        self.dataset_info = {}

    def add_parameter(self, name, value):
        self.parameters[name] = value

    def add_result(self, name, value):
        self.results[name] = value

    def add_timing(self, name, time_seconds):
        self.timing_info[name] = time_seconds

    def add_dataset_info(self, name, value):
        self.dataset_info[name] = value

    def display_complete_summary(self):
        print("\n" + "="*60)
        print("RESUMO COMPLETO DO EXPERIMENTO")
        print("="*60)
        
        print("\nPARÂMETROS:")
        print("-" * 40)
        for name, value in self.parameters.items():
            print(f"  • {name}: {value}")
        
        print("\nINFORMAÇÕES DO DATASET:")
        print("-" * 40)
        for name, value in self.dataset_info.items():
            print(f"  • {name}: {value}")
        
        print("\nRESULTADOS:")
        print("-" * 40)
        for name, value in self.results.items():
            if isinstance(value, float):
                print(f"  • {name}: {value:.4f}")
            else:
                print(f"  • {name}: {value}")
        
        print("\nTEMPOS DE EXECUÇÃO:")
        print("-" * 40)
        for name, time_val in self.timing_info.items():
            print(f"  • {name}: {time_val:.2f}s")
        
        print("="*60)

class Node:
    def __init__(self, feature_index=None, threshold=None, left_child=None, right_child=None, info_gain=None, value=None):
        self.feature_index = feature_index
        self.threshold = threshold
        self.left_child = left_child
        self.right_child = right_child
        self.info_gain = info_gain
        self.value = value

class DecisionTree:
    def __init__(self, min_samples_to_split=2, max_depth=100, n_features_per_split=None, min_samples_leaf=1, ccp_alpha=0.0):
        self.min_samples_to_split = min_samples_to_split
        self.max_depth = max_depth
        self.n_features_per_split = n_features_per_split
        self.min_samples_leaf = min_samples_leaf
        self.ccp_alpha = ccp_alpha
        self.root = None

    def fit(self, X, y):
        self.root = self._build_tree(X, y, 0)

    def _build_tree(self, X, y, depth):
        n_samples, n_features = X.shape
        
        if (n_samples < self.min_samples_to_split or
            (self.max_depth is not None and depth >= self.max_depth) or
            len(np.unique(y)) == 1 or
            n_samples < 2 * self.min_samples_leaf):
            leaf_value = self._calculate_leaf_value(y)
            return Node(value=leaf_value)

        if self.n_features_per_split is None or self.n_features_per_split == n_features:
            feature_indices = list(range(n_features))
        else:
            feature_indices = np.random.choice(n_features, self.n_features_per_split, replace=False)

        best_split = self._get_best_split(X, y, feature_indices)
        
        if best_split is None or best_split['info_gain'] <= 0:
            leaf_value = self._calculate_leaf_value(y)
            return Node(value=leaf_value)

        left_indices = X[:, best_split['feature_index']] <= best_split['threshold']
        right_indices = ~left_indices
        
        if np.sum(left_indices) < self.min_samples_leaf or np.sum(right_indices) < self.min_samples_leaf:
            leaf_value = self._calculate_leaf_value(y)
            return Node(value=leaf_value)

        left_child = self._build_tree(X[left_indices], y[left_indices], depth + 1)
        right_child = self._build_tree(X[right_indices], y[right_indices], depth + 1)

        return Node(
            feature_index=best_split['feature_index'],
            threshold=best_split['threshold'],
            left_child=left_child,
            right_child=right_child,
            info_gain=best_split['info_gain']
        )

    def predict(self, X):
        predictions = [self._traverse_tree(x, self.root) for x in X]
        return np.array(predictions)

    def _traverse_tree(self, x, node):
        if node.value is not None:
            return node.value
        
        if x[node.feature_index] <= node.threshold:
            return self._traverse_tree(x, node.left_child)
        else:
            return self._traverse_tree(x, node.right_child)

class DecisionTreeRegressor(DecisionTree):
    def _calculate_mse(self, y):
        if len(y) == 0:
            return 0
        mean_y = np.mean(y)
        return np.mean((y - mean_y) ** 2)

    def _get_best_split(self, X, y, feature_indices):
        best_split = None
        best_info_gain = -1

        for feature_index in feature_indices:
            feature_values = X[:, feature_index]
            possible_thresholds = np.unique(feature_values)
            
            for threshold in possible_thresholds:
                left_indices = feature_values <= threshold
                right_indices = ~left_indices
                
                if np.sum(left_indices) == 0 or np.sum(right_indices) == 0:
                    continue

                y_left, y_right = y[left_indices], y[right_indices]
                
                if len(y_left) < self.min_samples_leaf or len(y_right) < self.min_samples_leaf:
                    continue

                parent_mse = self._calculate_mse(y)
                left_mse = self._calculate_mse(y_left)
                right_mse = self._calculate_mse(y_right)

                n = len(y)
                n_left = len(y_left)
                n_right = len(y_right)

                weighted_avg_mse = (n_left / n) * left_mse + (n_right / n) * right_mse
                info_gain = parent_mse - weighted_avg_mse

                if info_gain > best_info_gain:
                    best_info_gain = info_gain
                    best_split = {
                        'feature_index': feature_index,
                        'threshold': threshold,
                        'info_gain': info_gain
                    }

        return best_split

    def _calculate_leaf_value(self, y):
        return np.mean(y)

class DecisionTreeClassifier(DecisionTree):
    def __init__(self, min_samples_to_split=2, max_depth=100, n_features_per_split=None, min_samples_leaf=1, ccp_alpha=0.0, criterion='gini'):
        super().__init__(min_samples_to_split, max_depth, n_features_per_split, min_samples_leaf, ccp_alpha)
        self.criterion = criterion

    def _calculate_gini_impurity(self, y):
        classes, counts = np.unique(y, return_counts=True)
        probabilities = counts / len(y)
        gini = 1 - np.sum(probabilities ** 2)
        return gini

    def _calculate_entropy(self, y):
        classes, counts = np.unique(y, return_counts=True)
        probabilities = counts / len(y)
        entropy = -np.sum(probabilities * np.log2(probabilities + 1e-10))
        return entropy

    def _get_best_split(self, X, y, feature_indices):
        best_split = None
        best_info_gain = -1

        if self.criterion == 'gini':
            parent_impurity = self._calculate_gini_impurity(y)
        else:
            parent_impurity = self._calculate_entropy(y)

        for feature_index in feature_indices:
            feature_values = X[:, feature_index]
            possible_thresholds = np.unique(feature_values)
            
            for threshold in possible_thresholds:
                left_indices = feature_values <= threshold
                right_indices = ~left_indices
                
                if np.sum(left_indices) == 0 or np.sum(right_indices) == 0:
                    continue

                y_left, y_right = y[left_indices], y[right_indices]
                
                if len(y_left) < self.min_samples_leaf or len(y_right) < self.min_samples_leaf:
                    continue

                if self.criterion == 'gini':
                    left_impurity = self._calculate_gini_impurity(y_left)
                    right_impurity = self._calculate_gini_impurity(y_right)
                else:
                    left_impurity = self._calculate_entropy(y_left)
                    right_impurity = self._calculate_entropy(y_right)

                n = len(y)
                n_left = len(y_left)
                n_right = len(y_right)

                weighted_avg_impurity = (n_left / n) * left_impurity + (n_right / n) * right_impurity
                info_gain = parent_impurity - weighted_avg_impurity

                if info_gain > best_info_gain:
                    best_info_gain = info_gain
                    best_split = {
                        'feature_index': feature_index,
                        'threshold': threshold,
                        'info_gain': info_gain
                    }

        return best_split

    def _calculate_leaf_value(self, y):
        classes, counts = np.unique(y, return_counts=True)
        most_common_index = np.argmax(counts)
        return classes[most_common_index]

class RandomForest:
    def __init__(self, n_trees=100, min_samples_to_split=2, max_depth=100, 
                 n_features_per_tree='sqrt', tree_model_class=None, random_state=None, 
                 min_samples_leaf=1, ccp_alpha=0.0):
        self.n_trees = n_trees
        self.min_samples_to_split = min_samples_to_split
        self.max_depth = max_depth
        self.n_features_per_tree = n_features_per_tree
        self.tree_model_class = tree_model_class
        self.random_state = random_state
        self.min_samples_leaf = min_samples_leaf
        self.ccp_alpha = ccp_alpha
        self.trees = []
        self.feature_importances_ = None

        if self.random_state is not None:
            np.random.seed(self.random_state)

    def _bootstrap_sample(self, X, y):
        n_samples = X.shape[0]
        indices = np.random.choice(n_samples, n_samples, replace=True)
        return X[indices], y[indices]

    def fit(self, X_df, y_series):
        X = X_df.values if isinstance(X_df, pd.DataFrame) else X_df
        y = y_series.values if isinstance(y_series, pd.Series) else y_series
        
        n_features = X.shape[1]
        if self.n_features_per_tree == 'sqrt':
            self.n_features_per_tree_actual = int(np.sqrt(n_features))
        elif self.n_features_per_tree == 'all':
            self.n_features_per_tree_actual = n_features
        else:
            self.n_features_per_tree_actual = min(self.n_features_per_tree, n_features)
        
        print(f"  Treinando {self.n_trees} árvores...")
        self.trees = []
        feature_importances = np.zeros(X.shape[1])
        
        for i in range(self.n_trees):
            X_sample, y_sample = self._bootstrap_sample(X, y)
            
            tree = self.tree_model_class(
                min_samples_to_split=self.min_samples_to_split,
                max_depth=self.max_depth,
                n_features_per_split=self.n_features_per_tree_actual,
                min_samples_leaf=self.min_samples_leaf,
                ccp_alpha=self.ccp_alpha
            )
            tree.fit(X_sample, y_sample)
            self.trees.append(tree)
            
            if tree.root:
                self._accumulate_feature_importance(tree.root, feature_importances)
        
        if np.sum(feature_importances) > 0:
            self.feature_importances_ = feature_importances / np.sum(feature_importances)
        else:
            self.feature_importances_ = np.zeros(X.shape[1])
        
        print(f"  Treinamento de {self.n_trees} árvores concluído!")

    def _accumulate_feature_importance(self, node, importances):
        if node.feature_index is not None and node.info_gain is not None:
            importances[node.feature_index] += node.info_gain
            if node.left_child:
                self._accumulate_feature_importance(node.left_child, importances)
            if node.right_child:
                self._accumulate_feature_importance(node.right_child, importances)

    def predict(self, X_df):
        X = X_df.values if isinstance(X_df, pd.DataFrame) else X_df
        tree_predictions_list = []
        
        print(f"  Realizando predições com {len(self.trees)} árvores...")
        for tree in self.trees:
            tree_predictions_list.append(tree.predict(X))
        
        if not tree_predictions_list:
            return np.array([])

        tree_preds_stacked = np.stack(tree_predictions_list, axis=0)
        tree_preds_transposed = tree_preds_stacked.T
        
        predictions = self._aggregate_predictions(tree_preds_transposed)
        print(f"  Predições concluídas para {len(predictions)} amostras!")
        return predictions

    def get_feature_importance(self, feature_names):
        if self.feature_importances_ is not None:
            importance_dict = {}
            for i, importance in enumerate(self.feature_importances_):
                importance_dict[feature_names[i]] = importance
            return importance_dict
        return {}

    def _aggregate_predictions(self, tree_preds_transposed):
        raise NotImplementedError

class RandomForestRegressor(RandomForest):
    def __init__(self, n_trees=100, min_samples_to_split=2, max_depth=100, n_features_per_tree='sqrt', random_state=None, min_samples_leaf=1, ccp_alpha=0.0):
        super().__init__(n_trees, min_samples_to_split, max_depth, n_features_per_tree, 
                         tree_model_class=DecisionTreeRegressor, random_state=random_state, 
                         min_samples_leaf=min_samples_leaf, ccp_alpha=ccp_alpha)

    def _aggregate_predictions(self, tree_preds_transposed):
        return np.mean(tree_preds_transposed, axis=1)

class RandomForestClassifier(RandomForest):
    def __init__(self, n_trees=100, min_samples_to_split=2, max_depth=100, n_features_per_tree='sqrt', random_state=None, min_samples_leaf=1, ccp_alpha=0.0, criterion='gini'):
        super().__init__(n_trees, min_samples_to_split, max_depth, n_features_per_tree,
                         tree_model_class=DecisionTreeClassifier, random_state=random_state, 
                         min_samples_leaf=min_samples_leaf, ccp_alpha=ccp_alpha)
        self.criterion = criterion

    def fit(self, X_df, y_series):
        print(f"  Treinando {self.n_trees} árvores com critério '{self.criterion}'...")
        self.trees = []
        
        X = X_df.values if isinstance(X_df, pd.DataFrame) else X_df
        y = y_series.values if isinstance(y_series, pd.Series) else y_series
        
        n_features = X.shape[1]
        if self.n_features_per_tree == 'sqrt':
            self.n_features_per_tree_actual = int(np.sqrt(n_features))
        elif self.n_features_per_tree == 'all':
            self.n_features_per_tree_actual = n_features
        else:
            self.n_features_per_tree_actual = min(self.n_features_per_tree, n_features)
        
        feature_importances = np.zeros(X.shape[1])
        
        for i in range(self.n_trees):
            X_sample, y_sample = self._bootstrap_sample(X, y)
            
            tree = DecisionTreeClassifier(
                min_samples_to_split=self.min_samples_to_split,
                max_depth=self.max_depth,
                n_features_per_split=self.n_features_per_tree_actual,
                min_samples_leaf=self.min_samples_leaf,
                ccp_alpha=self.ccp_alpha,
                criterion=self.criterion
            )
            tree.fit(X_sample, y_sample)
            self.trees.append(tree)
            
            if tree.root:
                self._accumulate_feature_importance(tree.root, feature_importances)
        
        if np.sum(feature_importances) > 0:
            self.feature_importances_ = feature_importances / np.sum(feature_importances)
        else:
            self.feature_importances_ = np.zeros(X.shape[1])
        
        print(f"  Treinamento de {self.n_trees} árvores concluído!")

    def _accumulate_feature_importance(self, node, importances):
        if node.feature_index is not None and node.info_gain is not None:
            importances[node.feature_index] += node.info_gain
            if node.left_child:
                self._accumulate_feature_importance(node.left_child, importances)
            if node.right_child:
                self._accumulate_feature_importance(node.right_child, importances)

    def predict(self, X_df):
        X = X_df.values if isinstance(X_df, pd.DataFrame) else X_df
        tree_predictions_list = []
        
        print(f"  Realizando predições com {len(self.trees)} árvores...")
        for tree in self.trees:
            tree_predictions_list.append(tree.predict(X))
        
        if not tree_predictions_list:
            return np.array([])

        tree_preds_stacked = np.stack(tree_predictions_list, axis=0)
        tree_preds_transposed = tree_preds_stacked.T
        
        predictions = self._aggregate_predictions(tree_preds_transposed)
        print(f"  Predições concluídas para {len(predictions)} amostras!")
        return predictions

    def get_feature_importance(self, feature_names):
        if self.feature_importances_ is not None:
            importance_dict = {}
            for i, importance in enumerate(self.feature_importances_):
                importance_dict[feature_names[i]] = importance
            return importance_dict
        return {}

    def _aggregate_predictions(self, tree_preds_transposed):
        y_pred_aggregated = []
        for sample_predictions in tree_preds_transposed:
            most_common = Counter(sample_predictions).most_common(1)
            y_pred_aggregated.append(most_common[0][0] if most_common else None)
        return np.array(y_pred_aggregated)

def display_feature_importance(rf_model, feature_names, model_name):
    """Exibe importância das features"""
    importance_dict = rf_model.get_feature_importance(feature_names)
    if importance_dict:
        print(f"\nImportância das Features - {model_name}:")
        print("-" * 40)
        sorted_features = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
        for feature, importance in sorted_features:
            print(f"  • {feature}: {importance:.4f}")
        print()

def evaluate_model_performance(y_true, y_pred, model_type="Modelo"):
    """Avalia performance do modelo com métricas detalhadas"""
    print(f"\nAnálise Detalhada - {model_type}:")
    print("-" * 40)
    
    if model_type.lower().find('regressor') != -1 or model_type.lower().find('regressão') != -1:
        # Métricas de regressão
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_true, y_pred)
        mae = np.mean(np.abs(y_true - y_pred))
        
        print(f"  • RMSE: {rmse:.4f}")
        print(f"  • MSE: {mse:.4f}")
        print(f"  • MAE: {mae:.4f}")
        print(f"  • R²: {r2:.4f}")
        
        # Análise dos resíduos
        residuals = y_true - y_pred
        print(f"  • Resíduo Médio: {np.mean(residuals):.4f}")
        print(f"  • Desvio Padrão dos Resíduos: {np.std(residuals):.4f}")
        print(f"  • Resíduo Mín: {np.min(residuals):.4f}")
        print(f"  • Resíduo Máx: {np.max(residuals):.4f}")
        
        return {'rmse': rmse, 'mse': mse, 'mae': mae, 'r2': r2}
    
    else:
        # Métricas de classificação
        accuracy = accuracy_score(y_true, y_pred)
        print(f"  • Acurácia: {accuracy:.4f}")
        
        # Distribuição de classes
        unique_true = np.unique(y_true)
        unique_pred = np.unique(y_pred)
        print(f"  • Classes Verdadeiras: {unique_true}")
        print(f"  • Classes Preditas: {unique_pred}")
        
        return {'accuracy': accuracy}

def cross_validate_model(rf_model, X, y, cv_folds=5, model_type="Modelo"):
    """Realiza validação cruzada"""
    print(f"\nValidação Cruzada ({cv_folds} folds) - {model_type}:")
    print("-" * 40)
    
    try:
        if model_type.lower().find('regressor') != -1:
            # Para regressão, usar RMSE negativo
            scores = []
            kf = KFold(n_splits=cv_folds, shuffle=True, random_state=RANDOM_SEED)
            
            for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
                X_train_fold = X[train_idx] if isinstance(X, np.ndarray) else X.iloc[train_idx]
                X_val_fold = X[val_idx] if isinstance(X, np.ndarray) else X.iloc[val_idx]
                y_train_fold = y[train_idx] if isinstance(y, np.ndarray) else y.iloc[train_idx]
                y_val_fold = y[val_idx] if isinstance(y, np.ndarray) else y.iloc[val_idx]
                
                fold_model = type(rf_model)(
                    n_trees=rf_model.n_trees,
                    max_depth=rf_model.max_depth,
                    min_samples_to_split=rf_model.min_samples_to_split,
                    n_features_per_tree=rf_model.n_features_per_tree,
                    random_state=rf_model.random_state,
                    min_samples_leaf=rf_model.min_samples_leaf,
                    ccp_alpha=rf_model.ccp_alpha
                )
                fold_model.fit(X_train_fold, y_train_fold)
                y_pred_fold = fold_model.predict(X_val_fold)
                rmse_fold = np.sqrt(mean_squared_error(y_val_fold, y_pred_fold))
                scores.append(rmse_fold)
                print(f"  • Fold {fold+1}: RMSE = {rmse_fold:.4f}")
            
            mean_score = np.mean(scores)
            std_score = np.std(scores)
            print(f"  • RMSE Médio: {mean_score:.4f} (±{std_score:.4f})")
            return mean_score, std_score
        
        else:
            # Para classificação
            scores = []
            skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=RANDOM_SEED)
            
            for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
                X_train_fold = X[train_idx] if isinstance(X, np.ndarray) else X.iloc[train_idx]
                X_val_fold = X[val_idx] if isinstance(X, np.ndarray) else X.iloc[val_idx]
                y_train_fold = y[train_idx] if isinstance(y, np.ndarray) else y.iloc[train_idx]
                y_val_fold = y[val_idx] if isinstance(y, np.ndarray) else y.iloc[val_idx]
                
                fold_model = type(rf_model)(
                    n_trees=rf_model.n_trees,
                    max_depth=rf_model.max_depth,
                    min_samples_to_split=rf_model.min_samples_to_split,
                    n_features_per_tree=rf_model.n_features_per_tree,
                    random_state=rf_model.random_state,
                    min_samples_leaf=rf_model.min_samples_leaf,
                    ccp_alpha=rf_model.ccp_alpha,
                    criterion=getattr(rf_model, 'criterion', 'gini')
                )
                fold_model.fit(X_train_fold, y_train_fold)
                y_pred_fold = fold_model.predict(X_val_fold)
                acc_fold = accuracy_score(y_val_fold, y_pred_fold)
                scores.append(acc_fold)
                print(f"  • Fold {fold+1}: Acurácia = {acc_fold:.4f}")
            
            mean_score = np.mean(scores)
            std_score = np.std(scores)
            print(f"  • Acurácia Média: {mean_score:.4f} (±{std_score:.4f})")
            return mean_score, std_score
            
    except Exception as e:
        print(f"  Erro na validação cruzada: {e}")
        return None, None

def run_random_forest_algorithm():
    overall_script_start_time = time.time()
    print("INICIANDO ALGORITMO RANDOM FOREST (IMPLEMENTAÇÃO PRÓPRIA)")
    print("="*80)

    # Configurações fixas
    features_cols = ['si3_qPA', 'si4_pulso', 'si5_resp'] 
    target_reg_col = 'g1_gravid'
    target_cls_col = 'y1_classe'
    test_size = 0.2
    cv_folds = 5

    # Usar hiperparâmetros fixos separados
    reg_params = FIXED_HYPERPARAMS_REGRESSOR
    clf_params = FIXED_HYPERPARAMS_CLASSIFIER

    print(f"\nHiperparâmetros Regressor:")
    for key, value in reg_params.items():
        print(f"  • {key}: {value}")
    
    print(f"\nHiperparâmetros Classificador:")
    for key, value in clf_params.items():
        print(f"  • {key}: {value}")

    experiment_start_time = time.time()
    summary = ResultsSummary()

    # Salvar parâmetros no resumo
    summary.add_parameter("Regressor - Número de Árvores", reg_params["n_trees"])
    summary.add_parameter("Regressor - Profundidade Máxima", reg_params["max_depth"])
    summary.add_parameter("Regressor - Min Samples Split", reg_params["min_samples_to_split"])
    summary.add_parameter("Regressor - Min Samples Leaf", reg_params["min_samples_leaf"])
    summary.add_parameter("Regressor - Features por Árvore", reg_params["n_features_per_tree"])
    summary.add_parameter("Regressor - Random State", reg_params["random_state"])
    summary.add_parameter("Regressor - CCP Alpha", reg_params["ccp_alpha"])
    
    summary.add_parameter("Classificador - Número de Árvores", clf_params["n_trees"])
    summary.add_parameter("Classificador - Profundidade Máxima", clf_params["max_depth"])
    summary.add_parameter("Classificador - Min Samples Split", clf_params["min_samples_to_split"])
    summary.add_parameter("Classificador - Min Samples Leaf", clf_params["min_samples_leaf"])
    summary.add_parameter("Classificador - Features por Árvore", clf_params["n_features_per_tree"])
    summary.add_parameter("Classificador - Random State", clf_params["random_state"])
    summary.add_parameter("Classificador - CCP Alpha", clf_params["ccp_alpha"])
    summary.add_parameter("Classificador - Critério", clf_params["criterion"])
    
    summary.add_parameter("Test Size", test_size)
    summary.add_parameter("CV Folds", cv_folds)
    summary.add_parameter("Features Utilizadas", features_cols)

    # Carregar dados dos arquivos CSV
    print(f"\nCarregando dados dos arquivos CSV...")
    try:
        # Ler arquivo com labels (treino_sinais_vitais_com_label.csv)
        df_hist = pd.read_csv('treino_sinais_vitais_com_label.csv', header=None)
        
        # Definir nomes das colunas baseado na estrutura observada
        df_hist.columns = ['i', 'si1_pSist', 'si2_pDiast', 'si3_qPA', 'si4_pulso', 'si5_resp', 'g1_gravid', 'y1_classe']
        
        load_time = time.time() - experiment_start_time
        print(f"Dados carregados! Formato: {df_hist.shape}, Tempo: {load_time:.2f}s")
        print(f"Colunas: {list(df_hist.columns)}")
        print(f"Primeiras 3 linhas:")
        print(df_hist.head(3))
        
    except Exception as e:
        print(f"Erro ao carregar dados: {e}")
        print("Verifique se o arquivo 'treino_sinais_vitais_com_label.csv' existe no diretório atual.")
        return

    # Preparar dados (X, y)
    X_df = df_hist[features_cols]
    y_reg_series = df_hist[target_reg_col]
    y_cls_series = df_hist[target_cls_col].astype(int)
    unique_classes = sorted(y_cls_series.unique())
    class_names = [f"Class {c}" for c in unique_classes]

    print(f"\nEstatísticas dos dados:")
    print(f"  • Total de amostras: {len(df_hist)}")
    print(f"  • Features: {features_cols}")
    print(f"  • Classes únicas: {unique_classes}")
    print(f"  • Distribuição de classes: {dict(y_cls_series.value_counts().sort_index())}")
    print(f"  • Estatísticas do target de regressão:")
    print(f"    - Média: {y_reg_series.mean():.2f}")
    print(f"    - Min: {y_reg_series.min():.2f}")
    print(f"    - Max: {y_reg_series.max():.2f}")

    summary.add_dataset_info("Total de Amostras", len(df_hist))
    summary.add_dataset_info("Número de Features", len(features_cols))
    summary.add_dataset_info("Classes Únicas", unique_classes)
    summary.add_dataset_info("Distribuição Classes", dict(y_cls_series.value_counts().sort_index()))

    # Divisão treino/validação
    print(f"\nDividindo dados (treino: {1-test_size:.0%}, validação: {test_size:.0%})...")
    split_start = time.time()
    stratify_option = y_cls_series if len(unique_classes) > 1 and y_cls_series.value_counts().min() >= cv_folds and y_cls_series.value_counts().min() > 1 else None
    
    X_train_df, X_val_df, y_train_reg, y_val_reg, y_train_cls, y_val_cls = train_test_split(
        X_df, y_reg_series, y_cls_series, test_size=test_size, 
        random_state=reg_params["random_state"], stratify=stratify_option
    )
    split_time = time.time() - split_start
    summary.add_timing("Divisão dos Dados", split_time)
    summary.add_dataset_info("Amostras Treino", len(X_train_df))
    summary.add_dataset_info("Amostras Validação", len(X_val_df))
    print(f"Divisão concluída! Treino: {len(X_train_df)}, Validação: {len(X_val_df)}. Tempo: {split_time:.2f}s")

    # REGRESSÃO
    print("\n" + "-"*40); print("TAREFA DE REGRESSÃO"); print("-"*40)
    reg_start = time.time()
    rf_regressor = RandomForestRegressor(
        n_trees=reg_params["n_trees"], 
        max_depth=reg_params["max_depth"], 
        min_samples_to_split=reg_params["min_samples_to_split"],
        n_features_per_tree=reg_params["n_features_per_tree"], 
        random_state=reg_params["random_state"],
        min_samples_leaf=reg_params["min_samples_leaf"], 
        ccp_alpha=reg_params["ccp_alpha"]
    )
    rf_regressor.fit(X_train_df, y_train_reg)
    y_pred_reg_val = rf_regressor.predict(X_val_df)
    reg_metrics = evaluate_model_performance(y_val_reg, y_pred_reg_val, "Random Forest Regressor")
    reg_train_time = time.time() - reg_start
    summary.add_timing("Treinamento Regressor", reg_train_time)
    
    cv_mean_reg, cv_std_reg = cross_validate_model(rf_regressor, X_df, y_reg_series, cv_folds, "Regressor")
    display_feature_importance(rf_regressor, features_cols, "Random Forest Regressor")
    
    summary.add_result("RMSE Regressor (Validação)", reg_metrics['rmse'])
    summary.add_result("R² Regressor", reg_metrics['r2'])
    summary.add_result("MAE Regressor", reg_metrics['mae'])
    if cv_mean_reg is not None: summary.add_result("RMSE CV Médio (Regressor)", cv_mean_reg)
    if cv_std_reg is not None: summary.add_result("RMSE CV Std (Regressor)", cv_std_reg)

    # CLASSIFICAÇÃO
    print("\n" + "-"*40); print("TAREFA DE CLASSIFICAÇÃO"); print("-"*40)
    clf_start = time.time()
    rf_classifier = RandomForestClassifier(
        n_trees=clf_params["n_trees"], 
        max_depth=clf_params["max_depth"], 
        min_samples_to_split=clf_params["min_samples_to_split"],
        n_features_per_tree=clf_params["n_features_per_tree"], 
        random_state=clf_params["random_state"],
        min_samples_leaf=clf_params["min_samples_leaf"], 
        ccp_alpha=clf_params["ccp_alpha"], 
        criterion=clf_params["criterion"]
    )
    rf_classifier.fit(X_train_df, y_train_cls)
    y_pred_cls_val = rf_classifier.predict(X_val_df)
    clf_metrics = evaluate_model_performance(y_val_cls, y_pred_cls_val, "Random Forest Classifier")
    
    y_pred_cls_val_int = np.array(y_pred_cls_val).astype(int)
    y_val_cls_np_int = y_val_cls.values.astype(int)
    print(classification_report(y_val_cls_np_int, y_pred_cls_val_int, labels=unique_classes, target_names=class_names, zero_division=0))
    cm_rf = confusion_matrix(y_val_cls_np_int, y_pred_cls_val_int, labels=unique_classes)
    print(pd.DataFrame(cm_rf, index=[f"Real {c}" for c in unique_classes], columns=[f"Pred {c}" for c in unique_classes]))
    
    clf_train_time = time.time() - clf_start
    summary.add_timing("Treinamento Classificador", clf_train_time)

    cv_mean_clf, cv_std_clf = cross_validate_model(rf_classifier, X_df, y_cls_series, cv_folds, "Classificador")
    display_feature_importance(rf_classifier, features_cols, "Random Forest Classifier")

    summary.add_result("Acurácia Classificador (Validação)", clf_metrics['accuracy'])
    if cv_mean_clf is not None: summary.add_result("Acurácia CV Média (Classificador)", cv_mean_clf)
    if cv_std_clf is not None: summary.add_result("Acurácia CV Std (Classificador)", cv_std_clf)

    experiment_run_time = time.time() - experiment_start_time
    summary.add_timing("Tempo Total Experimento", experiment_run_time)

    summary.display_complete_summary()
    print(f"\nEXPERIMENTO CONCLUÍDO. Tempo: {experiment_run_time:.2f}s")

    overall_script_run_time = time.time() - overall_script_start_time
    print(f"Tempo total de execução: {overall_script_run_time:.2f}s")
    print("="*80)

if __name__ == "__main__":
    run_random_forest_algorithm()