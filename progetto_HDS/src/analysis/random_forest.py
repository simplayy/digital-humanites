"""
Implementazione di un modello Random Forest per l'analisi della relazione
tra feature di sentiment e veridicità delle notizie.

Questo modulo estende l'analisi statistica con approcci di machine learning
non lineari, per verificare se possono catturare pattern più complessi.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Any, Optional, Union
from pathlib import Path

# ML imports
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    roc_curve, precision_recall_curve, average_precision_score
)
from sklearn.inspection import permutation_importance


# Imposta lo stile di seaborn
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12

# Definizione percorsi
ROOT_DIR = Path(__file__).parent.parent.parent
DATA_DIR = ROOT_DIR / "data"
PROCESSED_DIR = DATA_DIR / "processed"
RESULTS_DIR = ROOT_DIR / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
TABLES_DIR = RESULTS_DIR / "tables"

# Assicurarsi che le directory esistano
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
TABLES_DIR.mkdir(parents=True, exist_ok=True)


def load_feature_matrix() -> pd.DataFrame:
    """
    Carica la matrice delle feature dal file CSV processato.
    
    Returns:
        pd.DataFrame: DataFrame contenente la matrice delle feature.
    """
    feature_matrix_path = PROCESSED_DIR / "pheme_feature_matrix.csv"
    if not feature_matrix_path.exists():
        raise FileNotFoundError(f"File non trovato: {feature_matrix_path}")
    
    return pd.read_csv(feature_matrix_path)


def prepare_data(df: pd.DataFrame, 
                target_col: str = 'veracity',
                feature_cols: Optional[List[str]] = None,
                test_size: float = 0.3,
                random_state: int = 42,
                stratify: bool = True) -> Dict[str, Any]:
    """
    Prepara i dati per l'addestramento del Random Forest.
    
    Args:
        df (pd.DataFrame): DataFrame contenente i dati.
        target_col (str): Nome della colonna target.
        feature_cols (List[str], optional): Lista di colonne da usare come feature.
        test_size (float): Proporzione del dataset da usare come test set.
        random_state (int): Seed per la riproducibilità.
        stratify (bool): Se True, stratifica il train/test split in base al target.
        
    Returns:
        Dict[str, Any]: Dizionario contenente i dati preparati.
    """
    # Verifica la presenza della colonna target
    if target_col not in df.columns:
        raise ValueError(f"La colonna target '{target_col}' non è presente nel DataFrame")
    
    # Se necessario, converti il target in formato numerico
    df_processed = df.copy()
    if target_col == 'veracity':
        # Trasforma il target in forma binaria (True=1, False=0)
        veracity_values = df[target_col].unique()
        
        # Verifica se ci sono valori 'true' e 'false'
        has_true_false = 'true' in veracity_values and 'false' in veracity_values
        
        if not has_true_false:
            # Prova con lowercase
            if 'true' in df[target_col].str.lower().unique() and 'false' in df[target_col].str.lower().unique():
                # Converti a lowercase
                df_processed[target_col] = df_processed[target_col].str.lower()
            else:
                raise ValueError(f"La colonna '{target_col}' non contiene i valori 'true' e 'false' necessari per l'analisi")
        
        # Filtra solo le righe con valori 'true' o 'false'
        df_processed = df_processed[df_processed[target_col].isin(['true', 'false'])]
        
        # Converti in formato numerico
        df_processed[target_col + '_numeric'] = df_processed[target_col].apply(lambda x: 1 if x == 'true' else 0)
        target_col = target_col + '_numeric'  # Aggiorna il nome della colonna target
    
    # Se non sono specificate le feature, seleziona automaticamente
    if feature_cols is None:
        # Seleziona tutte le colonne numeriche eccetto il target
        feature_cols = df_processed.select_dtypes(include=['number']).columns.tolist()
        # Esclude il target e altre colonne non rilevanti
        feature_cols = [col for col in feature_cols if col != target_col and not col.endswith('_numeric')]
        # Esclude colonne categoriche one-hot encoded
        feature_cols = [col for col in feature_cols if not (col.startswith('sentiment_category_') or 
                                                           col.startswith('stance_category_') or 
                                                           col.startswith('readability_category_'))]
    else:
        # Filtra solo le colonne che esistono nel DataFrame
        feature_cols = [col for col in feature_cols if col in df_processed.columns]
    
    if not feature_cols:
        raise ValueError("Nessuna feature valida specificata o trovata")
    
    # Rimuovi righe con valori nulli nelle feature o nel target
    cols_to_check = feature_cols + [target_col]
    df_clean = df_processed[cols_to_check].dropna()
    
    if len(df_clean) == 0:
        raise ValueError("Dopo la rimozione dei valori nulli, non ci sono righe rimanenti")
    
    # Separa feature e target
    X = df_clean[feature_cols]
    y = df_clean[target_col]
    
    # Split in training e test set
    if stratify:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
    
    # Standardizza le feature
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Prepara il risultato
    result = {
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'X_train_scaled': X_train_scaled,
        'X_test_scaled': X_test_scaled,
        'feature_cols': feature_cols,
        'target_col': target_col,
        'scaler': scaler,
        'n_samples': len(df_clean),
        'n_features': len(feature_cols),
        'target_distribution': y.value_counts(normalize=True).to_dict()
    }
    
    return result


def train_random_forest(data: Dict[str, Any], 
                      n_estimators: int = 100,
                      max_depth: Optional[int] = None,
                      min_samples_split: int = 2,
                      random_state: int = 42,
                      class_weight: Optional[str] = 'balanced') -> Dict[str, Any]:
    """
    Addestra un modello Random Forest sui dati forniti.
    
    Args:
        data (Dict[str, Any]): Dati preparati (output di prepare_data).
        n_estimators (int): Numero di alberi nel forest.
        max_depth (int, optional): Profondità massima degli alberi.
        min_samples_split (int): Numero minimo di campioni richiesti per dividere un nodo.
        random_state (int): Seed per la riproducibilità.
        class_weight (str, optional): Pesi delle classi ('balanced' o None).
        
    Returns:
        Dict[str, Any]: Dizionario con il modello e i risultati.
    """
    # Estrai i dati
    X_train = data['X_train_scaled']
    y_train = data['y_train']
    X_test = data['X_test_scaled']
    y_test = data['y_test']
    feature_cols = data['feature_cols']
    
    # Crea e addestra il modello
    rf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        random_state=random_state,
        class_weight=class_weight,
        n_jobs=-1  # Usa tutti i core disponibili
    )
    
    rf.fit(X_train, y_train)
    
    # Generazione predizioni
    y_pred = rf.predict(X_test)
    y_pred_proba = rf.predict_proba(X_test)[:, 1]  # Probabilità della classe positiva
    
    # Calcola le metriche di valutazione
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1_score': f1_score(y_test, y_pred),
        'roc_auc': roc_auc_score(y_test, y_pred_proba),
        'avg_precision': average_precision_score(y_test, y_pred_proba)
    }
    
    # Matrice di confusione
    conf_matrix = confusion_matrix(y_test, y_pred)
    
    # Feature importance
    importances = rf.feature_importances_
    
    # Calcola feature importance con permutation method (più robusto)
    perm_importance = permutation_importance(rf, X_test, y_test, 
                                           n_repeats=10, 
                                           random_state=random_state)
    
    # Combina i risultati in un DataFrame
    feature_importance_df = pd.DataFrame({
        'feature': feature_cols,
        'importance': importances,
        'perm_importance_mean': perm_importance.importances_mean,
        'perm_importance_std': perm_importance.importances_std
    }).sort_values('perm_importance_mean', ascending=False)
    
    # Valutazione con cross-validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    cv_scores = cross_val_score(rf, data['X_train_scaled'], data['y_train'], 
                              cv=cv, scoring='roc_auc')
    
    # Prepara il risultato
    results = {
        'model': rf,
        'metrics': metrics,
        'cv_scores': cv_scores,
        'cv_mean': cv_scores.mean(),
        'cv_std': cv_scores.std(),
        'confusion_matrix': conf_matrix,
        'feature_importance': feature_importance_df,
        'classification_report': classification_report(y_test, y_pred, output_dict=True),
        'y_pred': y_pred,
        'y_pred_proba': y_pred_proba,
        'y_test': y_test
    }
    
    return results


def plot_feature_importance(results: Dict[str, Any], 
                          top_n: int = 15,
                          use_permutation: bool = True,
                          filename: Optional[str] = None) -> plt.Figure:
    """
    Visualizza l'importanza delle feature nel modello Random Forest.
    
    Args:
        results (Dict[str, Any]): Risultati del Random Forest.
        top_n (int): Numero di feature da visualizzare.
        use_permutation (bool): Se True, usa l'importanza calcolata con il metodo permutation.
        filename (str, optional): Nome del file per salvare il grafico.
        
    Returns:
        plt.Figure: Figura matplotlib.
    """
    # Estrai l'importanza delle feature
    importance_df = results['feature_importance']
    
    # Ordina e seleziona le top_n feature
    if use_permutation:
        importance_df = importance_df.sort_values('perm_importance_mean', ascending=False).head(top_n)
        importance_values = importance_df['perm_importance_mean']
        importance_std = importance_df['perm_importance_std']
        importance_type = 'Permutation Importance'
    else:
        importance_df = importance_df.sort_values('importance', ascending=False).head(top_n)
        importance_values = importance_df['importance']
        importance_std = None
        importance_type = 'Feature Importance'
    
    # Crea il grafico
    plt.figure(figsize=(12, 8))
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Plot a barre
    y_pos = np.arange(len(importance_df))
    bars = ax.barh(y_pos, importance_values, align='center', alpha=0.8)
    
    # Aggiungi barre di errore se disponibili
    if importance_std is not None:
        ax.errorbar(importance_values, y_pos, xerr=importance_std, fmt='o', 
                   color='black', alpha=0.7, capsize=5)
    
    # Aggiungi etichette
    ax.set_yticks(y_pos)
    ax.set_yticklabels(importance_df['feature'])
    ax.invert_yaxis()  # Le feature con importanza maggiore in alto
    ax.set_xlabel(importance_type)
    ax.set_title(f'Top {top_n} Feature per Importanza nel Random Forest')
    
    # Etichette con il valore di importanza
    for i, v in enumerate(importance_values):
        ax.text(v + 0.01, i, f'{v:.4f}', va='center')
    
    plt.tight_layout()
    
    # Salva il grafico se richiesto
    if filename:
        save_path = FIGURES_DIR / filename
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_roc_curve_comparison(rf_results: Dict[str, Any], 
                            lr_results: Optional[Dict[str, Any]] = None,
                            filename: Optional[str] = None) -> plt.Figure:
    """
    Confronta le curve ROC del Random Forest e della regressione logistica (se disponibile).
    
    Args:
        rf_results (Dict[str, Any]): Risultati del Random Forest.
        lr_results (Dict[str, Any], optional): Risultati della regressione logistica.
        filename (str, optional): Nome del file per salvare il grafico.
        
    Returns:
        plt.Figure: Figura matplotlib.
    """
    # Estrai i dati per il Random Forest
    y_test_rf = rf_results['y_test']
    y_pred_proba_rf = rf_results['y_pred_proba']
    roc_auc_rf = rf_results['metrics']['roc_auc']
    
    # Calcola la curva ROC per il RF
    fpr_rf, tpr_rf, _ = roc_curve(y_test_rf, y_pred_proba_rf)
    
    # Crea il grafico
    plt.figure(figsize=(10, 8))
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Plot per il Random Forest
    ax.plot(fpr_rf, tpr_rf, label=f'Random Forest (AUC = {roc_auc_rf:.3f})', 
           linewidth=2, color='forestgreen')
    
    # Se disponibile, aggiungi la curva ROC per la regressione logistica
    if lr_results is not None:
        y_test_lr = lr_results['y_test']
        y_pred_proba_lr = lr_results['y_pred_proba']
        roc_auc_lr = lr_results['metrics']['auc']
        
        # Verifica che i test set siano gli stessi
        if len(y_test_lr) == len(y_test_rf) and (y_test_lr == y_test_rf).all():
            # Calcola la curva ROC per la LR
            fpr_lr, tpr_lr, _ = roc_curve(y_test_lr, y_pred_proba_lr)
            
            # Plot per la regressione logistica
            ax.plot(fpr_lr, tpr_lr, label=f'Regressione Logistica (AUC = {roc_auc_lr:.3f})', 
                   linewidth=2, color='royalblue', linestyle='--')
    
    # Aggiungi la linea diagonale (random classifier)
    ax.plot([0, 1], [0, 1], linestyle='--', color='gray', alpha=0.7, label='Random Classifier')
    
    # Aggiungi etichette e titolo
    ax.set_title('Curva ROC - Confronto tra Modelli')
    ax.set_xlabel('Tasso di falsi positivi (1 - Specificità)')
    ax.set_ylabel('Tasso di veri positivi (Sensibilità)')
    
    # Aggiungi griglia e legenda
    ax.grid(True, alpha=0.3)
    ax.legend(loc='lower right')
    
    # Miglioramenti estetici
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    
    plt.tight_layout()
    
    # Salva il grafico se richiesto
    if filename:
        save_path = FIGURES_DIR / filename
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def save_model_results(results: Dict[str, Any], 
                     metrics_filename: str = "random_forest_metrics.csv",
                     importance_filename: str = "random_forest_importance.csv") -> Tuple[str, str]:
    """
    Salva i risultati del modello Random Forest.
    
    Args:
        results (Dict[str, Any]): Risultati del Random Forest.
        metrics_filename (str): Nome del file per le metriche.
        importance_filename (str): Nome del file per l'importanza delle feature.
        
    Returns:
        Tuple[str, str]: Percorsi dei file salvati.
    """
    # Salva le metriche
    metrics_df = pd.DataFrame(results['metrics'], index=[0])
    metrics_df['cv_mean'] = results['cv_mean']
    metrics_df['cv_std'] = results['cv_std']
    
    metrics_path = TABLES_DIR / metrics_filename
    metrics_df.to_csv(metrics_path, index=False)
    
    # Salva l'importanza delle feature
    importance_path = TABLES_DIR / importance_filename
    results['feature_importance'].to_csv(importance_path, index=False)
    
    return str(metrics_path), str(importance_path)


def grid_search_rf(data: Dict[str, Any], 
                 param_grid: Dict[str, List[Any]],
                 cv: int = 5,
                 scoring: str = 'roc_auc') -> Dict[str, Any]:
    """
    Esegue una grid search per trovare i migliori parametri per il Random Forest.
    
    Args:
        data (Dict[str, Any]): Dati preparati (output di prepare_data).
        param_grid (Dict[str, List[Any]]): Griglia di parametri da testare.
        cv (int): Numero di fold per la cross-validation.
        scoring (str): Metrica da ottimizzare.
        
    Returns:
        Dict[str, Any]: Risultati della grid search.
    """
    from sklearn.model_selection import GridSearchCV
    
    # Estrai i dati
    X_train = data['X_train_scaled']
    y_train = data['y_train']
    
    # Crea il modello base
    rf = RandomForestClassifier(random_state=42, class_weight='balanced')
    
    # Configura e esegui la grid search
    grid_search = GridSearchCV(
        estimator=rf,
        param_grid=param_grid,
        cv=cv,
        scoring=scoring,
        n_jobs=-1,
        verbose=1,
        return_train_score=True
    )
    
    grid_search.fit(X_train, y_train)
    
    # Crea DataFrame con i risultati
    results_df = pd.DataFrame(grid_search.cv_results_)
    
    # Estrai i migliori parametri
    best_params = grid_search.best_params_
    best_score = grid_search.best_score_
    
    # Prepara il risultato
    results = {
        'grid_search': grid_search,
        'best_params': best_params,
        'best_score': best_score,
        'results_df': results_df,
        'best_estimator': grid_search.best_estimator_
    }
    
    return results


def compare_feature_sets(df: pd.DataFrame, 
                        feature_sets: Dict[str, List[str]],
                        target_col: str = 'veracity',
                        n_estimators: int = 100,
                        random_state: int = 42) -> Dict[str, Any]:
    """
    Confronta le performance del Random Forest con diversi set di feature.
    
    Args:
        df (pd.DataFrame): DataFrame contenente i dati.
        feature_sets (Dict[str, List[str]]): Dizionario con diversi set di feature da testare.
        target_col (str): Nome della colonna target.
        n_estimators (int): Numero di alberi nel forest.
        random_state (int): Seed per la riproducibilità.
        
    Returns:
        Dict[str, Any]: Risultati del confronto.
    """
    results = {}
    performance_metrics = []
    
    # Per ogni set di feature
    for set_name, features in feature_sets.items():
        print(f"\nTestando set di feature: {set_name} ({len(features)} features)")
        
        try:
            # Prepara i dati
            data = prepare_data(
                df, 
                target_col=target_col,
                feature_cols=features,
                random_state=random_state,
                stratify=True
            )
            
            # Addestra il modello
            model_results = train_random_forest(
                data,
                n_estimators=n_estimators,
                random_state=random_state
            )
            
            # Salva i risultati
            results[set_name] = model_results
            
            # Aggiungi le metriche principali al riepilogo
            metrics = model_results['metrics']
            perf = {
                'set_name': set_name,
                'n_features': len(features),
                'accuracy': metrics['accuracy'],
                'precision': metrics['precision'],
                'recall': metrics['recall'],
                'f1_score': metrics['f1_score'],
                'roc_auc': metrics['roc_auc'],
                'cv_score': model_results['cv_mean']
            }
            
            performance_metrics.append(perf)
            
            print(f"  ROC AUC: {metrics['roc_auc']:.4f} (CV: {model_results['cv_mean']:.4f})")
            print(f"  F1 Score: {metrics['f1_score']:.4f}")
            
        except Exception as e:
            print(f"Errore con il set {set_name}: {str(e)}")
            continue
    
    # Converti a DataFrame
    performance_df = pd.DataFrame(performance_metrics)
    
    return {
        'model_results': results,
        'performance_summary': performance_df
    }


if __name__ == "__main__":
    try:
        # Carica i dati
        print("Caricamento dei dati...")
        df = load_feature_matrix()
        print(f"Dati caricati con successo: {df.shape[0]} righe, {df.shape[1]} colonne")
        
        # 1. Prepara i dati
        print("\nPreparazione dei dati per il Random Forest...")
        prepared_data = prepare_data(df, stratify=True)
        
        # Visualizza la distribuzione del target
        print(f"Distribuzione del target:")
        for class_label, prop in prepared_data['target_distribution'].items():
            print(f"  Classe {class_label}: {prop:.4f} ({prop*100:.1f}%)")
        
        # 2. Addestra il Random Forest con parametri predefiniti
        print("\nAddestramento del modello Random Forest...")
        rf_results = train_random_forest(
            prepared_data,
            n_estimators=100,
            class_weight='balanced'
        )
        
        print("\nMetriche di performance del Random Forest:")
        for metric_name, metric_value in rf_results['metrics'].items():
            print(f"  {metric_name}: {metric_value:.4f}")
        
        print(f"\nCV Score (ROC AUC): {rf_results['cv_mean']:.4f} ± {rf_results['cv_std']:.4f}")
        
        # 3. Visualizza l'importanza delle feature
        print("\nCreazione del grafico di importanza delle feature...")
        plot_feature_importance(
            rf_results,
            top_n=15,
            use_permutation=True,
            filename="random_forest_feature_importance.png"
        )
        
        # 4. Confronta con la regressione logistica (se disponibile)
        try:
            from src.analysis.regression import load_feature_matrix, prepare_data_for_regression, fit_logistic_regression
            
            print("\nConfrontando con la regressione logistica...")
            lr_data = prepare_data_for_regression(df)
            lr_results = fit_logistic_regression(lr_data)
            
            plot_roc_curve_comparison(
                rf_results,
                lr_results,
                filename="roc_curve_comparison.png"
            )
            
            print("\nConfonto delle metriche:")
            print(f"  Random Forest - ROC AUC: {rf_results['metrics']['roc_auc']:.4f}")
            print(f"  Regressione Logistica - ROC AUC: {lr_results['metrics']['auc']:.4f}")
            
        except Exception as e:
            print(f"\nImpossibile confrontare con la regressione logistica: {str(e)}")
            print("Creazione del grafico ROC solo per Random Forest...")
            plot_roc_curve_comparison(rf_results, filename="random_forest_roc_curve.png")
        
        # 5. Salva i risultati
        print("\nSalvataggio dei risultati...")
        metrics_path, importance_path = save_model_results(rf_results)
        print(f"Metriche salvate in: {metrics_path}")
        print(f"Importanza delle feature salvata in: {importance_path}")
        
        # 6. Confronto tra diversi set di feature
        print("\nConfrontando diversi set di feature...")
        
        # Feature di sentiment
        sentiment_features = ['sentiment_polarity', 'sentiment_subjectivity']
        
        # Feature di stance
        stance_features = ['stance_score']
        
        # Feature di leggibilità
        readability_features = [
            'flesch_reading_ease', 'type_token_ratio', 'formal_language_score',
            'vocabulary_richness', 'avg_word_length', 'long_words_ratio', 'culture_score'
        ]
        
        # Combinazioni
        feature_sets = {
            'sentiment_only': sentiment_features,
            'stance_only': stance_features,
            'readability_only': readability_features,
            'sentiment_stance': sentiment_features + stance_features,
            'sentiment_readability': sentiment_features + readability_features,
            'all_features': sentiment_features + stance_features + readability_features
        }
        
        comparison_results = compare_feature_sets(
            df,
            feature_sets,
            n_estimators=100,
            random_state=42
        )
        
        # Salva i risultati del confronto
        comparison_summary = comparison_results['performance_summary']
        summary_path = TABLES_DIR / "feature_sets_comparison.csv"
        comparison_summary.to_csv(summary_path, index=False)
        print(f"\nRiepilogo del confronto salvato in: {summary_path}")
        
        # 7. Ottimizzazione iperparametri (se c'è abbastanza tempo)
        run_grid_search = False
        if run_grid_search:
            print("\nAvvio della Grid Search per ottimizzare gli iperparametri...")
            param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [None, 10, 20],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            }
            
            grid_results = grid_search_rf(prepared_data, param_grid, cv=5)
            
            print("\nParametri ottimali trovati:")
            for param, value in grid_results['best_params'].items():
                print(f"  {param}: {value}")
            print(f"Punteggio migliore: {grid_results['best_score']:.4f}")
            
            # Salva i risultati della grid search
            grid_results_path = TABLES_DIR / "random_forest_grid_search.csv"
            grid_results['results_df'].to_csv(grid_results_path, index=False)
            print(f"Risultati grid search salvati in: {grid_results_path}")
        
        print("\nAnalisi Random Forest completata con successo!")
        
    except Exception as e:
        print(f"Errore nell'esecuzione: {str(e)}")
