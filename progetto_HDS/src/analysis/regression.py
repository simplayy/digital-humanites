"""
Modulo per l'analisi di regressione e modelli predittivi.

Questo modulo contiene funzioni per implementare modelli di regressione
che analizzano la relazione tra feature di sentiment e veridicità delle notizie.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Any, Optional, Union
from pathlib import Path
import scipy.stats as stats
import statsmodels.api as sm
from statsmodels.formula.api import ols, logit
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, roc_curve
)
from sklearn.preprocessing import StandardScaler

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


def prepare_data_for_regression(df: pd.DataFrame, 
                              target_col: str = 'veracity',
                              feature_cols: Optional[List[str]] = None,
                              test_size: float = 0.3,
                              random_state: int = 42) -> Dict[str, Any]:
    """
    Prepara i dati per l'analisi di regressione.
    
    Args:
        df (pd.DataFrame): DataFrame contenente i dati.
        target_col (str): Nome della colonna target (es. 'veracity').
        feature_cols (List[str], optional): Lista di colonne da usare come feature.
            Se None, vengono selezionate automaticamente le colonne numeriche.
        test_size (float): Proporzione del dataset da usare come test set.
        random_state (int): Seed per la riproducibilità.
    
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
        # Esclude il target se è presente
        feature_cols = [col for col in feature_cols if col != target_col]
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
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    # Standardizza le feature
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Crea DataFrame con le feature standardizzate
    X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=feature_cols, index=X_train.index)
    X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=feature_cols, index=X_test.index)
    
    # Prepara il risultato
    result = {
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'X_train_scaled': X_train_scaled_df,
        'X_test_scaled': X_test_scaled_df,
        'feature_cols': feature_cols,
        'target_col': target_col,
        'scaler': scaler,
        'n_samples': len(df_clean),
        'n_features': len(feature_cols),
        'target_distribution': y.value_counts(normalize=True).to_dict()
    }
    
    return result


def fit_logistic_regression(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Implementa un modello di regressione logistica per predire la veridicità.
    
    Args:
        data (Dict[str, Any]): Dizionario con i dati preparati da prepare_data_for_regression().
        
    Returns:
        Dict[str, Any]: Risultati della regressione logistica.
    """
    # Estrai i dati dal dizionario
    X_train = data['X_train_scaled']
    y_train = data['y_train']
    X_test = data['X_test_scaled']
    y_test = data['y_test']
    feature_cols = data['feature_cols']
    
    # Aggiungi costante per l'intercetta
    X_train_sm = sm.add_constant(X_train)
    X_test_sm = sm.add_constant(X_test)
    
    # Crea e adatta il modello
    model = sm.Logit(y_train, X_train_sm)
    
    try:
        result = model.fit(disp=0)  # disp=0 per disabilitare l'output dettagliato
    except Exception as e:
        print(f"Errore nel fit del modello: {str(e)}")
        print("Tentativo con metodo alternativo...")
        try:
            result = model.fit_regularized(disp=0)
        except Exception as e2:
            raise ValueError(f"Impossibile adattare il modello: {str(e2)}")
    
    # Predizioni
    y_pred_proba = result.predict(X_test_sm)
    y_pred = (y_pred_proba >= 0.5).astype(int)
    
    # Metriche di valutazione
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred_proba)
    conf_matrix = confusion_matrix(y_test, y_pred)
    
    # Coefficienti e p-value
    coef_df = pd.DataFrame({
        'feature': ['const'] + feature_cols,
        'coefficient': result.params,
        'std_err': result.bse,
        'z_value': result.tvalues,
        'p_value': result.pvalues,
        'odds_ratio': np.exp(result.params),
        'CI_lower_95': np.exp(result.params - 1.96 * result.bse),
        'CI_upper_95': np.exp(result.params + 1.96 * result.bse)
    })
    
    # Ordina i coefficienti per valore assoluto decrescente
    coef_df['abs_coefficient'] = coef_df['coefficient'].abs()
    coef_df = coef_df.sort_values('abs_coefficient', ascending=False)
    coef_df.drop('abs_coefficient', axis=1, inplace=True)
    
    # Significatività dei coefficienti
    coef_df['significant'] = coef_df['p_value'] < 0.05
    
    # Prepara il risultato
    results = {
        'model': result,
        'summary': result.summary(),
        'coefficients': coef_df,
        'metrics': {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'auc': auc,
        },
        'confusion_matrix': conf_matrix,
        'y_pred': y_pred,
        'y_pred_proba': y_pred_proba,
        'y_test': y_test
    }
    
    return results


def save_regression_results(results: Dict[str, Any], 
                          coefs_filename: str = "logistic_regression_coefficients.csv",
                          metrics_filename: str = "logistic_regression_metrics.csv") -> Tuple[str, str]:
    """
    Salva i risultati della regressione logistica.
    
    Args:
        results (Dict[str, Any]): Risultati della regressione logistica.
        coefs_filename (str): Nome del file per i coefficienti.
        metrics_filename (str): Nome del file per le metriche.
        
    Returns:
        Tuple[str, str]: Percorsi dei file salvati.
    """
    # Salva i coefficienti
    coefs_path = TABLES_DIR / coefs_filename
    results['coefficients'].to_csv(coefs_path, index=False)
    
    # Salva le metriche
    metrics_df = pd.DataFrame(results['metrics'], index=[0])
    metrics_path = TABLES_DIR / metrics_filename
    metrics_df.to_csv(metrics_path, index=False)
    
    return str(coefs_path), str(metrics_path)


def plot_coefficients(results: Dict[str, Any], 
                    top_n: int = 10,
                    filename: Optional[str] = None) -> plt.Figure:
    """
    Visualizza i coefficienti più importanti del modello di regressione logistica.
    
    Args:
        results (Dict[str, Any]): Risultati della regressione logistica.
        top_n (int): Numero di coefficienti da visualizzare.
        filename (str, optional): Nome del file per salvare il grafico.
        
    Returns:
        plt.Figure: Figura matplotlib.
    """
    # Estrai i coefficienti
    coefs = results['coefficients']
    
    # Filtra i coefficienti escludendo la costante
    coefs_without_const = coefs[coefs['feature'] != 'const']
    
    # Prendi i top_n coefficienti per valore assoluto
    top_coefs = coefs_without_const.head(top_n)
    
    # Crea il grafico
    plt.figure(figsize=(14, 8))
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Ordina per valore del coefficiente
    top_coefs = top_coefs.sort_values('coefficient')
    
    # Crea il grafico a barre orizzontale
    colors = ['green' if c > 0 else 'red' for c in top_coefs['coefficient']]
    bars = ax.barh(top_coefs['feature'], top_coefs['coefficient'], color=colors, alpha=0.7)
    
    # Aggiungi intervalli di confidenza
    for i, (_, row) in enumerate(top_coefs.iterrows()):
        ci_lower = row['coefficient'] - 1.96 * row['std_err']
        ci_upper = row['coefficient'] + 1.96 * row['std_err']
        ax.plot([ci_lower, ci_upper], [i, i], color='black', linewidth=2, alpha=0.7)
    
    # Aggiungi etichette con odds ratio
    for i, bar in enumerate(bars):
        odds_ratio = top_coefs.iloc[i]['odds_ratio']
        p_value = top_coefs.iloc[i]['p_value']
        significant = top_coefs.iloc[i]['significant']
        
        # Formatta il testo in base alla significatività
        text = f"OR: {odds_ratio:.2f}"
        if significant:
            text += f" (p < 0.05)"
        else:
            text += f" (p = {p_value:.3f})"
        
        # Posiziona l'annotazione
        x_pos = bar.get_width() * 1.05 if bar.get_width() >= 0 else bar.get_width() * 0.95
        ax.annotate(text,
                  xy=(x_pos, bar.get_y() + bar.get_height() / 2),
                  va='center',
                  ha='left' if bar.get_width() >= 0 else 'right',
                  fontsize=9,
                  alpha=0.8)
    
    # Aggiungi linea verticale a zero
    ax.axvline(x=0, color='black', linestyle='-', alpha=0.3)
    
    # Aggiungi titolo e etichette
    ax.set_title('Coefficienti più importanti del modello di regressione logistica')
    ax.set_xlabel('Coefficiente (log-odds)')
    ax.set_ylabel('Feature')
    
    # Aggiungi interpretazione
    plt.figtext(0.5, 0.01, 
               "I coefficienti positivi indicano un aumento della probabilità che la notizia sia vera.\n"
               "I coefficienti negativi indicano una diminuzione della probabilità che la notizia sia vera.\n"
               "OR = Odds Ratio, valori > 1 indicano associazione positiva, < 1 indicano associazione negativa.",
               ha='center', fontsize=10, style='italic', alpha=0.7)
    
    plt.tight_layout(rect=[0, 0.05, 1, 1])
    
    # Salva il grafico se richiesto
    if filename:
        save_path = FIGURES_DIR / filename
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_roc_curve(results: Dict[str, Any], 
                 filename: Optional[str] = None) -> plt.Figure:
    """
    Visualizza la curva ROC del modello di regressione logistica.
    
    Args:
        results (Dict[str, Any]): Risultati della regressione logistica.
        filename (str, optional): Nome del file per salvare il grafico.
        
    Returns:
        plt.Figure: Figura matplotlib.
    """
    # Estrai i dati
    y_test = results['y_test']
    y_pred_proba = results['y_pred_proba']
    auc = results['metrics']['auc']
    
    # Calcola la curva ROC
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
    
    # Crea il grafico
    plt.figure(figsize=(10, 8))
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Disegna la curva ROC
    ax.plot(fpr, tpr, label=f'AUC = {auc:.3f}', linewidth=2)
    
    # Aggiungi la linea diagonale (random classifier)
    ax.plot([0, 1], [0, 1], linestyle='--', color='gray', alpha=0.7, label='Random Classifier')
    
    # Aggiungi etichette e titolo
    ax.set_title('Curva ROC per la previsione della veridicità')
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


def plot_confusion_matrix(results: Dict[str, Any], 
                        filename: Optional[str] = None) -> plt.Figure:
    """
    Visualizza la matrice di confusione del modello di regressione logistica.
    
    Args:
        results (Dict[str, Any]): Risultati della regressione logistica.
        filename (str, optional): Nome del file per salvare il grafico.
        
    Returns:
        plt.Figure: Figura matplotlib.
    """
    # Estrai la matrice di confusione
    conf_matrix = results['confusion_matrix']
    
    # Calcola le percentuali
    conf_matrix_norm = conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, np.newaxis]
    
    # Crea il grafico
    plt.figure(figsize=(10, 8))
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Disegna la matrice di confusione
    im = ax.imshow(conf_matrix_norm, interpolation='nearest', cmap='Blues')
    
    # Aggiungi colorbar
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel('Percentuale', rotation=-90, va='bottom')
    
    # Aggiungi etichette
    classes = ['False', 'True']
    tick_marks = np.arange(len(classes))
    ax.set_xticks(tick_marks)
    ax.set_yticks(tick_marks)
    ax.set_xticklabels(classes)
    ax.set_yticklabels(classes)
    
    # Aggiungi titolo e etichette
    ax.set_title('Matrice di Confusione Normalizzata')
    ax.set_ylabel('Veridicità Reale')
    ax.set_xlabel('Veridicità Prevista')
    
    # Aggiungi i valori nella matrice
    thresh = conf_matrix_norm.max() / 2.0
    for i in range(conf_matrix_norm.shape[0]):
        for j in range(conf_matrix_norm.shape[1]):
            value = f"{conf_matrix[i, j]} ({conf_matrix_norm[i, j]:.2f})"
            ax.text(j, i, value, ha="center", va="center",
                   color="white" if conf_matrix_norm[i, j] > thresh else "black")
    
    plt.tight_layout()
    
    # Salva il grafico se richiesto
    if filename:
        save_path = FIGURES_DIR / filename
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


if __name__ == "__main__":
    try:
        # Carica i dati
        print("Caricamento dei dati...")
        df = load_feature_matrix()
        print(f"Dati caricati con successo: {df.shape[0]} righe, {df.shape[1]} colonne")
        
        # Lista delle principali feature numeriche
        numerical_features = [
            'sentiment_polarity', 'sentiment_subjectivity',
            'stance_score', 'flesch_reading_ease',
            'type_token_ratio', 'formal_language_score',
            'vocabulary_richness', 'avg_word_length',
            'long_words_ratio', 'culture_score'
        ]
        
        # 1. Prepara i dati per la regressione
        print("\nPreparazione dei dati per la regressione logistica...")
        prepared_data = prepare_data_for_regression(df, feature_cols=numerical_features)
        print(f"Dati preparati: {prepared_data['n_samples']} campioni, {prepared_data['n_features']} feature")
        print(f"Distribuzione target: {prepared_data['target_distribution']}")
        
        # 2. Implementa il modello di regressione logistica
        print("\nImplementazione del modello di regressione logistica...")
        try:
            logistic_results = fit_logistic_regression(prepared_data)
            print(f"Modello implementato con successo.")
            print(f"Metriche di valutazione:")
            for metric, value in logistic_results['metrics'].items():
                print(f"  {metric}: {value:.4f}")
            
            # 3. Salva i risultati
            print("\nSalvataggio dei risultati...")
            coefs_path, metrics_path = save_regression_results(logistic_results)
            print(f"Coefficienti salvati in: {coefs_path}")
            print(f"Metriche salvate in: {metrics_path}")
            
            # 4. Visualizza i coefficienti più importanti
            print("\nCreazione visualizzazione dei coefficienti...")
            plot_coefficients(logistic_results, filename="logistic_regression_coefficients.png")
            
            # 5. Visualizza la curva ROC
            print("\nCreazione visualizzazione della curva ROC...")
            plot_roc_curve(logistic_results, filename="logistic_regression_roc.png")
            
            # 6. Visualizza la matrice di confusione
            print("\nCreazione visualizzazione della matrice di confusione...")
            plot_confusion_matrix(logistic_results, filename="logistic_regression_confusion_matrix.png")
            
            print("\nCompletata l'analisi di regressione!")
            
        except Exception as e:
            print(f"Errore nell'implementazione del modello: {str(e)}")
            print("Questo può accadere se ci sono problemi di convergenza o multicollinearità nelle feature.")
            print("Prova a rimuovere alcune feature correlate o ad aumentare la regolarizzazione.")
        
    except Exception as e:
        print(f"Errore durante l'esecuzione: {str(e)}")
