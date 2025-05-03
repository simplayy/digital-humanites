"""
Modulo per l'analisi statistica descrittiva dei dati.

Questo modulo contiene funzioni per calcolare statistiche descrittive
sui dati del dataset PHEME, visualizzare le distribuzioni principali
e identificare eventuali outlier.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Any, Optional, Union
from pathlib import Path

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


def calculate_descriptive_stats(df: pd.DataFrame, 
                               numerical_cols: Optional[List[str]] = None,
                               group_by: Optional[str] = None) -> pd.DataFrame:
    """
    Calcola le statistiche descrittive per le colonne numeriche del DataFrame.
    
    Args:
        df (pd.DataFrame): DataFrame contenente i dati.
        numerical_cols (List[str], optional): Lista di colonne numeriche da analizzare.
            Se None, vengono selezionate automaticamente.
        group_by (str, optional): Colonna per raggruppare i dati prima del calcolo.
    
    Returns:
        pd.DataFrame: DataFrame contenente le statistiche descrittive.
    """
    # Se non sono specificate le colonne numeriche, seleziona automaticamente
    if numerical_cols is None:
        numerical_cols = df.select_dtypes(include=['number']).columns.tolist()
    
    # Filtra solo le colonne specificate che esistono nel DataFrame
    numerical_cols = [col for col in numerical_cols if col in df.columns]
    
    if not numerical_cols:
        raise ValueError("Nessuna colonna numerica trovata nel DataFrame")
    
    # Se è specificato un raggruppamento
    if group_by and group_by in df.columns:
        stats = df.groupby(group_by)[numerical_cols].agg([
            'count', 'mean', 'std', 'min', 'median', 'max'
        ])
    else:
        stats = df[numerical_cols].agg([
            'count', 'mean', 'std', 'min', 'median', 'max'
        ]).T
    
    return stats


def save_descriptive_stats(stats: pd.DataFrame, filename: str = "descriptive_stats.csv") -> str:
    """
    Salva le statistiche descrittive in un file CSV.
    
    Args:
        stats (pd.DataFrame): DataFrame contenente le statistiche descrittive.
        filename (str): Nome del file di output.
        
    Returns:
        str: Percorso del file salvato.
    """
    output_path = TABLES_DIR / filename
    stats.to_csv(output_path)
    return str(output_path)


def plot_distribution(df: pd.DataFrame, 
                     column: str, 
                     by_veracity: bool = True, 
                     bins: int = 30,
                     kde: bool = True,
                     title: Optional[str] = None,
                     filename: Optional[str] = None) -> plt.Figure:
    """
    Crea un grafico della distribuzione di una colonna numerica.
    
    Args:
        df (pd.DataFrame): DataFrame contenente i dati.
        column (str): Nome della colonna da visualizzare.
        by_veracity (bool): Se True, mostra distribuzioni separate per veridicità.
        bins (int): Numero di bin per l'istogramma.
        kde (bool): Se True, mostra la stima della densità del kernel.
        title (str, optional): Titolo del grafico.
        filename (str, optional): Nome del file per salvare il grafico.
        
    Returns:
        plt.Figure: Figura matplotlib.
    """
    fig, ax = plt.subplots(figsize=(12, 8))
    
    if by_veracity and 'veracity' in df.columns:
        # Creazione della distribuzione per ciascun valore di veridicità
        for veracity, sub_df in df.groupby('veracity'):
            sns.histplot(data=sub_df, x=column, kde=kde, label=veracity, bins=bins, alpha=0.6)
    else:
        # Creazione della distribuzione generale
        sns.histplot(data=df, x=column, kde=kde, bins=bins)
    
    # Titolo e etichette
    if title:
        ax.set_title(title)
    else:
        ax.set_title(f"Distribuzione di {column}")
    
    ax.set_xlabel(column)
    ax.set_ylabel("Frequenza")
    
    if by_veracity and 'veracity' in df.columns:
        plt.legend(title="Veridicità")
    
    # Salva il grafico se richiesto
    if filename:
        save_path = FIGURES_DIR / filename
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def identify_outliers(df: pd.DataFrame, 
                     column: str, 
                     method: str = 'iqr', 
                     threshold: float = 1.5) -> pd.DataFrame:
    """
    Identifica gli outlier in una colonna numerica del DataFrame.
    
    Args:
        df (pd.DataFrame): DataFrame contenente i dati.
        column (str): Nome della colonna da analizzare.
        method (str): Metodo per identificare gli outlier ('iqr' o 'zscore').
        threshold (float): Soglia per l'identificazione degli outlier.
            - Per IQR: multiplo di IQR oltre il quale un valore è considerato outlier.
            - Per Z-score: numero di deviazioni standard oltre il quale un valore è outlier.
    
    Returns:
        pd.DataFrame: DataFrame contenente solo le righe identificate come outlier.
    """
    if column not in df.columns:
        raise ValueError(f"La colonna {column} non esiste nel DataFrame")
    
    # Crea una copia del DataFrame originale
    result_df = df.copy()
    
    if method == 'iqr':
        # Metodo IQR (Interquartile Range)
        Q1 = np.percentile(df[column].dropna(), 25)
        Q3 = np.percentile(df[column].dropna(), 75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - threshold * IQR
        upper_bound = Q3 + threshold * IQR
        
        # Crea una colonna che indica se una riga è un outlier
        result_df['is_outlier'] = (df[column] < lower_bound) | (df[column] > upper_bound)
        
    elif method == 'zscore':
        # Metodo Z-score
        mean = df[column].mean()
        std = df[column].std()
        
        # Calcola lo Z-score
        z_scores = np.abs((df[column] - mean) / std)
        
        # Crea una colonna che indica se una riga è un outlier
        result_df['is_outlier'] = z_scores > threshold
        
    else:
        raise ValueError(f"Metodo non supportato: {method}. Usare 'iqr' o 'zscore'.")
    
    # Ritorna solo le righe che sono outlier
    outliers = result_df[result_df['is_outlier']]
    
    # Aggiungi le statistiche degli outlier
    if len(outliers) > 0:
        print(f"Identificati {len(outliers)} outlier su {len(df)} righe ({len(outliers)/len(df)*100:.2f}%)")
        print(f"Valore minimo degli outlier: {outliers[column].min()}")
        print(f"Valore massimo degli outlier: {outliers[column].max()}")
        print(f"Media degli outlier: {outliers[column].mean()}")
    else:
        print("Nessun outlier identificato.")
        
    return outliers


def plot_boxplots(df: pd.DataFrame, 
                 columns: List[str], 
                 by_veracity: bool = True,
                 filename: Optional[str] = None) -> plt.Figure:
    """
    Crea box plot per visualizzare la distribuzione e gli outlier delle colonne specificate.
    
    Args:
        df (pd.DataFrame): DataFrame contenente i dati.
        columns (List[str]): Lista di colonne da visualizzare.
        by_veracity (bool): Se True, raggruppa i box plot per veridicità.
        filename (str, optional): Nome del file per salvare il grafico.
        
    Returns:
        plt.Figure: Figura matplotlib.
    """
    # Filtra solo le colonne che esistono nel DataFrame
    valid_columns = [col for col in columns if col in df.columns]
    
    if not valid_columns:
        raise ValueError("Nessuna colonna valida specificata")
    
    # Decidi il layout del grafico in base al numero di colonne
    n_cols = len(valid_columns)
    n_rows = 1
    
    if n_cols > 3:
        n_rows = (n_cols + 2) // 3  # Arrotonda per eccesso
        n_cols = 3
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6*n_cols, 6*n_rows))
    
    # Assicurati che axes sia sempre un array 2D
    if n_rows == 1 and n_cols == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = np.array([axes])
    
    # Crea un box plot per ogni colonna
    for i, column in enumerate(valid_columns):
        row = i // 3
        col = i % 3
        
        if by_veracity and 'veracity' in df.columns:
            sns.boxplot(data=df, x='veracity', y=column, ax=axes[row, col])
            axes[row, col].set_title(f"Box Plot di {column} per Veridicità")
        else:
            sns.boxplot(data=df, y=column, ax=axes[row, col])
            axes[row, col].set_title(f"Box Plot di {column}")
    
    # Nasconde gli assi vuoti
    for i in range(len(valid_columns), n_rows * n_cols):
        row = i // 3
        col = i % 3
        axes[row, col].set_visible(False)
    
    plt.tight_layout()
    
    # Salva il grafico se richiesto
    if filename:
        save_path = FIGURES_DIR / filename
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def create_correlation_heatmap(df: pd.DataFrame, 
                              columns: Optional[List[str]] = None,
                              method: str = 'pearson',
                              cmap: str = 'coolwarm',
                              filename: Optional[str] = None) -> plt.Figure:
    """
    Crea una heatmap delle correlazioni tra le colonne numeriche.
    
    Args:
        df (pd.DataFrame): DataFrame contenente i dati.
        columns (List[str], optional): Lista di colonne da includere.
            Se None, vengono selezionate tutte le colonne numeriche.
        method (str): Metodo di correlazione ('pearson', 'kendall', o 'spearman').
        cmap (str): Mappa colori per la heatmap.
        filename (str, optional): Nome del file per salvare il grafico.
        
    Returns:
        plt.Figure: Figura matplotlib.
    """
    # Se non sono specificate le colonne, seleziona le colonne numeriche
    if columns is None:
        columns = df.select_dtypes(include=['number']).columns.tolist()
    else:
        # Filtra solo le colonne numeriche che sono anche nell'elenco specificato
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        columns = [col for col in columns if col in numeric_cols]
    
    if not columns:
        raise ValueError("Nessuna colonna numerica trovata o specificata")
    
    # Calcola la matrice di correlazione
    corr_matrix = df[columns].corr(method=method)
    
    # Crea la heatmap
    plt.figure(figsize=(14, 12))
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    
    fig, ax = plt.subplots(figsize=(14, 12))
    
    # Disegna la heatmap
    sns.heatmap(
        corr_matrix,
        mask=mask,
        cmap=cmap,
        annot=True,
        fmt=".2f",
        center=0,
        square=True,
        linewidths=0.5,
        cbar_kws={"shrink": 0.8},
        ax=ax
    )
    
    plt.title(f"Matrice di Correlazione ({method.capitalize()})")
    
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
        
        # 1. Calcola statistiche descrittive generali
        print("\nCalcolo statistiche descrittive generali...")
        stats_general = calculate_descriptive_stats(df, numerical_features)
        save_path = save_descriptive_stats(stats_general, "descriptive_stats_general.csv")
        print(f"Statistiche generali salvate in: {save_path}")
        
        # 2. Calcola statistiche descrittive raggruppate per veridicità
        if 'veracity' in df.columns:
            print("\nCalcolo statistiche descrittive per veridicità...")
            stats_by_veracity = calculate_descriptive_stats(df, numerical_features, group_by='veracity')
            save_path = save_descriptive_stats(stats_by_veracity, "descriptive_stats_by_veracity.csv")
            print(f"Statistiche per veridicità salvate in: {save_path}")
        
        # 3. Visualizza le distribuzioni principali
        print("\nCreazione grafici di distribuzione...")
        for feature in numerical_features:
            if feature in df.columns:
                print(f"  Creazione grafico per {feature}...")
                plot_distribution(
                    df, 
                    feature, 
                    by_veracity=True, 
                    bins=30,
                    title=f"Distribuzione di {feature} per Veridicità",
                    filename=f"distribution_{feature}.png"
                )
        
        # 4. Crea box plot per visualizzare outlier
        print("\nCreazione box plot per visualizzare outlier...")
        plot_boxplots(
            df,
            numerical_features,
            by_veracity=True,
            filename="boxplots_by_veracity.png"
        )
        
        # 5. Crea matrice di correlazione
        print("\nCreazione matrice di correlazione...")
        create_correlation_heatmap(
            df,
            columns=numerical_features,
            method='pearson',
            filename="correlation_heatmap.png"
        )
        
        # 6. Identifica outlier per ogni feature numerica
        print("\nIdentificazione degli outlier...")
        outlier_results = {}
        for feature in numerical_features:
            if feature in df.columns:
                print(f"\nAnalisi outlier per {feature}:")
                outliers = identify_outliers(df, feature, method='iqr', threshold=1.5)
                outlier_results[feature] = len(outliers)
        
        print("\nCompletata l'analisi descrittiva!")
        
    except Exception as e:
        print(f"Errore durante l'esecuzione: {str(e)}")
