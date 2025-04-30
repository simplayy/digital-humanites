#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Modulo per la creazione della matrice di feature integrata che combina
sentiment, stance e leggibilità in un unico dataset normalizzato e
pronto per l'analisi statistica.

Questo script prende i dataset con le singole feature e li unisce,
normalizzando i valori numerici e selezionando le feature più rilevanti.
"""

import pandas as pd
import numpy as np
import logging
from pathlib import Path
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# Configurazione del logger
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_dataset(file_path):
    """
    Carica il dataset completo con tutte le feature.
    
    Args:
        file_path (str): Percorso al file CSV del dataset completo
        
    Returns:
        pd.DataFrame: Il dataset caricato
    """
    logger.info(f"Caricamento del dataset da {file_path}")
    try:
        df = pd.read_csv(file_path)
        logger.info(f"Dataset caricato con successo. Dimensioni: {df.shape}")
        return df
    except Exception as e:
        logger.error(f"Errore nel caricamento del dataset: {e}")
        return None

def select_features(df):
    """
    Seleziona e organizza le feature più rilevanti dal dataset completo.
    
    Args:
        df (pd.DataFrame): Il dataset completo
        
    Returns:
        pd.DataFrame: Dataset con le feature selezionate
    """
    logger.info("Selezione delle feature più rilevanti...")
    
    # Feature di base per l'identificazione
    base_cols = ['thread_id', 'event', 'tweet_id', 'text', 'timestamp', 
                 'veracity', 'is_rumour', 'is_source', 'reaction_index']
    
    # Feature di sentiment
    sentiment_cols = ['sentiment_polarity', 'sentiment_subjectivity', 'sentiment_category']
    
    # Feature di stance
    stance_cols = ['stance_score', 'stance_category']
    
    # Feature di leggibilità e acculturazione
    readability_cols = ['culture_score', 'readability_category', 'flesch_reading_ease', 
                        'type_token_ratio', 'formal_language_score', 'vocabulary_richness',
                        'avg_word_length', 'long_words_ratio']
    
    # Combina tutte le colonne necessarie
    selected_cols = base_cols + sentiment_cols + stance_cols + readability_cols
    
    # Verifica che tutte le colonne esistano nel dataframe
    existing_cols = [col for col in selected_cols if col in df.columns]
    missing_cols = [col for col in selected_cols if col not in df.columns]
    
    if missing_cols:
        logger.warning(f"Le seguenti colonne non sono presenti nel dataset: {missing_cols}")
    
    logger.info(f"Selezionate {len(existing_cols)} colonne per la matrice di feature")
    
    return df[existing_cols].copy()

def normalize_numeric_features(df):
    """
    Normalizza le feature numeriche del dataset.
    
    Args:
        df (pd.DataFrame): Dataset con feature selezionate
        
    Returns:
        pd.DataFrame: Dataset con feature numeriche normalizzate
    """
    logger.info("Normalizzazione delle feature numeriche...")
    
    # Colonne da normalizzare con StandardScaler (media 0, deviazione standard 1)
    std_scale_cols = ['sentiment_polarity', 'sentiment_subjectivity', 'stance_score']
    
    # Colonne da normalizzare con MinMaxScaler (da 0 a 1)
    minmax_scale_cols = ['flesch_reading_ease', 'type_token_ratio', 
                         'formal_language_score', 'vocabulary_richness',
                         'avg_word_length', 'long_words_ratio']
    
    # Crea una copia del dataframe
    normalized_df = df.copy()
    
    # Applica StandardScaler
    std_cols = [col for col in std_scale_cols if col in df.columns]
    if std_cols:
        std_scaler = StandardScaler()
        normalized_df[std_cols] = std_scaler.fit_transform(df[std_cols].fillna(0))
    
    # Applica MinMaxScaler
    minmax_cols = [col for col in minmax_scale_cols if col in df.columns]
    if minmax_cols:
        minmax_scaler = MinMaxScaler()
        normalized_df[minmax_cols] = minmax_scaler.fit_transform(df[minmax_cols].fillna(0))
    
    logger.info("Normalizzazione completata")
    return normalized_df

def encode_categorical_features(df):
    """
    Codifica le feature categoriche usando one-hot encoding.
    
    Args:
        df (pd.DataFrame): Dataset con feature normalizzate
        
    Returns:
        pd.DataFrame: Dataset con feature categoriche codificate
    """
    logger.info("Codifica delle feature categoriche...")
    
    # Colonne da codificare
    cat_cols = ['sentiment_category', 'stance_category', 'readability_category']
    
    # Crea una copia del dataframe
    encoded_df = df.copy()
    
    # Applica one-hot encoding per ogni colonna categorica
    for col in cat_cols:
        if col in df.columns:
            dummies = pd.get_dummies(df[col], prefix=col)
            encoded_df = pd.concat([encoded_df, dummies], axis=1)
    
    logger.info("Codifica completata")
    return encoded_df

def create_feature_matrix(input_file, output_file):
    """
    Crea la matrice di feature integrata dal file di input.
    
    Args:
        input_file (str): Percorso al file CSV del dataset completo
        output_file (str): Percorso dove salvare la matrice di feature
        
    Returns:
        pd.DataFrame: La matrice di feature creata
    """
    # Carica il dataset
    df = load_dataset(input_file)
    if df is None:
        return None
    
    # Seleziona le feature rilevanti
    selected_df = select_features(df)
    
    # Normalizza le feature numeriche
    normalized_df = normalize_numeric_features(selected_df)
    
    # Codifica le feature categoriche
    feature_matrix = encode_categorical_features(normalized_df)
    
    # Salva la matrice di feature
    try:
        feature_matrix.to_csv(output_file, index=False)
        logger.info(f"Matrice di feature salvata in {output_file}")
    except Exception as e:
        logger.error(f"Errore nel salvataggio della matrice di feature: {e}")
    
    return feature_matrix

def create_correlation_heatmap(feature_matrix, output_file):
    """
    Crea e salva una heatmap delle correlazioni tra feature numeriche.
    
    Args:
        feature_matrix (pd.DataFrame): La matrice di feature
        output_file (str): Percorso dove salvare la heatmap
    """
    logger.info("Creazione della heatmap di correlazione...")
    
    # Seleziona solo le colonne numeriche
    numeric_cols = feature_matrix.select_dtypes(include=['float64', 'int64']).columns
    numeric_cols = [col for col in numeric_cols if not col.startswith('thread_id') 
                   and not col.startswith('tweet_id') 
                   and not col.startswith('user_id')
                   and not col.startswith('user_followers')
                   and not col.startswith('favorites')
                   and not col.startswith('retweets')
                   and not col.startswith('reaction_index')]
    
    # Calcola la matrice di correlazione
    corr_matrix = feature_matrix[numeric_cols].corr()
    
    # Crea la figura
    plt.figure(figsize=(14, 12))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1, fmt='.2f', linewidths=0.5)
    plt.title('Correlazione tra Feature Numeriche', fontsize=16)
    plt.tight_layout()
    
    # Salva la figura
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    logger.info(f"Heatmap di correlazione salvata in {output_file}")
    plt.close()

def summarize_feature_matrix(feature_matrix):
    """
    Crea un riepilogo statistico della matrice di feature.
    
    Args:
        feature_matrix (pd.DataFrame): La matrice di feature
        
    Returns:
        pd.DataFrame: Statistiche descrittive delle feature
    """
    logger.info("Generazione del riepilogo statistico...")
    
    # Calcola le statistiche descrittive
    stats = feature_matrix.describe(include='all').T
    
    # Aggiungi informazioni sul tipo di dati e valori mancanti
    stats['dtype'] = feature_matrix.dtypes
    stats['missing'] = feature_matrix.isnull().sum()
    stats['missing_pct'] = feature_matrix.isnull().mean() * 100
    
    logger.info("Riepilogo statistico completato")
    return stats

def main():
    """Funzione principale per l'esecuzione dello script."""
    # Definizione dei percorsi dei file
    project_dir = Path(__file__).resolve().parents[2]
    data_dir = project_dir / "data"
    processed_dir = data_dir / "processed"
    results_dir = project_dir / "results"
    figures_dir = results_dir / "figures"
    tables_dir = results_dir / "tables"
    
    # Crea le directory se non esistono
    figures_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)
    
    # Definisci i percorsi dei file
    input_file = processed_dir / "pheme_with_readability.csv"
    output_file = processed_dir / "pheme_feature_matrix.csv"
    correlation_file = figures_dir / "feature_correlation_heatmap.png"
    stats_file = tables_dir / "feature_matrix_stats.csv"
    
    # Verifica che il file di input esista
    if not input_file.exists():
        logger.error(f"Il file {input_file} non esiste. Eseguire prima l'estrazione di tutte le feature.")
        return
    
    # Crea la matrice di feature
    feature_matrix = create_feature_matrix(input_file, output_file)
    if feature_matrix is None:
        return
    
    # Crea una heatmap di correlazione
    create_correlation_heatmap(feature_matrix, correlation_file)
    
    # Genera statistiche descrittive
    stats = summarize_feature_matrix(feature_matrix)
    stats.to_csv(stats_file)
    logger.info(f"Statistiche descrittive salvate in {stats_file}")
    
    logger.info("Creazione della matrice di feature completata!")

if __name__ == "__main__":
    main()
