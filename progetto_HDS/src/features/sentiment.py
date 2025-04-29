#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Modulo per l'estrazione del sentiment dai commenti del dataset PHEME.

Questo script implementa funzioni per analizzare il sentiment (positività/negatività)
dei commenti nelle conversazioni Twitter e calcolare la polarità del sentiment.

Utilizza la libreria TextBlob per un'analisi del sentiment semplice ed efficace.
"""

import pandas as pd
import numpy as np
import logging
from pathlib import Path
from textblob import TextBlob
import nltk
from tqdm import tqdm

# Configurazione del logger
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Verifica se i dati necessari per NLTK sono già scaricati, altrimenti li scarica
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    logger.info("Downloading NLTK punkt tokenizer...")
    nltk.download('punkt')

def analyze_sentiment_textblob(text):
    """
    Analizza il sentiment del testo utilizzando TextBlob.
    
    Args:
        text (str): Il testo da analizzare
        
    Returns:
        dict: Un dizionario contenente la polarità e la soggettività del testo
    """
    if not text or not isinstance(text, str):
        return {'polarity': 0.0, 'subjectivity': 0.0}
    
    try:
        analysis = TextBlob(text)
        return {
            'polarity': analysis.sentiment.polarity,
            'subjectivity': analysis.sentiment.subjectivity
        }
    except Exception as e:
        logger.warning(f"Errore nell'analisi del sentiment per il testo: {e}")
        return {'polarity': 0.0, 'subjectivity': 0.0}

def categorize_sentiment(polarity):
    """
    Categorizza il sentiment basato sulla polarità in positivo, neutro o negativo.
    
    Args:
        polarity (float): La polarità del testo (-1.0 a 1.0)
        
    Returns:
        str: La categoria del sentiment ('positive', 'neutral', o 'negative')
    """
    if polarity > 0.05:
        return 'positive'
    elif polarity < -0.05:
        return 'negative'
    else:
        return 'neutral'

def process_dataset(file_path):
    """
    Elabora il dataset PHEME preprocessato e aggiunge feature di sentiment.
    
    Args:
        file_path (str): Percorso al file CSV del dataset preprocessato
        
    Returns:
        pd.DataFrame: Il dataset con le colonne di sentiment aggiunte
    """
    logger.info(f"Caricamento del dataset da {file_path}")
    try:
        df = pd.read_csv(file_path)
        logger.info(f"Dataset caricato con successo. Dimensioni: {df.shape}")
    except Exception as e:
        logger.error(f"Errore nel caricamento del dataset: {e}")
        return None

    # Inizializza le nuove colonne
    df['sentiment_polarity'] = 0.0
    df['sentiment_subjectivity'] = 0.0
    df['sentiment_category'] = 'neutral'
    
    # Elabora i testi e aggiungi le feature di sentiment
    logger.info("Elaborazione del sentiment per ogni tweet...")
    
    for idx, row in tqdm(df.iterrows(), total=df.shape[0], desc="Analisi del sentiment"):
        text = row['text']
        sentiment = analyze_sentiment_textblob(text)
        df.at[idx, 'sentiment_polarity'] = sentiment['polarity']
        df.at[idx, 'sentiment_subjectivity'] = sentiment['subjectivity']
        df.at[idx, 'sentiment_category'] = categorize_sentiment(sentiment['polarity'])
    
    logger.info("Analisi del sentiment completata")
    return df

def save_sentiment_features(df, output_path):
    """
    Salva il dataset con le feature di sentiment.
    
    Args:
        df (pd.DataFrame): Il dataset con le feature di sentiment
        output_path (str): Percorso dove salvare il dataset arricchito
    """
    try:
        df.to_csv(output_path, index=False)
        logger.info(f"Dataset con feature di sentiment salvato in {output_path}")
    except Exception as e:
        logger.error(f"Errore nel salvataggio del dataset: {e}")

def analyze_sentiment_aggregates(df):
    """
    Calcola statistiche aggregate sul sentiment per thread e per veracity.
    
    Args:
        df (pd.DataFrame): Il dataset con le feature di sentiment
        
    Returns:
        tuple: (sentiment_per_thread, sentiment_per_veracity)
    """
    logger.info("Calcolo delle statistiche aggregate di sentiment...")
    
    # Aggregazione per thread
    thread_sentiment = df.groupby('thread_id').agg({
        'sentiment_polarity': ['mean', 'std', 'min', 'max'],
        'sentiment_subjectivity': ['mean', 'std'],
        'sentiment_category': lambda x: x.value_counts().to_dict()
    })
    
    # Aggregazione per veridicità (separatamente per tweet sorgente e reazioni)
    source_tweets = df[df['is_source'] == True]
    veracity_sentiment = df.groupby(['veracity', 'is_rumour']).agg({
        'sentiment_polarity': ['mean', 'std', 'count'],
        'sentiment_subjectivity': ['mean', 'std'],
        'sentiment_category': lambda x: pd.Series(x).value_counts().to_dict()
    })
    
    logger.info("Statistiche aggregate calcolate")
    return thread_sentiment, veracity_sentiment

def main():
    """Funzione principale per l'esecuzione dello script."""
    # Definizione dei percorsi dei file
    project_dir = Path(__file__).resolve().parents[2]
    data_dir = project_dir / "data"
    processed_dir = data_dir / "processed"
    
    input_file = processed_dir / "pheme_processed.csv"
    output_file = processed_dir / "pheme_with_sentiment.csv"
    output_stats = project_dir / "results" / "tables" / "sentiment_stats.csv"
    
    # Verifica che il file di input esista
    if not input_file.exists():
        logger.error(f"Il file {input_file} non esiste. Eseguire prima il preprocessing.")
        return
    
    # Elabora il dataset e aggiungi feature di sentiment
    enriched_df = process_dataset(input_file)
    if enriched_df is None:
        return
    
    # Salva il dataset arricchito
    save_sentiment_features(enriched_df, output_file)
    
    # Calcola e salva statistiche aggregate
    thread_stats, veracity_stats = analyze_sentiment_aggregates(enriched_df)
    
    # Salva le statistiche aggregate
    thread_stats_file = project_dir / "results" / "tables" / "thread_sentiment_stats.csv"
    veracity_stats_file = project_dir / "results" / "tables" / "veracity_sentiment_stats.csv"
    
    # Crea la directory results/tables se non esiste
    (project_dir / "results" / "tables").mkdir(parents=True, exist_ok=True)
    
    # Salva le statistiche in formato CSV
    thread_stats.to_csv(thread_stats_file)
    veracity_stats.to_csv(veracity_stats_file)
    logger.info(f"Statistiche thread salvate in {thread_stats_file}")
    logger.info(f"Statistiche veracity salvate in {veracity_stats_file}")
    
    logger.info("Processo di estrazione del sentiment completato!")

if __name__ == "__main__":
    main()
