#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Modulo per l'analisi della stance (posizione) dei commenti rispetto alle notizie nel dataset PHEME.

Questo script implementa funzioni per determinare se i commenti sono favorevoli o contrari 
alla notizia principale e per classificare le reazioni in base al loro atteggiamento 
verso la veridicità della notizia.
"""

import pandas as pd
import numpy as np
import logging
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.corpus import stopwords
from tqdm import tqdm

# Configurazione del logger
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Verifica se i dati necessari per NLTK sono già scaricati, altrimenti li scarica
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    logger.info("Downloading NLTK stopwords...")
    nltk.download('stopwords')

def preprocess_text(text):
    """
    Preprocessa il testo per l'analisi della stance.
    
    Args:
        text (str): Testo da preprocessare
        
    Returns:
        str: Testo preprocessato
    """
    if not text or not isinstance(text, str):
        return ""
    
    # Converte in minuscolo
    text = text.lower()
    
    # Rimuove stopwords (parole comuni non informative)
    stop_words = set(stopwords.words('english'))
    words = text.split()
    words = [word for word in words if word not in stop_words]
    
    return " ".join(words)

def calculate_stance_tfidf_similarity(source_text, reaction_text):
    """
    Calcola la stance basata sulla similarità TF-IDF tra il testo sorgente e la reazione.
    
    Args:
        source_text (str): Testo del tweet sorgente
        reaction_text (str): Testo della reazione
        
    Returns:
        float: Similarità tra -1 e 1 (negativa per stance contraria, positiva per stance favorevole)
    """
    if not source_text or not reaction_text:
        return 0.0
    
    # Preprocessa i testi
    source_processed = preprocess_text(source_text)
    reaction_processed = preprocess_text(reaction_text)
    
    if not source_processed or not reaction_processed:
        return 0.0
    
    # Crea il vettorizzatore TF-IDF
    vectorizer = TfidfVectorizer()
    
    try:
        # Trasforma i testi in vettori TF-IDF
        tfidf_matrix = vectorizer.fit_transform([source_processed, reaction_processed])
        
        # Calcola la similarità del coseno
        similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
        
        return similarity
    except Exception as e:
        logger.warning(f"Errore nel calcolo della similarità: {e}")
        return 0.0

def determine_stance_sentiment(similarity, sentiment_polarity):
    """
    Determina la stance combinando la similarità del testo con il sentiment.
    
    Args:
        similarity (float): Similarità tra testo sorgente e reazione
        sentiment_polarity (float): Polarità del sentiment della reazione
        
    Returns:
        tuple: (stance_score, stance_category)
    """
    # Combina similarità e sentiment per ottenere uno score di stance
    # Alta similarità + sentiment positivo = forte supporto
    # Alta similarità + sentiment negativo = supporto critico
    # Bassa similarità + sentiment positivo = supporto indiretto
    # Bassa similarità + sentiment negativo = opposizione
    
    stance_score = similarity * sentiment_polarity
    
    # Categorizza la stance
    if similarity > 0.2:  # Alta similarità tematica
        if sentiment_polarity > 0.1:
            stance_category = "supportive"
        elif sentiment_polarity < -0.1:
            stance_category = "critical"
        else:
            stance_category = "neutral"
    else:  # Bassa similarità tematica
        if sentiment_polarity > 0.1:
            stance_category = "indirect_supportive"
        elif sentiment_polarity < -0.1:
            stance_category = "opposing"
        else:
            stance_category = "unrelated"
            
    return stance_score, stance_category

def analyze_thread_stance(df, thread_id):
    """
    Analizza la stance per un thread specifico.
    
    Args:
        df (pd.DataFrame): Il dataset completo
        thread_id (str): ID del thread da analizzare
        
    Returns:
        pd.DataFrame: Un dataframe con i tweet del thread e le loro stance
    """
    # Filtra il thread
    thread_df = df[df['thread_id'] == thread_id].copy()
    
    # Se non ci sono tweet nel thread, ritorna un dataframe vuoto
    if thread_df.empty:
        logger.warning(f"Thread {thread_id} non trovato nel dataset")
        return pd.DataFrame()
    
    # Ottieni il tweet sorgente
    source_tweet = thread_df[thread_df['is_source'] == True]
    if source_tweet.empty:
        logger.warning(f"Tweet sorgente non trovato per il thread {thread_id}")
        return thread_df
    
    source_text = source_tweet.iloc[0]['text']
    
    # Per ogni reazione, calcola la stance
    stance_scores = []
    stance_categories = []
    
    for idx, row in thread_df.iterrows():
        if row['is_source']:
            # Il tweet sorgente ha stance neutra rispetto a sé stesso
            stance_score = 0.0
            stance_category = "source"
        else:
            reaction_text = row['text']
            similarity = calculate_stance_tfidf_similarity(source_text, reaction_text)
            stance_score, stance_category = determine_stance_sentiment(
                similarity, row.get('sentiment_polarity', 0.0)
            )
        
        stance_scores.append(stance_score)
        stance_categories.append(stance_category)
    
    # Aggiungi le stance al dataframe
    thread_df['stance_score'] = stance_scores
    thread_df['stance_category'] = stance_categories
    
    return thread_df

def process_dataset_stance(input_file):
    """
    Elabora l'intero dataset per calcolare la stance di ogni tweet.
    
    Args:
        input_file (str): Percorso al file CSV del dataset con sentiment
        
    Returns:
        pd.DataFrame: Il dataset con le colonne di stance aggiunte
    """
    logger.info(f"Caricamento del dataset da {input_file}")
    try:
        df = pd.read_csv(input_file)
        logger.info(f"Dataset caricato con successo. Dimensioni: {df.shape}")
    except Exception as e:
        logger.error(f"Errore nel caricamento del dataset: {e}")
        return None

    # Inizializza le nuove colonne
    df['stance_score'] = 0.0
    df['stance_category'] = 'neutral'
    
    # Ottieni la lista di thread unici
    unique_threads = df['thread_id'].unique()
    logger.info(f"Elaborazione della stance per {len(unique_threads)} thread unici...")
    
    # Processa ogni thread separatamente
    processed_dfs = []
    for thread_id in tqdm(unique_threads, desc="Analisi della stance per thread"):
        processed_thread = analyze_thread_stance(df, thread_id)
        processed_dfs.append(processed_thread)
    
    # Combina tutti i dataframe processati
    result_df = pd.concat(processed_dfs, ignore_index=False)
    
    # Ordina il dataframe come era originariamente
    result_df = result_df.sort_index()
    
    logger.info("Analisi della stance completata")
    return result_df

def aggregate_stance_statistics(df):
    """
    Calcola statistiche aggregate sulla stance per veracity category.
    
    Args:
        df (pd.DataFrame): Il dataset con le stance calcolate
        
    Returns:
        pd.DataFrame: Statistiche aggregate
    """
    logger.info("Calcolo delle statistiche aggregate per stance...")
    
    # Aggregazione per veracity
    veracity_stance = df.groupby(['veracity', 'is_rumour']).agg({
        'stance_score': ['mean', 'std', 'count'],
        'stance_category': lambda x: pd.Series(x).value_counts().to_dict()
    })
    
    # Aggregazione per evento
    event_stance = df.groupby(['event']).agg({
        'stance_score': ['mean', 'std', 'count'],
        'stance_category': lambda x: pd.Series(x).value_counts().to_dict()
    })
    
    logger.info("Statistiche aggregate per stance calcolate")
    return veracity_stance, event_stance

def main():
    """Funzione principale per l'esecuzione dello script."""
    # Definizione dei percorsi dei file
    project_dir = Path(__file__).resolve().parents[2]
    data_dir = project_dir / "data"
    processed_dir = data_dir / "processed"
    
    input_file = processed_dir / "pheme_with_sentiment.csv"
    output_file = processed_dir / "pheme_with_stance.csv"
    
    # Verifica che il file di input esista
    if not input_file.exists():
        logger.error(f"Il file {input_file} non esiste. Eseguire prima l'analisi del sentiment.")
        return
    
    # Elabora il dataset e aggiungi feature di stance
    enriched_df = process_dataset_stance(input_file)
    if enriched_df is None:
        return
    
    # Salva il dataset arricchito
    try:
        enriched_df.to_csv(output_file, index=False)
        logger.info(f"Dataset con feature di stance salvato in {output_file}")
    except Exception as e:
        logger.error(f"Errore nel salvataggio del dataset: {e}")
        return
    
    # Calcola e salva statistiche aggregate
    veracity_stance, event_stance = aggregate_stance_statistics(enriched_df)
    
    # Salva le statistiche aggregate
    tables_dir = project_dir / "results" / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    
    veracity_stance_file = tables_dir / "veracity_stance_stats.csv"
    event_stance_file = tables_dir / "event_stance_stats.csv"
    
    veracity_stance.to_csv(veracity_stance_file)
    event_stance.to_csv(event_stance_file)
    logger.info(f"Statistiche stance per veracity salvate in {veracity_stance_file}")
    logger.info(f"Statistiche stance per evento salvate in {event_stance_file}")
    
    logger.info("Processo di analisi della stance completato!")

if __name__ == "__main__":
    main()
