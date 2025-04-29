#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Modulo per analizzare la leggibilità e il livello di acculturazione nei commenti del dataset PHEME.

Questo script implementa funzioni per calcolare vari indici di leggibilità,
complessità lessicale e ricchezza del vocabolario che possono essere utilizzati
come indicatori del livello di acculturazione dell'autore.
"""

import pandas as pd
import numpy as np
import re
import logging
import math
from pathlib import Path
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from collections import Counter
from tqdm import tqdm
import textstat

# Configurazione del logger
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Verifica se i dati necessari per NLTK sono già scaricati, altrimenti li scarica
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
except LookupError:
    logger.info("Downloading NLTK resources...")
    nltk.download('punkt')
    nltk.download('stopwords')

def count_syllables(word):
    """
    Conta le sillabe in una parola inglese.
    
    Args:
        word (str): La parola di cui contare le sillabe
        
    Returns:
        int: Numero di sillabe
    """
    # Conversione in minuscolo e rimozione dei caratteri non alfabetici
    word = re.sub(r'[^a-zA-Z]', '', word.lower())
    
    # Caso speciale per parole vuote
    if not word:
        return 0
        
    # Conta vocali con regole basiche
    count = 0
    vowels = "aeiouy"
    if word[0] in vowels:
        count += 1
    for index in range(1, len(word)):
        if word[index] in vowels and word[index - 1] not in vowels:
            count += 1
    
    # Correzioni per casi particolari
    if word.endswith("e"):
        count -= 1
    if word.endswith("le") and len(word) > 2 and word[-3] not in vowels:
        count += 1
    if count == 0:
        count = 1
        
    return count

def calculate_readability_metrics(text):
    """
    Calcola vari indici di leggibilità per un testo.
    
    Args:
        text (str): Il testo da analizzare
        
    Returns:
        dict: Dizionario contenente gli indici di leggibilità
    """
    if not text or not isinstance(text, str) or len(text) < 10:
        return {
            "flesch_reading_ease": 0,
            "flesch_kincaid_grade": 0,
            "smog_index": 0,
            "gunning_fog": 0,
            "automated_readability_index": 0
        }
    
    try:
        # Utilizza textstat per il calcolo degli indici di leggibilità
        metrics = {
            "flesch_reading_ease": textstat.flesch_reading_ease(text),
            "flesch_kincaid_grade": textstat.flesch_kincaid_grade(text),
            "smog_index": textstat.smog_index(text),
            "gunning_fog": textstat.gunning_fog(text),
            "automated_readability_index": textstat.automated_readability_index(text)
        }
        return metrics
    
    except Exception as e:
        logger.warning(f"Errore nel calcolo delle metriche di leggibilità: {e}")
        return {
            "flesch_reading_ease": 0,
            "flesch_kincaid_grade": 0,
            "smog_index": 0,
            "gunning_fog": 0,
            "automated_readability_index": 0
        }

def calculate_lexical_complexity(text):
    """
    Calcola metriche di complessità lessicale.
    
    Args:
        text (str): Il testo da analizzare
        
    Returns:
        dict: Dizionario contenente metriche di complessità lessicale
    """
    if not text or not isinstance(text, str) or len(text) < 10:
        return {
            "type_token_ratio": 0,
            "vocabulary_richness": 0,
            "avg_word_length": 0,
            "long_words_ratio": 0
        }
    
    try:
        # Tokenizza il testo in modo semplice (fallback sicuro)
        try:
            words = word_tokenize(text.lower())
        except:
            # Fallback se word_tokenize non funziona
            words = re.findall(r'\b[a-zA-Z]+\b', text.lower())
            
        words = [word for word in words if word.isalpha()]  # Solo parole alfabetiche
        
        if not words:
            return {
                "type_token_ratio": 0,
                "vocabulary_richness": 0,
                "avg_word_length": 0,
                "long_words_ratio": 0
            }
            
        # Type-Token Ratio (diversità lessicale)
        unique_words = set(words)
        ttr = len(unique_words) / len(words) if words else 0
        
        # Ricchezza del vocabolario (hapax legomena ratio)
        word_freq = Counter(words)
        hapax = sum(1 for word, freq in word_freq.items() if freq == 1)
        hapax_ratio = hapax / len(words) if words else 0
        
        # Lunghezza media delle parole
        avg_word_length = sum(len(word) for word in words) / len(words) if words else 0
        
        # Proporzione di parole lunghe (>6 caratteri)
        long_words = sum(1 for word in words if len(word) > 6)
        long_words_ratio = long_words / len(words) if words else 0
        
        return {
            "type_token_ratio": ttr,
            "vocabulary_richness": hapax_ratio,
            "avg_word_length": avg_word_length,
            "long_words_ratio": long_words_ratio
        }
    
    except Exception as e:
        logger.warning(f"Errore nel calcolo della complessità lessicale: {e}")
        return {
            "type_token_ratio": 0,
            "vocabulary_richness": 0,
            "avg_word_length": 0,
            "long_words_ratio": 0
        }

def calculate_formal_language_indicators(text):
    """
    Calcola indicatori di uso di linguaggio formale.
    
    Args:
        text (str): Il testo da analizzare
        
    Returns:
        dict: Dizionario contenente indicatori di linguaggio formale
    """
    if not text or not isinstance(text, str) or len(text) < 10:
        return {
            "formal_language_score": 0,
            "abbreviation_ratio": 0
        }
    
    try:
        # Tokenizza il testo in modo semplice (fallback sicuro)
        try:
            words = word_tokenize(text.lower())
        except:
            # Fallback se word_tokenize non funziona
            words = re.findall(r'\b[a-zA-Z]+\b', text.lower())
        if not words:
            return {
                "formal_language_score": 0,
                "abbreviation_ratio": 0
            }
        
        # Lista di parole informali comuni (slang, abbreviazioni)
        informal_words = set([
            "lol", "omg", "btw", "tbh", "idk", "lmao", "rofl", "wtf", 
            "gonna", "wanna", "gotta", "cuz", "cause", "u", "r", "y", "k"
        ])
        
        # Controlla la presenza di abbreviazioni ed emoticon
        abbreviation_pattern = re.compile(r'\b[a-z]{1,2}\b|\b[a-z]+[0-9]+[a-z]*\b')
        emoticon_pattern = re.compile(r'[:;=]-?[\)\(DP]')
        
        # Conta elementi informali
        informal_count = sum(1 for word in words if word in informal_words)
        abbreviation_count = sum(1 for word in words if abbreviation_pattern.match(word))
        emoticon_count = len(emoticon_pattern.findall(text))
        
        total_informal = informal_count + abbreviation_count + emoticon_count
        
        # Calcola il punteggio di formalità (inverso dell'informalità)
        formality_score = max(0, 1 - (total_informal / len(words)))
        
        # Rapporto di abbreviazioni
        abbreviation_ratio = abbreviation_count / len(words) if words else 0
        
        return {
            "formal_language_score": formality_score,
            "abbreviation_ratio": abbreviation_ratio
        }
    
    except Exception as e:
        logger.warning(f"Errore nel calcolo degli indicatori di linguaggio formale: {e}")
        return {
            "formal_language_score": 0,
            "abbreviation_ratio": 0
        }

def analyze_text_complexity(text):
    """
    Analizza un testo per determinare la sua complessità e livello di acculturazione.
    
    Args:
        text (str): Testo da analizzare
        
    Returns:
        dict: Un dizionario con tutte le metriche calcolate
    """
    if not text or not isinstance(text, str):
        return {
            "culture_score": 0,
            "readability_category": "unknown"
        }
    
    # Calcola tutte le metriche
    readability = calculate_readability_metrics(text)
    lexical = calculate_lexical_complexity(text)
    formal = calculate_formal_language_indicators(text)
    
    # Calcola un punteggio complessivo di acculturazione
    # Normalizza i punteggi di leggibilità
    flesch_normalized = min(1, max(0, readability["flesch_reading_ease"] / 100))
    fog_normalized = min(1, max(0, 1 - (readability["gunning_fog"] / 20)))
    
    # Combina metriche in un punteggio di acculturazione
    culture_score = (
        0.3 * flesch_normalized +
        0.2 * fog_normalized +
        0.2 * lexical["type_token_ratio"] +
        0.1 * lexical["vocabulary_richness"] +
        0.1 * lexical["long_words_ratio"] +
        0.1 * formal["formal_language_score"]
    )
    
    # Categorizza il livello di leggibilità
    if readability["flesch_reading_ease"] >= 90:
        readability_category = "very_easy"
    elif readability["flesch_reading_ease"] >= 70:
        readability_category = "easy"
    elif readability["flesch_reading_ease"] >= 50:
        readability_category = "moderate"
    elif readability["flesch_reading_ease"] >= 30:
        readability_category = "difficult"
    else:
        readability_category = "very_difficult"
    
    # Aggrega tutti i risultati
    results = {
        "culture_score": culture_score,
        "readability_category": readability_category
    }
    results.update(readability)
    results.update(lexical)
    results.update(formal)
    
    return results

def process_dataset(file_path):
    """
    Elabora il dataset PHEME con stance e aggiunge feature di leggibilità e cultura.
    
    Args:
        file_path (str): Percorso al file CSV del dataset con stance
        
    Returns:
        pd.DataFrame: Il dataset con le colonne di leggibilità e cultura aggiunte
    """
    logger.info(f"Caricamento del dataset da {file_path}")
    try:
        df = pd.read_csv(file_path)
        logger.info(f"Dataset caricato con successo. Dimensioni: {df.shape}")
    except Exception as e:
        logger.error(f"Errore nel caricamento del dataset: {e}")
        return None
    
    # Inizializza le nuove colonne principali
    df['culture_score'] = 0.0
    df['readability_category'] = 'unknown'
    df['flesch_reading_ease'] = 0.0
    df['type_token_ratio'] = 0.0
    df['formal_language_score'] = 0.0
    
    # Elabora i testi e aggiungi le feature di leggibilità e cultura
    logger.info("Analisi della leggibilità e cultura per ogni tweet...")
    
    results = []
    for idx, row in tqdm(df.iterrows(), total=df.shape[0], desc="Analisi di leggibilità e cultura"):
        text = row['text']
        metrics = analyze_text_complexity(text)
        results.append(metrics)
    
    # Converte la lista di dizionari in un dataframe e uniscilo al dataframe originale
    results_df = pd.DataFrame(results)
    
    # Assicurati che l'indice sia compatibile con df
    results_df.index = df.index
    
    # Unisci i dataframe
    for column in results_df.columns:
        df[column] = results_df[column]
    
    logger.info("Analisi della leggibilità e cultura completata")
    return df

def aggregate_readability_statistics(df):
    """
    Calcola statistiche aggregate sulla leggibilità e cultura per veracity.
    
    Args:
        df (pd.DataFrame): Il dataset con le feature di leggibilità
        
    Returns:
        pd.DataFrame: Statistiche aggregate
    """
    logger.info("Calcolo delle statistiche aggregate per leggibilità e cultura...")
    
    # Aggregazione per veracity
    veracity_readability = df.groupby(['veracity', 'is_rumour']).agg({
        'culture_score': ['mean', 'std', 'min', 'max'],
        'flesch_reading_ease': ['mean', 'std'],
        'type_token_ratio': ['mean', 'std'],
        'readability_category': lambda x: pd.Series(x).value_counts().to_dict()
    })
    
    # Aggregazione per evento
    event_readability = df.groupby(['event']).agg({
        'culture_score': ['mean', 'std'],
        'flesch_reading_ease': ['mean', 'std'],
        'type_token_ratio': ['mean', 'std'],
        'readability_category': lambda x: pd.Series(x).value_counts().to_dict()
    })
    
    logger.info("Statistiche aggregate per leggibilità e cultura calcolate")
    return veracity_readability, event_readability

def main():
    """Funzione principale per l'esecuzione dello script."""
    # Definizione dei percorsi dei file
    project_dir = Path(__file__).resolve().parents[2]
    data_dir = project_dir / "data"
    processed_dir = data_dir / "processed"
    
    input_file = processed_dir / "pheme_with_stance.csv"
    output_file = processed_dir / "pheme_with_readability.csv"
    
    # Verifica che il file di input esista
    if not input_file.exists():
        logger.error(f"Il file {input_file} non esiste. Eseguire prima l'analisi della stance.")
        return
    
    # Elabora il dataset e aggiungi feature di leggibilità e cultura
    enriched_df = process_dataset(input_file)
    if enriched_df is None:
        return
    
    # Salva il dataset arricchito
    try:
        enriched_df.to_csv(output_file, index=False)
        logger.info(f"Dataset con feature di leggibilità salvato in {output_file}")
    except Exception as e:
        logger.error(f"Errore nel salvataggio del dataset: {e}")
        return
    
    # Calcola e salva statistiche aggregate
    veracity_readability, event_readability = aggregate_readability_statistics(enriched_df)
    
    # Salva le statistiche aggregate
    tables_dir = project_dir / "results" / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    
    veracity_readability_file = tables_dir / "veracity_readability_stats.csv"
    event_readability_file = tables_dir / "event_readability_stats.csv"
    
    veracity_readability.to_csv(veracity_readability_file)
    event_readability.to_csv(event_readability_file)
    logger.info(f"Statistiche di leggibilità per veracity salvate in {veracity_readability_file}")
    logger.info(f"Statistiche di leggibilità per evento salvate in {event_readability_file}")
    
    logger.info("Processo di analisi della leggibilità e cultura completato!")

if __name__ == "__main__":
    main()
