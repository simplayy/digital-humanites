#!/usr/bin/env python
"""
Script per il preprocessing del dataset PHEME per analisi del sentiment nei commenti e fake news detection
"""

import os
import sys
import json
import pandas as pd
import numpy as np
from pathlib import Path
import logging
from tqdm import tqdm
import re
from datetime import datetime

# Configurare il logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("preprocess.log"),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)

# Percorsi dei dati
DATA_DIR = Path(__file__).resolve().parents[2] / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

# Percorso specifico del dataset PHEME
PHEME_DATA_DIR = RAW_DATA_DIR / "pheme" / "veracity_dataset"

# Eventi disponibili nel dataset PHEME
EVENTS = [
    "charliehebdo-all-rnr-threads", 
    "ferguson-all-rnr-threads", 
    "germanwings-crash-all-rnr-threads", 
    "ottawashooting-all-rnr-threads", 
    "sydneysiege-all-rnr-threads",
    # Eventi aggiuntivi disponibili nel dataset
    "ebola-essien-all-rnr-threads",
    "gurlitt-all-rnr-threads",
    "prince-toronto-all-rnr-threads",
    "putinmissing-all-rnr-threads"
]
# Etichette di veridicità
VERACITY_LABELS = ["true", "false", "unverified"]


def load_tweet_thread(thread_dir):
    """
    Carica il tweet principale e le relative risposte da una directory di thread
    
    Args:
        thread_dir (Path): Directory del thread specifico
    
    Returns:
        dict: Dizionario con il tweet principale e le risposte
    """
    if not thread_dir.exists() or not thread_dir.is_dir():
        logger.warning(f"Directory thread non trovata: {thread_dir}")
        return None
    
    try:
        # Carica il tweet principale (source-tweet)
        source_tweet_file = thread_dir / "source-tweet" / f"{thread_dir.name}.json"
        
        if not source_tweet_file.exists():
            logger.warning(f"File source-tweet non trovato: {source_tweet_file}")
            return None
        
        with open(source_tweet_file, 'r', encoding='utf-8') as f:
            source_tweet = json.load(f)
        
        # Carica le risposte (reaction-tweets)
        reactions_dir = thread_dir / "reactions"
        reactions = []
        
        if reactions_dir.exists() and reactions_dir.is_dir():
            for reaction_file in reactions_dir.glob("*.json"):
                try:
                    with open(reaction_file, 'r', encoding='utf-8') as f:
                        reaction = json.load(f)
                        reactions.append(reaction)
                except (json.JSONDecodeError, UnicodeDecodeError) as e:
                    logger.warning(f"Errore nel parsing del file {reaction_file}: {e}")
        
        return {
            "source_tweet": source_tweet,
            "reactions": reactions
        }
        
    except (json.JSONDecodeError, UnicodeDecodeError) as e:
        logger.warning(f"Errore nel parsing dei file nel thread {thread_dir}: {e}")
        return None


def load_veracity_annotation(event_dir, thread_name):
    """
    Carica l'annotazione di veridicità per un thread specifico
    
    Args:
        event_dir (Path): Directory dell'evento
        thread_name (str): Nome del thread
    
    Returns:
        str: Etichetta di veridicità ('true', 'false', 'unverified') o None se non trovata
    """
    annotation_file = event_dir / "en" / f"{thread_name}.json"
    
    if not annotation_file.exists():
        logger.warning(f"File di annotazione non trovato: {annotation_file}")
        return None
    
    try:
        with open(annotation_file, 'r', encoding='utf-8') as f:
            annotation = json.load(f)
            return annotation.get("veracity")
    except (json.JSONDecodeError, UnicodeDecodeError) as e:
        logger.warning(f"Errore nel parsing del file di annotazione {annotation_file}: {e}")
        return None


def clean_text(text):
    """
    Pulisce il testo rimuovendo URL, menzioni, caratteri speciali, ecc.
    
    Args:
        text (str): Testo da pulire
        
    Returns:
        str: Testo pulito
    """
    if not isinstance(text, str):
        return ""
    
    # Rimuovi URL
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    
    # Rimuovi menzioni (@username)
    text = re.sub(r'@\w+', '', text)
    
    # Rimuovi hashtag ma mantieni il testo (#example -> example)
    text = re.sub(r'#(\w+)', r'\1', text)
    
    # Rimuovi caratteri speciali e numeri
    text = re.sub(r'[^\w\s]', '', text)
    
    # Rimuovi spazi multipli
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text


def extract_tweet_features(tweet):
    """
    Estrae le caratteristiche rilevanti da un tweet
    
    Args:
        tweet (dict): Oggetto tweet
        
    Returns:
        dict: Caratteristiche estratte
    """
    if not tweet:
        return None
    
    # Estrai informazioni base
    text = clean_text(tweet.get("text", ""))
    tweet_id = tweet.get("id_str", "")
    
    # Metadati del tweet
    created_at_str = tweet.get("created_at", "")
    try:
        created_at = datetime.strptime(created_at_str, "%a %b %d %H:%M:%S %z %Y") if created_at_str else None
        timestamp = created_at.isoformat() if created_at else ""
    except ValueError:
        timestamp = ""
    
    # Informazioni sull'utente
    user = tweet.get("user", {})
    user_id = user.get("id_str", "")
    user_name = user.get("screen_name", "")
    user_followers = user.get("followers_count", 0)
    user_friends = user.get("friends_count", 0)
    user_favourites = user.get("favourites_count", 0)
    user_verified = user.get("verified", False)
    
    # Metriche di engagement
    favorites = tweet.get("favorite_count", 0)
    retweets = tweet.get("retweet_count", 0)
    
    return {
        "tweet_id": tweet_id,
        "text": text,
        "timestamp": timestamp,
        "user_id": user_id,
        "user_name": user_name,
        "user_followers": user_followers,
        "user_friends": user_friends,
        "user_favourites": user_favourites,
        "user_verified": user_verified,
        "favorites": favorites,
        "retweets": retweets
    }


def process_thread(thread_data, veracity):
    """
    Processa un thread completo (tweet principale + risposte) e lo prepara per l'analisi
    
    Args:
        thread_data (dict): Dati del thread con tweet principale e risposte
        veracity (str): Etichetta di veridicità ('true', 'false', 'unverified')
        
    Returns:
        dict: Thread processato con features estratte
    """
    if not thread_data or "source_tweet" not in thread_data:
        return None
    
    # Estrai features dal tweet principale
    source_tweet = extract_tweet_features(thread_data["source_tweet"])
    if not source_tweet:
        return None
    
    # Estrai features dalle risposte
    reactions = []
    for reaction in thread_data.get("reactions", []):
        reaction_features = extract_tweet_features(reaction)
        if reaction_features:
            reactions.append(reaction_features)
    
    # Ordina le risposte in ordine cronologico
    reactions.sort(key=lambda x: x.get("timestamp", ""))
    
    return {
        "source_tweet": source_tweet,
        "reactions": reactions,
        "veracity": veracity,
        "reactions_count": len(reactions)
    }


def process_event(event):
    """
    Processa tutti i thread per un evento specifico
    
    Args:
        event (str): Nome dell'evento (es. 'charliehebdo', 'ferguson', ecc.)
        
    Returns:
        list: Lista di thread processati
    """
    event_dir = PHEME_DATA_DIR / "all-rnr-annotated-threads" / event
    
    if not event_dir.exists() or not event_dir.is_dir():
        logger.error(f"Directory evento non trovata: {event_dir}")
        return []
    
    logger.info(f"Processando i thread dell'evento: {event}")
    processed_threads = []
    
    # Processa i rumours (potrebbero essere veri, falsi o non verificati)
    rumours_dir = event_dir / "rumours"
    if rumours_dir.exists() and rumours_dir.is_dir():
        for thread_dir in tqdm(list(rumours_dir.iterdir()), desc=f"{event} - rumours"):
            if thread_dir.is_dir():
                # Carica il thread e l'annotazione di veridicità
                thread_data = load_tweet_thread(thread_dir)
                veracity = load_veracity_annotation(event_dir, thread_dir.name)
                
                if thread_data and veracity:
                    processed_thread = process_thread(thread_data, veracity)
                    if processed_thread:
                        processed_thread["event"] = event
                        processed_thread["thread_id"] = thread_dir.name
                        processed_thread["is_rumour"] = True
                        processed_threads.append(processed_thread)
    
    # Processa i non-rumours (considerati sempre veri)
    non_rumours_dir = event_dir / "non-rumours"
    if non_rumours_dir.exists() and non_rumours_dir.is_dir():
        for thread_dir in tqdm(list(non_rumours_dir.iterdir()), desc=f"{event} - non-rumours"):
            if thread_dir.is_dir():
                # Carica il thread (per i non-rumours, la veridicità è sempre "true")
                thread_data = load_tweet_thread(thread_dir)
                
                if thread_data:
                    processed_thread = process_thread(thread_data, "true")
                    if processed_thread:
                        processed_thread["event"] = event
                        processed_thread["thread_id"] = thread_dir.name
                        processed_thread["is_rumour"] = False
                        processed_threads.append(processed_thread)
    
    logger.info(f"Processati {len(processed_threads)} thread per l'evento {event}")
    return processed_threads


def preprocess_dataset():
    """
    Preprocessa l'intero dataset PHEME e salva i dati in formato strutturato
    
    Returns:
        Path: Percorso al file preprocessato
    """
    all_threads = []
    
    # Processa tutti gli eventi disponibili
    for event in EVENTS:
        event_threads = process_event(event)
        all_threads.extend(event_threads)
    
    if not all_threads:
        logger.error("Nessun thread processato!")
        return None
    
    logger.info(f"Numero totale di thread processati: {len(all_threads)}")
    
    # Crea una struttura più piatta per l'analisi in pandas
    flattened_data = []
    for thread in all_threads:
        source_tweet = thread["source_tweet"]
        source_row = {
            "thread_id": thread["thread_id"],
            "event": thread["event"],
            "tweet_id": source_tweet["tweet_id"],
            "text": source_tweet["text"],
            "timestamp": source_tweet["timestamp"],
            "user_id": source_tweet["user_id"],
            "user_name": source_tweet["user_name"],
            "user_followers": source_tweet["user_followers"],
            "user_verified": source_tweet["user_verified"],
            "favorites": source_tweet["favorites"],
            "retweets": source_tweet["retweets"],
            "veracity": thread["veracity"],
            "is_rumour": thread["is_rumour"],
            "reactions_count": thread["reactions_count"],
            "is_source": True
        }
        flattened_data.append(source_row)
        
        # Aggiungi le risposte con metadati del thread
        for i, reaction in enumerate(thread["reactions"]):
            reaction_row = {
                "thread_id": thread["thread_id"],
                "event": thread["event"],
                "tweet_id": reaction["tweet_id"],
                "text": reaction["text"],
                "timestamp": reaction["timestamp"],
                "user_id": reaction["user_id"],
                "user_name": reaction["user_name"],
                "user_followers": reaction["user_followers"],
                "user_verified": reaction["user_verified"],
                "favorites": reaction["favorites"],
                "retweets": reaction["retweets"],
                "veracity": thread["veracity"],  # La veridicità del thread principale
                "is_rumour": thread["is_rumour"],
                "reactions_count": thread["reactions_count"],
                "is_source": False,
                "reaction_index": i
            }
            flattened_data.append(reaction_row)
    
    # Converti in DataFrame
    df = pd.DataFrame(flattened_data)
    
    # Salva i dati in diversi formati
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    # Dataset completo
    csv_path = PROCESSED_DATA_DIR / "pheme_processed.csv"
    df.to_csv(csv_path, index=False)
    
    # Solo i tweet principali
    source_df = df[df["is_source"] == True].copy()
    source_csv_path = PROCESSED_DATA_DIR / "pheme_source_tweets.csv"
    source_df.to_csv(source_csv_path, index=False)
    
    # Solo le reazioni/commenti
    reactions_df = df[df["is_source"] == False].copy()
    reactions_csv_path = PROCESSED_DATA_DIR / "pheme_reactions.csv"
    reactions_df.to_csv(reactions_csv_path, index=False)
    
    # Conteggi per categorie
    veracity_counts = source_df["veracity"].value_counts()
    logger.info(f"Distribuzione delle etichette di veridicità:")
    for label, count in veracity_counts.items():
        logger.info(f"- {label}: {count}")
    
    logger.info(f"Numero totale di tweet principali: {len(source_df)}")
    logger.info(f"Numero totale di reazioni/commenti: {len(reactions_df)}")
    logger.info(f"Numero medio di commenti per thread: {source_df['reactions_count'].mean():.2f}")
    
    # Salva anche una versione JSON più completa con la struttura gerarchica
    json_output = PROCESSED_DATA_DIR / "pheme_processed.json"
    with open(json_output, 'w', encoding='utf-8') as f:
        json.dump(all_threads, f, ensure_ascii=False, indent=2)
    
    logger.info(f"Dataset CSV salvato in {csv_path}")
    logger.info(f"Dataset JSON salvato in {json_output}")
    
    return csv_path


if __name__ == "__main__":
    logger.info("Iniziando il preprocessing del dataset PHEME...")
    preprocess_dataset()
    logger.info("Preprocessing completato!")
