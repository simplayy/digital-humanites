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
EVENTS = ["charliehebdo", "ferguson", "germanwings-crash", "ottawashooting", "sydneysiege"]
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
        with open(content_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    except (json.JSONDecodeError, UnicodeDecodeError) as e:
        logger.warning(f"Errore nella lettura di {content_file}: {e}")
        return None


def load_tweets(news_dir):
    """
    Carica i tweet associati a una notizia
    
    Args:
        news_dir (Path): Directory della notizia specifica
    
    Returns:
        list: Lista di tweet o lista vuota se non trovati
    """
    tweets_dir = news_dir / "tweets"
    if not tweets_dir.exists() or not tweets_dir.is_dir():
        return []
    
    tweets = []
    for tweet_file in tweets_dir.glob("*.json"):
        try:
            with open(tweet_file, 'r', encoding='utf-8') as f:
                tweet_data = json.load(f)
                tweets.append(tweet_data)
        except (json.JSONDecodeError, UnicodeDecodeError) as e:
            logger.warning(f"Errore nella lettura di {tweet_file}: {e}")
            continue
    
    return tweets


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


def extract_news_features(news_content):
    """
    Estrae le caratteristiche rilevanti dal contenuto di una notizia
    
    Args:
        news_content (dict): Contenuto della notizia
        
    Returns:
        dict: Caratteristiche estratte
    """
    if not news_content:
        return {
            "title": "",
            "text": "",
            "meta_data": {}
        }
    
    title = clean_text(news_content.get("title", ""))
    text = clean_text(news_content.get("text", ""))
    
    # Metadati aggiuntivi
    meta_data = {
        "source": news_content.get("source", ""),
        "publish_date": news_content.get("publish_date", ""),
        "authors": news_content.get("authors", []),
        "url": news_content.get("url", "")
    }
    
    return {
        "title": title,
        "text": text,
        "meta_data": meta_data
    }


def extract_tweet_features(tweets):
    """
    Estrae le caratteristiche dai tweet associati a una notizia
    
    Args:
        tweets (list): Lista di tweet
        
    Returns:
        list: Lista di commenti estratti dai tweet
    """
    comments = []
    for tweet in tweets:
        if not tweet:
            continue
            
        # Estrai il testo del tweet
        text = clean_text(tweet.get("text", ""))
        if not text:
            continue
        
        # Metadati del tweet
        user = tweet.get("user", {})
        timestamp = tweet.get("created_at", "")
        favorites = tweet.get("favorite_count", 0)
        retweets = tweet.get("retweet_count", 0)
        
        comment = {
            "text": text,
            "timestamp": timestamp,
            "user_id": user.get("id_str", ""),
            "user_followers": user.get("followers_count", 0),
            "favorites": favorites,
            "retweets": retweets
        }
        
        comments.append(comment)
    
    return comments


def process_news_category(category, label):
    """
    Processa tutte le notizie di una categoria e etichetta specifiche
    
    Args:
        category (str): Categoria ('politifact', 'gossipcop')
        label (str): Etichetta ('fake', 'real')
        
    Returns:
        list: Lista di dizionari con le informazioni delle notizie
    """
    dataset = []
    
    # Determina la directory in base alla categoria e all'etichetta
    if label == "fake":
        base_dir = RAW_DATA_DIR / category / "fake"
    else:
        base_dir = RAW_DATA_DIR / f"{category}_real" / "real"
    
    if not base_dir.exists():
        logger.warning(f"Directory non trovata: {base_dir}")
        return dataset
    
    # Scansiona tutte le notizie nella categoria
    news_dirs = [d for d in base_dir.iterdir() if d.is_dir()]
    logger.info(f"Processando {len(news_dirs)} notizie da {category} ({label})")
    
    for news_dir in tqdm(news_dirs, desc=f"{category}_{label}"):
        news_id = news_dir.name
        
        # Carica il contenuto della notizia
        news_content = load_news_content(news_dir)
        news_features = extract_news_features(news_content)
        
        # Carica i tweet associati
        tweets = load_tweets(news_dir)
        comments = extract_tweet_features(tweets)
        
        # Crea un entry per il dataset
        entry = {
            "news_id": news_id,
            "category": category,
            "label": label,
            "title": news_features["title"],
            "text": news_features["text"],
            "meta_data": news_features["meta_data"],
            "comments": comments,
            "comments_count": len(comments)
        }
        
        dataset.append(entry)
    
    return dataset


def preprocess_dataset():
    """
    Preelabora l'intero dataset FakeNewsNet e lo salva in formato strutturato
    
    Returns:
        Path: Percorso al file preprocessato
    """
    full_dataset = []
    
    # Processa ogni categoria e etichetta
    for category in CATEGORIES:
        for label in LABELS:
            logger.info(f"Processando {category} - {label}")
            category_data = process_news_category(category, label)
            full_dataset.extend(category_data)
    
    # Converte in DataFrame per facilitare l'analisi
    df = pd.json_normalize(full_dataset)
    
    # Salva il dataset preprocessato
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    output_file = PROCESSED_DATA_DIR / "fakenewsnet_processed.csv"
    df.to_csv(output_file, index=False)
    
    # Salva anche una versione JSON più completa (con commenti nested)
    json_output = PROCESSED_DATA_DIR / "fakenewsnet_processed.json"
    with open(json_output, 'w', encoding='utf-8') as f:
        json.dump(full_dataset, f, ensure_ascii=False, indent=2)
    
    logger.info(f"Dataset preprocessato salvato in {output_file}")
    logger.info(f"Dataset JSON salvato in {json_output}")
    
    return output_file


if __name__ == "__main__":
    logger.info("Iniziando il preprocessing del dataset FakeNewsNet...")
    preprocess_dataset()
    logger.info("Preprocessing completato!")
