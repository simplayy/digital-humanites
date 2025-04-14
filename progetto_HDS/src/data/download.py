#!/usr/bin/env python
"""
Scrip# Eventi disponibili nel dataset PHEME
PHEME_EVENTS = [
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
]aricare il dataset PHEME
"""

import os
import sys
import json
import requests
import zipfile
from pathlib import Path
import logging
from tqdm import tqdm

# Configurare il logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("download.log"),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)

# URL per scaricare il dataset PHEME
PHEME_FIGSHARE_URL = "https://figshare.com/articles/dataset/PHEME_dataset_for_Rumour_Detection_and_Veracity_Classification/6392078"
PHEME_DATASET_URL = "https://figshare.com/ndownloader/articles/6392078/versions/1"
PHEME_DATASET_FILENAME = "PHEME_veracity.zip"

# Definizione degli eventi disponibili nel dataset PHEME
PHEME_EVENTS = [
    "charliehebdo", 
    "ferguson", 
    "germanwings-crash", 
    "ottawashooting", 
    "sydneysiege"
]

# Percorso dove salvare i dati scaricati
DATA_DIR = Path(__file__).resolve().parents[2] / "data"
RAW_DATA_DIR = DATA_DIR / "raw"


def download_file(url, destination):
    """
    Scarica un file da una URL e lo salva nella destinazione specificata,
    mostrando una barra di progresso.
    
    Args:
        url (str): URL del file da scaricare
        destination (Path): Percorso di destinazione dove salvare il file
    
    Returns:
        bool: True se il download è completato con successo, False altrimenti
    """
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        # Ottieni la dimensione del file in bytes
        total_size = int(response.headers.get('content-length', 0))
        
        # Assicurati che la directory di destinazione esista
        destination.parent.mkdir(parents=True, exist_ok=True)
        
        # Scarica il file con una barra di progresso
        with open(destination, 'wb') as f, tqdm(
            desc=f"Downloading {destination.name}",
            total=total_size,
            unit='B',
            unit_scale=True,
            unit_divisor=1024,
        ) as bar:
            for chunk in response.iter_content(chunk_size=8192):
                size = f.write(chunk)
                bar.update(size)
        
        logger.info(f"File scaricato con successo: {destination}")
        return True
    
    except Exception as e:
        logger.error(f"Errore durante il download da {url}: {e}")
        return False


def extract_zip(zip_path, extract_to):
    """
    Estrae un file zip nella directory specificata.
    
    Args:
        zip_path (Path): Percorso al file zip
        extract_to (Path): Directory dove estrarre i contenuti
    
    Returns:
        bool: True se l'estrazione è completata con successo, False altrimenti
    """
    try:
        # Assicurati che la directory di estrazione esista
        extract_to.mkdir(parents=True, exist_ok=True)
        
        if zip_path.suffix == ".zip":
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                # Ottieni il numero totale di file nel zip
                total_files = len(zip_ref.filelist)
                
                # Estrai i file con una barra di progresso
                for i, file in enumerate(zip_ref.filelist):
                    zip_ref.extract(file, extract_to)
                    sys.stdout.write(f"\rEstracting: {i+1}/{total_files} files")
                    sys.stdout.flush()
                
                sys.stdout.write("\n")
        elif zip_path.suffix == ".bz2" or str(zip_path).endswith(".tar.bz2"):
            # Per file tar.bz2, usa il comando tar
            import subprocess
            
            logger.info(f"Estrazione del file tar.bz2: {zip_path}")
            try:
                result = subprocess.run(
                    ["tar", "-xjf", str(zip_path), "-C", str(extract_to)],
                    check=True,
                    capture_output=True
                )
                logger.info("Estrazione tar.bz2 completata con successo")
            except subprocess.CalledProcessError as e:
                logger.error(f"Errore nell'estrazione tar.bz2: {e}")
                logger.error(f"Output: {e.output.decode('utf-8')}")
                logger.error(f"Error: {e.stderr.decode('utf-8')}")
                return False
        else:
            logger.error(f"Formato di file non supportato: {zip_path}")
            return False
        
        logger.info(f"File estratto con successo: {zip_path} -> {extract_to}")
        return True
    
    except Exception as e:
        logger.error(f"Errore durante l'estrazione di {zip_path}: {e}")
        return False


def download_pheme_dataset():
    """
    Scarica ed estrae il dataset PHEME per l'analisi dei commenti e veridicità delle notizie.
    Il dataset contiene conversazioni Twitter e le relative etichette di veridicità.
    
    Returns:
        bool: True se il download è completato con successo, False altrimenti
    """
    logger.info("Scaricamento del dataset PHEME da Figshare...")
    
    # Percorsi per salvare il dataset
    pheme_dir = RAW_DATA_DIR / "pheme"
    zip_path = pheme_dir / PHEME_DATASET_FILENAME
    
    # Assicurati che la directory esista
    pheme_dir.mkdir(parents=True, exist_ok=True)
    
    # Scarica il file zip
    if not download_file(PHEME_DATASET_URL, zip_path):
        return False
    
    # Estrai il file zip
    extract_to = pheme_dir / "veracity_dataset"
    if not extract_zip(zip_path, extract_to):
        return False
    
    logger.info(f"Dataset PHEME scaricato con successo in {extract_to}")
    logger.info(f"Il dataset contiene i seguenti eventi: {', '.join(PHEME_EVENTS)}")
    
    return True


def verify_pheme_dataset(pheme_dir):
    """
    Verifica l'integrità del dataset PHEME scaricato.
    
    Args:
        pheme_dir (Path): Directory principale del dataset PHEME
    
    Returns:
        bool: True se la verifica è completata con successo, False altrimenti
    """
    veracity_dir = pheme_dir / "veracity_dataset"
    
    if not veracity_dir.exists() or not veracity_dir.is_dir():
        logger.error(f"Directory {veracity_dir} non trovata!")
        return False
    
    # Verifica che tutti gli eventi siano presenti
    all_events_exist = True
    for event in PHEME_EVENTS:
        event_dir = veracity_dir / "all-rnr-annotated-threads" / event
        if not event_dir.exists() or not event_dir.is_dir():
            logger.error(f"Directory dell'evento {event} non trovata: {event_dir}")
            all_events_exist = False
    
    if not all_events_exist:
        return False
    
    # Verifica che ci siano sia thread veri che falsi
    counts = {"true": 0, "false": 0, "unverified": 0}
    
    for event in PHEME_EVENTS:
        event_dir = veracity_dir / "all-rnr-annotated-threads" / event
        for veracity in ["rumours", "non-rumours"]:
            if (event_dir / veracity).exists():
                for thread in (event_dir / veracity).iterdir():
                    if thread.is_dir():
                        # Il thread è un rumour, quindi cerchiamo l'annotazione di veracity
                        if veracity == "rumours":
                            # Verifica l'etichetta di veridicità nel file di annotazione
                            annotation_file = event_dir / "en" / f"{thread.name}.json"
                            if annotation_file.exists():
                                with open(annotation_file, 'r') as f:
                                    try:
                                        annotation = json.load(f)
                                        if "veracity" in annotation:
                                            counts[annotation["veracity"]] += 1
                                    except json.JSONDecodeError:
                                        logger.warning(f"Errore nel parsing del file {annotation_file}")
                        else:
                            # I non-rumours sono considerati veri per default
                            counts["true"] += 1
    
    logger.info(f"Statistiche del dataset PHEME:")
    logger.info(f"- Thread veri (true): {counts['true']}")
    logger.info(f"- Thread falsi (false): {counts['false']}")
    logger.info(f"- Thread non verificati (unverified): {counts['unverified']}")
    
    # Verifica che ci siano dati in entrambe le categorie
    if counts["true"] == 0 or counts["false"] == 0:
        logger.warning("Non ci sono abbastanza dati in entrambe le categorie!")
        return False
    
    return True


if __name__ == "__main__":
    # Assicurati che le directory esistano
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    logger.info("Iniziando il download del dataset PHEME...")
    
    if download_pheme_dataset():
        logger.info("Download completato con successo!")
        
        # Verifica l'integrità del dataset
        pheme_dir = RAW_DATA_DIR / "pheme"
        if verify_pheme_dataset(pheme_dir):
            logger.info("Verifica dell'integrità del dataset completata con successo!")
        else:
            logger.error("La verifica dell'integrità del dataset ha rilevato problemi.")
    else:
        logger.error("Si è verificato un errore durante il download del dataset.")
    
    logger.info("Operazioni completate!")
