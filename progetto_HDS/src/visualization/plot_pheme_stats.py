#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Script per visualizzare statistiche di base sul dataset PHEME preprocessato
"""

import pandas as pd
import matplotlib
# Forza matplotlib a usare il backend Agg (non richiede un server X)
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
import logging
from pathlib import Path

# Configurazione logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                   stream=sys.stdout)
logger = logging.getLogger(__name__)

# Imposta lo stile dei grafici
plt.style.use('ggplot')
sns.set_palette("deep")

def load_dataset(filepath):
    """Carica il dataset CSV"""
    logger.info(f"Caricamento del dataset da {filepath}")
    df = pd.read_csv(filepath)
    return df

def create_output_dir(output_dir):
    """Crea la directory di output se non esiste"""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        logger.info(f"Creata directory {output_dir}")

def plot_event_distribution(df, output_dir):
    """Visualizza la distribuzione degli eventi nel dataset"""
    plt.figure(figsize=(12, 8))
    
    # Conta eventi principali (source tweets)
    event_counts = df[df['is_source'] == True]['event'].value_counts()
    
    # Visualizza grafico a barre
    ax = sns.barplot(x=event_counts.index, y=event_counts.values)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, horizontalalignment='right')
    
    plt.title('Distribuzione degli Eventi nel Dataset PHEME', fontsize=16)
    plt.xlabel('Eventi')
    plt.ylabel('Numero di Thread')
    plt.tight_layout()
    
    # Salva il grafico
    output_file = os.path.join(output_dir, 'event_distribution.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    logger.info(f"Grafico di distribuzione eventi salvato in {output_file}")
    plt.close()

def plot_veracity_distribution(df, output_dir):
    """Visualizza la distribuzione della veridicità nel dataset"""
    plt.figure(figsize=(10, 8))
    
    # Conta solo i tweet principali (source tweets)
    source_df = df[df['is_source'] == True]
    veracity_counts = source_df['veracity'].value_counts()
    
    # Crea grafico a torta
    plt.pie(veracity_counts.values, 
            labels=veracity_counts.index, 
            autopct='%1.1f%%',
            startangle=90,
            shadow=True,
            explode=(0.1, 0, 0) if len(veracity_counts) == 3 else None)
    
    plt.axis('equal')
    plt.title('Distribuzione della Veridicità', fontsize=16)
    
    # Salva il grafico
    output_file = os.path.join(output_dir, 'veracity_distribution.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    logger.info(f"Grafico di distribuzione veridicità salvato in {output_file}")
    plt.close()

def plot_reactions_distribution(df, output_dir):
    """Visualizza la distribuzione del numero di reazioni per thread"""
    plt.figure(figsize=(12, 8))
    
    # Filtra solo i tweet principali
    source_df = df[df['is_source'] == True]
    
    # Crea grafico a istogramma
    sns.histplot(source_df['reactions_count'], kde=True, bins=30)
    plt.title('Distribuzione del Numero di Reazioni per Thread', fontsize=16)
    plt.xlabel('Numero di Reazioni')
    plt.ylabel('Conteggio')
    plt.grid(True)
    
    # Salva il grafico
    output_file = os.path.join(output_dir, 'reactions_distribution.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    logger.info(f"Grafico di distribuzione reazioni salvato in {output_file}")
    plt.close()

def plot_veracity_by_event(df, output_dir):
    """Visualizza la distribuzione della veridicità per ogni evento"""
    plt.figure(figsize=(14, 10))
    
    # Filtra solo i tweet principali e crea tabella pivot
    source_df = df[df['is_source'] == True]
    event_veracity = pd.crosstab(source_df['event'], source_df['veracity'])
    
    # Normalizza i dati per evento
    event_veracity_norm = event_veracity.div(event_veracity.sum(axis=1), axis=0)
    
    # Crea grafico a barre impilate
    event_veracity_norm.plot(kind='bar', stacked=True, figsize=(14, 10))
    plt.title('Distribuzione della Veridicità per Evento', fontsize=16)
    plt.xlabel('Evento')
    plt.ylabel('Proporzione')
    plt.xticks(rotation=45, ha='right')
    plt.legend(title='Veridicità')
    plt.grid(True, axis='y')
    plt.tight_layout()
    
    # Salva il grafico
    output_file = os.path.join(output_dir, 'veracity_by_event.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    logger.info(f"Grafico di veracity per evento salvato in {output_file}")
    plt.close()

def plot_rumour_vs_nonrumour(df, output_dir):
    """Visualizza la proporzione di rumours vs non-rumours nel dataset"""
    plt.figure(figsize=(10, 8))
    
    # Conta i tweet principali
    source_df = df[df['is_source'] == True]
    rumour_counts = source_df['is_rumour'].value_counts()
    
    # Converti i valori booleani a stringhe per l'etichetta
    labels = ['Rumour' if x else 'Non-Rumour' for x in rumour_counts.index]
    
    # Crea grafico a torta
    plt.pie(rumour_counts.values, 
            labels=labels, 
            autopct='%1.1f%%',
            startangle=90,
            shadow=True,
            colors=['#ff9999','#66b3ff'])
    
    plt.axis('equal')
    plt.title('Proporzione di Rumours vs Non-Rumours', fontsize=16)
    
    # Salva il grafico
    output_file = os.path.join(output_dir, 'rumour_distribution.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    logger.info(f"Grafico rumour vs non-rumour salvato in {output_file}")
    plt.close()

def main():
    try:
        # Percorso al dataset
        project_dir = Path(__file__).resolve().parents[2]
        logger.info(f"Project directory: {project_dir}")
        data_file = project_dir / 'data' / 'processed' / 'pheme_processed.csv'
        logger.info(f"Data file path: {data_file}")
        output_dir = project_dir / 'reports' / 'figures'
        logger.info(f"Output directory: {output_dir}")
        
        # Crea directory di output se non esiste
        create_output_dir(output_dir)
        
        # Verifica che il file esista
        if not os.path.exists(data_file):
            logger.error(f"File non trovato: {data_file}")
            return
        
        # Carica dataset
        df = load_dataset(data_file)
        logger.info(f"Dataset caricato con successo. Dimensioni: {df.shape}")
        
        # Genera solo un grafico di test
        logger.info("Generazione grafico di test...")
        
        # Semplice grafico di test per verificare se funziona
        plt.figure(figsize=(8, 6))
        plt.plot([1, 2, 3, 4, 5], [10, 5, 8, 12, 3])
        plt.title('Grafico di Test')
        plt.xlabel('X')
        plt.ylabel('Y')
        test_output = os.path.join(output_dir, 'test_graph.png')
        logger.info(f"Salvando il grafico di test in: {test_output}")
        plt.savefig(test_output)
        plt.close()
        
        logger.info("Grafico di test salvato con successo")
        
        # Se funziona, genera gli altri grafici
        logger.info("Generazione degli altri grafici...")
        try:
            plot_event_distribution(df, output_dir)
            plot_veracity_distribution(df, output_dir)
            plot_reactions_distribution(df, output_dir)
            plot_veracity_by_event(df, output_dir)
            plot_rumour_vs_nonrumour(df, output_dir)
            logger.info("Tutte le visualizzazioni sono state generate con successo!")
        except Exception as e:
            logger.error(f"Errore durante la generazione dei grafici: {e}")
    
    except Exception as e:
        logger.error(f"Errore generale: {e}")

if __name__ == "__main__":
    main()
