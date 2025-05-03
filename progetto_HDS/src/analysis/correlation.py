"""
Modulo per l'analisi di correlazione tra feature.

Questo modulo contiene funzioni per analizzare le correlazioni tra le
varie feature estratte dal dataset PHEME, con particolare attenzione
alla relazione tra sentiment dei commenti e veridicità delle notizie.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Any, Optional, Union
from pathlib import Path
import scipy.stats as stats
from statsmodels.stats.multitest import multipletests

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


def calculate_correlation_matrix(df: pd.DataFrame, 
                               columns: Optional[List[str]] = None,
                               method: str = 'pearson') -> pd.DataFrame:
    """
    Calcola la matrice di correlazione per le variabili selezionate.
    
    Args:
        df (pd.DataFrame): DataFrame contenente i dati.
        columns (List[str], optional): Lista di colonne da includere.
            Se None, vengono utilizzate tutte le colonne numeriche.
        method (str): Metodo di correlazione ('pearson', 'kendall', o 'spearman').
    
    Returns:
        pd.DataFrame: Matrice di correlazione.
    """
    # Se non sono specificate le colonne, seleziona le colonne numeriche
    if columns is None:
        columns = df.select_dtypes(include=['number']).columns.tolist()
    else:
        # Filtra solo colonne valide
        columns = [col for col in columns if col in df.columns]
    
    if not columns:
        raise ValueError("Nessuna colonna numerica trovata o specificata")
    
    # Calcola la matrice di correlazione
    corr_matrix = df[columns].corr(method=method)
    
    return corr_matrix


def calculate_significance_matrix(df: pd.DataFrame, 
                                columns: Optional[List[str]] = None,
                                method: str = 'pearson',
                                alpha: float = 0.05) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Calcola la matrice di correlazione e la relativa matrice di significatività statistica.
    
    Args:
        df (pd.DataFrame): DataFrame contenente i dati.
        columns (List[str], optional): Lista di colonne da includere.
        method (str): Metodo di correlazione ('pearson', 'kendall', o 'spearman').
        alpha (float): Livello di significatività.
    
    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: Matrice di correlazione e matrice di p-value.
    """
    # Se non sono specificate le colonne, seleziona le colonne numeriche
    if columns is None:
        columns = df.select_dtypes(include=['number']).columns.tolist()
    else:
        # Filtra solo colonne valide
        columns = [col for col in columns if col in df.columns]
    
    if not columns:
        raise ValueError("Nessuna colonna numerica trovata o specificata")
    
    # Inizializza le matrici risultato
    n = len(columns)
    corr_matrix = pd.DataFrame(np.zeros((n, n)), index=columns, columns=columns)
    p_matrix = pd.DataFrame(np.zeros((n, n)), index=columns, columns=columns)
    
    # Calcola correlazione e p-value per ogni coppia di variabili
    for i, col1 in enumerate(columns):
        for j, col2 in enumerate(columns):
            if i == j:
                # Diagonale: correlazione perfetta, p-value = 0
                corr_matrix.iloc[i, j] = 1.0
                p_matrix.iloc[i, j] = 0.0
                continue
                
            # Rimuovi i valori NaN per le due variabili
            valid_data = df[[col1, col2]].dropna()
            
            if len(valid_data) < 2:
                # Non abbastanza dati per calcolare la correlazione
                corr_matrix.iloc[i, j] = np.nan
                p_matrix.iloc[i, j] = np.nan
                continue
            
            if method == 'pearson':
                corr, p = stats.pearsonr(valid_data[col1], valid_data[col2])
            elif method == 'spearman':
                corr, p = stats.spearmanr(valid_data[col1], valid_data[col2])
            elif method == 'kendall':
                corr, p = stats.kendalltau(valid_data[col1], valid_data[col2])
            else:
                raise ValueError(f"Metodo non supportato: {method}. Usare 'pearson', 'spearman' o 'kendall'.")
            
            corr_matrix.iloc[i, j] = corr
            p_matrix.iloc[i, j] = p
    
    return corr_matrix, p_matrix


def get_significant_correlations(corr_matrix: pd.DataFrame, 
                               p_matrix: pd.DataFrame,
                               alpha: float = 0.05,
                               correct_multiple_tests: bool = True) -> pd.DataFrame:
    """
    Estrae le correlazioni statisticamente significative dalla matrice di correlazione.
    
    Args:
        corr_matrix (pd.DataFrame): Matrice di correlazione.
        p_matrix (pd.DataFrame): Matrice di p-value.
        alpha (float): Livello di significatività.
        correct_multiple_tests (bool): Se True, applica la correzione per test multipli.
        
    Returns:
        pd.DataFrame: DataFrame con le correlazioni significative.
    """
    # Verifica che le matrici abbiano le stesse dimensioni
    if corr_matrix.shape != p_matrix.shape:
        raise ValueError("Le matrici di correlazione e p-value devono avere le stesse dimensioni")
    
    # Estrai i valori dal triangolo inferiore delle matrici
    indices = np.tril_indices_from(corr_matrix, k=-1)
    correlations = corr_matrix.values[indices]
    p_values = p_matrix.values[indices]
    pairs = [(corr_matrix.index[i], corr_matrix.columns[j]) for i, j in zip(*indices)]
    
    # Crea DataFrame per i risultati
    results = pd.DataFrame({
        'var1': [p[0] for p in pairs],
        'var2': [p[1] for p in pairs],
        'correlation': correlations,
        'p_value': p_values
    })
    
    # Applica correzione per test multipli se richiesto
    if correct_multiple_tests and len(results) > 1:
        _, p_corrected, _, _ = multipletests(results['p_value'], alpha=alpha, method='fdr_bh')
        results['p_value_corrected'] = p_corrected
        results['significant'] = p_corrected < alpha
    else:
        results['significant'] = results['p_value'] < alpha
    
    # Filtra solo le correlazioni significative
    significant_corrs = results[results['significant']].copy()
    
    # Aggiungi colonna per l'interpretazione della forza della correlazione
    def interpret_correlation(r):
        if abs(r) < 0.1:
            return "trascurabile"
        elif abs(r) < 0.3:
            return "debole"
        elif abs(r) < 0.5:
            return "moderata"
        elif abs(r) < 0.7:
            return "forte"
        else:
            return "molto forte"
    
    significant_corrs['strength'] = significant_corrs['correlation'].apply(interpret_correlation)
    significant_corrs['direction'] = significant_corrs['correlation'].apply(lambda x: "positiva" if x > 0 else "negativa")
    
    # Ordina per forza di correlazione (valore assoluto)
    significant_corrs['abs_correlation'] = significant_corrs['correlation'].abs()
    significant_corrs = significant_corrs.sort_values('abs_correlation', ascending=False)
    significant_corrs.drop('abs_correlation', axis=1, inplace=True)
    
    return significant_corrs


def plot_correlation_heatmap(corr_matrix: pd.DataFrame,
                           p_matrix: Optional[pd.DataFrame] = None,
                           alpha: float = 0.05,
                           cmap: str = 'coolwarm',
                           mask_insignificant: bool = True,
                           filename: Optional[str] = None) -> plt.Figure:
    """
    Visualizza una heatmap della matrice di correlazione, evidenziando le correlazioni significative.
    
    Args:
        corr_matrix (pd.DataFrame): Matrice di correlazione.
        p_matrix (pd.DataFrame, optional): Matrice di p-value.
        alpha (float): Livello di significatività.
        cmap (str): Mappa colori per la heatmap.
        mask_insignificant (bool): Se True e p_matrix è fornito, maschera le correlazioni non significative.
        filename (str, optional): Nome del file per salvare il grafico.
        
    Returns:
        plt.Figure: Figura matplotlib.
    """
    # Crea una copia della matrice di correlazione per la visualizzazione
    corr_display = corr_matrix.copy()
    
    # Se la matrice p è fornita e mask_insignificant è True, maschera le correlazioni non significative
    if p_matrix is not None and mask_insignificant:
        insignificant = p_matrix > alpha
        corr_display = corr_display.mask(insignificant, np.nan)
    
    # Crea la maschera per il triangolo superiore
    mask = np.triu(np.ones_like(corr_display, dtype=bool))
    
    # Crea il grafico
    plt.figure(figsize=(14, 12))
    fig, ax = plt.subplots(figsize=(14, 12))
    
    # Disegna la heatmap
    sns.heatmap(
        corr_display,
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
    
    # Aggiungi titolo
    title = "Matrice di Correlazione"
    if p_matrix is not None and mask_insignificant:
        title += f" (solo correlazioni con p < {alpha})"
    plt.title(title)
    
    # Salva il grafico se richiesto
    if filename:
        save_path = FIGURES_DIR / filename
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_correlation_network(corr_matrix: pd.DataFrame,
                           p_matrix: Optional[pd.DataFrame] = None,
                           alpha: float = 0.05,
                           min_correlation: float = 0.3,
                           filename: Optional[str] = None) -> plt.Figure:
    """
    Visualizza una rete di correlazioni, dove i nodi sono le variabili e gli archi rappresentano correlazioni significative.
    
    Args:
        corr_matrix (pd.DataFrame): Matrice di correlazione.
        p_matrix (pd.DataFrame, optional): Matrice di p-value.
        alpha (float): Livello di significatività.
        min_correlation (float): Correlazione minima (in valore assoluto) da rappresentare.
        filename (str, optional): Nome del file per salvare il grafico.
        
    Returns:
        plt.Figure: Figura matplotlib.
    """
    try:
        import networkx as nx
    except ImportError:
        raise ImportError("Questa funzione richiede la libreria networkx. Installala con 'pip install networkx'")
    
    # Crea un grafo non direzionato
    G = nx.Graph()
    
    # Aggiungi i nodi (variabili)
    for col in corr_matrix.columns:
        G.add_node(col)
    
    # Aggiungi gli archi (correlazioni)
    for i, var1 in enumerate(corr_matrix.columns):
        for j, var2 in enumerate(corr_matrix.columns):
            if i >= j:  # Evita duplicati e auto-correlazioni
                continue
            
            corr = corr_matrix.iloc[i, j]
            
            # Controlla se la correlazione è significativa
            is_significant = True
            if p_matrix is not None:
                p_value = p_matrix.iloc[i, j]
                is_significant = p_value < alpha
            
            # Aggiungi l'arco solo se la correlazione è sufficientemente forte e significativa
            if abs(corr) >= min_correlation and is_significant:
                G.add_edge(var1, var2, weight=abs(corr), sign=1 if corr > 0 else -1)
    
    # Crea il grafico
    plt.figure(figsize=(16, 12))
    fig, ax = plt.subplots(figsize=(16, 12))
    
    # Posiziona i nodi
    pos = nx.spring_layout(G, k=0.5, seed=42)
    
    # Disegna i nodi
    nx.draw_networkx_nodes(G, pos, node_size=2000, node_color='lightblue', alpha=0.8, ax=ax)
    
    # Disegna le etichette dei nodi
    nx.draw_networkx_labels(G, pos, font_size=10, ax=ax)
    
    # Disegna gli archi, distinguendo tra correlazioni positive e negative
    positive_edges = [(u, v) for u, v, d in G.edges(data=True) if d['sign'] > 0]
    negative_edges = [(u, v) for u, v, d in G.edges(data=True) if d['sign'] < 0]
    
    # Calcola lo spessore degli archi in base alla correlazione
    pos_edge_weights = [G[u][v]['weight'] * 5 for u, v in positive_edges]
    neg_edge_weights = [G[u][v]['weight'] * 5 for u, v in negative_edges]
    
    nx.draw_networkx_edges(G, pos, edgelist=positive_edges, width=pos_edge_weights, edge_color='green', alpha=0.6, ax=ax)
    nx.draw_networkx_edges(G, pos, edgelist=negative_edges, width=neg_edge_weights, edge_color='red', alpha=0.6, ax=ax)
    
    # Aggiungi legenda
    plt.plot([0], [0], color='green', linewidth=5, label='Correlazione positiva', alpha=0.6)
    plt.plot([0], [0], color='red', linewidth=5, label='Correlazione negativa', alpha=0.6)
    plt.legend()
    
    plt.title("Rete delle Correlazioni Significative")
    plt.axis('off')
    
    # Salva il grafico se richiesto
    if filename:
        save_path = FIGURES_DIR / filename
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def analyze_correlations_with_veracity(df: pd.DataFrame,
                                      numerical_features: Optional[List[str]] = None,
                                      alpha: float = 0.05) -> pd.DataFrame:
    """
    Analizza la correlazione tra feature numeriche e veridicità delle notizie.
    
    Args:
        df (pd.DataFrame): DataFrame contenente i dati.
        numerical_features (List[str], optional): Lista di feature numeriche da analizzare.
        alpha (float): Livello di significatività.
        
    Returns:
        pd.DataFrame: DataFrame con i risultati dell'analisi.
    """
    if 'veracity' not in df.columns:
        raise ValueError("La colonna 'veracity' non è presente nel DataFrame")
    
    # Se non sono specificate le feature numeriche, seleziona automaticamente
    if numerical_features is None:
        numerical_features = df.select_dtypes(include=['number']).columns.tolist()
    
    # Filtra solo le colonne che esistono nel DataFrame
    numerical_features = [col for col in numerical_features if col in df.columns]
    
    # Verifica se la colonna veracity ha valori 'true' e 'false'
    veracity_values = df['veracity'].unique()
    has_true_false = 'true' in veracity_values and 'false' in veracity_values
    
    if not has_true_false:
        # Prova con lowercase
        if 'true' in df['veracity'].str.lower().unique() and 'false' in df['veracity'].str.lower().unique():
            # Converti a lowercase
            df = df.copy()  # Crea una copia per non modificare l'originale
            df['veracity'] = df['veracity'].str.lower()
        else:
            raise ValueError("La colonna 'veracity' non contiene i valori 'true' e 'false' necessari per l'analisi")
    
    # Filtra solo le righe con veracity 'true' o 'false'
    df_filtered = df[df['veracity'].isin(['true', 'false'])]
    
    # Crea una variabile dummy per la veridicità (0 = false, 1 = true)
    df_filtered['veracity_numeric'] = df_filtered['veracity'].apply(lambda x: 1 if x == 'true' else 0)
    
    # Inizializza lista per i risultati
    results = []
    
    # Calcola la correlazione tra ciascuna feature e la veridicità
    for feature in numerical_features:
        # Rimuovi le righe con valori mancanti
        valid_data = df_filtered[['veracity_numeric', feature]].dropna()
        
        if len(valid_data) < 2:
            # Non abbastanza dati per calcolare la correlazione
            continue
        
        # Pearson (correlazione lineare)
        pearson_corr, pearson_p = stats.pearsonr(valid_data['veracity_numeric'], valid_data[feature])
        
        # Spearman (correlazione di rango)
        spearman_corr, spearman_p = stats.spearmanr(valid_data['veracity_numeric'], valid_data[feature])
        
        # Point-biserial (specifico per variabile dicotomica e continua)
        # In realtà coincide con Pearson in questo caso
        
        results.append({
            'feature': feature,
            'pearson_corr': pearson_corr,
            'pearson_p': pearson_p,
            'pearson_significant': pearson_p < alpha,
            'spearman_corr': spearman_corr,
            'spearman_p': spearman_p,
            'spearman_significant': spearman_p < alpha,
        })
    
    # Converti a DataFrame
    results_df = pd.DataFrame(results)
    
    # Applica correzione per test multipli
    if len(results_df) > 1:
        # Correzione per i p-value di Pearson
        _, p_corrected_pearson, _, _ = multipletests(results_df['pearson_p'], alpha=alpha, method='fdr_bh')
        results_df['pearson_p_corrected'] = p_corrected_pearson
        results_df['pearson_significant_corrected'] = p_corrected_pearson < alpha
        
        # Correzione per i p-value di Spearman
        _, p_corrected_spearman, _, _ = multipletests(results_df['spearman_p'], alpha=alpha, method='fdr_bh')
        results_df['spearman_p_corrected'] = p_corrected_spearman
        results_df['spearman_significant_corrected'] = p_corrected_spearman < alpha
    
    # Aggiungi colonne per l'interpretazione della forza della correlazione
    def interpret_correlation(r):
        if abs(r) < 0.1:
            return "trascurabile"
        elif abs(r) < 0.3:
            return "debole"
        elif abs(r) < 0.5:
            return "moderata"
        elif abs(r) < 0.7:
            return "forte"
        else:
            return "molto forte"
    
    results_df['pearson_strength'] = results_df['pearson_corr'].apply(interpret_correlation)
    results_df['spearman_strength'] = results_df['spearman_corr'].apply(interpret_correlation)
    
    # Ordina per forza di correlazione (valore assoluto)
    results_df['abs_pearson_corr'] = results_df['pearson_corr'].abs()
    results_df = results_df.sort_values('abs_pearson_corr', ascending=False)
    results_df.drop('abs_pearson_corr', axis=1, inplace=True)
    
    return results_df


def save_correlation_results(results: pd.DataFrame, filename: str) -> str:
    """
    Salva i risultati dell'analisi di correlazione in un file CSV.
    
    Args:
        results (pd.DataFrame): DataFrame con i risultati dell'analisi.
        filename (str): Nome del file di output.
        
    Returns:
        str: Percorso del file salvato.
    """
    output_path = TABLES_DIR / filename
    results.to_csv(output_path, index=False)
    return str(output_path)


def plot_top_correlations(corr_results: pd.DataFrame, 
                        top_n: int = 10,
                        use_corrected: bool = True,
                        filename: Optional[str] = None) -> plt.Figure:
    """
    Visualizza le correlazioni più forti con la veridicità.
    
    Args:
        corr_results (pd.DataFrame): DataFrame con i risultati dell'analisi di correlazione.
        top_n (int): Numero di correlazioni da visualizzare.
        use_corrected (bool): Se True, usa i p-value corretti per determinare la significatività.
        filename (str, optional): Nome del file per salvare il grafico.
        
    Returns:
        plt.Figure: Figura matplotlib.
    """
    if len(corr_results) == 0:
        raise ValueError("Nessun risultato da visualizzare")
    
    # Seleziona le colonne di interesse
    if use_corrected and 'pearson_p_corrected' in corr_results.columns:
        # Seleziona correlazioni significative dopo la correzione
        significant = corr_results[corr_results['pearson_significant_corrected']]
    else:
        # Seleziona correlazioni significative senza correzione
        significant = corr_results[corr_results['pearson_significant']]
    
    # Se non ci sono correlazioni significative, prendi le più forti
    if len(significant) == 0:
        significant = corr_results.copy()
    
    # Ordina per forza di correlazione (valore assoluto)
    significant['abs_pearson_corr'] = significant['pearson_corr'].abs()
    significant = significant.sort_values('abs_pearson_corr', ascending=False)
    significant.drop('abs_pearson_corr', axis=1, inplace=True)
    
    # Prendi le top N
    top = significant.head(top_n)
    
    # Se non ci sono abbastanza correlazioni, prendi tutte quelle disponibili
    if len(top) == 0:
        print("Nessuna correlazione significativa trovata")
        return None
    
    # Crea il grafico
    plt.figure(figsize=(14, 8))
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Estrai le feature e le correlazioni
    features = top['feature']
    correlations = top['pearson_corr']
    
    # Crea il grafico a barre
    bars = ax.barh(features, correlations, color=['green' if c > 0 else 'red' for c in correlations], alpha=0.7)
    
    # Aggiungi etichette e titolo
    ax.set_xlabel('Coefficiente di Correlazione con la Veridicità')
    ax.set_ylabel('Feature')
    title = f"Top {len(top)} Correlazioni con la Veridicità"
    if use_corrected and 'pearson_p_corrected' in corr_results.columns:
        title += " (p-value corretto)"
    ax.set_title(title)
    
    # Aggiungi i p-value come annotazioni
    for i, bar in enumerate(bars):
        p_value = top.iloc[i]['pearson_p_corrected'] if use_corrected and 'pearson_p_corrected' in top.columns else top.iloc[i]['pearson_p']
        ax.annotate(f'p = {p_value:.4f}',
                   xy=(bar.get_width() * 0.5 if bar.get_width() >= 0 else bar.get_width() * 1.1,
                       bar.get_y() + bar.get_height() * 0.5),
                   ha='center' if bar.get_width() >= 0 else 'left',
                   va='center',
                   color='white' if bar.get_width() >= 0 else 'black',
                   fontsize=10)
    
    # Aggiungi linee di riferimento
    ax.axvline(x=0, color='black', linestyle='-', alpha=0.3)
    ax.axvline(x=0.1, color='gray', linestyle='--', alpha=0.3, label='Correlazione debole')
    ax.axvline(x=0.3, color='gray', linestyle='--', alpha=0.3, label='Correlazione moderata')
    ax.axvline(x=0.5, color='gray', linestyle='--', alpha=0.3, label='Correlazione forte')
    ax.axvline(x=-0.1, color='gray', linestyle='--', alpha=0.3)
    ax.axvline(x=-0.3, color='gray', linestyle='--', alpha=0.3)
    ax.axvline(x=-0.5, color='gray', linestyle='--', alpha=0.3)
    
    plt.legend()
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
        
        # 1. Calcola matrice di correlazione tra le feature
        print("\nCalcolo della matrice di correlazione tra le feature...")
        corr_matrix, p_matrix = calculate_significance_matrix(df, numerical_features)
        
        # 2. Visualizza la matrice di correlazione
        print("\nCreazione heatmap delle correlazioni...")
        plot_correlation_heatmap(
            corr_matrix,
            p_matrix,
            alpha=0.05,
            mask_insignificant=True,
            filename="correlation_heatmap_significant.png"
        )
        
        # 3. Estrai e salva le correlazioni significative
        print("\nEstrazione delle correlazioni significative...")
        significant_corrs = get_significant_correlations(corr_matrix, p_matrix)
        save_path = save_correlation_results(significant_corrs, "significant_correlations.csv")
        print(f"Correlazioni significative salvate in: {save_path}")
        
        # 4. Analizza correlazioni con la veridicità
        print("\nAnalisi delle correlazioni con la veridicità...")
        veracity_corr = analyze_correlations_with_veracity(df, numerical_features)
        save_path = save_correlation_results(veracity_corr, "veracity_correlations.csv")
        print(f"Correlazioni con la veridicità salvate in: {save_path}")
        
        # 5. Visualizza le correlazioni più forti con la veridicità
        print("\nVisualizzazione delle correlazioni più forti con la veridicità...")
        plot_top_correlations(
            veracity_corr,
            top_n=10,
            use_corrected=True,
            filename="top_veracity_correlations.png"
        )
        
        # 6. Prova a generare la rete di correlazioni
        try:
            print("\nCreazione della rete di correlazioni...")
            plot_correlation_network(
                corr_matrix,
                p_matrix,
                min_correlation=0.3,
                filename="correlation_network.png"
            )
        except ImportError:
            print("Impossibile creare la rete di correlazioni: networkx non è installato.")
            print("Installa networkx con 'pip install networkx' per utilizzare questa funzione.")
        
        print("\nCompletata l'analisi di correlazione!")
        
    except Exception as e:
        print(f"Errore durante l'esecuzione: {str(e)}")
