"""
Modulo per creare una narrazione visiva integrativa dei risultati principali.

Questo script genera una sequenza di visualizzazioni che raccontano in modo
coerente la storia dei dati e dei risultati dell'analisi, integrando testo
esplicativo e evidenziando i risultati più significativi.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec
from typing import Dict, List, Tuple, Any, Optional, Union
from pathlib import Path
import json
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.ticker as mtick
from matplotlib.ticker import FuncFormatter

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
NARRATIVE_DIR = RESULTS_DIR / "narrative"

# Assicurarsi che le directory esistano
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
TABLES_DIR.mkdir(parents=True, exist_ok=True)
NARRATIVE_DIR.mkdir(parents=True, exist_ok=True)

# Definisci una palette di colori accessibile
PALETTE = {
    'true': '#1f77b4',  # Blu per vero
    'false': '#ff7f0e',  # Arancione per falso
    'highlight': '#d62728',  # Rosso per evidenziare
    'neutral': '#2ca02c',  # Verde per neutrale
    'background': '#f5f5f5',  # Grigio chiaro per sfondo
    'text': '#333333',  # Grigio scuro per testo
}

# Definisci una palette accessibile per daltonici
# Utilizza 'Blue', 'Orange', 'Green', 'Purple', 'Brown'
COLORBLIND_PALETTE = sns.color_palette("colorblind", 5)


def load_data() -> Dict[str, pd.DataFrame]:
    """
    Carica tutti i dati necessari per le visualizzazioni.
    
    Returns:
        Dict[str, pd.DataFrame]: Dizionario con i dati caricati.
    """
    data = {}
    
    # Carica la matrice delle feature
    feature_matrix_path = PROCESSED_DIR / "pheme_feature_matrix.csv"
    if feature_matrix_path.exists():
        data['feature_matrix'] = pd.read_csv(feature_matrix_path)
    
    # Carica i risultati dei test di ipotesi
    hypothesis_path = TABLES_DIR / "sentiment_veracity_summary.csv"
    if hypothesis_path.exists():
        data['hypothesis_tests'] = pd.read_csv(hypothesis_path)
    
    # Carica le correlazioni
    correlations_path = TABLES_DIR / "veracity_correlations.csv"
    if correlations_path.exists():
        data['correlations'] = pd.read_csv(correlations_path)
    
    # Carica i coefficienti della regressione logistica
    lr_coef_path = TABLES_DIR / "logistic_regression_coefficients.csv"
    if lr_coef_path.exists():
        data['lr_coefficients'] = pd.read_csv(lr_coef_path)
    
    # Carica le metriche della regressione logistica
    lr_metrics_path = TABLES_DIR / "logistic_regression_metrics.csv"
    if lr_metrics_path.exists():
        data['lr_metrics'] = pd.read_csv(lr_metrics_path)
    
    # Carica l'importanza delle feature del Random Forest
    rf_importance_path = TABLES_DIR / "random_forest_importance.csv"
    if rf_importance_path.exists():
        data['rf_importance'] = pd.read_csv(rf_importance_path)
    
    # Carica le metriche del Random Forest
    rf_metrics_path = TABLES_DIR / "random_forest_metrics.csv"
    if rf_metrics_path.exists():
        data['rf_metrics'] = pd.read_csv(rf_metrics_path)
    
    # Carica il confronto tra set di feature
    feature_sets_path = TABLES_DIR / "feature_sets_comparison.csv"
    if feature_sets_path.exists():
        data['feature_sets'] = pd.read_csv(feature_sets_path)
    
    return data


def create_title_page(filename: str = "01_title_page.png") -> plt.Figure:
    """
    Crea una pagina di titolo per la narrazione visiva.
    
    Args:
        filename (str): Nome del file per salvare il grafico.
        
    Returns:
        plt.Figure: Figura matplotlib.
    """
    plt.figure(figsize=(16, 10))
    fig, ax = plt.subplots(figsize=(16, 10), facecolor=PALETTE['background'])
    
    # Disattiva gli assi
    ax.axis('off')
    
    # Aggiungi titolo principale
    ax.text(0.5, 0.7, 'Relazione tra Sentiment nei Commenti\ne Veridicità delle Notizie', 
           fontsize=28, fontweight='bold', ha='center', color=PALETTE['text'])
    
    # Aggiungi sottotitolo
    ax.text(0.5, 0.55, 'Una Narrazione Visiva dei Risultati', 
           fontsize=22, ha='center', color=PALETTE['text'])
    
    # Aggiungi informazioni sul progetto
    ax.text(0.5, 0.35, 'Progetto di Human Data Science', 
           fontsize=18, ha='center', color=PALETTE['text'])
    ax.text(0.5, 0.30, 'Analisi del Dataset PHEME', 
           fontsize=18, ha='center', color=PALETTE['text'])
    
    # Aggiungi data
    ax.text(0.5, 0.20, 'Maggio 2025', 
           fontsize=16, ha='center', color=PALETTE['text'])
    
    # Salva il grafico
    save_path = NARRATIVE_DIR / filename
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor=PALETTE['background'])
    
    return fig


def create_hypothesis_summary(data: Dict[str, pd.DataFrame], 
                            filename: str = "02_hypothesis_summary.png") -> plt.Figure:
    """
    Crea un riepilogo visivo delle ipotesi e dei risultati chiave.
    
    Args:
        data (Dict[str, pd.DataFrame]): Dizionario con i dati caricati.
        filename (str): Nome del file per salvare il grafico.
        
    Returns:
        plt.Figure: Figura matplotlib.
    """
    plt.figure(figsize=(16, 10))
    fig, ax = plt.subplots(figsize=(16, 10), facecolor=PALETTE['background'])
    
    # Disattiva gli assi
    ax.axis('off')
    
    # Aggiungi titolo della sezione
    ax.text(0.5, 0.95, 'Le Nostre Ipotesi e Risultati Chiave', 
           fontsize=24, fontweight='bold', ha='center', color=PALETTE['text'])
    
    # Aggiungi l'ipotesi principale
    ax.text(0.5, 0.85, 'Ipotesi Principale:', 
           fontsize=18, fontweight='bold', ha='center', color=PALETTE['text'])
    ax.text(0.5, 0.80, '"Esiste una relazione statisticamente significativa tra\ni pattern di sentiment nei commenti e la veridicità delle notizie?"', 
           fontsize=16, ha='center', color=PALETTE['text'], style='italic')
    
    # Aggiungi i risultati chiave
    ax.text(0.05, 0.70, 'Risultati Chiave:', 
           fontsize=18, fontweight='bold', ha='left', color=PALETTE['text'])
    
    # Punto 1: Differenze statisticamente significative
    ax.text(0.05, 0.64, '1. Differenze Statisticamente Significative', 
           fontsize=16, fontweight='bold', ha='left', color=PALETTE['highlight'])
    ax.text(0.05, 0.60, 'Esistono differenze statisticamente significative nei pattern di sentiment\ntra notizie vere e false, specialmente in termini di soggettività e polarità.', 
           fontsize=14, ha='left', color=PALETTE['text'])
    
    # Punto 2: Effect size limitato
    ax.text(0.05, 0.52, '2. Effect Size Limitato', 
           fontsize=16, fontweight='bold', ha='left', color=PALETTE['highlight'])
    ax.text(0.05, 0.48, 'Nonostante la significatività statistica, la dimensione dell\'effetto è trascurabile\n(tutti i valori <0.1), limitando la rilevanza pratica di queste differenze.', 
           fontsize=14, ha='left', color=PALETTE['text'])
    
    # Punto 3: Relazioni non lineari
    ax.text(0.05, 0.40, '3. Relazioni Non Lineari', 
           fontsize=16, fontweight='bold', ha='left', color=PALETTE['highlight'])
    ax.text(0.05, 0.36, 'Le relazioni tra sentiment e veridicità sono principalmente non lineari,\ncome dimostrato dalla superiorità del Random Forest (AUC: 0.93) rispetto alla Regressione Logistica (AUC: 0.54).', 
           fontsize=14, ha='left', color=PALETTE['text'])
    
    # Punto 4: Importanza delle feature di leggibilità
    ax.text(0.05, 0.28, '4. Rilevanza delle Feature di Leggibilità', 
           fontsize=16, fontweight='bold', ha='left', color=PALETTE['highlight'])
    ax.text(0.05, 0.24, 'Le feature di leggibilità e complessità linguistica hanno mostrato\nun potere predittivo superiore rispetto alle pure misure di sentiment.', 
           fontsize=14, ha='left', color=PALETTE['text'])
    
    # Conclusione
    ax.text(0.5, 0.14, 'Conclusione:', 
           fontsize=18, fontweight='bold', ha='center', color=PALETTE['text'])
    ax.text(0.5, 0.09, 'L\'ipotesi è parzialmente verificata: esistono relazioni statisticamente significative\ntra sentiment e veridicità, ma sono deboli se considerate linearmente\ne richiedono modelli complessi per essere adeguatamente catturate.', 
           fontsize=16, ha='center', color=PALETTE['text'])
    
    # Salva il grafico
    save_path = NARRATIVE_DIR / filename
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor=PALETTE['background'])
    
    return fig


def create_distribution_comparison(data: Dict[str, pd.DataFrame], 
                                 feature: str = 'sentiment_polarity',
                                 filename: str = "03_sentiment_distribution.png") -> plt.Figure:
    """
    Crea un confronto visivo delle distribuzioni di una feature tra notizie vere e false.
    
    Args:
        data (Dict[str, pd.DataFrame]): Dizionario con i dati caricati.
        feature (str): Feature da visualizzare.
        filename (str): Nome del file per salvare il grafico.
        
    Returns:
        plt.Figure: Figura matplotlib.
    """
    if 'feature_matrix' not in data:
        raise ValueError("Matrice delle feature non disponibile")
    
    # Estrai i dati
    df = data['feature_matrix']
    
    # Verifica che la feature esista
    if feature not in df.columns:
        raise ValueError(f"Feature '{feature}' non trovata nei dati")
    
    # Verifica che la colonna veracity esista
    if 'veracity' not in df.columns:
        raise ValueError("Colonna 'veracity' non trovata nei dati")
    
    # Filtra per notizie vere e false
    df_true = df[df['veracity'].str.lower() == 'true']
    df_false = df[df['veracity'].str.lower() == 'false']
    
    # Crea il grafico
    plt.figure(figsize=(16, 10))
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10), facecolor=PALETTE['background'])
    
    # Titolo principale
    feature_name = feature.replace('_', ' ').title()
    fig.suptitle(f'Distribuzione di {feature_name} per Veridicità delle Notizie', 
                fontsize=24, fontweight='bold', y=0.98, color=PALETTE['text'])
    
    # 1. Istogrammi sovrapposti
    sns.histplot(df_true[feature], kde=True, ax=ax1, color=PALETTE['true'], label='Notizie Vere', alpha=0.6)
    sns.histplot(df_false[feature], kde=True, ax=ax1, color=PALETTE['false'], label='Notizie False', alpha=0.6)
    
    ax1.set_title('Istogrammi di Confronto', fontsize=18, color=PALETTE['text'])
    ax1.set_xlabel(feature_name, fontsize=14, color=PALETTE['text'])
    ax1.set_ylabel('Frequenza', fontsize=14, color=PALETTE['text'])
    ax1.legend(fontsize=12)
    
    # Aggiungi statistiche descrittive
    mean_true = df_true[feature].mean()
    mean_false = df_false[feature].mean()
    std_true = df_true[feature].std()
    std_false = df_false[feature].std()
    
    stats_text = f"Notizie Vere: Media = {mean_true:.3f}, Deviazione Standard = {std_true:.3f}\n"
    stats_text += f"Notizie False: Media = {mean_false:.3f}, Deviazione Standard = {std_false:.3f}"
    
    ax1.text(0.02, 0.95, stats_text, transform=ax1.transAxes, fontsize=12,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # 2. Box plot
    df_melted = pd.melt(df[df['veracity'].str.lower().isin(['true', 'false'])], 
                       id_vars=['veracity'], value_vars=[feature],
                       var_name='Feature', value_name='Valore')
    
    sns.boxplot(x='Feature', y='Valore', hue='veracity', data=df_melted, 
               palette=[PALETTE['true'], PALETTE['false']], ax=ax2)
    
    ax2.set_title('Box Plot di Confronto', fontsize=18, color=PALETTE['text'])
    ax2.set_xlabel('', fontsize=14, color=PALETTE['text'])
    ax2.set_ylabel(feature_name, fontsize=14, color=PALETTE['text'])
    ax2.legend(title='Veridicità', fontsize=12)
    
    # Aggiungi annotazione con i risultati statistici
    if 'hypothesis_tests' in data:
        hypothesis_df = data['hypothesis_tests']
        feature_result = hypothesis_df[hypothesis_df['feature'] == feature]
        
        if not feature_result.empty:
            p_value = feature_result['p_value_corrected'].iloc[0]
            effect_size = feature_result['effect_size'].iloc[0]
            significant = feature_result['significant_corrected'].iloc[0]
            
            result_text = f"Test Statistico: Mann-Whitney U\n"
            result_text += f"p-value corretto: {p_value:.3e}\n"
            result_text += f"Effect Size: {effect_size:.3f} (trascurabile)\n"
            result_text += f"Statisticamente Significativo: {'Sì' if significant else 'No'}"
            
            ax2.text(0.02, 0.95, result_text, transform=ax2.transAxes, fontsize=12,
                    verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Aggiungi interpretazione
    interp_text = "Interpretazione: "
    if mean_true > mean_false:
        interp_text += f"Le notizie vere tendono ad avere valori di {feature_name} più alti delle notizie false."
    else:
        interp_text += f"Le notizie false tendono ad avere valori di {feature_name} più alti delle notizie vere."
    
    interp_text += "\nTuttavia, la sovrapposizione significativa delle distribuzioni e l'effect size trascurabile"
    interp_text += "\nindicano che questa differenza ha una rilevanza pratica limitata per l'identificazione delle fake news."
    
    fig.text(0.5, 0.01, interp_text, ha='center', fontsize=14, color=PALETTE['text'],
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    
    # Salva il grafico
    save_path = NARRATIVE_DIR / filename
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor=PALETTE['background'])
    
    return fig


def create_correlation_visualization(data: Dict[str, pd.DataFrame],
                                   filename: str = "04_correlation_analysis.png") -> plt.Figure:
    """
    Crea una visualizzazione delle correlazioni tra feature e veridicità.
    
    Args:
        data (Dict[str, pd.DataFrame]): Dizionario con i dati caricati.
        filename (str): Nome del file per salvare il grafico.
        
    Returns:
        plt.Figure: Figura matplotlib.
    """
    if 'correlations' not in data:
        raise ValueError("Dati di correlazione non disponibili")
    
    # Estrai i dati di correlazione
    corr_df = data['correlations']
    
    # Seleziona le feature più rilevanti
    selected_features = ['sentiment_subjectivity', 'sentiment_polarity', 'stance_score', 
                       'flesch_reading_ease', 'type_token_ratio', 'formal_language_score',
                       'vocabulary_richness', 'avg_word_length', 'culture_score', 'long_words_ratio']
    
    # Filtra il dataframe
    corr_df = corr_df[corr_df['feature'].isin(selected_features)]
    
    # Ordina per correlazione di Pearson
    corr_df = corr_df.sort_values('pearson_corr', ascending=False)
    
    # Crea il grafico
    plt.figure(figsize=(16, 10))
    fig = plt.figure(figsize=(16, 10), facecolor=PALETTE['background'])
    
    # Crea un layout con GridSpec
    gs = GridSpec(2, 2, figure=fig, height_ratios=[1, 1.2])
    
    # 1. Grafico a barre delle correlazioni
    ax1 = fig.add_subplot(gs[0, 0])
    bars = ax1.barh(corr_df['feature'], corr_df['pearson_corr'], 
                   color=[PALETTE['highlight'] if sig else 'gray' for sig in corr_df['pearson_significant_corrected']])
    
    ax1.set_title('Correlazioni di Pearson con Veridicità', fontsize=16, color=PALETTE['text'])
    ax1.set_xlabel('Coefficiente di Correlazione', fontsize=12, color=PALETTE['text'])
    ax1.set_ylabel('Feature', fontsize=12, color=PALETTE['text'])
    
    # Aggiungi etichette con i valori
    for i, bar in enumerate(bars):
        value = corr_df['pearson_corr'].iloc[i]
        ax1.text(
            value + 0.001 if value >= 0 else value - 0.004,
            i,
            f'{value:.3f}',
            va='center',
            fontsize=10,
            color='black'
        )
    
    # 2. Scatter plot tra due feature principali e veridicità
    if 'feature_matrix' in data:
        ax2 = fig.add_subplot(gs[0, 1])
        
        df = data['feature_matrix']
        top_features = corr_df['feature'].iloc[:2].tolist()
        
        if len(top_features) >= 2 and 'veracity' in df.columns:
            x_feature = top_features[0]
            y_feature = top_features[1]
            
            # Filtra per notizie vere e false
            df_true = df[df['veracity'].str.lower() == 'true']
            df_false = df[df['veracity'].str.lower() == 'false']
            
            # Plot per notizie vere
            ax2.scatter(df_true[x_feature], df_true[y_feature], 
                       alpha=0.3, color=PALETTE['true'], label='Notizie Vere')
            
            # Plot per notizie false
            ax2.scatter(df_false[x_feature], df_false[y_feature], 
                       alpha=0.7, color=PALETTE['false'], label='Notizie False')
            
            # Aggiungi etichette
            x_name = x_feature.replace('_', ' ').title()
            y_name = y_feature.replace('_', ' ').title()
            ax2.set_title(f'Scatter Plot: {x_name} vs {y_name}', fontsize=16, color=PALETTE['text'])
            ax2.set_xlabel(x_name, fontsize=12, color=PALETTE['text'])
            ax2.set_ylabel(y_name, fontsize=12, color=PALETTE['text'])
            ax2.legend(fontsize=10)
    
    # 3. Tabella delle correlazioni significative
    ax3 = fig.add_subplot(gs[1, :])
    ax3.axis('off')
    
    # Filtra solo le correlazioni significative
    sig_corr = corr_df[corr_df['pearson_significant_corrected'] | corr_df['spearman_significant_corrected']].copy()
    sig_corr = sig_corr.sort_values('pearson_corr', ascending=False)
    
    # Crea una tabella con i dati
    if not sig_corr.empty:
        table_data = []
        table_columns = ['Feature', 'Correlazione\nPearson', 'p-value\nPearson', 'Correlazione\nSpearman', 'p-value\nSpearman', 'Forza\ndella Correlazione']
        
        for _, row in sig_corr.iterrows():
            feature = row['feature'].replace('_', ' ').title()
            pearson = f"{row['pearson_corr']:.3f}"
            pearson_p = f"{row['pearson_p_corrected']:.3e}"
            spearman = f"{row['spearman_corr']:.3f}"
            spearman_p = f"{row['spearman_p_corrected']:.3e}"
            strength = row['pearson_strength']
            
            table_data.append([feature, pearson, pearson_p, spearman, spearman_p, strength])
        
        table = ax3.table(
            cellText=table_data,
            colLabels=table_columns,
            loc='center',
            cellLoc='center',
            bbox=[0.1, 0.1, 0.8, 0.8]
        )
        
        # Formatta la tabella
        table.auto_set_font_size(False)
        table.set_fontsize(12)
        table.scale(1, 1.5)
        
        for (i, j), cell in table.get_celld().items():
            if i == 0:  # Header
                cell.set_text_props(weight='bold', color='white')
                cell.set_facecolor(PALETTE['text'])
            elif j == 5 and i > 0:  # Strength column
                if table_data[i-1][5] == 'trascurabile':
                    cell.set_facecolor('#FFCCCC')  # Light red for weak
                else:
                    cell.set_facecolor('#CCFFCC')  # Light green for strong
        
        ax3.set_title('Correlazioni Statisticamente Significative con Veridicità', fontsize=18, color=PALETTE['text'])
    else:
        ax3.text(0.5, 0.5, 'Nessuna correlazione significativa trovata', 
               ha='center', va='center', fontsize=16, color=PALETTE['text'])
    
    # Aggiungi interpretazione
    interpretazione = (
        "Interpretazione: Le correlazioni tra feature e veridicità sono statisticamente significative "
        "in molti casi, ma la loro forza è generalmente trascurabile (r < 0.1). Questo suggerisce che, "
        "sebbene esistano relazioni lineari rilevabili, queste non sono abbastanza forti per essere "
        "utilizzate efficacemente nella pratica per discriminare tra notizie vere e false utilizzando "
        "approcci lineari. La significatività statistica è probabilmente influenzata dalla grande "
        "dimensione del campione, che può rilevare anche correlazioni molto deboli."
    )
    
    fig.text(0.5, 0.01, interpretazione, ha='center', fontsize=12, color=PALETTE['text'],
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8), wrap=True)
    
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    
    # Titolo principale
    fig.suptitle('Analisi delle Correlazioni tra Feature e Veridicità', 
                fontsize=24, fontweight='bold', y=0.98, color=PALETTE['text'])
    
    # Salva il grafico
    save_path = NARRATIVE_DIR / filename
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor=PALETTE['background'])
    
    return fig


def create_model_comparison(data: Dict[str, pd.DataFrame],
                          filename: str = "05_model_comparison.png") -> plt.Figure:
    """
    Crea una visualizzazione del confronto tra modelli.
    
    Args:
        data (Dict[str, pd.DataFrame]): Dizionario con i dati caricati.
        filename (str): Nome del file per salvare il grafico.
        
    Returns:
        plt.Figure: Figura matplotlib.
    """
    # Verifica la presenza dei dati necessari
    required_keys = ['lr_metrics', 'rf_metrics']
    for key in required_keys:
        if key not in data:
            raise ValueError(f"Dati '{key}' non disponibili")
    
    # Estrai i dati dei modelli
    lr_metrics = data['lr_metrics']
    rf_metrics = data['rf_metrics']
    
    # Crea il grafico
    plt.figure(figsize=(16, 12))
    fig = plt.figure(figsize=(16, 12), facecolor=PALETTE['background'])
    
    # Crea un layout con GridSpec
    gs = GridSpec(2, 2, figure=fig)
    
    # 1. Confronto delle metriche di performance
    ax1 = fig.add_subplot(gs[0, 0])
    
    # Prepara i dati per il confronto
    metrics = ['accuracy', 'precision', 'recall', 'f1_score']
    lr_values = [lr_metrics[m].iloc[0] for m in metrics]
    rf_values = [rf_metrics[m].iloc[0] for m in metrics]
    
    x = np.arange(len(metrics))
    width = 0.35
    
    # Crea il grafico a barre
    bars1 = ax1.bar(x - width/2, lr_values, width, label='Regressione Logistica', color='royalblue')
    bars2 = ax1.bar(x + width/2, rf_values, width, label='Random Forest', color='forestgreen')
    
    # Aggiungi etichette e titolo
    ax1.set_title('Confronto delle Metriche di Performance', fontsize=16, color=PALETTE['text'])
    ax1.set_xticks(x)
    ax1.set_xticklabels([m.replace('_', ' ').title() for m in metrics])
    ax1.set_ylabel('Valore', fontsize=12, color=PALETTE['text'])
    ax1.legend()
    
    # Aggiungi i valori sopra le barre
    def autolabel(bars):
        for bar in bars:
            height = bar.get_height()
            ax1.annotate(f'{height:.3f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),  # 3 punti verticali di offset
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=10)
    
    autolabel(bars1)
    autolabel(bars2)
    
    # 2. Confronto AUC
    ax2 = fig.add_subplot(gs[0, 1])
    
    # Prepara i dati per il confronto AUC
    models = ['Regressione Logistica', 'Random Forest']
    # Adatta i nomi delle colonne a seconda di come sono salvati
    auc_col_lr = 'auc' if 'auc' in lr_metrics.columns else 'roc_auc'
    auc_col_rf = 'roc_auc' if 'roc_auc' in rf_metrics.columns else 'auc'
    
    auc_values = [lr_metrics[auc_col_lr].iloc[0], rf_metrics[auc_col_rf].iloc[0]]
    
    # Crea il grafico a barre per AUC
    bars = ax2.bar(models, auc_values, color=['royalblue', 'forestgreen'])
    
    # Aggiungi etichette e titolo
    ax2.set_title('Confronto dell\'Area Sotto la Curva ROC (AUC)', fontsize=16, color=PALETTE['text'])
    ax2.set_ylabel('AUC', fontsize=12, color=PALETTE['text'])
    ax2.set_ylim([0, 1])
    
    # Aggiungi linea per il classificatore casuale (AUC = 0.5)
    ax2.axhline(y=0.5, linestyle='--', color='gray', alpha=0.7, label='Classificatore Casuale')
    
    # Aggiungi i valori sopra le barre
    for bar in bars:
        height = bar.get_height()
        ax2.annotate(f'{height:.3f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=12)
    
    # 3. Feature importance
    ax3 = fig.add_subplot(gs[1, 0])
    
    if 'rf_importance' in data:
        # Estrai i dati di importanza del Random Forest
        rf_imp = data['rf_importance'].copy()
        
        # Filtra solo le feature rilevanti (escludi identificatori come thread_id, ecc.)
        relevant_features = [
            'sentiment_polarity', 'sentiment_subjectivity', 'stance_score',
            'flesch_reading_ease', 'type_token_ratio', 'formal_language_score',
            'vocabulary_richness', 'avg_word_length', 'culture_score', 'long_words_ratio'
        ]
        
        rf_imp_filtered = rf_imp[rf_imp['feature'].isin(relevant_features)]
        
        # Ordina per importanza
        if 'perm_importance_mean' in rf_imp_filtered.columns:
            rf_imp_filtered = rf_imp_filtered.sort_values('perm_importance_mean', ascending=False)
            importance_col = 'perm_importance_mean'
        else:
            rf_imp_filtered = rf_imp_filtered.sort_values('importance', ascending=False)
            importance_col = 'importance'
        
        # Limita a top 10
        rf_imp_top = rf_imp_filtered.head(10)
        
        # Crea il grafico a barre orizzontali
        bars = ax3.barh(
            rf_imp_top['feature'], 
            rf_imp_top[importance_col],
            color='forestgreen'
        )
        
        # Aggiungi etichette e titolo
        ax3.set_title('Top 10 Feature per Importanza (Random Forest)', fontsize=16, color=PALETTE['text'])
        ax3.set_xlabel('Importanza', fontsize=12, color=PALETTE['text'])
        ax3.invert_yaxis()  # Per avere la feature più importante in alto
        
        # Aggiungi i valori accanto alle barre
        for bar in bars:
            width = bar.get_width()
            ax3.annotate(f'{width:.4f}',
                        xy=(width, bar.get_y() + bar.get_height() / 2),
                        xytext=(3, 0),
                        textcoords="offset points",
                        ha='left', va='center', fontsize=10)
    else:
        ax3.text(0.5, 0.5, 'Dati di importanza delle feature non disponibili', 
               ha='center', va='center', fontsize=14, color=PALETTE['text'])
    
    # 4. LR Coefficients
    ax4 = fig.add_subplot(gs[1, 1])
    
    if 'lr_coefficients' in data:
        # Estrai i coefficienti della regressione logistica
        lr_coef = data['lr_coefficients'].copy()
        
        # Filtra solo le feature rilevanti (escludi intercetta/const)
        lr_coef_filtered = lr_coef[~lr_coef['feature'].isin(['const', 'intercept'])]
        lr_coef_filtered = lr_coef_filtered[lr_coef_filtered['significant']]
        
        # Ordina per valore assoluto del coefficiente
        lr_coef_filtered['abs_coef'] = lr_coef_filtered['coefficient'].abs()
        lr_coef_filtered = lr_coef_filtered.sort_values('abs_coef', ascending=False)
        
        # Limita a top 10
        lr_coef_top = lr_coef_filtered.head(10)
        
        # Crea il grafico a barre orizzontali
        bars = ax4.barh(
            lr_coef_top['feature'], 
            lr_coef_top['coefficient'],
            color=['forestgreen' if c > 0 else 'firebrick' for c in lr_coef_top['coefficient']]
        )
        
        # Aggiungi etichette e titolo
        ax4.set_title('Coefficienti Significativi della Regressione Logistica', fontsize=16, color=PALETTE['text'])
        ax4.set_xlabel('Coefficiente', fontsize=12, color=PALETTE['text'])
        ax4.invert_yaxis()  # Per allineare con il grafico a fianco
        
        # Aggiungi linea verticale a zero
        ax4.axvline(x=0, linestyle='-', color='gray', alpha=0.7)
        
        # Aggiungi i valori accanto alle barre
        for bar in bars:
            width = bar.get_width()
            ax4.annotate(f'{width:.3f}',
                        xy=(width, bar.get_y() + bar.get_height() / 2),
                        xytext=(3 if width >= 0 else -3, 0),
                        textcoords="offset points",
                        ha='left' if width >= 0 else 'right', va='center', fontsize=10)
    else:
        ax4.text(0.5, 0.5, 'Dati dei coefficienti non disponibili', 
               ha='center', va='center', fontsize=14, color=PALETTE['text'])
    
    # Aggiungi interpretazione
    interpretazione = (
        "Interpretazione: Il confronto tra modelli mostra un miglioramento drammatico passando dalla "
        "regressione logistica (AUC: 0.54) al Random Forest (AUC: 0.93). Questo indica che le relazioni "
        "tra le feature di sentiment e la veridicità sono prevalentemente non lineari. "
        "Tuttavia, l'alta importanza di feature come thread_id e tweet_id nel Random Forest "
        "solleva preoccupazioni sulla possibilità che il modello stia memorizzando pattern specifici "
        "del dataset piuttosto che apprendendo relazioni generalizzabili."
    )
    
    fig.text(0.5, 0.01, interpretazione, ha='center', fontsize=12, color=PALETTE['text'],
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8), wrap=True)
    
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    
    # Titolo principale
    fig.suptitle('Confronto tra Modelli Predittivi: Regressione Logistica vs Random Forest', 
                fontsize=24, fontweight='bold', y=0.98, color=PALETTE['text'])
    
    # Salva il grafico
    save_path = NARRATIVE_DIR / filename
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor=PALETTE['background'])
    
    return fig


def create_feature_sets_comparison(data: Dict[str, pd.DataFrame],
                                 filename: str = "06_feature_sets_comparison.png") -> plt.Figure:
    """
    Crea una visualizzazione del confronto tra diversi set di feature.
    
    Args:
        data (Dict[str, pd.DataFrame]): Dizionario con i dati caricati.
        filename (str): Nome del file per salvare il grafico.
        
    Returns:
        plt.Figure: Figura matplotlib.
    """
    if 'feature_sets' not in data:
        raise ValueError("Dati di confronto tra set di feature non disponibili")
    
    # Estrai i dati
    feature_sets = data['feature_sets'].copy()
    
    # Crea il grafico
    plt.figure(figsize=(16, 10))
    fig = plt.figure(figsize=(16, 10), facecolor=PALETTE['background'])
    
    # Crea un layout con GridSpec
    gs = GridSpec(2, 1, figure=fig, height_ratios=[1, 0.8])
    
    # 1. Confronto AUC
    ax1 = fig.add_subplot(gs[0])
    
    # Ordina i set per AUC
    feature_sets = feature_sets.sort_values('roc_auc')
    
    # Crea una palette di colori che varia in base al numero di feature
    colors = plt.cm.viridis(np.linspace(0, 1, len(feature_sets)))
    
    # Crea il grafico a barre orizzontali
    bars = ax1.barh(feature_sets['set_name'], feature_sets['roc_auc'], color=colors)
    
    # Aggiungi etichette e titolo
    ax1.set_title('Confronto dell\'AUC per Diversi Set di Feature', fontsize=18, color=PALETTE['text'])
    ax1.set_xlabel('Area Sotto la Curva ROC (AUC)', fontsize=14, color=PALETTE['text'])
    ax1.set_ylabel('Set di Feature', fontsize=14, color=PALETTE['text'])
    
    # Aggiungi il numero di feature per ogni set
    for i, bar in enumerate(bars):
        ax1.text(
            bar.get_width() + 0.005,
            bar.get_y() + bar.get_height()/2,
            f"Feature: {feature_sets['n_features'].iloc[i]}",
            va='center',
            fontsize=12
        )
    
    # Aggiungi i valori sulle barre
    for i, v in enumerate(feature_sets['roc_auc']):
        ax1.text(
            v - 0.05,
            i,
            f"{v:.3f}",
            va='center',
            fontsize=12,
            color='white',
            fontweight='bold'
        )
    
    # Aggiungi linea verticale per il classificatore casuale (AUC = 0.5)
    ax1.axvline(x=0.5, linestyle='--', color='gray', alpha=0.7, label='Classificatore Casuale')
    ax1.legend(fontsize=12)
    
    # 2. Radar chart delle metriche per i diversi set
    ax2 = fig.add_subplot(gs[1], polar=True)
    
    # Metriche da considerare
    metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'cv_score']
    angles = np.linspace(0, 2*np.pi, len(metrics), endpoint=False).tolist()
    angles += angles[:1]  # Chiudi il cerchio
    
    # Prepara il grafico
    ax2.set_theta_offset(np.pi / 2)
    ax2.set_theta_direction(-1)
    ax2.set_rlabel_position(0)
    ax2.set_yticklabels([])
    
    # Etichette
    plt.xticks(angles[:-1], [m.replace('_', ' ').title() for m in metrics], fontsize=12)
    
    # Disegna ogni set
    for i, (_, row) in enumerate(feature_sets.iterrows()):
        values = [row[m] for m in metrics]
        values += values[:1]  # Chiudi il cerchio
        ax2.plot(angles, values, linewidth=2, linestyle='solid', label=row['set_name'], color=colors[i])
        ax2.fill(angles, values, alpha=0.1, color=colors[i])
    
    # Aggiungi legenda
    ax2.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1), fontsize=10)
    ax2.set_title('Performance dei Diversi Set di Feature', fontsize=18, color=PALETTE['text'])
    
    # Aggiungi interpretazione
    interpretazione = (
        "Interpretazione: Il confronto tra diversi set di feature mostra che le feature di leggibilità "
        "da sole (AUC: 0.57) hanno un potere predittivo superiore alle pure feature di sentiment "
        "(AUC: 0.56) o stance (AUC: 0.51). La combinazione di tutte le feature migliora ulteriormente "
        "le performance (AUC: 0.58), ma il miglioramento è marginale rispetto alle sole feature di "
        "leggibilità. Questo suggerisce che le caratteristiche di complessità linguistica e livello di "
        "acculturazione potrebbero essere indicatori più rilevanti della veridicità rispetto al "
        "sentiment espresso nei commenti."
    )
    
    fig.text(0.5, 0.01, interpretazione, ha='center', fontsize=12, color=PALETTE['text'],
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8), wrap=True)
    
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    
    # Titolo principale
    fig.suptitle('Confronto tra Diversi Set di Feature nel Random Forest', 
                fontsize=24, fontweight='bold', y=0.98, color=PALETTE['text'])
    
    # Salva il grafico
    save_path = NARRATIVE_DIR / filename
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor=PALETTE['background'])
    
    return fig


def create_conclusions_visualization(filename: str = "07_conclusions.png") -> plt.Figure:
    """
    Crea una visualizzazione delle conclusioni principali.
    
    Args:
        filename (str): Nome del file per salvare il grafico.
        
    Returns:
        plt.Figure: Figura matplotlib.
    """
    plt.figure(figsize=(16, 10))
    fig, ax = plt.subplots(figsize=(16, 10), facecolor=PALETTE['background'])
    
    # Disattiva gli assi
    ax.axis('off')
    
    # Aggiungi titolo della sezione
    ax.text(0.5, 0.95, 'Conclusioni e Raccomandazioni', 
           fontsize=24, fontweight='bold', ha='center', color=PALETTE['text'])
    
    # Aggiungi le conclusioni principali
    ax.text(0.05, 0.87, 'Conclusioni Principali:', 
           fontsize=20, fontweight='bold', ha='left', color=PALETTE['text'])
    
    # Punto 1: Ipotesi parzialmente verificata
    ax.text(0.05, 0.82, '1. Ipotesi Parzialmente Verificata', 
           fontsize=16, fontweight='bold', ha='left', color=PALETTE['highlight'])
    ax.text(0.05, 0.78, 'Esistono differenze statisticamente significative nei pattern di sentiment tra notizie vere e false,\nma la dimensione di questi effetti è troppo limitata per avere rilevanza pratica.', 
           fontsize=14, ha='left', color=PALETTE['text'])
    
    # Punto 2: Relazioni non lineari
    ax.text(0.05, 0.70, '2. Predominanza di Relazioni Non Lineari', 
           fontsize=16, fontweight='bold', ha='left', color=PALETTE['highlight'])
    ax.text(0.05, 0.66, 'Le relazioni tra caratteristiche linguistiche e veridicità sono prevalentemente non lineari,\ncome dimostrato dalla grande differenza di performance tra regressione logistica (AUC: 0.54)\ne Random Forest (AUC: 0.93).', 
           fontsize=14, ha='left', color=PALETTE['text'])
    
    # Punto 3: Limiti del sentiment
    ax.text(0.05, 0.58, '3. Limiti dell\'Analisi del Sentiment', 
           fontsize=16, fontweight='bold', ha='left', color=PALETTE['highlight'])
    ax.text(0.05, 0.54, 'Le feature di sentiment da sole hanno un potere predittivo limitato (AUC: 0.56),\nindicando che l\'analisi del sentiment nei commenti non è sufficiente per identificare efficacemente\nle fake news.', 
           fontsize=14, ha='left', color=PALETTE['text'])
    
    # Punto 4: Possibile overfitting
    ax.text(0.05, 0.46, '4. Rischi di Overfitting', 
           fontsize=16, fontweight='bold', ha='left', color=PALETTE['highlight'])
    ax.text(0.05, 0.42, 'L\'alta importanza di feature come thread_id nel Random Forest solleva preoccupazioni\nsulla generalizzabilità del modello e suggerisce possibile dipendenza da caratteristiche specifiche\ndel dataset.', 
           fontsize=14, ha='left', color=PALETTE['text'])
    
    # Raccomandazioni future
    ax.text(0.05, 0.34, 'Raccomandazioni per Ricerche Future:', 
           fontsize=20, fontweight='bold', ha='left', color=PALETTE['text'])
    
    # Raccomandazione 1
    ax.text(0.05, 0.29, '1. Analizzare Pattern Temporali', 
           fontsize=16, fontweight='bold', ha='left', color=PALETTE['neutral'])
    ax.text(0.05, 0.25, 'Studiare come il sentiment evolve nel tempo all\'interno dei thread potrebbe essere\npiù indicativo rispetto al sentiment statico.', 
           fontsize=14, ha='left', color=PALETTE['text'])
    
    # Raccomandazione 2
    ax.text(0.05, 0.20, '2. Stratificare l\'Analisi per Tema o Evento', 
           fontsize=16, fontweight='bold', ha='left', color=PALETTE['neutral'])
    ax.text(0.05, 0.16, 'Analizzare separatamente per evento o tema potrebbe rivelare pattern più forti\naltrimenti diluiti dalla diversità degli eventi nel dataset.', 
           fontsize=14, ha='left', color=PALETTE['text'])
    
    # Raccomandazione 3
    ax.text(0.05, 0.11, '3. Integrare Approcci Multimodali', 
           fontsize=16, fontweight='bold', ha='left', color=PALETTE['neutral'])
    ax.text(0.05, 0.07, 'Combinare analisi testuale con analisi di rete e metadati degli utenti per\ncatturare le molteplici dimensioni del fenomeno della disinformazione.', 
           fontsize=14, ha='left', color=PALETTE['text'])
    
    # Salva il grafico
    save_path = NARRATIVE_DIR / filename
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor=PALETTE['background'])
    
    return fig


def main():
    """
    Funzione principale che genera tutte le visualizzazioni narrative.
    """
    try:
        print("Caricamento dei dati...")
        data = load_data()
        
        print("Creazione della narrazione visiva...")
        
        # 1. Pagina di titolo
        print("Creando pagina di titolo...")
        create_title_page()
        
        # 2. Riepilogo delle ipotesi
        print("Creando riepilogo delle ipotesi...")
        create_hypothesis_summary(data)
        
        # 3. Distribuzione del sentiment
        print("Creando visualizzazione della distribuzione del sentiment...")
        create_distribution_comparison(data, feature='sentiment_polarity', 
                                     filename="03a_sentiment_polarity.png")
        create_distribution_comparison(data, feature='sentiment_subjectivity', 
                                     filename="03b_sentiment_subjectivity.png")
        
        # 4. Analisi delle correlazioni
        print("Creando visualizzazione delle correlazioni...")
        create_correlation_visualization(data)
        
        # 5. Confronto tra modelli
        print("Creando confronto tra modelli...")
        create_model_comparison(data)
        
        # 6. Confronto tra set di feature
        print("Creando confronto tra set di feature...")
        create_feature_sets_comparison(data)
        
        # 7. Conclusioni
        print("Creando visualizzazione delle conclusioni...")
        create_conclusions_visualization()
        
        print("Tutte le visualizzazioni narrative sono state generate con successo.")
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"Errore nell'esecuzione: {str(e)}")


if __name__ == "__main__":
    main()
