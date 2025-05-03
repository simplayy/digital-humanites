"""
Modulo per l'implementazione dei test di ipotesi statistici.

Questo modulo contiene funzioni per eseguire vari test statistici
per verificare le ipotesi sulla relazione tra sentiment dei commenti
e veridicità delle notizie nel dataset PHEME.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Any, Optional, Union
from pathlib import Path
import scipy.stats as stats
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.multicomp import pairwise_tukeyhsd
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


def test_normality(df: pd.DataFrame, 
                  column: str, 
                  test_type: str = 'shapiro',
                  alpha: float = 0.05) -> Dict[str, Any]:
    """
    Esegue un test di normalità su una colonna del DataFrame.
    
    Args:
        df (pd.DataFrame): DataFrame contenente i dati.
        column (str): Nome della colonna da testare.
        test_type (str): Tipo di test ('shapiro' per Shapiro-Wilk o 'ks' per Kolmogorov-Smirnov).
        alpha (float): Livello di significatività.
    
    Returns:
        Dict[str, Any]: Dizionario con i risultati del test.
    """
    if column not in df.columns:
        raise ValueError(f"La colonna {column} non esiste nel DataFrame")
    
    # Rimuove i valori nulli per il test
    data = df[column].dropna()
    
    result = {}
    
    if test_type == 'shapiro':
        # Test di Shapiro-Wilk (più potente per campioni piccoli e medi)
        stat, p_value = stats.shapiro(data)
        test_name = "Shapiro-Wilk"
    elif test_type == 'ks':
        # Test di Kolmogorov-Smirnov (confronta con una distribuzione normale)
        stat, p_value = stats.kstest(data, 'norm', args=(data.mean(), data.std()))
        test_name = "Kolmogorov-Smirnov"
    else:
        raise ValueError(f"Tipo di test non supportato: {test_type}. Usare 'shapiro' o 'ks'.")
    
    # Determina se i dati sono normalmente distribuiti
    is_normal = p_value > alpha
    
    result = {
        'test_name': test_name,
        'statistic': stat,
        'p_value': p_value,
        'is_normal': is_normal,
        'alpha': alpha,
        'interpretation': f"La distribuzione {'è' if is_normal else 'non è'} normale (p = {p_value:.4f})"
    }
    
    return result


def test_mean_difference(df: pd.DataFrame, 
                        column: str,
                        group_column: str = 'veracity',
                        parametric: bool = True,
                        alpha: float = 0.05) -> Dict[str, Any]:
    """
    Esegue un test per confrontare le medie di una feature tra gruppi.
    
    Args:
        df (pd.DataFrame): DataFrame contenente i dati.
        column (str): Nome della colonna con i valori da testare.
        group_column (str): Nome della colonna con i gruppi da confrontare.
        parametric (bool): Se True, usa test parametrici (t-test, ANOVA), altrimenti non parametrici.
        alpha (float): Livello di significatività.
    
    Returns:
        Dict[str, Any]: Dizionario con i risultati del test.
    """
    if column not in df.columns or group_column not in df.columns:
        raise ValueError(f"Colonna {column} o {group_column} non presente nel DataFrame")
    
    # Rimuovi righe con valori nulli nelle colonne di interesse
    data = df[[column, group_column]].dropna()
    
    # Ottieni i gruppi unici
    groups = data[group_column].unique()
    n_groups = len(groups)
    
    result = {
        'feature': column,
        'grouping_variable': group_column,
        'n_groups': n_groups,
        'groups': list(groups),
        'alpha': alpha,
        'sample_sizes': {},
        'group_means': {},
        'group_std': {}
    }
    
    # Calcola statistiche per gruppo
    for group in groups:
        group_data = data[data[group_column] == group][column]
        result['sample_sizes'][group] = len(group_data)
        result['group_means'][group] = group_data.mean()
        result['group_std'][group] = group_data.std()
    
    # Test diversi a seconda del numero di gruppi
    if n_groups == 2:
        # Due gruppi: t-test o Mann-Whitney U
        group1_data = data[data[group_column] == groups[0]][column]
        group2_data = data[data[group_column] == groups[1]][column]
        
        if parametric:
            # Test parametrico: t-test per campioni indipendenti
            stat, p_value = stats.ttest_ind(group1_data, group2_data, equal_var=False)
            test_name = "t-test di Welch per campioni indipendenti"
        else:
            # Test non parametrico: Mann-Whitney U
            stat, p_value = stats.mannwhitneyu(group1_data, group2_data, alternative='two-sided')
            test_name = "Mann-Whitney U"
        
        result.update({
            'test_name': test_name,
            'statistic': stat,
            'p_value': p_value,
            'significant': p_value < alpha,
            'interpretation': f"La differenza tra i gruppi {'è' if p_value < alpha else 'non è'} statisticamente significativa (p = {p_value:.4f})"
        })
        
    elif n_groups > 2:
        # Più di due gruppi: ANOVA o Kruskal-Wallis
        groups_data = [data[data[group_column] == group][column] for group in groups]
        
        if parametric:
            # Test parametrico: ANOVA
            stat, p_value = stats.f_oneway(*groups_data)
            test_name = "ANOVA a una via"
            
            # Se ANOVA è significativo, esegui test post-hoc di Tukey
            if p_value < alpha:
                # Prepara dati per test di Tukey
                posthoc_data = pd.DataFrame({
                    'value': data[column],
                    'group': data[group_column]
                })
                
                # Esegui test di Tukey
                tukey_result = pairwise_tukeyhsd(posthoc_data['value'], posthoc_data['group'], alpha=alpha)
                
                # Estrai risultati del test post-hoc
                posthoc_pairs = []
                for i in range(len(tukey_result.groupsunique)):
                    for j in range(i+1, len(tukey_result.groupsunique)):
                        idx = int(i * len(tukey_result.groupsunique) - i*(i+1)/2 + j - i - 1)
                        pair = {
                            'group1': tukey_result.groupsunique[i],
                            'group2': tukey_result.groupsunique[j],
                            'mean_diff': tukey_result.meandiffs[idx],
                            'p_value': tukey_result.pvalues[idx],
                            'significant': tukey_result.reject[idx]
                        }
                        posthoc_pairs.append(pair)
                
                result['posthoc_test'] = 'Tukey HSD'
                result['posthoc_results'] = posthoc_pairs
                
        else:
            # Test non parametrico: Kruskal-Wallis
            stat, p_value = stats.kruskal(*groups_data)
            test_name = "Kruskal-Wallis H"
            
            # Se Kruskal-Wallis è significativo, esegui test post-hoc di Dunn
            if p_value < alpha:
                # Implementazione semplificata del test di Dunn
                # Per una vera implementazione del test di Dunn, potrebbe essere necessario utilizzare
                # librerie esterne come scikit-posthocs
                result['posthoc_test'] = 'Dunn (non implementato)'
                result['posthoc_note'] = 'Per un test post-hoc più preciso, utilizzare scikit-posthocs.'
        
        result.update({
            'test_name': test_name,
            'statistic': stat,
            'p_value': p_value,
            'significant': p_value < alpha,
            'interpretation': f"La differenza tra i gruppi {'è' if p_value < alpha else 'non è'} statisticamente significativa (p = {p_value:.4f})"
        })
    
    else:
        # Meno di due gruppi: non è possibile eseguire un test di confronto
        result.update({
            'test_name': 'Nessun test eseguito',
            'error': 'Numero insufficiente di gruppi per eseguire un test di confronto',
        })
    
    return result


def test_correlation(df: pd.DataFrame, 
                    var1: str, 
                    var2: str, 
                    method: str = 'pearson',
                    alpha: float = 0.05) -> Dict[str, Any]:
    """
    Calcola la correlazione tra due variabili.
    
    Args:
        df (pd.DataFrame): DataFrame contenente i dati.
        var1 (str): Nome della prima variabile.
        var2 (str): Nome della seconda variabile.
        method (str): Metodo di correlazione ('pearson', 'spearman', o 'kendall').
        alpha (float): Livello di significatività.
    
    Returns:
        Dict[str, Any]: Dizionario con i risultati del test di correlazione.
    """
    if var1 not in df.columns or var2 not in df.columns:
        raise ValueError(f"Variabile {var1} o {var2} non presente nel DataFrame")
    
    # Rimuovi righe con valori nulli nelle colonne di interesse
    data = df[[var1, var2]].dropna()
    
    # Seleziona il metodo di correlazione
    if method == 'pearson':
        correlation, p_value = stats.pearsonr(data[var1], data[var2])
        test_name = "Coefficiente di correlazione di Pearson"
    elif method == 'spearman':
        correlation, p_value = stats.spearmanr(data[var1], data[var2])
        test_name = "Coefficiente di correlazione di Spearman"
    elif method == 'kendall':
        correlation, p_value = stats.kendalltau(data[var1], data[var2])
        test_name = "Coefficiente di correlazione tau di Kendall"
    else:
        raise ValueError(f"Metodo di correlazione non supportato: {method}. Usare 'pearson', 'spearman' o 'kendall'.")
    
    # Prepara i risultati
    result = {
        'var1': var1,
        'var2': var2,
        'test_name': test_name,
        'correlation': correlation,
        'p_value': p_value,
        'significant': p_value < alpha,
        'alpha': alpha,
        'interpretation': f"La correlazione {'è' if p_value < alpha else 'non è'} statisticamente significativa (p = {p_value:.4f})"
    }
    
    # Aggiungi l'interpretazione della forza della correlazione
    if abs(correlation) < 0.1:
        strength = "trascurabile"
    elif abs(correlation) < 0.3:
        strength = "debole"
    elif abs(correlation) < 0.5:
        strength = "moderata"
    elif abs(correlation) < 0.7:
        strength = "forte"
    else:
        strength = "molto forte"
    
    result['strength'] = strength
    result['direction'] = "positiva" if correlation > 0 else "negativa"
    result['interpretation'] += f"\nLa correlazione è {strength} e {result['direction']} (r = {correlation:.4f})"
    
    return result


def test_sentiment_by_veracity(df: pd.DataFrame, 
                              sentiment_cols: List[str] = None,
                              alpha: float = 0.05) -> pd.DataFrame:
    """
    Esegue test statistici per confrontare il sentiment tra notizie vere e false.
    
    Args:
        df (pd.DataFrame): DataFrame contenente i dati.
        sentiment_cols (List[str], optional): Lista di colonne di sentiment da analizzare.
        alpha (float): Livello di significatività.
    
    Returns:
        pd.DataFrame: DataFrame con i risultati dei test.
    """
    if 'veracity' not in df.columns:
        raise ValueError("La colonna 'veracity' non è presente nel DataFrame")
    
    # Se non sono specificate le colonne di sentiment, usa quelle predefinite
    if sentiment_cols is None:
        sentiment_cols = [
            'sentiment_polarity', 'sentiment_subjectivity',
            'stance_score', 'flesch_reading_ease',
            'type_token_ratio', 'formal_language_score',
            'vocabulary_richness', 'avg_word_length',
            'long_words_ratio', 'culture_score'
        ]
    
    # Filtra solo le colonne che esistono nel DataFrame
    sentiment_cols = [col for col in sentiment_cols if col in df.columns]
    
    # Verifica se il dataset contiene veracity come 'true' e 'false'
    veracity_values = df['veracity'].unique()
    has_true_false = 'true' in veracity_values and 'false' in veracity_values
    
    if not has_true_false:
        # Se non ci sono valori 'true' e 'false', verifica se ci sono valori numerici o booleani
        if 'true' in df['veracity'].str.lower().unique() and 'false' in df['veracity'].str.lower().unique():
            # Converti a lowercase se necessario
            df['veracity'] = df['veracity'].str.lower()
        else:
            raise ValueError("La colonna 'veracity' non contiene i valori 'true' e 'false' necessari per il confronto")
    
    # Filtra solo le righe con veracity 'true' o 'false'
    df_filtered = df[df['veracity'].isin(['true', 'false'])]
    
    # Prepara i risultati
    results = []
    
    # Esegui test per ogni colonna di sentiment
    for col in sentiment_cols:
        # Prima verifica la normalità della distribuzione
        normality_true = test_normality(df_filtered[df_filtered['veracity'] == 'true'], col)
        normality_false = test_normality(df_filtered[df_filtered['veracity'] == 'false'], col)
        
        # Decidi se usare test parametrico o non parametrico
        use_parametric = normality_true['is_normal'] and normality_false['is_normal']
        
        # Esegui il test di confronto delle medie
        test_result = test_mean_difference(
            df_filtered, 
            col, 
            'veracity', 
            parametric=use_parametric,
            alpha=alpha
        )
        
        # Aggiungi informazioni sulla normalità e sul tipo di test
        test_result['normal_distribution'] = {
            'true': normality_true['is_normal'],
            'false': normality_false['is_normal']
        }
        test_result['test_type'] = 'parametric' if use_parametric else 'non-parametric'
        
        # Calcola l'effect size (d di Cohen)
        group_true = df_filtered[df_filtered['veracity'] == 'true'][col].dropna()
        group_false = df_filtered[df_filtered['veracity'] == 'false'][col].dropna()
        
        mean_true = group_true.mean()
        mean_false = group_false.mean()
        std_pooled = np.sqrt(((len(group_true) - 1) * group_true.std()**2 + 
                             (len(group_false) - 1) * group_false.std()**2) / 
                            (len(group_true) + len(group_false) - 2))
        
        if std_pooled > 0:
            effect_size = abs(mean_true - mean_false) / std_pooled
            
            # Interpretazione dell'effect size (d di Cohen)
            if effect_size < 0.2:
                effect_interpretation = "trascurabile"
            elif effect_size < 0.5:
                effect_interpretation = "piccolo"
            elif effect_size < 0.8:
                effect_interpretation = "medio"
            else:
                effect_interpretation = "grande"
        else:
            effect_size = None
            effect_interpretation = "non calcolabile (divisione per zero)"
        
        test_result['effect_size'] = effect_size
        test_result['effect_interpretation'] = effect_interpretation
        
        results.append(test_result)
    
    # Correggi per test multipli
    if len(results) > 1:
        p_values = [result['p_value'] for result in results]
        significant_corrected, p_corrected, _, _ = multipletests(p_values, alpha=alpha, method='fdr_bh')
        
        for i, result in enumerate(results):
            result['p_value_corrected'] = p_corrected[i]
            result['significant_corrected'] = significant_corrected[i]
            result['correction_method'] = 'Benjamini-Hochberg FDR'
            result['interpretation_corrected'] = f"Dopo la correzione per test multipli, la differenza {'è' if significant_corrected[i] else 'non è'} statisticamente significativa (p corretto = {p_corrected[i]:.4f})"
    
    # Converti a DataFrame
    results_df = pd.DataFrame(results)
    
    return results_df


def create_summary_table(test_results: pd.DataFrame) -> pd.DataFrame:
    """
    Crea una tabella di riepilogo dei risultati dei test statistici.
    
    Args:
        test_results (pd.DataFrame): DataFrame con i risultati dei test.
    
    Returns:
        pd.DataFrame: Tabella di riepilogo.
    """
    # Estrai le colonne principali
    summary = test_results[[
        'feature', 'test_name', 'p_value', 'significant',
        'effect_size', 'effect_interpretation'
    ]].copy()
    
    # Aggiungi colonne corrette per test multipli se disponibili
    if 'p_value_corrected' in test_results.columns:
        summary['p_value_corrected'] = test_results['p_value_corrected']
        summary['significant_corrected'] = test_results['significant_corrected']
    
    # Ordina per significatività e dimensione dell'effetto
    summary = summary.sort_values(['significant', 'effect_size'], ascending=[False, False])
    
    return summary


def save_test_results(results: pd.DataFrame, filename: str = "test_results.csv") -> str:
    """
    Salva i risultati dei test statistici in un file CSV.
    
    Args:
        results (pd.DataFrame): DataFrame con i risultati dei test.
        filename (str): Nome del file di output.
        
    Returns:
        str: Percorso del file salvato.
    """
    output_path = TABLES_DIR / filename
    results.to_csv(output_path, index=False)
    return str(output_path)


def plot_test_results(results: pd.DataFrame, 
                    filename: Optional[str] = None) -> plt.Figure:
    """
    Visualizza i risultati dei test statistici.
    
    Args:
        results (pd.DataFrame): DataFrame con i risultati dei test.
        filename (str, optional): Nome del file per salvare il grafico.
        
    Returns:
        plt.Figure: Figura matplotlib.
    """
    if len(results) == 0:
        raise ValueError("Nessun risultato da visualizzare")
    
    # Estrai le feature e i p-value
    features = results['feature']
    p_values = results['p_value']
    
    # Usa p-value corretto se disponibile
    if 'p_value_corrected' in results.columns:
        p_values_corrected = results['p_value_corrected']
    else:
        p_values_corrected = None
    
    # Estrai effect size se disponibile
    if 'effect_size' in results.columns:
        effect_sizes = results['effect_size']
    else:
        effect_sizes = None
    
    # Crea il grafico
    fig, ax1 = plt.subplots(figsize=(14, 8))
    
    # Barre per i p-value
    x = np.arange(len(features))
    bars1 = ax1.bar(x, p_values, width=0.4, label='p-value', color='skyblue', alpha=0.7)
    
    if p_values_corrected is not None:
        bars2 = ax1.bar(x + 0.4, p_values_corrected, width=0.4, label='p-value corretto', color='lightgreen', alpha=0.7)
    
    # Aggiungi linea per la soglia di significatività
    ax1.axhline(y=0.05, color='red', linestyle='--', alpha=0.7, label='Soglia α = 0.05')
    
    # Configura l'asse y primario per i p-value
    ax1.set_xlabel('Feature')
    ax1.set_ylabel('p-value')
    ax1.set_title('Risultati dei test statistici per feature')
    ax1.set_xticks(x + 0.2)
    ax1.set_xticklabels(features, rotation=45, ha='right')
    ax1.legend(loc='upper left')
    
    # Se disponibile, aggiungi effect size su un asse secondario
    if effect_sizes is not None:
        ax2 = ax1.twinx()
        line = ax2.plot(x + 0.2, effect_sizes, 'o-', color='darkred', label="Effect size (d di Cohen)")
        ax2.set_ylabel("Effect size")
        
        # Aggiungi legenda per l'effect size
        lines, labels = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax2.legend(lines + lines2, labels + labels2, loc='upper right')
    
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
        
        # 1. Esegui test per confrontare il sentiment tra notizie vere e false
        print("\nEsecuzione test per confrontare il sentiment tra notizie vere e false...")
        sentiment_features = [
            'sentiment_polarity', 'sentiment_subjectivity',
            'stance_score', 'flesch_reading_ease',
            'type_token_ratio', 'formal_language_score',
            'vocabulary_richness', 'avg_word_length',
            'long_words_ratio', 'culture_score'
        ]
        test_results = test_sentiment_by_veracity(df, sentiment_features)
        
        # 2. Crea tabella di riepilogo
        print("\nCreazione tabella di riepilogo...")
        summary_table = create_summary_table(test_results)
        
        # 3. Salva i risultati
        print("\nSalvataggio dei risultati...")
        save_path = save_test_results(test_results, "sentiment_veracity_tests.csv")
        print(f"Risultati completi salvati in: {save_path}")
        
        summary_path = save_test_results(summary_table, "sentiment_veracity_summary.csv")
        print(f"Tabella di riepilogo salvata in: {summary_path}")
        
        # 4. Visualizza i risultati
        print("\nCreazione visualizzazione dei risultati...")
        plot_test_results(test_results, "sentiment_veracity_tests.png")
        
        # 5. Esegui test di correlazione tra le feature principali
        print("\nEsecuzione test di correlazione tra le feature principali...")
        correlation_results = []
        
        for i, feature1 in enumerate(sentiment_features[:-1]):
            for feature2 in sentiment_features[i+1:]:
                if feature1 in df.columns and feature2 in df.columns:
                    print(f"  Calcolo correlazione tra {feature1} e {feature2}...")
                    corr_result = test_correlation(df, feature1, feature2)
                    correlation_results.append(corr_result)
        
        # Converti i risultati di correlazione in DataFrame
        correlation_df = pd.DataFrame(correlation_results)
        
        # Salva i risultati di correlazione
        corr_path = save_test_results(correlation_df, "feature_correlations.csv")
        print(f"Risultati di correlazione salvati in: {corr_path}")
        
        print("\nCompletati tutti i test statistici!")
        
    except Exception as e:
        print(f"Errore durante l'esecuzione: {str(e)}")
