#!/bin/bash

# Script per generare un report scientifico completo in PDF dai file markdown

echo "Generazione del report scientifico completo in PDF..."

# Verifica se pandoc è installato
if ! command -v pandoc &> /dev/null; then
    echo "Errore: pandoc non è installato. È necessario installarlo per generare il PDF."
    echo "Puoi installarlo con: brew install pandoc"
    exit 1
fi

# Verifica se LaTeX è installato (necessario per la conversione PDF)
if ! command -v pdflatex &> /dev/null; then
    echo "Avviso: pdflatex non è installato. È necessario per una corretta formattazione PDF."
    echo "Puoi installare BasicTeX con: brew install --cask basictex"
    echo "Procedendo comunque, ma la formattazione potrebbe non essere ottimale."
fi

# Directory del report scientifico
REPORT_DIR="./report_scientifico"
OUTPUT_DIR="."
TEMP_DIR=$(mktemp -d)
echo "Directory temporanea creata: $TEMP_DIR"

# Copiamo tutti i file markdown nella directory temporanea
echo "Preparazione dei file markdown..."
cp -r "$REPORT_DIR"/*.md "$TEMP_DIR/"

# Copia le immagini necessarie in una struttura facilmente accessibile
mkdir -p "$TEMP_DIR/images/figures"
mkdir -p "$TEMP_DIR/images/narrative"

echo "Copia delle immagini nella directory temporanea..."
cp -r ../results/figures/* "$TEMP_DIR/images/figures/" 2>/dev/null || true
cp -r ../results/narrative/* "$TEMP_DIR/images/narrative/" 2>/dev/null || true

# Modifica i percorsi delle immagini nei file markdown
echo "Aggiornamento dei percorsi delle immagini nei file markdown..."
for file in "$TEMP_DIR"/*.md; do
    # Sostituisci ../../results/figures/ con ./images/figures/
    sed -i '' 's|../../results/figures/|./images/figures/|g' "$file"
    # Sostituisci ../../results/narrative/ con ./images/narrative/
    sed -i '' 's|../../results/narrative/|./images/narrative/|g' "$file"
done

# Verifica che la directory esista
if [ ! -d "$REPORT_DIR" ]; then
    echo "Errore: La directory del report scientifico non esiste."
    exit 1
fi

# Crea un file temporaneo per il frontespizio
TEMP_FRONTESPIZIO=$(mktemp)
cat > "$TEMP_FRONTESPIZIO" << EOF
---
title: "Relazione tra Sentiment nei Commenti e Veridicità delle Notizie: Analisi del Dataset PHEME"
author: "Simone"
date: "3 maggio 2025"
geometry: margin=2.5cm
colorlinks: true
linkcolor: blue
urlcolor: cyan
toccolor: blue
toc: true
toc-depth: 3
documentclass: report
---

EOF

# Crea un file temporaneo per lo stile
TEMP_STYLE=$(mktemp)
cat > "$TEMP_STYLE" << EOF
---
header-includes:
  - \usepackage{fancyhdr}
  - \usepackage{graphicx}
  - \usepackage{titlesec}
  - \usepackage{booktabs}
  - \usepackage{float}
  - \floatplacement{figure}{H}
  - \pagestyle{fancy}
  - \fancyhead[LE,RO]{Studio sulla Relazione tra Sentiment e Veridicità}
  - \fancyfoot[CE,CO]{Università - Digital Humanities}
  - \fancyfoot[LE,RO]{\thepage}
  - \titleformat{\chapter}{\normalfont\huge\bfseries}{\thechapter.}{20pt}{\Huge}
---

EOF

echo "Combinazione dei file markdown..."

# Ordine dei file markdown
FILES=(
    "sommario_esecutivo.md"
    "01_introduzione.md"
    "02_dataset_metodologia.md"
    "03_analisi_esplorativa.md"
    "04_analisi_statistica.md"
    "05_modelli_predittivi.md"
    "06_risultati_discussione.md"
    "07_limitazioni_validazione.md"
    "08_conclusioni.md"
    "09_bibliografia.md"
)

# Combinare i file markdown in un unico PDF
# Creiamo un array di percorsi file con le virgolette per gestire gli spazi
FILE_PATHS=()
for file in "${FILES[@]}"; do
  FILE_PATHS+=("$TEMP_DIR/$file")
done

echo "Tentativo di generazione PDF con pdflatex..."
# Prova con pdflatex
if command -v pdflatex &> /dev/null; then
    pandoc "$TEMP_FRONTESPIZIO" "$TEMP_STYLE" "${FILE_PATHS[@]}" \
      -o "$OUTPUT_DIR/report_completo.pdf" \
      --from markdown \
      --pdf-engine=pdflatex \
      --variable documentclass=report \
      --variable papersize=a4 \
      --variable fontsize=12pt \
      --variable links-as-notes=true \
      --variable toc-depth=3 \
      --number-sections \
      --highlight-style=tango
else
    echo "pdflatex non trovato, tentativo con alternative..."
    # Se pdflatex non è disponibile, prova con wkhtmltopdf
    if command -v wkhtmltopdf &> /dev/null; then
        echo "Usando wkhtmltopdf..."
        # Prima converti in HTML
        pandoc "$TEMP_FRONTESPIZIO" "$TEMP_STYLE" "${FILE_PATHS[@]}" \
          -o "$TEMP_DIR/report.html" \
          --from markdown \
          --to html \
          --standalone \
          --toc \
          --toc-depth=3 \
          --highlight-style=tango
        
        # Poi da HTML a PDF
        wkhtmltopdf --enable-local-file-access "$TEMP_DIR/report.html" "$OUTPUT_DIR/report_completo.pdf"
    else
        echo "Nessun motore PDF trovato, generazione file HTML..."
        # Se nessuna alternativa è disponibile, genera almeno un HTML
        pandoc "$TEMP_FRONTESPIZIO" "$TEMP_STYLE" "${FILE_PATHS[@]}" \
          -o "$OUTPUT_DIR/report_completo.html" \
          --from markdown \
          --to html \
          --standalone \
          --toc \
          --toc-depth=3 \
          --highlight-style=tango
        echo "Generato report in formato HTML: $OUTPUT_DIR/report_completo.html"
        echo "Per generare un PDF, installa uno di questi pacchetti: pdflatex (brew install --cask basictex) o wkhtmltopdf (brew install wkhtmltopdf)"
    fi
fi

# Verifica se la generazione è andata a buon fine
if [ -f "$OUTPUT_DIR/report_completo.pdf" ]; then
    echo "Il report completo è stato generato con successo: $OUTPUT_DIR/report_completo.pdf"
elif [ -f "$OUTPUT_DIR/report_completo.html" ]; then
    echo "Il report completo è stato generato in formato HTML: $OUTPUT_DIR/report_completo.html"
else
    echo "Si è verificato un errore durante la generazione del report."
    exit 1
fi

# Pulizia file temporanei
echo "Pulizia dei file temporanei..."
rm -f "$TEMP_FRONTESPIZIO" "$TEMP_STYLE"
rm -rf "$TEMP_DIR"

echo "Processo completato."
