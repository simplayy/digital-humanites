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
REPORT_DIR="/Users/simone/Documents/magistrale/digital humanites/progetto_HDS/reports/report_scientifico"
OUTPUT_DIR="/Users/simone/Documents/magistrale/digital humanites/progetto_HDS/reports"

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
pandoc "$TEMP_FRONTESPIZIO" "$TEMP_STYLE" ${FILES[@]/#/"$REPORT_DIR/"} \
  -o "$OUTPUT_DIR/report_completo.pdf" \
  --from markdown \
  --pdf-engine=xelatex \
  --variable documentclass=report \
  --variable papersize=a4 \
  --variable fontsize=12pt \
  --variable links-as-notes=true \
  --variable toc-depth=3 \
  --number-sections \
  --highlight-style=tango

# Verifica se la generazione è andata a buon fine
if [ $? -eq 0 ]; then
    echo "Il report completo è stato generato con successo: $OUTPUT_DIR/report_completo.pdf"
else
    echo "Si è verificato un errore durante la generazione del report."
    exit 1
fi

# Pulizia file temporanei
rm -f "$TEMP_FRONTESPIZIO" "$TEMP_STYLE"

echo "Processo completato."
