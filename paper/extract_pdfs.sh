#!/bin/bash

# Script pour extraire le texte de tous les PDFs dans le dossier paper/articles/
# et créer des fichiers .txt correspondants

PDF_DIR="articles"
TXT_DIR="articles_txt"

# Créer le dossier de sortie s'il n'existe pas
mkdir -p "$TXT_DIR"

# Compter le nombre total de PDFs
total_pdfs=$(find "$PDF_DIR" -name "*.pdf" | wc -l)
echo "Nombre total de PDFs trouvés: $total_pdfs"

# Extraire le texte de chaque PDF
counter=0
for pdf_file in "$PDF_DIR"/*.pdf; do
    if [ -f "$pdf_file" ]; then
        counter=$((counter + 1))
        filename=$(basename "$pdf_file" .pdf)
        txt_file="$TXT_DIR/${filename}.txt"

        echo "[$counter/$total_pdfs] Extraction de: $filename.pdf"

        # Extraire le texte avec pdftotext
        if command -v pdftotext >/dev/null 2>&1; then
            pdftotext "$pdf_file" "$txt_file" 2>/dev/null

            # Vérifier si l'extraction a réussi
            if [ -s "$txt_file" ]; then
                lines=$(wc -l < "$txt_file")
                echo "  ✓ Succès: $lines lignes extraites"
            else
                echo "  ⚠ Avertissement: fichier texte vide ou extraction échouée"
                # Essayer avec strings comme alternative
                strings "$pdf_file" > "$txt_file" 2>/dev/null
                lines=$(wc -l < "$txt_file")
                echo "  ✓ Alternative (strings): $lines lignes extraites"
            fi
        else
            echo "  ❌ Erreur: pdftotext n'est pas installé"
            echo "  Installation: brew install poppler (macOS) ou apt-get install poppler-utils (Linux)"
            exit 1
        fi
    fi
done

echo ""
echo "=== Résumé ==="
echo "PDFs traités: $counter"
echo "Fichiers texte créés dans: $TXT_DIR"
echo ""
echo "=== Liste des fichiers texte créés ==="
ls -la "$TXT_DIR"/*.txt 2>/dev/null | head -20

# Optionnel: créer un fichier d'index avec les premières lignes de chaque fichier
echo ""
echo "=== Création d'un index des extractions ==="
INDEX_FILE="$TXT_DIR/00_index.txt"
echo "# Index des extractions PDF -> TXT" > "$INDEX_FILE"
echo "# Généré le: $(date)" >> "$INDEX_FILE"
echo "# Nombre de fichiers: $counter" >> "$INDEX_FILE"
echo "" >> "$INDEX_FILE"

for txt_file in "$TXT_DIR"/*.txt; do
    if [ "$(basename "$txt_file")" != "00_index.txt" ]; then
        echo "=== $(basename "$txt_file") ===" >> "$INDEX_FILE"
        head -5 "$txt_file" | sed 's/^/  /' >> "$INDEX_FILE"
        echo "" >> "$INDEX_FILE"
    fi
done

echo "Index créé: $INDEX_FILE"
