#!/usr/bin/env python3
"""
Script pour analyser les citations dans l'article et extraire les passages pertinents
des fichiers texte des références.
"""

import os
import re
from pathlib import Path

# Chemins
ARTICLE_TEX = "../paper/article.tex"
CITATION_VERIFICATION = "citation_verification.md"
TXT_DIR = "articles_txt"
OUTPUT_DIR = "citation_analysis"

# Créer le dossier de sortie
os.makedirs(OUTPUT_DIR, exist_ok=True)


def extract_citations_from_tex() -> list[str]:
    """Extraire toutes les citations de l'article LaTeX."""
    with open(ARTICLE_TEX, encoding="utf-8") as f:
        content = f.read()

    # Trouver toutes les commandes \cite{}
    cite_pattern = r"\\cite\{([^}]+)\}"
    matches = re.findall(cite_pattern, content)

    # Extraire les noms de citations individuels
    citations = set()
    for match in matches:
        # Les citations peuvent être séparées par des virgules
        for citation in match.split(","):
            citation = citation.strip()
            if citation:
                citations.add(citation)

    return sorted(citations)


def find_citation_context(citation: str, tex_content: str) -> list[tuple[int, str]]:
    """Trouver le contexte d'une citation dans l'article."""
    lines = tex_content.split("\n")
    contexts = []

    for i, line in enumerate(lines):
        if f"\\cite{{{citation}}}" in line or "\\cite{" in line and citation in line:
            # Prendre quelques lignes avant et après
            start = max(0, i - 3)
            end = min(len(lines), i + 4)
            context = "\n".join(lines[start:end])
            contexts.append((i + 1, context))  # +1 car les lignes commencent à 1

    return contexts


def extract_passages_for_citation(citation: str) -> tuple[list[str] | None, list[str]]:
    """Extraire des passages pertinents pour une citation."""
    txt_file = Path(TXT_DIR) / f"{citation}.txt"

    if not txt_file.exists():
        # Chercher des fichiers similaires
        similar = list(Path(TXT_DIR).glob(f"*{citation}*.txt"))
        if similar:
            txt_file = similar[0]
        else:
            return None, []

    with open(txt_file, encoding="utf-8", errors="ignore") as f:
        content = f.read()

    # Extraire les premières lignes
    lines = content.split("\n")
    head_lines = lines[:100]  # 100 premières lignes

    # Chercher des sections importantes
    current_section = []
    in_abstract = False
    in_introduction = False
    in_conclusion = False

    for line in lines:
        line_lower = line.lower()
        if "abstract" in line_lower:
            in_abstract = True
            current_section = [line]
        elif "introduction" in line_lower:
            in_introduction = True
            current_section = [line]
        elif "conclusion" in line_lower:
            in_conclusion = True
            current_section = [line]
        elif in_abstract and len(current_section) < 20:
            current_section.append(line)
        elif in_introduction and len(current_section) < 30:
            current_section.append(line)
        elif in_conclusion and len(current_section) < 20:
            current_section.append(line)

    sections_text = []
    if current_section:
        sections_text.append("\n".join(current_section[:50]))

    return head_lines, sections_text


def main() -> None:
    print("=== Analyse des citations ===")

    # Lire l'article
    with open(ARTICLE_TEX, encoding="utf-8") as f:
        tex_content = f.read()

    # Extraire les citations
    citations = extract_citations_from_tex()
    print(f"Nombre de citations trouvées: {len(citations)}")
    print("Citations:", ", ".join(citations))
    print()

    # Créer un rapport
    report_lines = []
    report_lines.append("=== Rapport d'analyse des citations ===")
    report_lines.append(f"Date: {os.popen('date').read().strip()}")
    report_lines.append(f"Nombre de citations: {len(citations)}")
    report_lines.append("")

    # Analyser chaque citation
    for citation in citations:
        print(f"=== Analyse de: {citation} ===")
        report_lines.append(f"=== {citation} ===")

        # Trouver le contexte
        contexts = find_citation_context(citation, tex_content)
        if contexts:
            for line_num, context in contexts:
                print(f"Ligne {line_num}:")
                print(context)
                print()
                report_lines.append(f"Ligne {line_num}:")
                report_lines.append(context)
                report_lines.append("")

        # Extraire les passages
        head_lines, sections = extract_passages_for_citation(citation)

        if head_lines:
            # Sauvegarder les premières lignes
            head_file = Path(OUTPUT_DIR) / f"{citation}_head.txt"
            with open(head_file, "w", encoding="utf-8") as f:
                f.write("\n".join(head_lines))
            print(f"  Fichier créé: {head_file}")

            # Ajouter au rapport
            report_lines.append(f"Premières lignes ({len(head_lines)} lignes):")
            report_lines.append("\n".join(head_lines[:10]))  # Juste les 10 premières
            report_lines.append("...")
        else:
            print(f"  ⚠ Fichier texte non trouvé pour {citation}")
            report_lines.append("⚠ Fichier texte non trouvé")

        if sections:
            sections_file = Path(OUTPUT_DIR) / f"{citation}_sections.txt"
            with open(sections_file, "w", encoding="utf-8") as f:
                for section in sections:
                    f.write(section)
                    f.write("\n" + "=" * 80 + "\n")
            print(f"  Fichier créé: {sections_file}")

        print()
        report_lines.append("")

    # Sauvegarder le rapport
    report_file = Path(OUTPUT_DIR) / "00_analysis_report.txt"
    with open(report_file, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))

    print("=== Analyse terminée ===")
    print(f"Rapport sauvegardé dans: {report_file}")
    print(f"Fichiers d'extraction dans: {OUTPUT_DIR}/")

    # Créer un fichier de synthèse pour la vérification
    print("\n=== Synthèse pour citation_verification.md ===")
    print("Voici les citations qui ont besoin de passages:")

    missing_passages = []
    for citation in citations:
        txt_file = Path(TXT_DIR) / f"{citation}.txt"
        if not txt_file.exists():
            similar = list(Path(TXT_DIR).glob(f"*{citation}*.txt"))
            if not similar:
                missing_passages.append(citation)

    if missing_passages:
        print(f"\nCitations sans fichiers texte ({len(missing_passages)}):")
        for citation in missing_passages:
            print(f"  - {citation}")
    else:
        print("\nToutes les citations ont des fichiers texte disponibles.")

    # Suggestions pour les passages à extraire
    print("\n=== Suggestions de recherche par citation ===")
    suggestion_file = Path(OUTPUT_DIR) / "01_search_suggestions.txt"
    with open(suggestion_file, "w", encoding="utf-8") as f:
        f.write("=== Suggestions de recherche par citation ===\n\n")

        for citation in citations:
            f.write(f"=== {citation} ===\n")

            # Basé sur le contexte, suggérer des termes de recherche
            contexts = find_citation_context(citation, tex_content)
            if contexts:
                for line_num, context in contexts:
                    f.write(f"Ligne {line_num}: {context[:100]}...\n")

                    # Extraire des mots-clés du contexte
                    words = re.findall(r"\b\w+\b", context.lower())
                    common_words = {
                        "the",
                        "and",
                        "of",
                        "in",
                        "to",
                        "a",
                        "is",
                        "that",
                        "for",
                        "on",
                        "with",
                        "as",
                        "by",
                        "this",
                        "are",
                        "or",
                        "an",
                        "be",
                        "from",
                        "at",
                        "which",
                        "it",
                        "has",
                        "have",
                        "was",
                        "were",
                        "but",
                        "not",
                        "they",
                        "their",
                        "we",
                        "can",
                        "will",
                        "if",
                        "then",
                        "so",
                        "because",
                        "when",
                        "where",
                        "how",
                        "why",
                        "what",
                        "who",
                        "whom",
                        "whose",
                    }
                    keywords = [w for w in words if w not in common_words and len(w) > 3]

                    if keywords:
                        f.write("Mots-clés suggérés: " + ", ".join(set(keywords[:5])) + "\n")
                        f.write(
                            "Commande grep: grep -i '"
                            + "\\|".join(set(keywords[:3]))
                            + f"' {TXT_DIR}/{citation}.txt\n"
                        )

            f.write("\n")


if __name__ == "__main__":
    main()
