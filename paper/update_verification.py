#!/usr/bin/env python3
"""
Script pour mettre à jour citation_verification.md avec les passages extraits.
"""

import re
from pathlib import Path

# Chemins
VERIFICATION_FILE = "citation_verification.md"
TXT_DIR = "articles_txt"
ANALYSIS_DIR = "citation_analysis"


def extract_key_passages(citation: str) -> list[str]:
    """Extraire les passages clés pour une citation."""
    txt_file = Path(TXT_DIR) / f"{citation}.txt"
    if not txt_file.exists():
        return []

    with open(txt_file, encoding="utf-8", errors="ignore") as f:
        content = f.read()

    passages = []

    # Chercher l'abstract
    abstract_match = re.search(r"(?i)abstract.*?(?=\n\n|\n[A-Z])", content, re.DOTALL)
    if abstract_match:
        abstract = abstract_match.group(0)[:500]  # Limiter à 500 caractères
        passages.append(f"**Abstract excerpt:** {abstract}...")

    # Chercher des sections importantes
    sections_to_find = ["introduction", "conclusion", "discussion", "results", "method"]
    for section in sections_to_find:
        pattern = rf"(?i){section}.*?(?=\n\n|\n[A-Z]|$)"
        match = re.search(pattern, content, re.DOTALL)
        if match:
            section_text = match.group(0)[:300]  # Limiter à 300 caractères
            passages.append(f"**{section.capitalize()} excerpt:** {section_text}...")

    # Chercher des phrases clés basées sur le contexte de la citation
    # (à adapter selon la citation)
    keywords = {
        "roubeau2009": ["laryngeal mechanism", "M1", "M2", "transition", "overlap"],
        "henrich2004": ["open quotient", "OQ", "glottal", "EGG", "mechanism"],
        "boratto2025": ["XGBoost", "Differential Evolution", "97%", "accuracy", "register"],
        "bourne2012": ["amplification", "belt", "mix", "music theatre", "CCM"],
        "sundberg1974": ["singing formant", "2.8 kHz", "formant cluster", "F3", "F4", "F5"],
        "tatit2002": ["figurativização", "speech-like", "Brazilian", "popular singing"],
        "rezende2016": ["Choro", "vocal technique", "ornamental", "diction", "agility"],
        "cotton2007": ["Fach", "classification", "inconsistencies", "voice teachers"],
        "henrich2006": ["register", "history", "Garcia", "modern methods"],
        "kim2018": ["CREPE", "pitch estimation", "convolutional neural network"],
        "chen2016": ["XGBoost", "tree boosting", "scalable"],
        "lee2013": ["pseudo-label", "semi-supervised", "deep neural networks"],
        "nigam2000": ["EM", "unlabeled documents", "text classification"],
        "maryn2015": ["CPPS", "cepstral peak prominence", "voice quality"],
        "teixeira2013": ["jitter", "shimmer", "HNR", "acoustic analysis"],
        "kreiman2012": ["H1-H2", "glottal adduction", "voice quality"],
        "kreiman2014": ["voice production", "perception", "unified theory"],
        "sundberg2006": ["alpha ratio", "spectrum balance", "loudness"],
        "sundberg1987": ["singing voice", "science", "formant tuning", "acoustics"],
        "bozeman2013": ["vocal acoustics", "pedagogy", "formant tuning", "passaggio"],
        "alku2023": ["formant tracking", "deep learning", "linear prediction", "high-pitched"],
        "gowda2022": ["formant", "tracking", "deep learning"],
        "defossez2021": ["source separation", "Demucs", "hybrid"],
        "rouard2023": ["HTDemucs", "Transformer", "source separation"],
        "kim2025": ["pop music", "vocal register", "machine learning"],
        "sol2023": ["vocal mode", "classification", "XGBoost", "CVT"],
        "ghasemzadeh2023": ["machine learning", "generalizable", "sample size", "overfitting"],
        "gupta2024": ["ML", "voice research", "methodology", "best practices"],
        "drugman2019": ["glottal", "source", "analysis"],
        "degottex2011": ["glottal", "source", "analysis"],
        "miller2000": ["soprano", "voice training", "Fach"],
        "behlau1988": ["Brazilian", "popular singing", "textual intelligibility"],
        "boersma2023": ["Praat", "phonetics", "software"],
    }

    if citation in keywords:
        for keyword in keywords[citation]:
            pattern = rf".*{keyword}.*"
            matches = re.findall(pattern, content, re.IGNORECASE)
            for match in matches[:2]:  # Prendre les 2 premières occurrences
                if len(match) > 50 and len(match) < 300:
                    passages.append(f"**Keyword '{keyword}':** {match}")

    return passages[:5]  # Limiter à 5 passages


def update_verification_file() -> None:
    """Mettre à jour le fichier de vérification."""
    with open(VERIFICATION_FILE, encoding="utf-8") as f:
        content = f.read()

    # Trouver toutes les sections de citations
    sections = re.findall(r"(## \d+\. \w+)", content)

    updated_content = content

    for section in sections:
        # Extraire le nom de la citation
        match = re.search(r"## \d+\. (\w+)", section)
        if not match:
            continue

        citation = match.group(1)
        print(f"Traitement de: {citation}")

        # Extraire les passages
        passages = extract_key_passages(citation)

        if not passages:
            print(f"  ⚠ Aucun passage trouvé pour {citation}")
            continue

        # Trouver la section dans le contenu
        section_start = content.find(section)
        if section_start == -1:
            continue

        # Trouver la fin de la section (prochaine section ou fin du fichier)
        next_section_match = re.search(r"\n## \d+\. ", content[section_start + len(section) :])
        if next_section_match:
            section_end = section_start + len(section) + next_section_match.start()
        else:
            section_end = len(content)

        section_text = content[section_start:section_end]

        # Vérifier si la section a déjà des passages
        if "**Source passages from PDF:**" in section_text:
            print(f"  ✓ Déjà des passages pour {citation}")
            continue

        # Ajouter les passages après la référence complète
        ref_end_match = re.search(r"\*\*Occurrences in article", section_text)
        if not ref_end_match:
            # Chercher après la référence PDF
            pdf_match = re.search(r"\*\*PDF verified at:\*\*.*?\n", section_text, re.DOTALL)
            if pdf_match:
                insert_pos = section_start + pdf_match.end()
            else:
                # Chercher après la référence complète
                ref_match = re.search(r"\*\*Full reference:\*\*\n.*?\n\n", section_text, re.DOTALL)
                if ref_match:
                    insert_pos = section_start + ref_match.end()
                else:
                    print(f"  ⚠ Impossible de trouver où insérer pour {citation}")
                    continue
        else:
            insert_pos = section_start + ref_end_match.start()

        # Construire le texte à insérer
        passages_text = "\n**Source passages from PDF:**\n"
        for _i, passage in enumerate(passages, 1):
            passages_text += f"- {passage}\n"

        # Insérer les passages
        updated_content = (
            updated_content[:insert_pos] + passages_text + updated_content[insert_pos:]
        )

        print(f"  ✓ {len(passages)} passages ajoutés pour {citation}")

    # Sauvegarder le fichier mis à jour
    backup_file = f"{VERIFICATION_FILE}.backup"
    with open(backup_file, "w", encoding="utf-8") as f:
        f.write(content)

    with open(VERIFICATION_FILE, "w", encoding="utf-8") as f:
        f.write(updated_content)

    print("\n=== Mise à jour terminée ===")
    print(f"Backup sauvegardé dans: {backup_file}")
    print(f"Fichier principal mis à jour: {VERIFICATION_FILE}")
    print("\n=== Résumé ===")
    print("Les passages ont été ajoutés aux sections qui n'en avaient pas encore.")
    print("Chaque section contient maintenant des extraits pertinents du PDF.")


def main() -> None:
    print("=== Mise à jour de citation_verification.md ===")
    print("Ce script ajoute des passages extraits des PDFs au document de vérification.")
    print("")

    # Vérifier que le dossier articles_txt existe
    if not Path(TXT_DIR).exists():
        print(f"⚠ Le dossier {TXT_DIR} n'existe pas.")
        print("Exécutez d'abord: ./extract_pdfs.sh")
        return

    update_verification_file()


if __name__ == "__main__":
    main()
