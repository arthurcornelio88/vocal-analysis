# Reference Verification Checklist

**Article:** Multidimensional Vocological Analysis (JoV submission)
**Companion to:** `citation_verification.md`
**Last updated:** 2026-02-11

---

## Status Legend

| Symbol | Meaning |
|--------|---------|
| V | Verified by author |
| - | Not applicable (book/software) |
| X | Not yet done |

---

## Checklist

| # | Ref key | Short reference | Type | PDF/Source | Passages extracted | Status |
|---|---------|----------------|------|:---:|:---:|--------|
| 1 | roubeau2009 | Roubeau, Henrich & Castellengo (2009) | article | V | V | Ready |
| 2 | henrich2004 | Henrich, d'Alessandro, Doval & Castellengo (2005) | article | V | V | Ready |
| 3 | henrich2006 | Henrich (2006) | article | V | V | Ready |
| 4 | kim2018 | Kim, Salamon, Li & Bello (2018) | inproceedings | V | V | Ready |
| 5 | boersma2023 | Boersma & Weenink (2023) | software | - | - | N/A |
| 6 | maryn2015 | Maryn & Weenink (2015) | article | V | V | Ready |
| 7 | teixeira2013 | Teixeira, Oliveira & Lopes (2013) | article | V | V | Ready |
| 8 | yousef2024 | Yousef & Hunter (2024) | article | V | V | Ready |
| 9 | kreiman2012 | Kreiman, Shue, Chen et al. (2012) | article | V | V | Ready |
| 10 | kreiman2014 | Kreiman, Gerratt, Garellek et al. (2014) | article | V | V | Ready |
| 11 | bourne2012 | Bourne & Garnier (2012) | article | V | V | Ready |
| 12 | behlau1988 | Behlau & Ziemer (1988) | book chapter | - | - | N/A (book) |
| 13 | bozeman2013 | Bozeman (2013) | book | - | - | N/A (book) |
| 14 | sundberg1987 | Sundberg (1987) | book | - | - | N/A (book) |
| 15 | sundberg1974 | Sundberg (1974) | article | V | V | Ready |
| 16 | miller2000 | Miller (2000) | book | - | - | N/A (book) |
| 17 | alku2023 | Alku, Kadiri & Gowda (2023) | article | V | V | Ready |
| 18 | gowda2022 | Gowda, Bollepalli, Kadiri & Alku (2022) | article | V | V | Ready |
| 19 | kim2025 | Kim & Botha (2025) | article | V | V | Ready |
| 20 | boratto2025 | Boratto, Costa, Meireles et al. (2025) | article | V | V | Ready |
| 21 | hinrichs2026 | Hinrichs, Stephan, Lange & Ostermann (2026) | article | V | V | Ready |
| 22 | chen2016 | Chen & Guestrin (2016) | inproceedings | V | V | Ready |
| 23 | lee2013 | Lee (2013) | inproceedings | V | V | Ready |
| 24 | nigam2000 | Nigam, McCallum, Thrun & Mitchell (2000) | article | V | V | Ready |
| 25 | defossez2021 | Defossez (2021) | inproceedings | V | V | Ready |
| 26 | rouard2023 | Rouard, Massa & Defossez (2023) | inproceedings | V | V | Ready |
| 27 | cotton2007 | Cotton (2007) | thesis | V | V | Ready |
| 28 | rezende2016 | Rezende (2016) | book | - | - | N/A (book) |
| 29 | tatit2002 | Tatit (2002) | book | - | - | N/A (book) |
| 30 | degottex2011 | Degottex (2011) | thesis | V | V | Ready |
| 31 | drugman2019 | Drugman, Bozkurt & Dutoit (2019) | article | V | V | Ready |

---

## Summary

| Category | Count | Refs |
|----------|:-----:|------|
| Ready (PDF + passages) | 23 | roubeau2009, henrich2004, henrich2006, kim2018, maryn2015, kreiman2012, kreiman2014, bourne2012, sundberg1974, gowda2022, kim2025, boratto2025, chen2016, lee2013, nigam2000, rouard2023, cotton2007, yousef2024, hinrichs2026, alku2023, defossez2021, degottex2011, drugman2019 |
| Books/software (N/A) | 7 | boersma2023, behlau1988, bozeman2013, sundberg1987, miller2000, rezende2016, tatit2002 |
| Needs TXT + extraction | 0 | — |
| Needs extraction only | 0 | — |

**Total active refs:** 31
**Fully verified:** 31/31 (23 ready + 7 books/software + 1 hanson1997 paywalled/noted)
**Remaining work:** 0 — all refs verified

---

## Workflow

For each remaining ref:
1. Convert PDF to TXT if needed (run `extract_pdfs.sh` or manual)
2. Identify all occurrences of `\cite{key}` in `article.tex`
3. Read source TXT and extract relevant verbatim passages
4. Add passages + assessment to `citation_verification.md`
5. Mark as `V` in this checklist
