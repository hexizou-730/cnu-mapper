# CNU Knowledge Base Data Provenance

## File Inventory

```text
cnu_knowledge_base_official.json   - clean official CNU data, 80 sections
rebuild_official_kb.py             - reproducible rebuild script
DATA_PROVENANCE.md                 - this provenance note
```

## Data Sources

The 80 CNU sections in this project come from official French government
regulations published by the French Ministry of Higher Education and Research
(MESR).

### Source 1: Arrêté du 18 décembre 2018 - General CNU, 57 sections

URL: <https://www.legifrance.gouv.fr/jorf/id/JORFTEXT000038015526>  
Practical section list: <https://www.galaxie.enseignementsup-recherche.gouv.fr/ensup/pdf/qualification/sections.pdf>

This source covers:

- Groupe I - Droit, économie, gestion (01-06)
- Groupe II - Lettres (07-15)
- Groupe III - Sciences humaines (16-24)
- Groupe IV - Mathématiques et informatique (25-27)
- Groupe V - Physique, chimie, sciences de la matière (28-37)
- Groupe VI - Sciences pour l'ingénieur (60-63)
- Groupe VII - Sciences de la vie (64-69)
- Groupe VIII - Sciences éducation, STAPS, etc. (70-74)
- Théologie (76-77)
- Pharmacie mono-appartenants (85-87)

### Source 2: Arrêté du 29 juin 1992, modified by Arrêté du 27 juin 2024 - CNU-Santé, 23 sections

URL: <https://www.legifrance.gouv.fr/loda/id/JORFTEXT000000174965>

This source covers:

- Disciplines médicales (42-55, 14 sections)
- Disciplines odontologiques (56-58, 3 sections)
- Pharmacie bi-appartenants (80-82, 3 sections)
- Autres santé mono-appartenants (90-92, 3 sections): Maïeutique, Rééducation, Sciences infirmières

## Data Fields

`cnu_knowledge_base_official.json` contains only official fields. It does not
include derived keywords, AI-generated content, or manually enriched
classification text.

| Field | Description |
|---|---|
| `code_section` | Official two-digit CNU section code |
| `code_groupe` | Official group code |
| `groupe_fr` | Official French group name |
| `section_fr` | Official French section name |
| `source` | `general` for the 2018 arrêté or `health` for the health sections |

## Relationship To Runtime Code

`cnu_knowledge_base_official.json` is the authoritative CNU reference used by
the project. It is suitable for academic reporting because it keeps only the
official section codes and French names.

The English display names used in terminal output and in the LLM prompt are
maintained separately in `dewey_to_cnu.py`. Those English names are project
helpers for readability; they are not part of the official data source.

## License

The official French government data is released under Licence Ouverte 2.0
(Open License 2.0). It can be reused, modified, and redistributed with proper
attribution.

<https://github.com/etalab/licence-ouverte/blob/master/LO.md>

## Reproducibility

To rebuild the official CNU JSON file, run:

```bash
python3 rebuild_official_kb.py
```

The rebuild script stores the official section list directly in code and
writes `cnu_knowledge_base_official.json`. The script header documents the
official URLs used as sources.

## Suggested Citation In Reports

Suggested wording for the data-source or experimental-setup section:

> The CNU knowledge base used in this project contains 80 sections. The section
> codes and French section names are taken from official French government
> regulations: (1) Arrêté du 18 décembre 2018 for the general CNU sections
> (57 sections), and (2) Arrêté du 29 juin 1992, as modified by Arrêté du
> 27 juin 2024, for the CNU-Santé sections (23 sections). The data is released
> under Licence Ouverte 2.0. See `cnu_knowledge_base_official.json`.
