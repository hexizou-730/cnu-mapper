"""
rebuild_official_kb.py
========================

从法国政府官方数据源重建 cnu_knowledge_base_official.json.

数据来源:
  1. Galaxie 官方 PDF (2018 Arrêté)
     https://www.galaxie.enseignementsup-recherche.gouv.fr/ensup/pdf/qualification/sections.pdf
     → 57 个"通用 CNU"sections (groupes I-VIII + théologie + pharmacie)

  2. Arrêté 1992 (modified 2019, 2024) — Disciplines de santé
     https://www.legifrance.gouv.fr/loda/id/JORFTEXT000000174965
     → 23 个健康学科 sections (médecine, odontologie, pharmacie bi-app, autres santé)

  总计: 80 个官方 CNU sections.

输出:
  cnu_knowledge_base_official.json  —— 纯净版, 仅含官方字段
"""

import json
from pathlib import Path

# ═══════════════════════════════════════════════════════════════════════════
# 第一部分: 57 个"通用 CNU"sections
# 来源: Galaxie 官方 PDF (DGRH, 高等教育研究部)
# https://www.galaxie.enseignementsup-recherche.gouv.fr/ensup/pdf/qualification/sections.pdf
# ═══════════════════════════════════════════════════════════════════════════

GENERAL_SECTIONS = [
    # Groupe I — Droit, économie et gestion
    ("01", "I", "Droit, économie et gestion", "Droit privé et sciences criminelles"),
    ("02", "I", "Droit, économie et gestion", "Droit public"),
    ("03", "I", "Droit, économie et gestion", "Histoire du droit et des institutions"),
    ("04", "I", "Droit, économie et gestion", "Science politique"),
    ("05", "I", "Droit, économie et gestion", "Sciences économiques"),
    ("06", "I", "Droit, économie et gestion", "Sciences de gestion"),

    # Groupe II — Lettres et sciences humaines
    ("07", "II", "Lettres et sciences humaines", "Sciences du langage : linguistique et phonétique générales"),
    ("08", "II", "Lettres et sciences humaines", "Langues et littératures anciennes"),
    ("09", "II", "Lettres et sciences humaines", "Langue et littérature françaises"),
    ("10", "II", "Lettres et sciences humaines", "Littératures comparées"),
    ("11", "II", "Lettres et sciences humaines", "Langues et littératures anglaises et anglo-saxonnes"),
    ("12", "II", "Lettres et sciences humaines", "Langues et littératures germaniques et scandinaves"),
    ("13", "II", "Lettres et sciences humaines", "Langues et littératures slaves"),
    ("14", "II", "Lettres et sciences humaines", "Langues et littératures romanes : espagnol, italien, portugais, autres langues romanes"),
    ("15", "II", "Lettres et sciences humaines", "Langues et littératures arabes, chinoises, japonaises, hébraïques, d'autres domaines linguistiques"),

    # Groupe III — Sciences humaines et humanités
    ("16", "III", "Sciences humaines et humanités", "Psychologie, psychologie clinique, psychologie sociale"),
    ("17", "III", "Sciences humaines et humanités", "Philosophie"),
    ("18", "III", "Sciences humaines et humanités", "Architecture (ses théories et ses pratiques), arts appliqués, arts plastiques, arts du spectacle, épistémologie des enseignements artistiques, esthétique, musicologie, musique, sciences de l'art"),
    ("19", "III", "Sciences humaines et humanités", "Sociologie, démographie"),
    ("20", "III", "Sciences humaines et humanités", "Anthropologie biologique, ethnologie, préhistoire"),
    ("21", "III", "Sciences humaines et humanités", "Histoire, civilisation, archéologie et art des mondes anciens et médiévaux"),
    ("22", "III", "Sciences humaines et humanités", "Histoire et civilisations : histoire des mondes modernes, histoire du monde contemporain, de l'art, de la musique"),
    ("23", "III", "Sciences humaines et humanités", "Géographie physique, humaine, économique et régionale"),
    ("24", "III", "Sciences humaines et humanités", "Aménagement de l'espace, urbanisme"),

    # Groupe IV — Mathématiques et informatique
    ("25", "IV", "Mathématiques et informatique", "Mathématiques"),
    ("26", "IV", "Mathématiques et informatique", "Mathématiques appliquées et applications des mathématiques"),
    ("27", "IV", "Mathématiques et informatique", "Informatique"),

    # Groupe V — Physique, chimie et sciences de la matière
    ("28", "V", "Physique, chimie et sciences de la matière", "Milieux denses et matériaux"),
    ("29", "V", "Physique, chimie et sciences de la matière", "Constituants élémentaires"),
    ("30", "V", "Physique, chimie et sciences de la matière", "Milieux dilués et optique"),
    ("31", "V", "Physique, chimie et sciences de la matière", "Chimie théorique, physique, analytique"),
    ("32", "V", "Physique, chimie et sciences de la matière", "Chimie organique, minérale, industrielle"),
    ("33", "V", "Physique, chimie et sciences de la matière", "Chimie des matériaux"),
    ("34", "V", "Physique, chimie et sciences de la matière", "Astronomie, astrophysique"),
    ("35", "V", "Physique, chimie et sciences de la matière", "Structure et évolution de la Terre et des autres planètes"),
    ("36", "V", "Physique, chimie et sciences de la matière", "Terre solide : géodynamique des enveloppes supérieures, paléo-biosphère"),
    ("37", "V", "Physique, chimie et sciences de la matière", "Météorologie, océanographie physique et physique de l'environnement"),

    # Groupe VI — Sciences pour l'ingénieur
    ("60", "VI", "Sciences pour l'ingénieur", "Mécanique, génie mécanique, génie civil"),
    ("61", "VI", "Sciences pour l'ingénieur", "Génie informatique, automatique et traitement du signal"),
    ("62", "VI", "Sciences pour l'ingénieur", "Énergétique, génie des procédés"),
    ("63", "VI", "Sciences pour l'ingénieur", "Génie électrique, électronique, photonique et systèmes"),

    # Groupe VII — Sciences de la vie
    ("64", "VII", "Sciences de la vie", "Biochimie et biologie moléculaire"),
    ("65", "VII", "Sciences de la vie", "Biologie cellulaire"),
    ("66", "VII", "Sciences de la vie", "Physiologie"),
    ("67", "VII", "Sciences de la vie", "Biologie des populations et écologie"),
    ("68", "VII", "Sciences de la vie", "Biologie des organismes"),
    ("69", "VII", "Sciences de la vie", "Neurosciences"),

    # Groupe VIII — Sciences de l'homme et humanités (sous-sections éducatives/sportives)
    ("70", "VIII", "Sciences humaines et éducation", "Sciences de l'éducation"),
    ("71", "VIII", "Sciences humaines et éducation", "Sciences de l'information et de la communication"),
    ("72", "VIII", "Sciences humaines et éducation", "Épistémologie, histoire des sciences et des techniques"),
    ("73", "VIII", "Sciences humaines et éducation", "Cultures et langues régionales"),
    ("74", "VIII", "Sciences humaines et éducation", "Sciences et techniques des activités physiques et sportives"),

    # Théologie (groupes spécifiques)
    ("76", "X", "Théologie", "Théologie catholique"),
    ("77", "X", "Théologie", "Théologie protestante"),

    # Pharmacie mono-appartenants (groupe XII dans le CNU général)
    ("85", "XII", "Pharmacie", "Sciences physico-chimiques et technologies pharmaceutiques"),
    ("86", "XII", "Pharmacie", "Sciences du médicament"),
    ("87", "XII", "Pharmacie", "Sciences biologiques pharmaceutiques"),
]

# ═══════════════════════════════════════════════════════════════════════════
# 第二部分: 23 个"CNU-Santé"sections (disciplines de santé)
# 来源: Arrêté du 29 juin 1992, modifié par Arrêté du 27 juin 2024
# https://www.legifrance.gouv.fr/loda/id/JORFTEXT000000174965
# ═══════════════════════════════════════════════════════════════════════════

HEALTH_SECTIONS = [
    # Disciplines médicales (14 sections — versions 2024)
    ("42", "M-I", "Disciplines médicales", "Morphologie et morphogenèse"),
    ("43", "M-I", "Disciplines médicales", "Biophysique et imagerie médicale"),
    ("44", "M-I", "Disciplines médicales", "Biochimie, biologie cellulaire et moléculaire, physiologie et nutrition"),
    ("45", "M-I", "Disciplines médicales", "Microbiologie, maladies transmissibles et hygiène"),
    ("46", "M-I", "Disciplines médicales", "Santé publique, environnement et société"),
    ("47", "M-I", "Disciplines médicales", "Cancérologie, génétique, hématologie, immunologie"),
    ("48", "M-I", "Disciplines médicales", "Anesthésiologie, réanimation, médecine d'urgence, pharmacologie et thérapeutique"),
    ("49", "M-I", "Disciplines médicales", "Pathologie nerveuse et musculaire, pathologie mentale, handicap et rééducation"),
    ("50", "M-I", "Disciplines médicales", "Pathologie ostéo-articulaire, dermatologie et chirurgie plastique"),
    ("51", "M-I", "Disciplines médicales", "Pathologie cardiorespiratoire et vasculaire"),
    ("52", "M-I", "Disciplines médicales", "Maladies des appareils digestif et urinaire"),
    ("53", "M-I", "Disciplines médicales", "Médecine interne, gériatrie et médecine générale"),
    ("54", "M-I", "Disciplines médicales", "Développement et pathologie de l'enfant, gynécologie-obstétrique, endocrinologie et reproduction"),
    ("55", "M-I", "Disciplines médicales", "Pathologie de la tête et du cou"),

    # Disciplines odontologiques (3 sections)
    ("56", "M-II", "Disciplines odontologiques", "Développement, croissance et prévention"),
    ("57", "M-II", "Disciplines odontologiques", "Chirurgie orale, parodontologie, biologie orale"),
    ("58", "M-II", "Disciplines odontologiques", "Réhabilitation orale"),

    # Disciplines pharmaceutiques bi-appartenants (3 sections)
    ("80", "M-III", "Pharmacie (bi-appartenants)", "Sciences physico-chimiques et ingénierie appliquée à la santé (bi-appartenants)"),
    ("81", "M-III", "Pharmacie (bi-appartenants)", "Sciences du médicament et des autres produits de santé (bi-appartenants)"),
    ("82", "M-III", "Pharmacie (bi-appartenants)", "Sciences biologiques, fondamentales et cliniques (bi-appartenants)"),

    # Autres disciplines de santé (mono-appartenants — créés par Arrêté 2019)
    ("90", "M-IV", "Autres disciplines de santé (mono-appartenants)", "Maïeutique"),
    ("91", "M-IV", "Autres disciplines de santé (mono-appartenants)", "Sciences de la rééducation et de réadaptation"),
    ("92", "M-IV", "Autres disciplines de santé (mono-appartenants)", "Sciences infirmières"),
]

# ═══════════════════════════════════════════════════════════════════════════
# 构造 JSON
# ═══════════════════════════════════════════════════════════════════════════

SOURCE_GENERAL = {
    "type": "arrêté",
    "name": "Arrêté du 18 décembre 2018 fixant la liste des groupes et des sections du CNU",
    "url": "https://www.legifrance.gouv.fr/jorf/id/JORFTEXT000038015526",
    "portal": "https://www.galaxie.enseignementsup-recherche.gouv.fr/ensup/pdf/qualification/sections.pdf",
}

SOURCE_HEALTH = {
    "type": "arrêté",
    "name": "Arrêté du 29 juin 1992 (modifié par Arrêté du 27 juin 2024) — Disciplines de santé",
    "url": "https://www.legifrance.gouv.fr/loda/id/JORFTEXT000000174965",
}

def build():
    out = {
        "_metadata": {
            "description": "Nomenclature officielle des sections du Conseil national des universités (CNU). Données strictement officielles, aucune dérivation ni enrichissement.",
            "total_sections": 0,
            "license": "Licence Ouverte 2.0 (Open License 2.0)",
            "sources": {
                "general_cnu": SOURCE_GENERAL,
                "health_cnu": SOURCE_HEALTH,
            },
            "fields": {
                "code_section":    "官方 2 位数 section 代码",
                "code_groupe":     "所属 groupe 罗马数字编号",
                "groupe_fr":       "groupe 法语名称",
                "section_fr":      "section 法语名称 (官方)",
                "source":          "数据来源: 'general' 或 'health'",
            },
        },
        "sections": []
    }

    for code, g, g_name, s_name in GENERAL_SECTIONS:
        out["sections"].append({
            "code_section": code,
            "code_groupe": g,
            "groupe_fr": g_name,
            "section_fr": s_name,
            "source": "general",
        })

    for code, g, g_name, s_name in HEALTH_SECTIONS:
        out["sections"].append({
            "code_section": code,
            "code_groupe": g,
            "groupe_fr": g_name,
            "section_fr": s_name,
            "source": "health",
        })

    out["_metadata"]["total_sections"] = len(out["sections"])

    # 按 code_section 排序
    out["sections"].sort(key=lambda s: s["code_section"])

    return out


if __name__ == "__main__":
    data = build()
    out_path = Path(__file__).parent / "cnu_knowledge_base_official.json"
    out_path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )
    print(f"Generated {out_path}")
    print(f"   Total official sections: {data['_metadata']['total_sections']}")
    general = sum(1 for s in data['sections'] if s['source'] == 'general')
    health = sum(1 for s in data['sections'] if s['source'] == 'health')
    print(f"   General CNU sections: {general} (groups I-VIII + X + XII)")
    print(f"   Health CNU sections: {health} (medicine, dentistry, pharmacy, other health sections)")
