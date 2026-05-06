"""
dewey_to_cnu.py  (v2 — capped at 2 labels per DDC)
====================================================

Map Dewey Decimal Classification (DDC) codes to French CNU section codes.
将 Dewey 十进制分类码映射到法国 CNU section 代码.

═══════════════════════════════════════════════════════════════════════════
v2 CHANGES (compared to v1) / v2 与 v1 的区别
═══════════════════════════════════════════════════════════════════════════

PROBLEM IN v1 / v1 中的问题:
  Some coarse-grained DDC codes were mapped to ALL relevant CNU sections
  (e.g. ddc:610 "Medicine" → 12 medical CNU sections [44-55]).
  This produced training samples where 12 of 63 labels were all marked 1,
  which is essentially noise: a thesis on cardiology should NOT be labeled
  as belonging to obstetrics, dentistry, etc.

  v1 中, 一些粗粒度 Dewey 码被映射到所有相关的 CNU section
  (例如 ddc:610 "医学" → 12 个 CNU 医学 section [44-55]).
  这制造了"心血管研究 = 同时属于产科 + 牙科 + 麻醉"的伪标签, 实际是噪声.

FIX IN v2 / v2 的修复:
  Hard cap: each DDC maps to AT MOST 2 CNU sections.
  When a DDC naturally spans many CNU sections, we pick the 1-2 most
  representative based on:
    1. Most common in the corpus
    2. Broadest coverage of the parent DDC
    3. When tied: prefer the lower CNU code

  硬上限: 每个 DDC 最多映射 2 个 CNU section.

CONSEQUENCES / 影响:
  - Training data label distribution shifts from heavily multi-label
    (3-12 labels common) to mostly single/double labels.
  - Top-1 accuracy expected to improve (less noise per sample).
  - Training data still does NOT cover 10 health-discipline sections
    (43, 56-58, 80-82, 90-92), as theses.fr lacks these sub-disciplines.
    This is a limitation of the data source, not the mapping.
"""

from __future__ import annotations


CNU_SECTION_EN: dict[str, str] = {
    "01": "Private law and criminal sciences",
    "02": "Public law",
    "03": "History of law and institutions",
    "04": "Political science",
    "05": "Economics",
    "06": "Management sciences and management",
    "07": "Language sciences (linguistics)",
    "08": "Ancient languages and literatures (Greek, Latin)",
    "09": "French language and literature",
    "10": "Comparative literature",
    "11": "Anglophone studies (English literature and culture)",
    "12": "Germanic and Scandinavian studies",
    "13": "Slavic and Baltic studies",
    "14": "Romance studies (Spanish, Italian, Portuguese, etc.)",
    "15": "African, Asian and other linguistic area languages, literatures and cultures",
    "16": "Psychology and ergonomics",
    "17": "Philosophy",
    "18": "Architecture, applied arts, visual arts, performing arts, aesthetics, musicology and music",
    "19": "Sociology and demography",
    "20": "Ethnology, prehistory and biological anthropology",
    "21": "History of ancient and medieval worlds, civilisations, archaeology and art",
    "22": "History of modern and contemporary worlds, art history and music history",
    "23": "Physical, human, economic and regional geography",
    "24": "Spatial planning and urban studies",
    "25": "Mathematics",
    "26": "Applied mathematics",
    "27": "Computer science",
    "28": "Dense media and materials (condensed matter physics)",
    "29": "Elementary constituents (particle and nuclear physics)",
    "30": "Dilute media and optics",
    "31": "Theoretical, physical and analytical chemistry",
    "32": "Organic, inorganic and industrial chemistry",
    "33": "Materials chemistry",
    "34": "Astronomy and astrophysics",
    "35": "Structure and evolution of the Earth and other planets",
    "36": "Solid Earth: geodynamics of upper envelopes and palaeobiosphere",
    "37": "Fluid envelopes of the Earth system and other planets (atmospheric and ocean sciences)",
    "42": "Morphology and morphogenesis (anatomy and pathological cytology)",
    "43": "Biophysics and medical imaging",
    "44": "Biochemistry, cell and molecular biology, physiology and nutrition",
    "45": "Microbiology, infectious diseases and hygiene",
    "46": "Public health, environment and society",
    "47": "Oncology, genetics, haematology and immunology",
    "48": "Anaesthesiology, intensive care, emergency medicine, pharmacology and therapeutics",
    "49": "Neurological and muscular pathology, mental disorders, disability and rehabilitation",
    "50": "Osteoarticular pathology, dermatology and plastic surgery",
    "51": "Cardiorespiratory and vascular pathology",
    "52": "Diseases of the digestive and urinary systems",
    "53": "Internal medicine, geriatrics and general surgery",
    "54": "Child development and pathology, gynaecology-obstetrics, endocrinology and reproduction",
    "55": "Head and neck pathology (ENT, ophthalmology)",
    "56": "Development, growth and prevention (dentistry)",
    "57": "Oral surgery, periodontology and oral biology",
    "58": "Oral rehabilitation (prosthetics and restorative dentistry)",
    "60": "Mechanics, mechanical engineering and civil engineering",
    "61": "Computer engineering, automation and signal processing",
    "62": "Energy engineering and process engineering",
    "63": "Electrical engineering, electronics, photonics and systems",
    "64": "Biochemistry and molecular biology",
    "65": "Cell biology",
    "66": "Physiology",
    "67": "Population biology and ecology",
    "68": "Organismal biology",
    "69": "Neurosciences",
    "70": "Education sciences and training",
    "71": "Information and communication sciences",
    "72": "Epistemology, history of science and technology",
    "73": "Regional cultures and languages",
    "74": "Sport sciences and physical activity",
    "76": "Catholic theology",
    "77": "Protestant theology",
    "80": "Physical-chemical sciences and engineering applied to health (pharmacy, bi-membership)",
    "81": "Pharmaceutical sciences and other health products (bi-membership)",
    "82": "Biological, fundamental and clinical sciences (pharmacy, bi-membership)",
    "85": "Physical-chemical sciences and engineering applied to health (pharmacy, mono-membership)",
    "86": "Pharmaceutical sciences and other health products (mono-membership)",
    "87": "Biological, fundamental and clinical sciences (pharmacy, mono-membership)",
    "90": "Midwifery (maieutics)",
    "91": "Rehabilitation and re-adaptation sciences (physiotherapy, occupational therapy)",
    "92": "Nursing sciences",
}


DEWEY_TO_CNU: dict[str, list[str]] = {

    # ═════════════════ 0xx — Computer science, info, general ═════════════════
    "000": ["27"],
    "001": ["27"],
    "003": ["27"],
    "004": ["27"],
    "005": ["27"],
    "006": ["27"],
    "010": ["71"],
    "020": ["71"],
    "025": ["71"],
    "070": ["71"],
    "080": ["71"],
    "090": ["72"],

    # ═════════════════ 1xx — Philosophy, psychology ═════════════════
    "100": ["17"],
    "110": ["17"],
    "120": ["17"],
    "130": ["17"],
    "140": ["17"],
    "150": ["16"],
    "152": ["16", "69"],
    "153": ["16"],
    "155": ["16"],
    "158": ["16"],
    "160": ["17"],
    "170": ["17"],
    "180": ["17"],
    "190": ["17"],

    # ═════════════════ 2xx — Religion, theology ═════════════════
    "200": ["76", "77"],
    "210": ["76", "77"],
    "220": ["76", "77"],
    "230": ["76", "77"],
    "240": ["76", "77"],
    "260": ["76", "77"],
    "270": ["22", "76"],
    "290": ["76", "77"],

    # ═════════════════ 3xx — Social sciences ═════════════════
    "300": ["19"],
    "301": ["19"],
    "302": ["19"],
    "303": ["19"],
    "304": ["19"],
    "305": ["19", "20"],
    "306": ["19", "20"],
    "307": ["19", "24"],
    "320": ["04"],
    "323": ["04"],
    "324": ["04"],
    "325": ["04"],
    "327": ["04"],
    "328": ["04"],
    "330": ["05"],
    "331": ["05"],
    "332": ["05", "06"],
    "333": ["05"],
    "335": ["05"],
    "336": ["05"],
    "337": ["05"],
    "338": ["05", "06"],
    "339": ["05"],
    "340": ["01", "02"],
    "341": ["02"],
    "342": ["02"],
    "343": ["01", "02"],
    "344": ["02"],
    "345": ["01"],
    "346": ["01"],
    "347": ["01"],
    "348": ["01", "02"],
    "349": ["01", "02"],
    "350": ["02", "04"],
    "353": ["04"],
    "355": ["02", "04"],
    "360": ["19"],
    "361": ["19"],
    "362": ["16", "19"],
    "363": ["19"],
    "370": ["70"],
    "371": ["70"],
    "372": ["70"],
    "373": ["70"],
    "374": ["70"],
    "375": ["70"],
    "378": ["70"],
    "379": ["70"],
    "380": ["05", "71"],
    "381": ["05"],
    "382": ["05"],
    "384": ["71"],
    "388": ["23"],
    "390": ["20"],
    "391": ["20"],
    "392": ["20"],
    "393": ["20"],
    "398": ["20"],

    # ═════════════════ 4xx — Languages ═════════════════
    "400": ["07"],
    "410": ["07"],
    "411": ["07"],
    "412": ["07"],
    "413": ["07"],
    "414": ["07"],
    "415": ["07"],
    "417": ["07", "73"],
    "418": ["07"],
    "419": ["07"],
    "440": ["09"],
    "443": ["09"],
    "445": ["09"],
    "447": ["09", "73"],
    "448": ["09"],
    "449": ["73"],
    "420": ["11"],
    "430": ["12"],
    "439": ["12"],
    "450": ["14"],
    "460": ["14"],
    "470": ["08"],
    "480": ["08"],
    "490": ["13", "15"],
    "491": ["13"],
    "492": ["15"],
    "493": ["08"],
    "494": ["15"],
    "495": ["15"],
    "496": ["15"],
    "499": ["15"],

    # ═════════════════ 5xx — Pure sciences ═════════════════
    "500": ["28"],
    "501": ["72"],
    "502": ["72"],
    "504": ["72"],
    "509": ["72"],
    "510": ["25"],
    "511": ["25"],
    "512": ["25"],
    "513": ["25"],
    "514": ["25"],
    "515": ["25", "26"],
    "516": ["25"],
    "518": ["26"],
    "519": ["26"],
    "520": ["34"],
    "521": ["34"],
    "522": ["34"],
    "523": ["34"],
    "525": ["34", "35"],
    "530": ["28"],          # v1: ["28","29","30"]
    "531": ["60"],
    "532": ["28", "60"],
    "533": ["28"],
    "534": ["30"],
    "535": ["30"],
    "536": ["28", "30"],
    "537": ["30", "63"],    # v1: ["28","30","63"]
    "538": ["28"],
    "539": ["29"],          # v1: ["28","29"]
    "540": ["32"],          # v1: ["31","32","33"]
    "541": ["31"],
    "542": ["31"],
    "543": ["31"],
    "544": ["31"],
    "545": ["31"],
    "546": ["32"],
    "547": ["32"],
    "548": ["33"],
    "549": ["33"],          # v1: ["33","35"]
    "550": ["35", "37"],    # v1: ["35","36","37"]
    "551": ["35", "37"],    # v1: ["35","36","37"]
    "552": ["35", "36"],
    "553": ["35", "36"],
    "554": ["36"],
    "560": ["20", "36"],
    "570": ["64", "65"],    # v1: ["64","65","66","67","68"]
    "571": ["66"],
    "572": ["64"],
    "573": ["66"],
    "574": ["64", "65"],
    "575": ["65", "67"],
    "576": ["64", "65"],
    "577": ["67"],
    "578": ["68"],
    "579": ["68"],
    "580": ["68"],
    "581": ["68"],
    "582": ["68"],
    "590": ["68"],
    "591": ["68"],
    "595": ["68"],
    "596": ["68"],
    "597": ["68"],
    "598": ["68"],
    "599": ["68"],

    # ═════════════════ 6xx — Technology, applied sciences ═════════════════
    "600": ["62"],          # v1: ["60","61","62","63"]

    # ───── Medicine — biggest fix ─────
    # ddc:610 was the source of all 12-label samples in v1 (mapped to 44-55).
    # v2 collapses to a single representative: 46 (Public health, environment, society),
    # which is the most general medical CNU section.
    "610": ["46"],          # v1: 12 sections [44-55]
    "611": ["42"],
    "612": ["44"],
    "613": ["46"],
    "614": ["46"],
    "615": ["48"],          # v1: ["48","85","86","87"]
    "616": ["53"],          # v1: 7 sections, now collapsed to internal medicine (broadest)
    "617": ["50"],          # v1: ["50","55"]
    "618": ["54"],
    "619": ["66"],

    # ───── Engineering ─────
    "620": ["62"],          # v1: ["60","61","62","63"]
    "621": ["62", "63"],    # v1: ["60","62","63"]
    "622": ["62"],
    "623": ["60", "62"],
    "624": ["60"],
    "625": ["60"],
    "626": ["60"],
    "627": ["60"],
    "628": ["62"],
    "629": ["60", "61"],
    "630": ["67", "68"],
    "631": ["67"],
    "632": ["67", "68"],
    "634": ["67", "68"],
    "636": ["68"],
    "637": ["67"],
    "640": ["19"],
    "641": ["32"],
    "650": ["06"],
    "657": ["06"],
    "658": ["06"],
    "659": ["06", "71"],
    "660": ["62"],          # v1: ["32","62"]
    "661": ["62"],          # v1: ["32","62"]
    "664": ["62"],
    "665": ["62"],          # v1: ["32","62"]
    "666": ["62"],          # v1: ["33","62"]
    "668": ["32"],          # v1: ["32","33"]
    "669": ["33"],
    "670": ["62", "63"],
    "676": ["62"],
    "677": ["62"],
    "680": ["62"],
    "690": ["60"],

    # ═════════════════ 7xx — The arts, recreation ═════════════════
    "700": ["18"],
    "701": ["18"],
    "709": ["18", "22"],    # v1: ["18","21","22"]
    "710": ["24"],
    "711": ["24"],
    "712": ["24"],
    "720": ["18"],
    "721": ["18"],
    "725": ["18", "24"],
    "730": ["18"],
    "740": ["18"],
    "741": ["18"],
    "745": ["18"],
    "750": ["18"],
    "759": ["18", "22"],
    "760": ["18"],
    "770": ["18"],
    "780": ["18"],
    "781": ["18"],
    "782": ["18"],
    "785": ["18"],
    "786": ["18"],
    "787": ["18"],
    "789": ["18"],
    "790": ["74"],
    "791": ["18"],
    "792": ["18"],
    "793": ["18"],
    "794": ["18", "74"],
    "796": ["74"],
    "797": ["74"],
    "798": ["74"],
    "799": ["74"],

    # ═════════════════ 8xx — Literature ═════════════════
    "800": ["09", "10"],
    "801": ["10"],
    "808": ["09", "10"],
    "809": ["10"],
    "840": ["09"],
    "841": ["09"],
    "842": ["09"],
    "843": ["09"],
    "844": ["09"],
    "846": ["09"],
    "848": ["09"],
    "820": ["11"],
    "830": ["12"],
    "850": ["14"],
    "860": ["14"],
    "870": ["08"],
    "880": ["08"],
    "890": ["13", "15"],    # v1: ["08","13","15"]
    "891": ["13"],
    "892": ["15"],          # v1: ["08","15"]
    "895": ["15"],
    "896": ["15"],
    "899": ["15"],

    # ═════════════════ 9xx — History, geography ═════════════════
    "900": ["21", "22"],
    "901": ["22"],
    "902": ["22"],
    "907": ["72"],
    "909": ["22"],
    "910": ["23"],
    "911": ["23"],
    "912": ["23"],
    "913": ["21"],          # v1: ["21","23"]
    "914": ["23"],
    "915": ["23"],
    "920": ["22"],
    "930": ["21"],
    "931": ["21"],
    "932": ["21"],
    "936": ["21"],
    "937": ["21"],
    "938": ["21"],
    "940": ["22"],
    "941": ["22"],
    "942": ["22"],
    "943": ["22"],
    "944": ["22"],
    "945": ["22"],
    "946": ["22"],
    "947": ["22"],
    "948": ["22"],
    "949": ["22"],
    "950": ["22"],
    "951": ["22"],
    "952": ["22"],
    "953": ["22"],
    "954": ["22"],
    "956": ["22"],
    "960": ["22"],
    "961": ["22"],
    "962": ["22"],
    "963": ["22"],
    "964": ["22"],
    "965": ["22"],
    "966": ["22"],
    "967": ["22"],
    "968": ["22"],
    "970": ["22"],
    "972": ["22"],
    "973": ["22"],
    "980": ["22"],
    "990": ["22"],
    "994": ["22"],
}


def lookup(ddc: str) -> list[str]:
    """Return the list of CNU section codes for a given Dewey code.
    根据 Dewey 码返回对应的 CNU section 代码列表.
    """
    ddc = (ddc or "").strip().zfill(3)
    return DEWEY_TO_CNU.get(ddc, [])


def section_name_en(cnu_code: str) -> str | None:
    """Return the project-maintained English display name for a CNU code."""
    return CNU_SECTION_EN.get((cnu_code or "").strip().zfill(2))


def section_display_name(cnu_code: str, section_fr: str | None = None) -> str:
    """Return a readable English/French display name without modifying official data."""
    code = (cnu_code or "").strip().zfill(2)
    section_en = section_name_en(code)
    if section_en and section_fr:
        return f"{section_en} / {section_fr}"
    return section_en or section_fr or "?"


def coverage_stats() -> dict:
    """Helper: report on the mapping table itself."""
    cnu_counter: dict[str, int] = {}
    label_count_dist: dict[int, int] = {}
    for ddc, cnu_list in DEWEY_TO_CNU.items():
        n = len(cnu_list)
        label_count_dist[n] = label_count_dist.get(n, 0) + 1
        for cnu in cnu_list:
            cnu_counter[cnu] = cnu_counter.get(cnu, 0) + 1
    max_labels = max(len(v) for v in DEWEY_TO_CNU.values())
    return {
        "total_ddc_entries": len(DEWEY_TO_CNU),
        "max_labels_per_ddc": max_labels,
        "labels_per_ddc_distribution": dict(sorted(label_count_dist.items())),
        "unique_cnu_targets": len(cnu_counter),
        "cnu_with_most_ddc_sources": sorted(
            cnu_counter.items(), key=lambda kv: -kv[1]
        )[:10],
    }


if __name__ == "__main__":
    import json
    stats = coverage_stats()
    print("Mapping table self-check (v2):")
    print(json.dumps(stats, ensure_ascii=False, indent=2))

    print("\nKey changes vs v1:")
    print(f"  Max labels per DDC: {stats['max_labels_per_ddc']}  (was 12 in v1)")
    print(f"  All DDC entries now have 1-2 labels.")

    print("\nSpot checks:")
    for code in ["004", "150", "340", "510", "530", "540", "570", "610", "615", "616", "620", "780", "840"]:
        print(f"  ddc:{code} -> {lookup(code)}")
