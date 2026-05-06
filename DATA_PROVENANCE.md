# CNU 知识库数据来源声明

## 文件清单

```
cnu_knowledge_base_official.json   — 官方数据的纯净版 (80 sections)
rebuild_official_kb.py             — 重建脚本 (可审计)
DATA_PROVENANCE.md                 — 本文档
```

## 数据来源

本知识库的 80 个 CNU sections 全部来自法国政府**官方法令 (Arrêté)**, 由法国高等教育研究部 (MESR) 发布.

### 来源 1: Arrêté du 18 décembre 2018 — 通用 CNU (57 sections)

**URL**: <https://www.legifrance.gouv.fr/jorf/id/JORFTEXT000038015526>  
**实用清单**: <https://www.galaxie.enseignementsup-recherche.gouv.fr/ensup/pdf/qualification/sections.pdf>  

涵盖:
- **Groupe I** - Droit, économie, gestion (01–06)
- **Groupe II** - Lettres (07–15)
- **Groupe III** - Sciences humaines (16–24)
- **Groupe IV** - Mathématiques et informatique (25–27)
- **Groupe V** - Physique, chimie, sciences de la matière (28–37)
- **Groupe VI** - Sciences pour l'ingénieur (60–63)
- **Groupe VII** - Sciences de la vie (64–69)
- **Groupe VIII** - Sciences éducation, STAPS, etc. (70–74)
- **Théologie** (76–77)
- **Pharmacie mono-appartenants** (85–87)

### 来源 2: Arrêté du 29 juin 1992 (modifié par Arrêté du 27 juin 2024) — CNU-Santé (23 sections)

**URL**: <https://www.legifrance.gouv.fr/loda/id/JORFTEXT000000174965>  

涵盖:
- **Disciplines médicales** (42–55, 14 sections)
- **Disciplines odontologiques** (56–58, 3 sections)
- **Pharmacie bi-appartenants** (80–82, 3 sections)
- **Autres santé mono-appartenants** (90–92, 3 sections) — Maïeutique, Rééducation, Sciences infirmières

## 数据字段

本 JSON 文件**只包含官方字段**, 不含任何派生或人工加工的内容:

| 字段 | 说明 |
|---|---|
| `code_section` | 官方 2 位数 section 代码 |
| `code_groupe` | 所属 groupe 罗马数字编号 |
| `groupe_fr` | Groupe 法语名称 |
| `section_fr` | Section 法语名称 (一字不差来自官方法令) |
| `source` | `general` (Arrêté 2018) 或 `health` (Arrêté 1992) |

## 与 v2 知识库的关系

原项目提供的 `cnu_knowledge_base_v2.json` 是一个**增广版**, 除了官方字段外还包含:

- `section_en`, `groupe_en`, `grande_discipline_en` — 非官方英文翻译 (人工)
- `keywords_en`, `keywords_fr` — 非官方关键词 (AI 生成)
- `text_en`, `text_fr`, `text_bilingual`, `text_fr_rich` — 上述字段的拼接派生

这些字段是为了辅助经典 NLP 方法 (TF-IDF, 关键词匹配) 而加入的. 它们**不是官方数据**.

**本文件 (`cnu_knowledge_base_official.json`) 仅保留官方部分**, 适合在学术报告中作为权威引用.

## 与 v2 的一致性检查

对所有 80 个 section 的 `section_fr` 与 v2 进行逐字对比, 发现:

- ✅ 79 条完全一致
- ⚠️ 1 条存在拼写差异:
  - Section 09: v2 = "Langue et littérature française" (singulier)
  - 官方 = "Langue et littérature **françaises**" (pluriel) — 这是官方法令用法

本文件采用官方写法.

## 许可证

法国政府 Open License 2.0 (Licence Ouverte 2.0), 可自由复用、修改、再分发, 注明出处即可.

<https://github.com/etalab/licence-ouverte/blob/master/LO.md>

## 可复现性

如需重新从官方数据重建 (例如法令更新后), 运行:

```bash
python3 rebuild_official_kb.py
```

此脚本的所有数据直接硬编码自官方 PDF + Légifrance 法令原文,
脚本顶部的文档字符串中标注了每个条目的具体来源 URL.

## 在报告中如何引用

建议在"数据来源"或"实验设置"章节中使用以下表述:

> 本项目使用的 CNU 知识库包含 80 个 sections, 其法语名称和代码完全来自法国政府官方法令:
> (1) Arrêté du 18 décembre 2018 (通用 CNU, 57 sections);
> (2) Arrêté du 29 juin 1992 modifié par Arrêté du 27 juin 2024 (CNU-Santé, 23 sections).
> 数据以 Licence Ouverte 2.0 发布. 详见 `cnu_knowledge_base_official.json`.
