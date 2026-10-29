"""BC5CDR: MeSH 2026 dictionaries and the CDR corpus (disease and chemical).

Descriptors are assigned by tree number (C and F03 for disease, D for
chemical), supplementary records by SCRClass. Each record keeps its MeSH
term types as name categories.
"""

import xml.etree.ElementTree as ET
from pathlib import Path

from pcde_el.data.annotations import (
    SPLITS,
    filter_and_dedup,
    iter_pubtator_entities,
    load_dictionary_ids,
    normalize_space,
    split_concept_ids,
)

DISEASE_TREE_ROOTS = ("C", "F03")  # F03: mental disorders
CHEMICAL_TREE_ROOTS = ("D",)
SCR_CLASS = {"chemical": "1", "disease": "3"}  # 2 (protein) and 4 (organism) are excluded

LEXICAL_TAG_CATEGORY = {
    "ABB": "abbreviation",
    "ABX": "abbreviation",
    "ACR": "acronym",
    "ACX": "acronym",
    "TRD": "trade_name",
    "NAM": "proper_name",
    "EPO": "spelling_variant",
    "FRN": "foreign_term",
}
# Lower wins when one label string is produced with several categories.
CATEGORY_PRIORITY = {
    "record_preferred_term": 0,
    "concept_preferred_term": 1,
    "permuted_term": 2,
    "abbreviation": 3,
    "acronym": 3,
    "trade_name": 3,
    "proper_name": 3,
    "spelling_variant": 3,
    "foreign_term": 3,
    "synonym": 4,
}

# The corpus was annotated against an older MeSH. These ids were merged into
# or replaced by the 2026 records below; each mapping was checked by hand.
CHEMICAL_LEGACY_TO_CURRENT_ID = {
    "MESH:C076029": "MESH:D000077152",
    "MESH:C400082": "MESH:D000069286",
    "MESH:C105934": "MESH:D000068579",
    "MESH:C081489": "MESH:D000068756",
    "MESH:C056507": "MESH:D000093542",
    "MESH:C069541": "MESH:D000069348",
    "MESH:C026098": "MESH:D000077287",
    "MESH:C030852": "MESH:D000077235",
    "MESH:C542870": "MESH:D019274",
    "MESH:C476513": "MESH:D000077554",
    "MESH:C035133": "MESH:D000078785",
    "MESH:C096012": "MESH:D000097629",
    "MESH:C107135": "MESH:D000068338",
    "MESH:C061870": "MESH:D000077123",
    "MESH:C043211": "MESH:D000077261",
    "MESH:C418563": "MESH:D000068698",
    "MESH:C118667": "MESH:D000077764",
    "MESH:C041359": "MESH:D000077291",
    "MESH:C098010": "MESH:D000069349",
    "MESH:C086816": "MESH:D000069468",
    "MESH:C065757": "MESH:D000077239",
    "MESH:C071741": "MESH:D000077208",
    "MESH:C055162": "MESH:D000077144",
    "MESH:C467567": "MESH:D000077269",
    "MESH:C065180": "MESH:D000077340",
    "MESH:C039726": "MESH:D000077332",
    "MESH:C051890": "MESH:D000077146",
    "MESH:C067431": "MESH:D000077289",
    "MESH:C019248": "MESH:D000077268",
    "MESH:C012052": "MESH:D000077582",
    "MESH:C036006": "MESH:D000078330",
    "MESH:C476756": "MESH:C429014",
    "MESH:C096918": "MESH:D000068698",
    "MESH:C043265": "MESH:D000078142",
    "MESH:C049430": "MESH:D000077590",
    "MESH:C034930": "MESH:D000077489",
    "MESH:C000873": "MESH:D000077555",
    "MESH:C010845": "MESH:D000077410",
    "MESH:C037663": "MESH:D000077212",
    "MESH:C064276": "MESH:D000077402",
    "MESH:C422649": "MESH:D000077613",
    "MESH:C065179": "MESH:D000069059",
    "MESH:C047781": "MESH:D000077213",
    "MESH:C045645": "MESH:D000077407",
    "MESH:C471405": "MESH:D000077157",
    "MESH:C047047": "MESH:D000077465",
    "MESH:C005435": "MESH:D000077553",
    "MESH:C059262": "MESH:D000077404",
    "MESH:C099041": "MESH:D000068737",
    "MESH:C047426": "MESH:D000069470",
    "MESH:C102006": "MESH:D000077559",
    "MESH:C052342": "MESH:D000077236",
    "MESH:C063008": "MESH:D009173",
    "MESH:C084178": "MESH:D000077333",
    "MESH:C110904": "MESH:D000069287",
    "MESH:C090450": "MESH:D000077384",
    "MESH:C009250": "MESH:D000077149",
    "MESH:C022189": "MESH:D000078305",
    "MESH:C476217": "MESH:D000069449",
    "MESH:C040029": "MESH:D000077206",
    "MESH:C048833": "MESH:D000077408",
    "MESH:C492458": "MESH:D000069499",
    "MESH:C067311": "MESH:D000077143",
    "MESH:C413408": "MESH:D000069446",
    "MESH:C032302": "MESH:D000077769",
    "MESH:C108606": "MESH:D000069438",
    "MESH:C094645": "MESH:D000068180",
    "MESH:C422923": "MESH:D000068718",
    "MESH:C081309": "MESH:D000077405",
    "MESH:C473478": "MESH:D000077210",
    "MESH:C032171": "MESH:C495469",
    "MESH:C076946": "MESH:D000077265",
    "MESH:C044650": "MESH:D000077728",
    "MESH:C085143": "MESH:D000077725",
    "MESH:D004070": "MESH:D004071",
    "MESH:C008281": "MESH:D000077222",
    "MESH:C014347": "MESH:D000077209",
}
DISEASE_LEGACY_TO_CURRENT_ID = {
    "MESH:C537630": "MESH:D000080365",
    "MESH:C562750": "MESH:D005902",
    "MESH:C531616": "MESH:D000075363",
    "MESH:C562386": "MESH:D000075222",
    "MESH:C531767": "MESH:D010211",
    "MESH:C562388": "MESH:D000082882",
    "MESH:C562729": "MESH:D000077277",
}
# Corpus ids that fall outside the target subset; their annotations are dropped.
CHEMICAL_EXCLUDED_UNMAPPABLE_IDS = {
    "MESH:D002188": "Plant genus (Cannabaceae), not a chemical.",
    "MESH:D001806": "Blood urea nitrogen is a measurement, not a chemical.",
    "MESH:C085788": "SCRClass 2 (protein), outside the chemical subset.",
    "MESH:C064227": "SCRClass 2 (protein), outside the chemical subset.",
    "MESH:C053519": "SCRClass 2 (protein), outside the chemical subset.",
    "MESH:C034868": "SCRClass 2 (protein), outside the chemical subset.",
    "MESH:D002952": "All tree numbers outside the D tree.",
}
DISEASE_EXCLUDED_UNMAPPABLE_IDS = {
    "MESH:D060845": "All tree numbers outside the disease trees.",
}
MISSING_ID_VALUES = frozenset({"", "-", "-1"})  # how the corpus marks "no id"


def _text(element: ET.Element, path: str) -> str:
    child = element.find(path)
    return normalize_space(child.text) if child is not None else ""


def _under(tree_number: str, root: str) -> bool:
    # "C" covers C01, C02...; "F03" covers F03 and F03.xxx but not F031.
    if len(root) == 1 and root.isalpha():
        return tree_number.startswith(root)
    return tree_number == root or tree_number.startswith(f"{root}.")


def _classify(term: ET.Element) -> str:
    """Name category of a MeSH term, from its flags and lexical tag."""
    yes =lambda attr: str(term.attrib.get(attr)).strip().upper() == "Y"
    if yes("RecordPreferredTermYN"):
        return "record_preferred_term"
    if yes("IsPermutedTermYN"):
        return "permuted_term"
    if yes("ConceptPreferredTermYN"):
        return "concept_preferred_term"
    return LEXICAL_TAG_CATEGORY.get(
        normalize_space(term.attrib.get("LexicalTag")), "synonym"
    )


def _record_rows(
    record: ET.Element, pid: str, record_name: str
) -> list[tuple[str, str, str]]:
    """One (name, pid, category) row per case-insensitively unique name.

    Names keep their first-seen order; a name listed under several
    categories gets the one with the highest priority.
    """
    labels: list[tuple[str, str]] = []
    if record_name:
        labels.append((record_name, "record_preferred_term"))
    for term in record.findall("./ConceptList/Concept/TermList/Term"):
        label = _text(term, "./String")
        if label:
            labels.append((label, _classify(term)))
    # Without a record name, promote the preferred term (or the first term).
    if not record_name:
        pref = next((l for l, c in labels if c == "record_preferred_term"), None)
        if pref is None and labels:
            pref = labels[0][0]
        if pref is not None:
            labels.insert(0, (pref, "record_preferred_term"))

    order, display, category = [], {}, {}
    for label, cat in labels:
        key = label.casefold()
        if key not in category:
            order.append(key)
            display[key] = label
            category[key] = cat
        elif CATEGORY_PRIORITY[cat] < CATEGORY_PRIORITY[category[key]]:
            category[key] = cat
    return [(display[k], pid, category[k]) for k in order]


def build_d0(desc_xml: Path, supp_xml: Path) -> dict[str, list[tuple[str, str, str]]]:
    """Returns {"disease": rows, "chemical": rows}, descriptors before
    supplementary records. A descriptor in both trees goes to both."""
    out: dict[str, list[tuple[str, str, str]]] = {"disease": [], "chemical": []}

    # iterparse + clear() keeps memory flat; the MeSH XML files are several hundred MB.
    for _, el in ET.iterparse(desc_xml, events=("end",)):
        if el.tag != "DescriptorRecord":
            continue
        trees = [
            normalize_space(t.text) for t in el.findall("./TreeNumberList/TreeNumber")
        ]
        is_disease = any(_under(t, r) for t in trees for r in DISEASE_TREE_ROOTS)
        is_chemical = any(_under(t, r) for t in trees for r in CHEMICAL_TREE_ROOTS)
        if is_disease or is_chemical:
            pid = f"MESH:{_text(el, './DescriptorUI')}"
            rows = _record_rows(el, pid, _text(el, "./DescriptorName/String"))
            if is_disease:
                out["disease"].extend(rows)
            if is_chemical:
                out["chemical"].extend(rows)
        el.clear()

    for _, el in ET.iterparse(supp_xml, events=("end",)):
        if el.tag != "SupplementalRecord":
            continue
        scr_class = normalize_space(el.attrib.get("SCRClass"))
        for entity, cls in SCR_CLASS.items():
            if scr_class == cls:
                pid = f"MESH:{_text(el, './SupplementalRecordUI')}"
                out[entity].extend(
                    _record_rows(el, pid, _text(el, "./SupplementalRecordName/String"))
                )
        el.clear()
    return out


def build_test(
    entity: str, d0_csv: Path, split_files: dict[str, Path]
) -> list[tuple[str, str, str]]:
    """(mention, gold_pid, split) rows for entity "disease" or "chemical"."""
    legacy = (
        CHEMICAL_LEGACY_TO_CURRENT_ID
        if entity == "chemical"
        else DISEASE_LEGACY_TO_CURRENT_ID
    )
    excluded = (
        CHEMICAL_EXCLUDED_UNMAPPABLE_IDS
        if entity == "chemical"
        else DISEASE_EXCLUDED_UNMAPPABLE_IDS
    )
    valid = load_dictionary_ids(d0_csv)
    out = []
    for split in SPLITS:
        annos = [
            (m, split_concept_ids(raw, MISSING_ID_VALUES))
            for m, etype, raw, _ in iter_pubtator_entities(split_files[split])
            if etype.casefold() == entity
        ]
        rows, stats = filter_and_dedup(
            annos, valid_ids=valid, legacy_to_current=legacy, excluded_ids=excluded
        )
        print(f"  [{entity}/{split}] {dict(stats)}")
        out.extend((m, gold, split) for m, gold in rows)
    return out
