"""
Synonym expansion table used for CBQA evaluation.

Reproduced from Table 12 of Ye et al. (2025)
"Analyzing the Effects of Supervised Fine-Tuning on Model Knowledge
from Token and Parameter Levels", EMNLP 2025.

The table maps canonical entity names to their common aliases so that
evaluation does not penalise correct answers that use alternate forms.
"""

from __future__ import annotations

# Mapping from canonical name → list of acceptable surface forms.
# The canonical name is always included in the list.
SYNONYM_TABLE: dict[str, list[str]] = {
    "United States of America": [
        "United States of America", "USA", "United States",
    ],
    "New York City": [
        "New York City", "New York",
    ],
    "University of Michigan": [
        "University of Michigan", "UMich",
    ],
    "South Korea": [
        "South Korea", "Republic of Korea", "Korea",
    ],
    "Saint Petersburg": [
        "Saint Petersburg", "St. Petersburg",
    ],
    "Buenos Aires": [
        "Buenos Aires", "Baires",
    ],
    "People's Republic of China": [
        "People's Republic of China", "PRC", "China",
    ],
    "Ohio State University": [
        "Ohio State University", "Ohio State",
    ],
    "Bosnia and Herzegovina": [
        "Bosnia and Herzegovina", "Bosnia", "Bosna i Hercegovina",
    ],
    "University of Texas at Austin": [
        "University of Texas at Austin", "University of Texas", "UT Austin",
    ],
    "University of Cambridge": [
        "University of Cambridge", "Cambridge University", "Cambridge",
    ],
    "United States Military Academy": [
        "United States Military Academy", "West Point",
    ],
    "Rio de Janeiro": [
        "Rio de Janeiro", "Rio de",
    ],
    "University of Edinburgh": [
        "University of Edinburgh", "Edinburgh University",
    ],
    "Museo del Prado": [
        "Museo del Prado", "Prado Museum", "Museo Nacional del Prado",
    ],
    "Salt Lake City": [
        "Salt Lake City", "Salt Lake",
    ],
    "North Carolina State University": [
        "North Carolina State University", "NC State",
    ],
    "University of Durham": [
        "University of Durham", "Durham University",
    ],
    "Harvard Law School": [
        "Harvard Law School", "Harvard University",
    ],
    "University of Paris (1896-1968)": [
        "University of Paris (1896-1968)", "Université de Paris",
        "University of Paris", "Paris University",
    ],
    "Newcastle upon Tyne": [
        "Newcastle upon Tyne", "Newcastle",
    ],
    "University of Oslo": [
        "University of Oslo", "Oslo University",
    ],
    "Hebrew University of Jerusalem": [
        "Hebrew University of Jerusalem", "University of Jerusalem",
        "Hebrew University",
    ],
    "Carnegie Mellon University": [
        "Carnegie Mellon University", "Carnegie Mellon",
    ],
    "University of Oxford": [
        "University of Oxford", "Oxford University",
    ],
    "Autodromo Nazionale Monza": [
        "Autodromo Nazionale Monza", "Monza",
    ],
    "Indiana State House": [
        "Indiana State House", "Indiana State",
    ],
    "Imperial College London": [
        "Imperial College London", "Imperial College",
    ],
    "United Arab Emirates": [
        "United Arab Emirates", "UAE",
    ],
}

# Build a flat map from each surface form → set of all synonyms for that entity.
# Used at evaluation time: if a gold answer appears as a key here, we check all
# its synonyms against the model prediction.
_FORM_TO_SYNONYMS: dict[str, list[str]] = {}
for _canonical, _forms in SYNONYM_TABLE.items():
    for _form in _forms:
        _FORM_TO_SYNONYMS[_form.lower()] = _forms


def get_all_synonyms(answers: list[str]) -> list[str]:
    """
    Given a list of gold answer strings, return the expanded set of all
    acceptable surface forms (including synonyms from the paper's table).

    The input answers list may itself contain multiple valid answers from
    the dataset; each is expanded individually and duplicates are removed.
    """
    expanded: set[str] = set()
    for ans in answers:
        expanded.add(ans)
        synonyms = _FORM_TO_SYNONYMS.get(ans.lower())
        if synonyms:
            expanded.update(synonyms)
    return list(expanded)
