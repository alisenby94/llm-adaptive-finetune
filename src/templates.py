"""
Prompt templates for:
  1. Mastery scoring  — cloze-style completion prompts used to evaluate the
                        pretrained model's factual recall (21 per relation).
  2. SFT training     — the instruction-style Q/A format used during fine-tuning.
  3. CBQA evaluation  — the prompt format used when evaluating the fine-tuned model.

Template design follows Ye et al. (2025) Appendix A/F.1.  The exact mastery-scoring
templates used in the paper are not published, so we design 21 linguistically varied
cloze prefixes per relation.  The SFT / eval formats are inferred from Figure 3 of
the paper, which shows the prompt pattern used during both training and testing.

Cloze templates take a single `{subject}` format argument.
"""

from __future__ import annotations

# ──────────────────────────────────────────────────────────────────────────────
# SFT / Evaluation prompt format
# ──────────────────────────────────────────────────────────────────────────────
# The paper uses a uniform template during training and evaluation (Figure 3).
# We use a simple Question/Answer format that works with any base LLM.

SFT_TEMPLATE = "Question: {question}\nAnswer:"


def make_sft_prompt(question: str) -> str:
    """Return the prompt string used during SFT and CBQA evaluation."""
    return SFT_TEMPLATE.format(question=question.strip())


def make_sft_full(question: str, answer: str) -> str:
    """Return the full prompt+answer string used as the SFT training target."""
    return f"{make_sft_prompt(question)} {answer.strip()}"


# ──────────────────────────────────────────────────────────────────────────────
# Per-relation cloze templates (21 per relation for mastery scoring)
# ──────────────────────────────────────────────────────────────────────────────
# Each template is a Python format string with a single `{subject}` placeholder.
# The pretrained model is expected to *complete* the prefix with the answer entity.

_TEMPLATES_P17 = [  # country
    "{subject} is located in the country of",
    "The country that {subject} is in is",
    "{subject} is in the nation of",
    "Nationally, {subject} belongs to",
    "The sovereign state encompassing {subject} is",
    "{subject} is within the borders of",
    "The country where {subject} is found is",
    "{subject} falls under the jurisdiction of",
    "In terms of national territory, {subject} is part of",
    "Which country? {subject} is in",
    "The nation-state that governs {subject} is",
    "{subject} is a place that belongs to the country called",
    "From a national standpoint, {subject} is in",
    "The political country of {subject} is",
    "As a geographic location, {subject} is in",
    "The homeland country for {subject} is",
    "{subject} is a territory of",
    "The international territory where {subject} sits is",
    "{subject} is geographically located within",
    "Identifying the country: {subject} is in",
    "The recognized country of {subject} is",
]

_TEMPLATES_P19 = [  # place of birth
    "{subject} was born in the city of",
    "The birthplace of {subject} is",
    "{subject}'s place of birth is",
    "Where was {subject} born? In",
    "{subject} first saw the world in",
    "The city where {subject} was born is",
    "{subject} came into the world in",
    "According to records, {subject} was born in",
    "{subject} was born in",
    "The birth city of {subject} is",
    "{subject}'s hometown at birth is",
    "The place of origin of {subject} is",
    "{subject} was a native of",
    "At birth, {subject} was in the city of",
    "Early life: {subject} was born in",
    "The location of {subject}'s birth is",
    "{subject} entered the world in",
    "Birth location for {subject}:",
    "City of birth for {subject} is",
    "{subject} originated from the city",
    "The documented birth location of {subject} is",
]

_TEMPLATES_P20 = [  # place of death
    "{subject} died in the city of",
    "The place of death of {subject} is",
    "{subject}'s death occurred in",
    "Where did {subject} die? In",
    "The city where {subject} passed away is",
    "{subject} passed away in",
    "{subject} died in",
    "Death location of {subject}:",
    "The city of {subject}'s death is",
    "{subject} breathed their last in",
    "The location of {subject}'s death is",
    "{subject} was pronounced dead in",
    "At the time of death, {subject} was in",
    "The place where {subject} died is",
    "According to records, {subject} died in",
    "{subject}'s final city was",
    "{subject} departed this world in",
    "The recorded death city of {subject} is",
    "{subject} met their end in",
    "The city that witnessed the death of {subject} is",
    "Where did {subject} pass away? The answer is",
]

_TEMPLATES_P69 = [  # educated at
    "{subject} was educated at",
    "The school where {subject} studied is",
    "{subject} attended the institution called",
    "The university that {subject} went to is",
    "{subject} received their education at",
    "Where did {subject} study? At",
    "{subject}'s alma mater is",
    "The educational institution attended by {subject} is",
    "{subject} completed their studies at",
    "The college {subject} enrolled in is",
    "{subject} trained at the institution known as",
    "For higher education, {subject} attended",
    "{subject} holds a degree from",
    "The academic institution associated with {subject} is",
    "Education history: {subject} attended",
    "{subject} graduated from",
    "The institution where {subject} earned their degree is",
    "{subject} studied at the renowned institution of",
    "Academically, {subject} is associated with",
    "The place of learning for {subject} is",
    "The documented educational institution of {subject} is",
]

_TEMPLATES_P276 = [  # location / venue
    "{subject} took place at",
    "The venue of {subject} is",
    "{subject} is located at",
    "The location of {subject} is",
    "{subject} happens at the place called",
    "The site of {subject} is",
    "{subject} is held at",
    "Where does {subject} take place? At",
    "The physical location of {subject} is",
    "{subject} is situated at",
    "The venue that hosts {subject} is",
    "{subject} can be found at",
    "The place associated with {subject} is",
    "Location of {subject}:",
    "{subject} occurs at",
    "The facility where {subject} is located is",
    "{subject} is based at",
    "The specific venue of {subject} is",
    "Geographically, {subject} is at",
    "The documented location of {subject} is",
    "The establishment known as {subject} is at",
]

_TEMPLATES_P36 = [  # capital
    "The capital of {subject} is",
    "The capital city of {subject} is",
    "{subject}'s capital is",
    "{subject} has its capital in",
    "The seat of government of {subject} is",
    "The official capital of {subject} is",
    "The city that serves as the capital of {subject} is",
    "The primary administrative center of {subject} is",
    "The governmental center of {subject} is",
    "The capital municipality of {subject} is",
    "The chief administrative city of {subject} is",
    "When asked about {subject}'s capital, the answer is",
    "The political capital of {subject} is",
    "The national capital of {subject} is",
    "{subject} is administered from its capital, which is",
    "The major capital city recognized for {subject} is",
    "In geography, the capital of {subject} is",
    "{subject} designates as its capital the city of",
    "The head city of the territory {subject} is",
    "The government of {subject} is seated in",
    "Politically speaking, {subject}'s capital city is",
]

_TEMPLATES_P131 = [  # located in administrative territorial entity
    "{subject} is located in",
    "{subject} can be found in the administrative region of",
    "{subject} is situated in",
    "{subject} is part of",
    "Geographically, {subject} lies in",
    "{subject} is a place within",
    "The region containing {subject} is",
    "{subject} is in the administrative district of",
    "Administratively, {subject} belongs to",
    "{subject} falls within the boundaries of",
    "{subject} is positioned in the territory of",
    "The administrative territory containing {subject} is",
    "{subject} is administratively part of",
    "The larger administrative entity containing {subject} is",
    "{subject} is a municipality in",
    "The district where {subject} is found is",
    "{subject} is classified as being in",
    "In terms of geographic placement, {subject} is in",
    "{subject}, the place, is located within the territory of",
    "The containing administrative unit of {subject} is",
    "{subject} falls under the administrative jurisdiction of",
]

_TEMPLATES_P159 = [  # headquarters location
    "The headquarters of {subject} is located in",
    "{subject}'s headquarters is in",
    "{subject} is headquartered in",
    "The main office of {subject} is in",
    "{subject} has its headquarters in the city of",
    "The principal office location of {subject} is",
    "{subject} runs its operations from",
    "Corporate headquarters for {subject} is based in",
    "The central offices of {subject} are located in",
    "{subject} is based out of",
    "Where is {subject} headquartered? In",
    "The registered base of {subject} is",
    "The operational headquarters of {subject} is in",
    "{subject}'s home base is",
    "The primary offices of {subject} are in",
    "The city that hosts the headquarters of {subject} is",
    "{subject} maintains its headquarters at",
    "HQ of {subject}:",
    "The nerve center of {subject} is located in",
    "{subject} is officially based in",
    "The official location of {subject}'s headquarters is",
]

_TEMPLATES_P495 = [  # country of origin
    "{subject} was developed in the country of",
    "The country of origin of {subject} is",
    "{subject} originated from",
    "{subject} was created in",
    "The origin country of {subject} is",
    "{subject} comes from the country of",
    "Country of origin for {subject}:",
    "{subject} was produced in",
    "The nation that produced {subject} is",
    "{subject} was made in",
    "The home country for {subject} is",
    "The national origin of {subject} is",
    "{subject} has its origins in",
    "As a cultural product, {subject} comes from",
    "The country responsible for creating {subject} is",
    "{subject} was developed by people in",
    "Origin country of {subject}:",
    "The production country of {subject} is",
    "In which country was {subject} produced?",
    "{subject} arose from the country of",
    "The country that gave rise to {subject} is",
]

_TEMPLATES_P740 = [  # location of formation
    "{subject} was formed in",
    "The city where {subject} was founded is",
    "{subject} was established in",
    "{subject} was created in the city of",
    "Formation location of {subject}:",
    "{subject} originated in the city of",
    "The founding location of {subject} is",
    "Where was {subject} formed? In",
    "{subject} began in the city of",
    "The place where {subject} got started is",
    "{subject} was founded in",
    "The city of formation for {subject} is",
    "According to records, {subject} was formed in",
    "{subject} traces its roots back to",
    "The location where {subject} was officially formed is",
    "{subject} first appeared in",
    "Founding city of {subject}:",
    "{subject} was brought together in the city of",
    "The geographic origin of {subject} is the city of",
    "{subject} was organized in",
    "The initial formation location of {subject} was",
]


_RELATION_TEMPLATES: dict[str, list[str]] = {
    "P17":  _TEMPLATES_P17,
    "P19":  _TEMPLATES_P19,
    "P20":  _TEMPLATES_P20,
    "P36":  _TEMPLATES_P36,
    "P69":  _TEMPLATES_P69,
    "P131": _TEMPLATES_P131,
    "P159": _TEMPLATES_P159,
    "P276": _TEMPLATES_P276,
    "P495": _TEMPLATES_P495,
    "P740": _TEMPLATES_P740,
}

# Generic fallback templates used when a relation is not in the table above.
_GENERIC_TEMPLATES = [
    "The answer related to {subject} is",
    "{subject} is associated with",
    "Regarding {subject}, the relevant fact is",
    "The key information about {subject} is",
    "For {subject}, the answer is",
    "Information about {subject}:",
    "{subject} corresponds to",
    "A fact about {subject} is",
    "Completing the sentence about {subject}:",
    "The relevant detail for {subject} is",
    "{subject} relates to",
    "When considering {subject}, we find",
    "The factual answer for {subject} is",
    "{subject} — the answer:",
    "Looking up {subject}:",
    "The data point about {subject} is",
    "Tell me about {subject}:",
    "What do we know about {subject}? The answer is",
    "The fact concerning {subject} is",
    "The entity linked to {subject} is",
    "For the question involving {subject}, the answer is",
]


def get_cloze_templates(relation: str, n: int = 21) -> list[str]:
    """
    Return exactly n cloze templates for the given Wikidata relation ID.

    If the relation has fewer than n defined templates, generic templates
    are appended to reach the desired count.  If more than n are defined,
    the first n are returned.
    """
    base = list(_RELATION_TEMPLATES.get(relation, []))
    if len(base) < n:
        needed = n - len(base)
        base = base + _GENERIC_TEMPLATES[:needed]
    return base[:n]
