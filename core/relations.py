"""Research-paper relation vocabulary, wording map, and fail-closed quote gate."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import re

_NON_ALPHANUMERIC = re.compile(r"[\W_]+", re.UNICODE)
_RELATION_NAME = re.compile(r"^[A-Z][A-Z0-9_]{0,63}$")
# Clause/closed-class tokens that must not edge a concept noun phrase.
_ENDPOINT_EDGE_WORDS = frozenset(
    {
        "that",
        "where",
        "since",
        "such",
        "but",
        "note",
        "their",
        "little",
    }
)
# Lone function words rejected as entire endpoints (generic closed class).
_LONE_FUNCTION_WORDS = frozenset(
    {
        "a",
        "an",
        "the",
        "of",
        "in",
        "on",
        "at",
        "to",
        "for",
        "by",
        "as",
        "and",
        "or",
        "but",
        "if",
        "then",
        "than",
        "that",
        "this",
        "these",
        "those",
        "it",
        "its",
        "they",
        "them",
        "their",
        "we",
        "you",
        "he",
        "she",
        "his",
        "her",
        "our",
        "your",
        "with",
        "from",
        "into",
        "onto",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "do",
        "does",
        "did",
        "not",
        "no",
        "nor",
        "so",
        "such",
        "where",
        "when",
        "while",
        "since",
        "because",
        "although",
        "though",
        "which",
        "who",
        "whom",
        "what",
        "how",
        "why",
        "also",
        "very",
        "more",
        "less",
        "most",
        "least",
        "many",
        "much",
        "some",
        "any",
        "all",
        "each",
        "every",
        "both",
        "either",
        "neither",
        "other",
        "another",
        "same",
        "own",
        "note",
        "little",
        "one",
        "two",
        "up",
        "out",
        "about",
        "over",
        "after",
        "before",
        "between",
        "under",
        "again",
        "further",
        "once",
        "here",
        "there",
        "thus",
        "therefore",
        "however",
        "moreover",
        "overall",
        "respectively",
    }
)
# HAS_FEATURE target must end with a technical-component head (kinds of parts).
_FEATURE_TARGET_HEADS = frozenset(
    {
        "connection",
        "connections",
        "encoding",
        "encodings",
        "layer",
        "layers",
        "head",
        "heads",
        "module",
        "modules",
        "parameter",
        "parameters",
        "weight",
        "weights",
        "attention",
        "bias",
        "biases",
        "residual",
        "residuals",
        "embedding",
        "embeddings",
        "unit",
        "units",
        "block",
        "blocks",
        "network",
        "networks",
        "mechanism",
        "mechanisms",
        "activation",
        "activations",
        "normalization",
        "normalisation",
        "normalizations",
        "normalisations",
        "dropout",
        "dropouts",
        "regularization",
        "regularisation",
        "regularizations",
        "regularisations",
        "initialization",
        "initialisation",
        "initializations",
        "initialisations",
        "projection",
        "projections",
        "dimension",
        "dimensions",
    }
)
# ACHIEVES target must be result-like, not a vague "point".
_RESULT_TARGET_HEADS = frozenset(
    {
        "performance",
        "accuracy",
        "result",
        "results",
        "score",
        "scores",
        "sota",
        "gain",
        "gains",
        "improvement",
        "improvements",
        "quality",
        "f1",
        "bleu",
        "rouge",
        "perplexity",
        "loss",
        "error",
        "errors",
        "precision",
        "recall",
        "auc",
        "metric",
        "metrics",
        "benchmark",
        "benchmarks",
        "state",  # state of the art (normalized tokens)
        "art",
    }
)
_CURRENCY_OR_NUMERIC = re.compile(
    r"^(?:[\d.,]+(?:\s*(?:k|m|b|%|percent))?|[$€£¥]\s*[\d.,]+|[\d.,]+\s*[$€£¥])$",
    re.IGNORECASE,
)

CANONICAL_RELATIONS = frozenset(
    {
        "PART_OF",
        "PREREQUISITE_OF",
        "DESCRIBES",
        "RELATES_TO",
        "USES",
        "EVALUATED_ON",
        "TRAINED_ON",
        "BASED_ON",
        "PROPOSES",
        "OUTPERFORMS",
        "COMPARES_TO",
        "ACHIEVES",
        "REQUIRES",
        "APPLIED_TO",
        "IMPROVES",
        "ENABLES",
        "PRODUCES",
        "HAS_FEATURE",
    }
)

# Surface / near-synonym labels collapse onto the canonical table.
RELATION_ALIASES = {
    "CONSISTS_OF": "PART_OF",
    "COMPOSED_OF": "PART_OF",
    "MADE_UP_OF": "PART_OF",
    "USED_FOR": "USES",
    "EMPLOYS": "USES",
    "UTILIZES": "USES",
    "LEVERAGES": "USES",
    "RELIES_ON": "USES",
    "EVALUATE_FOR": "EVALUATED_ON",
    "TESTED_ON": "EVALUATED_ON",
    "TESTED_AGAINST": "EVALUATED_ON",
    "BUILDS_ON": "BASED_ON",
    "EXTENDS": "BASED_ON",
    "EXTENDED": "BASED_ON",
    "INTRODUCES": "PROPOSES",
    "PRESENTS": "PROPOSES",
    "BEATS": "OUTPERFORMS",
    "SURPASSES": "OUTPERFORMS",
    "COMPARE": "COMPARES_TO",
    "VERSUS": "COMPARES_TO",
    "COMPARED_TO": "COMPARES_TO",
    "COMPARED_WITH": "COMPARES_TO",
    "NEEDS": "REQUIRES",
    "DEPENDS_ON": "REQUIRES",
    "APPLIES_TO": "APPLIED_TO",
    "ENHANCES": "IMPROVES",
    "ALLOWS": "ENABLES",
    "GENERATES": "PRODUCES",
    "YIELDS": "PRODUCES",
    "FEATURE_OF": "HAS_FEATURE",
    "HAS": "HAS_FEATURE",
    "EXHIBITS": "HAS_FEATURE",
    "RELATED_TO": "RELATES_TO",
}

RELATION_DEFINITIONS = {
    "PART_OF": "A is a component, part, layer, module, or element of B; B contains or consists of A",
    "PREREQUISITE_OF": "the source says understanding A is required before understanding B",
    "DESCRIBES": "A explains, defines, or describes B",
    "RELATES_TO": "the source explicitly says A and B are related (not composition, use, or evaluation)",
    "USES": "A uses, employs, utilizes, or relies on B",
    "EVALUATED_ON": "A is evaluated or tested on B",
    "TRAINED_ON": "A is trained on B",
    "BASED_ON": "A is based on, builds on, or extends B",
    "PROPOSES": "A proposes, introduces, or presents B",
    "OUTPERFORMS": "A outperforms, beats, or surpasses B",
    "COMPARES_TO": "A is compared to, with, or against B",
    "ACHIEVES": "A achieves, obtains, or reports B",
    "REQUIRES": "A requires or needs B as a technical dependency, not a learning prerequisite",
    "APPLIED_TO": "A is applied to B",
    "IMPROVES": "A improves or enhances B",
    "ENABLES": "A enables or allows B",
    "PRODUCES": "A produces, generates, or yields B",
    "HAS_FEATURE": "A has or exhibits B",
}

# Connector words that must not be absorbed into a local-scan noun phrase.
RELATION_CUE_WORDS = (
    "uses",
    "used",
    "employs",
    "utilizes",
    "leverages",
    "relies",
    "relied",
    "evaluated",
    "tested",
    "trained",
    "trains",
    "based",
    "builds",
    "built",
    "extends",
    "extended",
    "proposes",
    "propose",
    "proposed",
    "introduces",
    "introduce",
    "introduced",
    "presents",
    "present",
    "presented",
    "outperforms",
    "outperform",
    "outperformed",
    "beats",
    "beat",
    "surpasses",
    "surpassed",
    "compared",
    "versus",
    "vs",
    "achieves",
    "achieve",
    "achieved",
    "obtains",
    "obtain",
    "obtained",
    "reaches",
    "reached",
    "reports",
    "reported",
    "requires",
    "require",
    "required",
    "needs",
    "needed",
    "depends",
    "applied",
    "applies",
    "improves",
    "improve",
    "improved",
    "enhances",
    "enhance",
    "enhanced",
    "enables",
    "enable",
    "enabled",
    "allows",
    "allow",
    "allowed",
    "produces",
    "produce",
    "produced",
    "generates",
    "generate",
    "generated",
    "yields",
    "yielded",
    "has",
    "have",
    "had",
    "exhibits",
    "exhibit",
    "exhibited",
    "powering",
    "related",
    "relationship",
    "relation",
    "directly",
    "closely",
    "explains",
    "defines",
    "describes",
    "explained",
    "defined",
    "described",
    "contains",
    "includes",
    "comprises",
    "consists",
    "composed",
    "made",
    "forms",
    "form",
    "understanding",
    "learning",
    "understand",
    "learn",
    "prerequisite",
    "necessary",
)


@dataclass(frozen=True)
class RelationWording:
    """One surface grammar mapped onto a canonical relation.

    ``template`` uses ``{src}``, ``{tgt}``, and optional ``{art}``.
    ``composed_slot`` selects the looser noun-phrase matcher for local scan.
    """

    relation: str
    template: str
    composed_slot: str | None = None


# More specific grammars first so they win over broader ones (e.g. PREREQUISITE
# before REQUIRES).
RELATION_WORDING: tuple[RelationWording, ...] = (
    RelationWording(
        "PART_OF",
        r"{src}\s+(?:is|are|was|were)\s+(?:an?\s+|the\s+|one\s+of\s+the\s+)?"
        r"(?:components?|parts?|layers?|modules?|elements?|subcomponents?|"
        r"constituents?|subunits?)\s+(?:of|in|within)\s+{tgt}",
    ),
    RelationWording(
        "PART_OF",
        r"{src}\s+(?:forms?|form)\s+(?:an?\s+)?part\s+of\s+{tgt}",
    ),
    RelationWording(
        "PART_OF",
        r"{tgt}\s+(?:contains|includes|comprises)\s+{art}{src}",
    ),
    RelationWording(
        "PART_OF",
        r"{tgt}\s+(?:consists\s+of|is\s+composed\s+of|is\s+made\s+up\s+of)\s+"
        r"{art}{src}",
        composed_slot="src",
    ),
    RelationWording(
        "PART_OF",
        r"(?:(?:an?|the|its|their)\s+)?(?:key\s+)?(?:core\s+)?"
        r"components?\s+(?:of|powering)\s+{tgt}\s+"
        r"(?:is|are|include|includes|including)\s+{art}{src}",
    ),
    RelationWording(
        "PART_OF",
        r"{src}\s+(?:is|are)\s+(?:an?\s+|the\s+)?(?:key\s+)?"
        r"components?\s+(?:of|powering)\s+{tgt}",
    ),
    RelationWording(
        "PREREQUISITE_OF",
        r"(?:understanding|learning)(?:\s+of)?\s+{src}\s+(?:is\s+)?"
        r"(?:an?\s+)?(?:prerequisite|required|necessary)\s+(?:before|for)\s+"
        r"(?:understanding|learning)(?:\s+of)?\s+{tgt}",
    ),
    RelationWording(
        "PREREQUISITE_OF",
        r"(?:understanding|learning)(?:\s+of)?\s+{tgt}\s+"
        r"(?:requires|needs)\s+(?:understanding|learning)(?:\s+of)?\s+{src}",
    ),
    RelationWording(
        "PREREQUISITE_OF",
        r"before\s+(?:understanding|learning)(?:\s+of)?\s+{tgt}\s+"
        r"(?:one|you|we)\s+(?:must|need\s+to)\s+(?:understand|learn)\s+{src}",
    ),
    RelationWording(
        "DESCRIBES",
        r"{src}\s+(?:explains|defines|describes)\s+(?:how\s+)?{tgt}",
    ),
    RelationWording(
        "DESCRIBES",
        r"{tgt}\s+(?:is|are)\s+(?:explained|defined|described)\s+by\s+{src}",
    ),
    RelationWording(
        "RELATES_TO",
        r"{src}\s+(?:(?:is|are|was|were)\s+)?(?:(?:directly|closely)\s+)?"
        r"related\s+to\s+{tgt}",
    ),
    RelationWording(
        "RELATES_TO",
        r"{src}\s+and\s+{tgt}\s+(?:are|is)\s+related",
    ),
    RelationWording(
        "RELATES_TO",
        r"(?:relationship|relation)\s+between\s+{src}\s+and\s+{tgt}",
    ),
    RelationWording(
        "USES",
        r"{src}\s+(?:uses|employs|utilizes|leverages)\s+{art}{tgt}",
    ),
    RelationWording(
        "USES",
        r"{src}\s+(?:relies|relied)\s+on\s+{art}{tgt}",
    ),
    RelationWording(
        "USES",
        r"{tgt}\s+(?:is|are|was|were)\s+used\s+(?:by|in)\s+{art}{src}",
    ),
    RelationWording(
        "EVALUATED_ON",
        r"{src}\s+(?:is|are|was|were)\s+evaluated\s+(?:on|against)\s+{art}{tgt}",
    ),
    RelationWording(
        "EVALUATED_ON",
        r"{src}\s+(?:is|are|was|were)\s+tested\s+(?:on|against)\s+{art}{tgt}",
    ),
    RelationWording(
        "TRAINED_ON",
        r"{src}\s+(?:is|are|was|were)\s+trained\s+on\s+{art}{tgt}",
    ),
    RelationWording(
        "TRAINED_ON",
        r"{src}\s+(?:trains|trained)\s+on\s+{art}{tgt}",
    ),
    RelationWording(
        "BASED_ON",
        r"{src}\s+(?:is|are|was|were)\s+based\s+on\s+{art}{tgt}",
    ),
    RelationWording(
        "BASED_ON",
        r"{src}\s+(?:builds?|built)\s+on\s+{art}{tgt}",
    ),
    RelationWording(
        "BASED_ON",
        r"{src}\s+(?:extends?|extended)\s+{art}{tgt}",
    ),
    RelationWording(
        "PROPOSES",
        r"{src}\s+(?:proposes?|proposed|introduces?|introduced|presents?|"
        r"presented)\s+{art}{tgt}",
    ),
    RelationWording(
        "OUTPERFORMS",
        r"{src}\s+(?:outperforms?|outperformed|beats?|beat|surpasses?|"
        r"surpassed)\s+{art}{tgt}",
    ),
    RelationWording(
        "COMPARES_TO",
        r"{src}\s+(?:is|are|was|were)\s+compared\s+(?:to|with|against)\s+"
        r"{art}{tgt}",
    ),
    RelationWording(
        "COMPARES_TO",
        r"{src}\s+(?:versus|vs\.?)\s+{art}{tgt}",
    ),
    # ACHIEVES: result-like target only (not vague "point"); gate filters target head.
    RelationWording(
        "ACHIEVES",
        r"{src}\s+(?:achieves?|achieved|obtains?|obtained|reaches?|reached|"
        r"reports?|reported)\s+(?!(?:a\s+|an\s+|the\s+)?point\b){art}{tgt}",
    ),
    # REQUIRES: block less/more/that as the immediate object.
    RelationWording(
        "REQUIRES",
        r"{src}\s+(?:requires?|required|needs?|needed)\s+"
        r"(?!(?:less|more|that)\b){art}{tgt}",
    ),
    RelationWording(
        "REQUIRES",
        r"{src}\s+depends\s+on\s+(?!(?:less|more|that)\b){art}{tgt}",
    ),
    RelationWording(
        "APPLIED_TO",
        r"{src}\s+(?:is|are|was|were)\s+applied\s+to\s+{art}{tgt}",
    ),
    RelationWording(
        "APPLIED_TO",
        r"{src}\s+applies\s+to\s+{art}{tgt}",
    ),
    RelationWording(
        "IMPROVES",
        r"{src}\s+(?:improves?|improved|enhances?|enhanced)\s+{art}{tgt}",
    ),
    # ENABLES: do not treat a bare "that"-clause as the target object.
    RelationWording(
        "ENABLES",
        r"{src}\s+(?:enables?|enabled|allows?|allowed)\s+(?!that\b){art}{tgt}",
    ),
    RelationWording(
        "PRODUCES",
        r"{src}\s+(?:produces?|produced|generates?|generated|yields?|yielded)"
        r"\s+{art}{tgt}",
    ),
    # HAS_FEATURE: technical component possession; reject little/advantage/seen.
    RelationWording(
        "HAS_FEATURE",
        r"{src}\s+(?:has|have|had)\s+"
        r"(?!(?:little|seen)\b)(?!(?:an?\s+|the\s+)?advantage\b){art}{tgt}",
    ),
    RelationWording(
        "HAS_FEATURE",
        r"{src}\s+(?:exhibits?|exhibited)\s+"
        r"(?!(?:little|seen)\b)(?!(?:an?\s+|the\s+)?advantage\b){art}{tgt}",
    ),
)

_QUOTE_ARTICLE = (
    r"(?:(?:an?|the|each|every|this|that|these|those|any|some|all)\s+)?"
)


def is_well_formed_relation(value: str) -> bool:
    return bool(_RELATION_NAME.fullmatch(value))


def normalize_relation(value: Any) -> str:
    """Return a canonical or well-formed novel relation; empty if unusable."""
    raw = str(value or "").strip().upper().replace(" ", "_").replace("-", "_")
    raw = re.sub(r"_+", "_", raw).strip("_")
    if not is_well_formed_relation(raw):
        return ""
    return RELATION_ALIASES.get(raw, raw)


def normalized_phrase(value: Any) -> str:
    """Normalize model terms and source text for conservative phrase matching."""
    return " ".join(_NON_ALPHANUMERIC.sub(" ", str(value).casefold()).split())


def is_concept_endpoint(value: Any) -> bool:
    """True when value is a concise concept noun phrase after normalization.

    Rejects clause tails, closed-class edges, and lone function words. Generic:
    does not inspect paper names, titles, authors, or expected quotes.
    """
    phrase = normalized_phrase(value)
    if not phrase:
        return False
    tokens = phrase.split()
    if len(tokens) == 1 and tokens[0] in _LONE_FUNCTION_WORDS:
        return False
    if tokens[0] in _ENDPOINT_EDGE_WORDS or tokens[-1] in _ENDPOINT_EDGE_WORDS:
        return False
    # Clause fragments that end mid-subordinator (that/where/such).
    if tokens[-1] in {"that", "where", "such"}:
        return False
    return True


def _is_feature_like_target(value: Any) -> bool:
    """HAS_FEATURE targets must be technical-component heads, not money or idiom."""
    phrase = normalized_phrase(value)
    if not phrase:
        return False
    if _CURRENCY_OR_NUMERIC.fullmatch(phrase.replace(" ", "")):
        return False
    tokens = phrase.split()
    if not tokens:
        return False
    # Reject non-technical idiom objects even if they slipped past the template.
    if tokens[0] in {"little", "seen", "advantage"} or tokens[-1] in {
        "bearing",
        "advantage",
        "advantages",
    }:
        return False
    return tokens[-1] in _FEATURE_TARGET_HEADS


def _is_result_like_target(value: Any) -> bool:
    """ACHIEVES targets must be result-like nouns, not a vague 'point'."""
    phrase = normalized_phrase(value)
    if not phrase:
        return False
    tokens = phrase.split()
    if not tokens or tokens[-1] == "point" or tokens == ["point"]:
        return False
    return any(token in _RESULT_TARGET_HEADS for token in tokens)


def evidence_term(value: str) -> str:
    """Match a normalized endpoint without allowing it to absorb surrounding words."""
    return rf"(?<!\w)(?:an? |the )?{re.escape(normalized_phrase(value))}(?!\w)"


def _template_group_order(template: str) -> tuple[int, int]:
    src_pos = template.find("{src}")
    tgt_pos = template.find("{tgt}")
    if src_pos < tgt_pos:
        return 1, 2
    return 2, 1


def compile_local_pair_patterns(
    noun_phrase: str,
    optional_article: str,
    composed_part: str,
) -> list[tuple[re.Pattern[str], str, int, int]]:
    """Compile the wording table against the caller's noun-phrase macros."""
    compiled: list[tuple[re.Pattern[str], str, int, int]] = []
    for item in RELATION_WORDING:
        src = composed_part if item.composed_slot == "src" else noun_phrase
        tgt = composed_part if item.composed_slot == "tgt" else noun_phrase
        pattern = item.template.format(src=src, tgt=tgt, art=optional_article)
        source_group, target_group = _template_group_order(item.template)
        compiled.append(
            (
                re.compile(pattern, re.IGNORECASE),
                item.relation,
                source_group,
                target_group,
            )
        )
    return compiled


def _novel_relation_patterns(
    relation: str, source_term: str, target_term: str
) -> list[str]:
    tokens = [token for token in relation.lower().split("_") if token]
    if not tokens:
        return []
    spaced = r"\s+".join(re.escape(token) for token in tokens)
    joined = re.escape("".join(tokens))
    predicate = rf"(?:{spaced}|{joined})"
    optional_be = r"(?:(?:is|are|was|were)\s+)?"
    prep = r"(?:by\s+|with\s+|from\s+|on\s+|to\s+)?"
    return [
        rf"{source_term}\s+{optional_be}{predicate}\s+{prep}{target_term}",
        rf"{target_term}\s+{optional_be}{predicate}\s+{prep}{source_term}",
    ]


def quote_supports_relation(
    quote: str, source: str, relation: Any, target: str
) -> bool:
    """Fail closed unless the quote directly asserts the proposed edge."""
    normalized = normalize_relation(relation)
    if not normalized:
        return False
    if not is_concept_endpoint(source) or not is_concept_endpoint(target):
        return False
    if normalized == "HAS_FEATURE" and not _is_feature_like_target(target):
        return False
    if normalized == "ACHIEVES" and not _is_result_like_target(target):
        return False
    if normalized == "REQUIRES":
        target_tokens = normalized_phrase(target).split()
        if target_tokens and target_tokens[0] in {"less", "more", "that"}:
            return False
    source_term = evidence_term(source)
    target_term = evidence_term(target)
    relation_quote = normalized_phrase(
        re.sub(r"[.!?;:]+", " relationboundary ", quote)
    )
    if not relation_quote:
        return False

    if normalized in CANONICAL_RELATIONS:
        patterns = [
            item.template.format(src=source_term, tgt=target_term, art=_QUOTE_ARTICLE)
            for item in RELATION_WORDING
            if item.relation == normalized
        ]
        # RELATES_TO is non-directional: accept either endpoint order.
        if normalized == "RELATES_TO":
            patterns.extend(
                item.template.format(
                    src=target_term, tgt=source_term, art=_QUOTE_ARTICLE
                )
                for item in RELATION_WORDING
                if item.relation == "RELATES_TO"
            )
    else:
        patterns = _novel_relation_patterns(normalized, source_term, target_term)
    return any(re.search(pattern, relation_quote) for pattern in patterns)


def preferred_relation_prompt_block() -> str:
    """Prompt fragment listing preferred relations and the novel-relation fallback."""
    lines = [
        "- Prefer one of these research relations when the source wording matches:"
    ]
    for name in (
        "PART_OF",
        "PREREQUISITE_OF",
        "DESCRIBES",
        "RELATES_TO",
        "USES",
        "EVALUATED_ON",
        "TRAINED_ON",
        "BASED_ON",
        "PROPOSES",
        "OUTPERFORMS",
        "COMPARES_TO",
        "ACHIEVES",
        "REQUIRES",
        "APPLIED_TO",
        "IMPROVES",
        "ENABLES",
        "PRODUCES",
        "HAS_FEATURE",
    ):
        lines.append(f"  {name}: {RELATION_DEFINITIONS[name]}.")
    lines.extend(
        [
            "- Map source wording onto the closest preferred relation. Examples: "
            '"System A uses technique B" is USES, not PART_OF; "The model is '
            'evaluated on a benchmark" is EVALUATED_ON, not RELATES_TO; '
            '"The approach is based on self-attention" is BASED_ON; '
            '"Quantization is applied to the weights" is APPLIED_TO; '
            '"The model has positional encodings" is HAS_FEATURE; '
            '"Technique B is a layer of System A" is PART_OF.',
            "- For PART_OF, the edge direction is always part to whole. When the "
            "source says a whole contains, includes, consists of, comprises, is "
            "composed of, or is powered by a part, emit the part as source and "
            "the whole as target.",
            "- A PREREQUISITE_OF B means the source explicitly says understanding A "
            "is required before understanding B; architectural reliance is not a "
            "learning prerequisite. RELATES_TO is only for an explicit "
            "non-directional conceptual connection; it does not imply direction, "
            "precedence, or composition.",
            "- If none of the preferred relations match the source predicate, you "
            "MAY emit one new UPPER_SNAKE_CASE relation named after that "
            "predicate (for example FINE_TUNES). Do not relabel an unmatched or "
            "unsupported relation as RELATES_TO.",
            "- Usage, reliance, evaluation, application, or possession never "
            "establish PART_OF, PREREQUISITE_OF, DESCRIBES, or RELATES_TO. The "
            "same source must independently satisfy the matching rule for the "
            "emitted relation.",
            '- Hard exclusion example: "System A uses technique B" does not '
            "support technique B PART_OF System A. Only an explicit whole-part "
            'statement such as "Technique B is a layer of System A" supports '
            "that PART_OF edge.",
        ]
    )
    return "\n".join(lines)


def verifier_relation_prompt_block() -> str:
    """Prompt fragment for approving an already-proposed relation."""
    return """
- Approve a candidate only when its resolved evidence span explicitly supports its
  existing relation and direction and names both endpoints.
- For a preferred relation, the span must match that relation's wording (for
  example USES requires uses/employs/relies on; EVALUATED_ON requires evaluated
  or tested on; PART_OF requires an explicit whole-part assertion). For
  PREREQUISITE_OF, the span must explicitly state that understanding A is required
  before understanding B; architectural reliance is not a learning prerequisite. For
  PART_OF, it must explicitly identify A as a component, part, layer, module, or element
  of B. For DESCRIBES, it must explicitly say A explains, defines, or describes B.
  RELATES_TO requires an explicit non-directional conceptual relationship and does not
  imply precedence, composition, or direction.
- Usage, reliance, architectural basis, addition, application, possession, capability,
  property, or evaluation must be approved only as USES, BASED_ON, APPLIED_TO,
  HAS_FEATURE, or EVALUATED_ON. Those statements do not by themselves support
  PART_OF, PREREQUISITE_OF, DESCRIBES, or RELATES_TO.
- Hard exclusion example: candidate technique B PART_OF System A with evidence "System A
  uses technique B." MUST be omitted; it has no whole-part assertion. Candidate technique
  B PART_OF System A with evidence "Technique B is a layer of System A." MAY be approved.
  Candidate System A USES technique B with evidence "System A uses technique B." MAY be
  approved. Never invent, paraphrase, or replace the resolved evidence span.
- For a novel UPPER_SNAKE_CASE relation that is not in the preferred table, approve it
  only when the span names both endpoints and the relation's words appear as the
  connecting predicate. Do not approve co-occurrence.
""".strip()
