"""Promise generation — Claude writes Hormozi-style headline promises.

The agent takes a free-form `context` blob (the more the better: offer, target,
pain, dream outcome, mechanism, proof, constraints) and an optional `references`
blob (structures or example headlines the operator wants the agent to learn from).

Output: a list of `Promise` objects, each with the four-layer structure plus
the rhetorical levers pulled and a short rationale.

Hormozi's Value Equation drives the system prompt:
    Value = (Dream Outcome × Perceived Likelihood) / (Time Delay × Effort & Sacrifice)
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass

from anthropic import Anthropic

CLAUDE_MODEL = "claude-sonnet-4-6"

MIN_HEADLINES = 10
MAX_HEADLINES = 25


@dataclass(frozen=True)
class Promise:
    """A four-layer promise.

    All four layers are MANDATORY. None of them can be empty.

    Roles:
      - `pre_headline`: qualifica il target (a chi parli, in che situazione)
      - `usp_name`:     NOME U.S.P. — etichetta-marchio brevissima, memorabile,
                        spesso ALL CAPS (es. "LIBERI COL MATTONE",
                        "IMPRENDITOR.I.A."). Non e` una descrizione, e` un brand
                        del metodo/offerta.
      - `headline`:     la promessa nuda — esplode il beneficio (outcome, numeri,
                        tempo)
      - `sub_headline`: corta (max ~90 caratteri), serve SOLO a disinnescare le
                        obiezioni che vengono in mente leggendo la promessa.
                        Non aggiunge benefit nuovi, non rispiega la promessa.
    """

    pre_headline: str
    usp_name: str
    headline: str
    sub_headline: str
    structure: str
    levers: tuple[str, ...]
    rationale: str


SYSTEM_PROMPT = (
    "Sei lo headline writer di Alex Hormozi. Scrivi promesse 'Grand Slam Offer' "
    "in italiano, calibrate sulla Value Equation:\n"
    "    Value = (Dream Outcome x Perceived Likelihood) / (Time Delay x Effort & Sacrifice)\n\n"
    "## STRUTTURA OBBLIGATORIA — QUATTRO LIVELLI\n"
    "Ogni promessa e composta da QUATTRO livelli, TUTTI obbligatori, in questo ordine:\n\n"
    "  1. PRE-HEADLINE -> qualifica il TARGET. Chi sta leggendo? In che situazione?\n"
    "     Es. 'A te coach 1-1 che vendi ancora a chiamata fredda',\n"
    "         'Per imprenditori 35-55 che hanno provato Meta Ads e si sono bruciati'.\n\n"
    "  2. USP-NAME -> il NOME U.S.P. dell'offerta. Un'etichetta-marchio brevissima\n"
    "     (1-4 parole, spesso ALL CAPS) che riassume tutta l'offerta in un'espressione\n"
    "     memorabile, come fosse il 'nome commerciale' del metodo. Punta a ritmo,\n"
    "     allitterazione, gioco di parole, doppi sensi, contrasto.\n"
    "     Es. 'LIBERI COL MATTONE', 'IMPRENDITOR.I.A.', 'MOLLA O RADDOPPIA',\n"
    "         'FUORI DAL TEMPO', 'IL MATTONE INTELLIGENTE'.\n"
    "     NON e una descrizione. NON usare punteggiatura standard a meno che non\n"
    "     sia parte di un gioco grafico (come gli intercalari di IMPRENDITOR.I.A.).\n\n"
    "  3. HEADLINE -> la PROMESSA nuda. Esplode il beneficio: numeri, tempo,\n"
    "     outcome specifico. Es. 'Riempi l'agenda di 5 nuovi clienti al mese in 90 giorni'.\n\n"
    "  4. SUB-HEADLINE -> CORTA (target: max ~90 caratteri). Serve SOLO a\n"
    "     disinnescare le obiezioni che vengono in mente subito dopo aver letto\n"
    "     la promessa. NON aggiungere benefit nuovi, NON rispiegare la promessa,\n"
    "     NON elencare meccanismi: togli SOLO il dubbio.\n"
    "     Es. 'Senza ads, senza chiamate fredde', 'Anche se parti da zero',\n"
    "         'Anche se hai gia provato e fallito', 'Anche se non sei tecnico'.\n\n"
    "REGOLA FERREA:\n"
    "  - Tutti e 4 i livelli sono SEMPRE presenti e mai vuoti.\n"
    "  - Se ti manca materiale dal context per uno dei 4, inventalo dentro lo\n"
    "    spirito del context — meglio una promessa completa che un trittico monco.\n\n"
    "## VALUE EQUATION\n"
    "Massimizza numeratore, minimizza denominatore:\n"
    "  - DREAM OUTCOME: risultato specifico e desiderabile (numeri, mai benefit vaghi)\n"
    "  - PERCEIVED LIKELIHOOD: meccanismo, garanzia, prova sociale, specificita\n"
    "  - TIME DELAY: tempo breve e definito ('in 30 giorni', 'entro venerdi')\n"
    "  - EFFORT & SACRIFICE: rimuovi attriti ('senza X', 'anche se Y')\n\n"
    "## PATTERN STRUTTURALI per la HEADLINE (mixa, non ripetere)\n"
    "  1. Aiuto [target] a [outcome] in [tempo], senza [sacrificio]\n"
    "  2. Come [outcome] in [tempo] anche se [obiezione]\n"
    "  3. Da [pain] a [dream] in [N giorni/settimane], garantito\n"
    "  4. [Numero] [unita] in [tempo] o [garanzia]\n"
    "  5. L'unico modo per [outcome] senza [pain comune dell'industria]\n\n"
    "## VIETATO\n"
    "  - verbi vaghi: migliora, ottimizza, scopri, esplora, potenzia\n"
    "  - weasel words: potresti, forse, magari, eventualmente\n"
    "  - feature senza payoff ('pacchetti', 'soluzioni', 'sistemi' generici)\n"
    "  - superlativi non quantificati ('il migliore', 'il piu efficace')\n"
    "  - claim non supportati dal context (mai numeri inventati per la HEADLINE)\n"
    "  - sub-headline lunghe o che ripetono la promessa\n"
    "  - usp-name descrittivi tipo 'METODO MARKETING DIGITALE' — sono morti\n\n"
    "## OBBLIGATORIO\n"
    "  - numeri, date, nomi specifici quando il context li fornisce\n"
    "  - tono diretto 'tu/tuo', mai 'voi' o 'lei'\n"
    "  - parole della lingua del prospect (riprendi i pain literali dal context)\n"
    "  - varieta di leve: non fare 10 promesse tutte basate sul tempo\n"
    "  - varieta di usp-name: non fare 10 nomi tutti con lo stesso pattern\n\n"
    "## OUTPUT\n"
    "Rispondi SOLO con un array JSON, niente prosa, niente markdown fences.\n"
    "Schema di ogni elemento:\n"
    '  {"pre_headline": "stringa NON vuota — qualifica il target",\n'
    '   "usp_name":     "stringa NON vuota — etichetta-marchio breve",\n'
    '   "headline":     "stringa NON vuota — la promessa nuda",\n'
    '   "sub_headline": "stringa NON vuota — anti-obiezione corta",\n'
    '   "structure":    "etichetta del pattern es. Outcome+Tempo+Anti-sacrificio",\n'
    '   "levers":       ["specificity", "time-bound", "objection-removal", ...],\n'
    '   "rationale":    "perche funziona su questo target (max 200 char)"}\n'
)


def _extract_json_array(raw: str) -> list[dict]:
    """Pull the JSON array out of a Claude reply, tolerating optional code fences."""
    raw = raw.strip()
    fence = re.match(r"^```(?:json)?\s*(.*?)\s*```$", raw, re.DOTALL)
    if fence:
        raw = fence.group(1).strip()
    return json.loads(raw)


def _section(label: str, body: str) -> str:
    """Render an optional input section, or empty string if blank."""
    body = (body or "").strip()
    if not body:
        return ""
    return f"\n## {label}\n{body}\n"


def _build_user_prompt(
    *,
    context: str,
    references: str,
    target_audience: str,
    brand_voice: str,
    n_headlines: int,
    extra_instructions: str = "",
) -> str:
    parts: list[str] = []
    parts.append(_section("Target audience", target_audience))
    parts.append(_section("Brand voice", brand_voice))
    parts.append(_section("Context (offerta, dream outcome, pain, meccanismo, prove)", context))
    parts.append(
        _section(
            "Reference (strutture o esempi che il copywriter deve usare come ispirazione)",
            references,
        )
    )
    parts.append(_section("Istruzioni extra", extra_instructions))
    parts.append(
        f"\n## Task\n"
        f"Scrivi esattamente {n_headlines} headline-promesse, ognuna con una leva "
        f"diversa o un pattern strutturale diverso. Ogni promessa deve avere TUTTI "
        f"e 4 i livelli (pre_headline, usp_name, headline, sub_headline) compilati "
        f"e non vuoti. Restituisci solo l'array JSON.\n"
    )
    return "".join(p for p in parts if p)


def _parse_promises(raw_items: list[dict]) -> list[Promise]:
    """Parse JSON items into `Promise` objects.

    All four fields (pre_headline, usp_name, headline, sub_headline) must be
    present and non-blank. Items missing any of the four are filtered out.
    """
    promises: list[Promise] = []
    for item in raw_items:
        pre = str(item.get("pre_headline", "")).strip()
        usp = str(item.get("usp_name", "")).strip()
        headline = str(item.get("headline", "")).strip()
        sub = str(item.get("sub_headline", "")).strip()
        # Strict: every layer must be present.
        if not pre or not usp or not headline or not sub:
            continue
        levers_raw = item.get("levers", []) or []
        if isinstance(levers_raw, str):
            levers_raw = [levers_raw]
        levers = tuple(str(lev).strip() for lev in levers_raw if str(lev).strip())
        promises.append(
            Promise(
                pre_headline=pre,
                usp_name=usp,
                headline=headline,
                sub_headline=sub,
                structure=str(item.get("structure", "")).strip(),
                levers=levers,
                rationale=str(item.get("rationale", "")).strip(),
            )
        )
    return promises


def write_promises(
    *,
    api_key: str,
    context: str,
    references: str = "",
    target_audience: str = "",
    brand_voice: str = "",
    n_headlines: int = 10,
    extra_instructions: str = "",
) -> list[Promise]:
    """Generate `n_headlines` Hormozi-style promises from the operator context.

    Args:
        api_key: Anthropic API key.
        context: Free-form blob describing the offer, target, pain, dream, mechanism,
            proof, constraints. The richer, the better — this is the entire grounding.
        references: Optional blob of structural references or example headlines the
            agent should learn the rhythm from.
        target_audience: One-line audience description (overrides if also in context).
        brand_voice: One-line brand voice description.
        n_headlines: How many headlines to generate. Bounded to [MIN_HEADLINES, MAX_HEADLINES].
        extra_instructions: Optional last-mile steering (e.g. "evita garanzie monetarie").

    Returns:
        A list of `Promise` dataclass instances, one per generated headline.

    Raises:
        ValueError: if `context` is empty or `n_headlines` is out of bounds.
        json.JSONDecodeError: if Claude's reply is not valid JSON.
    """
    if not context.strip():
        raise ValueError("context is required and cannot be empty")
    if n_headlines < MIN_HEADLINES or n_headlines > MAX_HEADLINES:
        raise ValueError(
            f"n_headlines must be in [{MIN_HEADLINES}, {MAX_HEADLINES}], got {n_headlines}"
        )

    user_prompt = _build_user_prompt(
        context=context,
        references=references,
        target_audience=target_audience,
        brand_voice=brand_voice,
        n_headlines=n_headlines,
        extra_instructions=extra_instructions,
    )

    client = Anthropic(api_key=api_key)
    msg = client.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=4000,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_prompt}],
    )
    text = "".join(b.text for b in msg.content if getattr(b, "type", "") == "text")
    return _parse_promises(_extract_json_array(text))


def regenerate_one(
    *,
    api_key: str,
    original: Promise,
    feedback: str,
    context: str,
    references: str = "",
    target_audience: str = "",
    brand_voice: str = "",
) -> Promise:
    """Regenerate a single promise given operator feedback on the original."""
    if not feedback.strip():
        raise ValueError("feedback is required to regenerate a promise")

    original_block = (
        f"  PRE:      {original.pre_headline}\n"
        f"  USP NAME: {original.usp_name}\n"
        f"  HEADLINE: {original.headline}\n"
        f"  SUB:      {original.sub_headline}"
    )

    instructions = (
        "Stai riscrivendo UNA singola promessa-quaternio. Versione originale:\n"
        f"{original_block}\n"
        f"  (struttura: {original.structure}; leve: {', '.join(original.levers)})\n\n"
        "Feedback dell'operatore su cosa cambiare:\n"
        f"  {feedback.strip()}\n\n"
        "Restituisci un array JSON con UN SOLO elemento (la nuova promessa-quaternio). "
        "Rispetta la regola: tutti e 4 i livelli (pre_headline, usp_name, headline, "
        "sub_headline) sono obbligatori e non vuoti."
    )
    user_prompt = _build_user_prompt(
        context=context,
        references=references,
        target_audience=target_audience,
        brand_voice=brand_voice,
        n_headlines=1,
        extra_instructions=instructions,
    )

    client = Anthropic(api_key=api_key)
    msg = client.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=600,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_prompt}],
    )
    text = "".join(b.text for b in msg.content if getattr(b, "type", "") == "text")
    promises = _parse_promises(_extract_json_array(text))
    if not promises:
        raise ValueError("regeneration returned no usable promise")
    return promises[0]
