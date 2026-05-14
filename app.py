"""Promise Writer Agent — Streamlit single-page UI.

The operator pastes everything they know about the offer, target and proof into
a big context textarea, optionally drops some structural references, picks how
many headlines to generate, and the agent returns Hormozi-style promises that
can be regenerated one-by-one with feedback.
"""
from __future__ import annotations

import os
import traceback

import streamlit as st
from dotenv import load_dotenv

from dataclasses import asdict

from agent.promise import (
    MAX_HEADLINES,
    MIN_HEADLINES,
    Promise,
    regenerate_field,
    regenerate_one,
    write_promises,
)
from agent.store import BriefRow, BriefStore


# Etichette per i bottoni "rigenera campo X"
FIELD_BUTTONS: list[tuple[str, str, str]] = [
    ("pre_headline", "PRE", "Rigenera solo la PRE-headline (qualifica del target)"),
    ("usp_name", "USP", "Rigenera solo il nome U.S.P. (etichetta-marchio)"),
    ("headline", "HEAD", "Rigenera solo la headline (la promessa)"),
    ("sub_headline", "SUB", "Rigenera solo la sub-headline (anti-obiezione)"),
]


# ── Config ─────────────────────────────────────────────────────────
load_dotenv()


def _secret(key: str, default: str = "") -> str:
    """Read from env first, then st.secrets (Streamlit Cloud)."""
    val = os.getenv(key)
    if val:
        return val
    try:
        return st.secrets.get(key, default)
    except (FileNotFoundError, AttributeError):
        return default


ANTHROPIC_API_KEY = _secret("ANTHROPIC_API_KEY")
APP_PASSWORD = _secret("APP_PASSWORD")

st.set_page_config(page_title="Promise Writer Agent", layout="wide", page_icon="🪄")


# ── Password gate ──────────────────────────────────────────────────
def _password_gate() -> None:
    if not APP_PASSWORD:
        return
    if st.session_state.get("authed"):
        return
    st.title("Promise Writer Agent")
    pw = st.text_input("Password", type="password", key="pw_input")
    if st.button("Enter"):
        if pw == APP_PASSWORD:
            st.session_state.authed = True
            st.rerun()
        else:
            st.error("Wrong password")
    st.stop()


_password_gate()


# ── State init ─────────────────────────────────────────────────────
DEFAULT_STATE: dict[str, object] = {
    "promises": None,
    "last_inputs": None,
    "error": None,
    "loaded_brief_id": None,    # id Supabase del brief attualmente in editing
    "prefill": None,            # dict per pre-popolare i campi del form al rerun
    "info": None,               # messaggio info verde (es. brief salvato)
    "promise_history": None,    # list[list[Promise]]: storia versioni precedenti per ogni promessa (per undo)
}
for k, v in DEFAULT_STATE.items():
    if k not in st.session_state:
        st.session_state[k] = v


# ── Store ──────────────────────────────────────────────────────────
def _store() -> BriefStore | None:
    if "_brief_store" not in st.session_state:
        try:
            st.session_state._brief_store = BriefStore.from_env()
        except Exception:
            st.session_state._brief_store = None
    return st.session_state._brief_store


def _promise_to_dict(p: Promise) -> dict:
    return {
        "pre_headline": p.pre_headline,
        "usp_name": p.usp_name,
        "headline": p.headline,
        "sub_headline": p.sub_headline,
        "structure": p.structure,
        "levers": list(p.levers or ()),
        "rationale": p.rationale,
    }


def _dict_to_promise(d: dict) -> Promise:
    return Promise(
        pre_headline=d.get("pre_headline", ""),
        usp_name=d.get("usp_name", ""),
        headline=d.get("headline", ""),
        sub_headline=d.get("sub_headline", ""),
        structure=d.get("structure", ""),
        levers=tuple(d.get("levers") or ()),
        rationale=d.get("rationale", ""),
    )


def _show_error_if_any() -> None:
    err = st.session_state.get("error")
    if err:
        st.error(err)


def _show_info_if_any() -> None:
    info = st.session_state.get("info")
    if info:
        st.success(info)
        st.session_state.info = None


def _autosave_brief(*, brief: dict, promises: list[Promise]) -> None:
    """Salva (insert o update) il brief in Supabase. Silenzioso se store non disponibile."""
    store = _store()
    if not store:
        return
    payload_promises = [_promise_to_dict(p) for p in promises]
    try:
        if st.session_state.get("loaded_brief_id"):
            row = store.update(st.session_state.loaded_brief_id, brief, payload_promises)
            st.session_state.info = f"Brief aggiornato in archivio (`{row.id[:8]}…`)."
        else:
            row = store.insert(brief, payload_promises)
            st.session_state.loaded_brief_id = row.id
            st.session_state.info = f"Brief salvato in archivio (`{row.id[:8]}…`)."
    except Exception as e:
        st.session_state.info = f"⚠️ Brief generato ma archivio non aggiornato: {e}"


# ── Sidebar: defaults you reuse across runs ────────────────────────
def _sidebar() -> dict[str, str]:
    st.sidebar.header("⚙️ Setup")
    if not ANTHROPIC_API_KEY:
        st.sidebar.error(
            "Manca `ANTHROPIC_API_KEY`. Settala in `.env` (locale) o in "
            "Streamlit Cloud → Settings → Secrets."
        )

    # Se c'e` un brief appena caricato dall'archivio, ricarico target/voice
    # dal prefill: cancello le keys dei widget cosi` Streamlit rigenera
    # i widget col value= dal prefill (Streamlit usa value= solo se la key
    # non e` ancora in session_state).
    target_default = st.session_state.get("_sb_target", "")
    voice_default = st.session_state.get("_sb_voice", "")
    if st.session_state.get("_consume_sidebar_prefill"):
        pre = st.session_state.get("prefill") or {}
        target_default = pre.get("target_audience", "")
        voice_default = pre.get("brand_voice", "")
        st.session_state.pop("_sb_target", None)
        st.session_state.pop("_sb_voice", None)
        st.session_state._consume_sidebar_prefill = False

    target_audience = st.sidebar.text_area(
        "Target audience (1 frase)",
        value=target_default,
        placeholder="Es. coach 1-1 che vendono percorsi 1.500-3.000€ a imprenditori 35-55",
        height=80,
        key="_sb_target",
        help="Tono diretto e specifico: piu il target e definito, piu le promesse sono mirate.",
    )
    brand_voice = st.sidebar.text_area(
        "Brand voice (1 frase)",
        value=voice_default,
        placeholder="Es. diretto, pragmatico, italiano semplice, no anglicismi",
        height=80,
        key="_sb_voice",
        help="Entra nel system prompt: regola il registro delle promesse.",
    )

    if st.sidebar.button("🔄 Reset session", use_container_width=True):
        for k in DEFAULT_STATE:
            st.session_state[k] = DEFAULT_STATE[k]
        st.rerun()

    _sidebar_archive()

    return {
        "target_audience": (target_audience or "").strip(),
        "brand_voice": (brand_voice or "").strip(),
    }


def _sidebar_archive() -> None:
    """Lista brief precedenti con load/duplicate/delete."""
    store = _store()
    st.sidebar.divider()
    st.sidebar.header("📚 Archivio brief")
    if not store:
        st.sidebar.caption(
            "Archivio disabilitato: mancano `SUPABASE_URL` e `SUPABASE_SECRET_KEY` "
            "nei secrets. Aggiungile per abilitarlo."
        )
        return

    try:
        rows = store.list_recent(limit=50)
    except Exception as e:
        st.sidebar.error(f"Errore archivio: {e}")
        return

    if not rows:
        st.sidebar.caption("_Nessun brief salvato ancora._")
        return

    if st.session_state.loaded_brief_id:
        st.sidebar.caption(f"📌 Editing brief `{st.session_state.loaded_brief_id[:8]}…`")
        if st.sidebar.button("🆕 Esci editing (nuovo brief)", use_container_width=True):
            st.session_state.loaded_brief_id = None
            st.session_state.promises = None
            st.session_state.last_inputs = None
            st.session_state.prefill = None
            st.rerun()

    for row in rows:
        with st.sidebar.container(border=True):
            label = row.title or "(senza titolo)"
            st.markdown(f"**{label[:60]}**")
            st.caption(
                f"{(row.updated_at or row.created_at)[:16].replace('T', ' ')} · "
                f"{len(row.promises)} promesse"
            )
            c1, c2, c3 = st.columns([1, 1, 1])
            if c1.button("📂", key=f"open_{row.id}", help="Apri / modifica"):
                _load_brief_into_form(row)
                st.rerun()
            if c2.button("📄", key=f"dup_{row.id}", help="Duplica come nuovo brief"):
                _load_brief_into_form(row, as_duplicate=True)
                st.rerun()
            if c3.button("🗑️", key=f"del_{row.id}", help="Elimina"):
                try:
                    store.delete(row.id)
                    if st.session_state.loaded_brief_id == row.id:
                        st.session_state.loaded_brief_id = None
                        st.session_state.promises = None
                        st.session_state.last_inputs = None
                    st.session_state.info = f"Brief `{label[:40]}…` eliminato."
                    st.rerun()
                except Exception as e:
                    st.sidebar.error(f"Delete fallito: {e}")


def _load_brief_into_form(row: BriefRow, as_duplicate: bool = False) -> None:
    brief = row.brief or {}
    st.session_state.loaded_brief_id = None if as_duplicate else row.id
    st.session_state.prefill = {
        "context": brief.get("context", ""),
        "references": brief.get("references", ""),
        "target_audience": brief.get("target_audience", ""),
        "brand_voice": brief.get("brand_voice", ""),
        "n_headlines": int(brief.get("n_headlines", 10) or 10),
        "extra_instructions": brief.get("extra_instructions", ""),
    }
    # Carico anche le promesse pregresse cosi` l'utente le rivede subito
    if row.promises and not as_duplicate:
        st.session_state.promises = [_dict_to_promise(p) for p in row.promises]
        st.session_state.promise_history = [[] for _ in row.promises]
        st.session_state.last_inputs = {
            k: brief.get(k, "")
            for k in ("context", "references", "target_audience", "brand_voice")
        }
    else:
        st.session_state.promises = None
        st.session_state.promise_history = None
        st.session_state.last_inputs = None
    # Segnalo a _sidebar di ri-applicare target/voice dal prefill al prossimo run
    # (non posso settare direttamente _sb_target qui: i widget sono gia` instanziati).
    st.session_state._consume_sidebar_prefill = True
    st.session_state.info = (
        f"Brief `{row.title[:40]}` " + ("duplicato come nuovo." if as_duplicate else "caricato.")
    )


# ── Main page ──────────────────────────────────────────────────────
def _main(sidebar: dict[str, str]) -> None:
    st.title("🪄 Promise Writer Agent")
    st.caption(
        "Copywriter specializzato in **promesse Hormozi-style**, sempre strutturate a "
        "**quattro livelli**:\n\n"
        "**1. Pre-headline** (qualifica il target) → "
        "**2. Nome U.S.P.** (etichetta-marchio breve e memorabile tipo *LIBERI COL MATTONE* o "
        "*IMPRENDITOR.I.A.*) → "
        "**3. Promessa** (esplode il beneficio) → "
        "**4. Sub-headline** (corta, solo anti-obiezione). "
        "Tutti e 4 sempre presenti."
    )

    if st.session_state.promises is None:
        _input_form(sidebar)
    else:
        _output_panel(sidebar)


def _input_form(sidebar: dict[str, str]) -> None:
    pre = st.session_state.get("prefill") or {}
    is_editing = bool(st.session_state.get("loaded_brief_id"))
    if is_editing:
        st.info(
            f"📌 Stai modificando un brief esistente (`{st.session_state.loaded_brief_id[:8]}…`). "
            "Al prossimo **Genera promesse** verra` aggiornato in archivio."
        )
    elif pre:
        st.info("📄 Brief caricato dall'archivio come **nuovo** (duplicato). Verra` creato un nuovo record.")
    with st.form("promise_form"):
        context = st.text_area(
            "📥 Context — tutto quello che sai sull'offerta, target e prove",
            value=pre.get("context", ""),
            height=320,
            placeholder=(
                "Carica qui TUTTO quello che hai. Piu dai, meglio scrive. Mappa di cosa "
                "alimenta cosa nei 4 livelli:\n\n"
                "→ ALIMENTA LA PRE-HEADLINE (chi e il target, in che situazione):\n"
                "  • Chi e il prospect (eta, ruolo, contesto, livello di awareness)\n"
                "  • Cosa ha gia provato e fallito\n"
                "  • Frustrazione attuale (le parole esatte che usa)\n\n"
                "→ ALIMENTA IL NOME U.S.P. (etichetta-marchio del metodo):\n"
                "  • Parole chiave del settore, mondo del prospect (es. mattone, agenda, ads)\n"
                "  • Concetti opposti o paradossi che vorresti esprimere (es. 'libero ma stabile')\n"
                "  • Eventuali nomi a cui sei gia affezionato (li uso come riferimento o ne propongo altri)\n\n"
                "→ ALIMENTA LA HEADLINE (la promessa nuda):\n"
                "  • Cosa vendi (prodotto/servizio in 2-3 righe)\n"
                "  • Dream outcome (numeri, KPI, sensazioni concrete)\n"
                "  • Tempo realistico per il risultato\n\n"
                "→ ALIMENTA LA SUB-HEADLINE (anti-obiezione corta):\n"
                "  • Obiezioni tipiche del prospect quando legge promesse cosi\n"
                "    (es. 'sara per chi parte gia avanti', 'serve essere tecnici',\n"
                "    'mi tocca fare chiamate fredde', 'costera un sacco')\n\n"
                "→ VINCOLI / CONTESTO:\n"
                "  • Cosa NON dire (claim non sostenibili, competitor da non citare)\n"
                "  • Eventuali prove/case study/testimonianze (informativi, non vanno in sub)"
            ),
            help=(
                "Questo blob va dritto nel prompt. Non riassumere — meglio rumoroso ma completo "
                "che pulito ma povero. L'agente filtra lui cosa usare."
            ),
        )
        references = st.text_area(
            "📚 Reference — strutture o esempi headline che il copywriter deve studiare",
            value=pre.get("references", ""),
            height=180,
            placeholder=(
                "Opzionale. Esempi:\n"
                "• 'Aiutiamo i coach 1-1 a fare 10K/mese in 90 giorni, senza ads e senza chiamate fredde'\n"
                "• 'Da 0 a 5K follower IG in 30 giorni — anche se parti senza audience'\n"
                "• Pattern: '[Numero] [outcome] in [tempo] o [garanzia]'\n"
                "Piu pattern dai, piu varia il ritmo delle promesse generate."
            ),
            help="Il modello impara cadenza e ritmo da queste reference. Niente plagio: solo struttura.",
        )

        cols = st.columns([1, 1, 2])
        n_headlines = cols[0].slider(
            "Quante promesse?",
            min_value=MIN_HEADLINES,
            max_value=MAX_HEADLINES,
            value=int(pre.get("n_headlines", 10) or 10),
            help=f"Default {MIN_HEADLINES}. Sopra le 15 il modello varia di piu ma puo ripetersi.",
        )
        extra_instructions = cols[2].text_input(
            "Indicazioni extra (opzionale)",
            value=pre.get("extra_instructions", ""),
            placeholder="Es. 'evita garanzie monetarie', 'mantieni sotto i 90 caratteri'",
        )

        submitted = st.form_submit_button("✨ Genera promesse", type="primary", use_container_width=True)

    if submitted:
        if not context.strip():
            st.error("Devi compilare almeno il **Context**.")
            return
        if not ANTHROPIC_API_KEY:
            st.error("Manca `ANTHROPIC_API_KEY`.")
            return
        with st.spinner(f"Lo Scrittore di Promesse sta lavorando ({n_headlines} headline)…"):
            try:
                promises = write_promises(
                    api_key=ANTHROPIC_API_KEY,
                    context=context,
                    references=references,
                    target_audience=sidebar["target_audience"],
                    brand_voice=sidebar["brand_voice"],
                    n_headlines=n_headlines,
                    extra_instructions=extra_instructions,
                )
                st.session_state.promises = promises
                # Reset history: ogni promessa parte senza versioni precedenti
                st.session_state.promise_history = [[] for _ in promises]
                st.session_state.last_inputs = {
                    "context": context,
                    "references": references,
                    "target_audience": sidebar["target_audience"],
                    "brand_voice": sidebar["brand_voice"],
                }
                # Auto-save in archivio Supabase (se configurato).
                _autosave_brief(
                    brief={
                        "context": context,
                        "references": references,
                        "target_audience": sidebar["target_audience"],
                        "brand_voice": sidebar["brand_voice"],
                        "n_headlines": n_headlines,
                        "extra_instructions": extra_instructions,
                    },
                    promises=promises,
                )
                # Il prefill ha finito il suo lavoro
                st.session_state.prefill = None
                st.rerun()
            except ValueError as e:
                st.session_state.error = f"Errore: {e}"
            except Exception as e:
                st.session_state.error = (
                    f"Generazione fallita: {e}\n\n{traceback.format_exc()}"
                )


def _push_history(i: int, snapshot: Promise) -> None:
    """Salva una versione precedente di promises[i] nella history (per undo)."""
    if st.session_state.get("promise_history") is None:
        st.session_state.promise_history = [[] for _ in (st.session_state.promises or [])]
    h: list[Promise] = st.session_state.promise_history[i]
    h.append(snapshot)
    # Cap a 10 versioni per promessa
    if len(h) > 10:
        h.pop(0)


def _undo_promise(i: int) -> bool:
    """Pop l'ultima versione e la rimette come corrente. False se nulla da annullare."""
    hist = st.session_state.get("promise_history")
    if not hist:
        return False
    h = hist[i]
    if not h:
        return False
    st.session_state.promises[i] = h.pop()
    return True


def _persist_promises() -> None:
    """Aggiorna il record Supabase se siamo in modalita` editing."""
    if not st.session_state.get("loaded_brief_id"):
        return
    store = _store()
    if not store:
        return
    try:
        store.update(
            st.session_state.loaded_brief_id,
            st.session_state.last_inputs or {},
            [_promise_to_dict(x) for x in st.session_state.promises],
        )
    except Exception:
        pass  # silenzioso: la sessione ha gia` il valore corretto


def _output_panel(sidebar: dict[str, str]) -> None:
    promises: list[Promise] = st.session_state.promises
    last = st.session_state.last_inputs or {}

    st.success(f"Generate **{len(promises)}** promesse. Copia quelle che ti servono o rigenera per campo.")

    cols = st.columns([1, 1, 4])
    if cols[0].button("⬅️ Nuovo brief"):
        st.session_state.promises = None
        st.session_state.last_inputs = None
        st.session_state.loaded_brief_id = None
        st.session_state.prefill = None
        st.rerun()
    if cols[1].button("🔁 Rigenera tutte"):
        # mantieni il prefill cosi` il form ricarica gli stessi input
        st.session_state.prefill = {
            "context": last.get("context", ""),
            "references": last.get("references", ""),
            "target_audience": last.get("target_audience", ""),
            "brand_voice": last.get("brand_voice", ""),
        }
        st.session_state.promises = None
        st.rerun()

    st.divider()

    for i, p in enumerate(promises):
        with st.container(border=True):
            st.markdown(f"**Promessa #{i + 1}**")
            if p.pre_headline:
                st.caption(f"_{p.pre_headline}_")
            if p.usp_name:
                st.markdown(
                    f"<div style='font-size:1.5rem; font-weight:800; "
                    f"letter-spacing:0.05em; color:#16a34a; "
                    f"margin: 0.2rem 0 0.4rem 0;'>"
                    f"{p.usp_name}</div>",
                    unsafe_allow_html=True,
                )
            st.markdown(f"### {p.headline}")
            if p.sub_headline:
                st.markdown(f"_{p.sub_headline}_")
            st.markdown("")  # spacing
            meta_cols = st.columns([2, 3])
            with meta_cols[0]:
                st.caption(f"**Struttura**: {p.structure or '—'}")
            with meta_cols[1]:
                if p.levers:
                    st.caption("**Leve**: " + ", ".join(f"`{lev}`" for lev in p.levers))
            if p.rationale:
                with st.expander("Perche dovrebbe funzionare"):
                    st.markdown(p.rationale)

            # ── Controls: per-field regen + undo + rigenera intera ──────
            hist_i: list[Promise] = (
                st.session_state.promise_history[i]
                if st.session_state.get("promise_history") is not None
                else []
            )
            feedback_key = f"fb_{i}"
            feedback = st.text_input(
                "Feedback (opzionale, applicato alla prossima rigenerazione)",
                placeholder="Es. 'piu` corto', 'usa il pain literal', 'evita la garanzia'",
                key=feedback_key,
            )
            ctrl_cols = st.columns([1, 1, 1, 1, 1, 1.5])
            # 4 bottoni per-campo
            for col_idx, (field, label, tooltip) in enumerate(FIELD_BUTTONS):
                if ctrl_cols[col_idx].button(
                    f"🔁 {label}",
                    key=f"regen_{field}_{i}",
                    help=tooltip,
                    use_container_width=True,
                ):
                    with st.spinner(f"Rigenero {label}…"):
                        try:
                            new_p = regenerate_field(
                                api_key=ANTHROPIC_API_KEY,
                                original=p,
                                field=field,
                                feedback=feedback or "",
                                context=last.get("context", ""),
                                references=last.get("references", ""),
                                target_audience=last.get("target_audience", ""),
                                brand_voice=last.get("brand_voice", ""),
                            )
                            _push_history(i, p)
                            st.session_state.promises[i] = new_p
                            _persist_promises()
                            st.rerun()
                        except Exception as e:
                            st.error(f"Rigenerazione {label} fallita: {e}")
            # Undo
            if ctrl_cols[4].button(
                "↶ Undo",
                key=f"undo_{i}",
                disabled=not hist_i,
                help=(f"Torna alla versione precedente ({len(hist_i)} disponibili)" if hist_i else "Nessuna versione precedente"),
                use_container_width=True,
            ):
                _undo_promise(i)
                _persist_promises()
                st.rerun()
            # Rigenera intera promessa (vecchio comportamento)
            if ctrl_cols[5].button(
                "🔄 Tutta",
                key=f"regen_all_{i}",
                disabled=not (feedback or "").strip(),
                help="Rigenera tutti e 4 i livelli usando il feedback. Richiede feedback non vuoto.",
                use_container_width=True,
            ):
                with st.spinner("Rigenerazione intera…"):
                    try:
                        new_p = regenerate_one(
                            api_key=ANTHROPIC_API_KEY,
                            original=p,
                            feedback=feedback,
                            context=last.get("context", ""),
                            references=last.get("references", ""),
                            target_audience=last.get("target_audience", ""),
                            brand_voice=last.get("brand_voice", ""),
                        )
                        _push_history(i, p)
                        st.session_state.promises[i] = new_p
                        _persist_promises()
                        st.rerun()
                    except Exception as e:
                        st.error(f"Rigenerazione fallita: {e}")
            if hist_i:
                st.caption(f"🕘 {len(hist_i)} versioni precedenti in memoria (undo disponibile).")


# ── Render ────────────────────────────────────────────────────────
sidebar_state = _sidebar()
_show_info_if_any()
_show_error_if_any()
_main(sidebar_state)
