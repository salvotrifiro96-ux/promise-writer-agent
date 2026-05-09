"""Pure-helper tests for agent.promise — no Anthropic API calls."""
from __future__ import annotations

import json

import pytest

from agent.promise import (
    MAX_HEADLINES,
    MIN_HEADLINES,
    Promise,
    _build_user_prompt,
    _extract_json_array,
    _parse_promises,
    _section,
    write_promises,
)


# ── _section ──────────────────────────────────────────────────────
@pytest.mark.unit
def test_section_returns_empty_for_blank_body():
    assert _section("Label", "") == ""
    assert _section("Label", "   \n  \t ") == ""


@pytest.mark.unit
def test_section_renders_label_and_body():
    out = _section("Target", "coach 1-1")
    assert "## Target" in out
    assert "coach 1-1" in out
    assert out.startswith("\n")
    assert out.endswith("\n")


# ── _extract_json_array ───────────────────────────────────────────
@pytest.mark.unit
def test_extract_json_array_plain():
    raw = '[{"headline": "x"}]'
    assert _extract_json_array(raw) == [{"headline": "x"}]


@pytest.mark.unit
def test_extract_json_array_strips_json_fence():
    raw = '```json\n[{"headline": "x"}]\n```'
    assert _extract_json_array(raw) == [{"headline": "x"}]


@pytest.mark.unit
def test_extract_json_array_strips_bare_fence():
    raw = '```\n[{"headline": "x"}]\n```'
    assert _extract_json_array(raw) == [{"headline": "x"}]


@pytest.mark.unit
def test_extract_json_array_invalid_raises():
    with pytest.raises(json.JSONDecodeError):
        _extract_json_array("not json at all")


# ── _parse_promises ───────────────────────────────────────────────
@pytest.mark.unit
def test_parse_promises_full_trio():
    raw = [
        {
            "pre_headline": "A te coach 1-1 che vendi a chiamata fredda",
            "headline": "Riempi l'agenda di 5 nuovi clienti al mese in 90 giorni",
            "sub_headline": "Senza ads, senza chiamate fredde, o ti rimborso",
            "structure": "PRE+HEADLINE+SUB / Outcome+Tempo+Anti-sacrificio",
            "levers": ["specificity", "time-bound", "objection-removal"],
            "rationale": "Tocca outcome misurabile, tempo e rimozione del pain piu odiato.",
        }
    ]
    out = _parse_promises(raw)
    assert len(out) == 1
    p = out[0]
    assert isinstance(p, Promise)
    assert p.pre_headline.startswith("A te coach")
    assert p.headline.startswith("Riempi")
    assert p.sub_headline.startswith("Senza")
    assert p.structure.startswith("PRE+HEADLINE+SUB")
    assert p.levers == ("specificity", "time-bound", "objection-removal")


@pytest.mark.unit
def test_parse_promises_pre_plus_headline_no_sub():
    raw = [
        {
            "pre_headline": "Per imprenditori 35-55 che si sono bruciati con Meta Ads",
            "headline": "5 lead caldi al giorno con €30 di budget",
            "sub_headline": "",
        }
    ]
    out = _parse_promises(raw)
    assert len(out) == 1
    assert out[0].pre_headline != ""
    assert out[0].sub_headline == ""


@pytest.mark.unit
def test_parse_promises_headline_plus_sub_no_pre():
    raw = [
        {
            "pre_headline": "",
            "headline": "Da 0 a 10K follower in 90 giorni",
            "sub_headline": "Anche se parti senza audience, anche se odi ballare su TikTok",
        }
    ]
    out = _parse_promises(raw)
    assert len(out) == 1
    assert out[0].pre_headline == ""
    assert out[0].sub_headline.startswith("Anche")


@pytest.mark.unit
def test_parse_promises_skips_blank_headline():
    raw = [
        {"pre_headline": "x", "headline": "  ", "sub_headline": "y"},
        {"pre_headline": "x", "headline": "Valida", "sub_headline": "y"},
    ]
    out = _parse_promises(raw)
    assert len(out) == 1
    assert out[0].headline == "Valida"


@pytest.mark.unit
def test_parse_promises_skips_when_only_headline_no_pre_no_sub():
    """The trio rule: at least one of pre/sub must be present, never just headline alone."""
    raw = [
        {"pre_headline": "", "headline": "Solo headline, senza pre ne sub", "sub_headline": ""},
        {"pre_headline": "ok", "headline": "Promessa valida", "sub_headline": ""},
    ]
    out = _parse_promises(raw)
    assert len(out) == 1
    assert out[0].headline == "Promessa valida"


@pytest.mark.unit
def test_parse_promises_coerces_string_levers_to_tuple():
    raw = [
        {
            "pre_headline": "ok",
            "headline": "x",
            "sub_headline": "",
            "levers": "specificity",
        }
    ]
    out = _parse_promises(raw)
    assert out[0].levers == ("specificity",)


@pytest.mark.unit
def test_parse_promises_handles_missing_optional_fields():
    raw = [{"pre_headline": "ok", "headline": "Solo headline + pre"}]
    out = _parse_promises(raw)
    assert out[0].headline == "Solo headline + pre"
    assert out[0].pre_headline == "ok"
    assert out[0].sub_headline == ""
    assert out[0].structure == ""
    assert out[0].levers == ()
    assert out[0].rationale == ""


@pytest.mark.unit
def test_parse_promises_filters_empty_lever_strings():
    raw = [
        {
            "pre_headline": "ok",
            "headline": "x",
            "sub_headline": "",
            "levers": ["valid", "", "  ", "another"],
        }
    ]
    out = _parse_promises(raw)
    assert out[0].levers == ("valid", "another")


@pytest.mark.unit
def test_promise_is_immutable():
    p = Promise(
        pre_headline="a",
        headline="x",
        sub_headline="b",
        structure="y",
        levers=("a",),
        rationale="z",
    )
    with pytest.raises(Exception):
        p.headline = "changed"  # type: ignore[misc]


# ── _build_user_prompt ────────────────────────────────────────────
@pytest.mark.unit
def test_build_user_prompt_includes_required_pieces():
    out = _build_user_prompt(
        context="vendo coaching 1-1 a 2K€",
        references="",
        target_audience="coach 35-55",
        brand_voice="diretto",
        n_headlines=12,
    )
    assert "## Target audience" in out
    assert "coach 35-55" in out
    assert "## Brand voice" in out
    assert "diretto" in out
    assert "## Context" in out
    assert "vendo coaching 1-1" in out
    assert "12 headline-promesse" in out


@pytest.mark.unit
def test_build_user_prompt_omits_blank_optional_sections():
    out = _build_user_prompt(
        context="ctx",
        references="",
        target_audience="",
        brand_voice="",
        n_headlines=10,
    )
    assert "## Target audience" not in out
    assert "## Brand voice" not in out
    assert "## Reference" not in out
    assert "## Context" in out
    assert "10 headline-promesse" in out


@pytest.mark.unit
def test_build_user_prompt_includes_references_when_present():
    out = _build_user_prompt(
        context="ctx",
        references="esempio: 'da 0 a 10K in 90gg'",
        target_audience="",
        brand_voice="",
        n_headlines=10,
    )
    assert "## Reference" in out
    assert "da 0 a 10K" in out


# ── write_promises validation ─────────────────────────────────────
@pytest.mark.unit
def test_write_promises_rejects_blank_context():
    with pytest.raises(ValueError, match="context"):
        write_promises(api_key="x", context="   ", n_headlines=10)


@pytest.mark.unit
@pytest.mark.parametrize("n", [0, MIN_HEADLINES - 1, MAX_HEADLINES + 1, 100])
def test_write_promises_rejects_out_of_bounds_n_headlines(n):
    with pytest.raises(ValueError, match="n_headlines"):
        write_promises(api_key="x", context="real context", n_headlines=n)


@pytest.mark.unit
def test_bounds_constants_are_consistent():
    assert MIN_HEADLINES >= 1
    assert MAX_HEADLINES > MIN_HEADLINES
