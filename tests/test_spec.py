"""Candidate specs and verdicts across any number of views."""

from __future__ import annotations

import json

import pytest

from ava.spec import CandidateSpec, RunState, Verdict, apply_style

# -- style assembly -------------------------------------------------------------


def test_prefix_style_assembles_like_generate_py() -> None:
    assert apply_style("an oil painting of", "a horse") == "an oil painting of a horse"
    assert apply_style("", "a horse") == "a horse"


def test_template_style_wraps_the_subject() -> None:
    assert (
        apply_style("oil painting style, {}", "a bazaar")
        == "oil painting style, a bazaar"
    )
    assert apply_style("{}, oil painting style", "tiger") == "tiger, oil painting style"


def test_full_prompts_follow_slot_order() -> None:
    spec = CandidateSpec("flip", ("a duck", "a rabbit"), "a lithograph of")
    assert spec.full_prompts == ["a lithograph of a duck", "a lithograph of a rabbit"]
    assert spec.n_views == 2


# -- identity -------------------------------------------------------------------


def test_uid_is_stable_across_json_round_trip() -> None:
    spec = CandidateSpec("three_view", ("a", "b", "c"), "x", seed=3)
    again = CandidateSpec(**json.loads(json.dumps(spec.__dict__)))
    assert again == spec
    assert again.uid() == spec.uid()
    assert isinstance(again.prompts, tuple)


def test_uid_depends_on_task_prompts_style_and_seed() -> None:
    base = CandidateSpec("flip", ("a", "b"))
    assert CandidateSpec("rotate_cw", ("a", "b")).uid() != base.uid()
    assert CandidateSpec("flip", ("b", "a")).uid() != base.uid()
    assert CandidateSpec("flip", ("a", "b"), "s").uid() != base.uid()
    assert CandidateSpec("flip", ("a", "b"), seed=1).uid() != base.uid()
    assert CandidateSpec("flip", ("a", "b"), ref_image="x.png").uid() != base.uid()


def test_a_single_prompt_is_not_an_illusion() -> None:
    with pytest.raises(ValueError):
        CandidateSpec("flip", ("alone",))


# -- verdicts -------------------------------------------------------------------


def two_view(
    s_far_low: float, s_far_high: float, s_near_low: float, s_near_high: float
):
    scores = [[s_far_low, s_far_high], [s_near_low, s_near_high]]
    p = [0.5, 0.5]  # not used by the margin logic
    return Verdict(
        uid="x",
        task="hybrid",
        slots=["low", "high"],
        scores=scores,
        p=p,
        j=0.5,
        alignment=min(s_far_low, s_near_high),
        concealment=0.5,
    )


def test_two_view_margins_match_the_old_definitions() -> None:
    """sep[0] was `sep_far`, sep[1] was `sep_near`."""
    v = two_view(0.30, 0.18, 0.21, 0.30)
    assert v.sep[0] == pytest.approx(0.30 - 0.18)
    assert v.sep[1] == pytest.approx(0.30 - 0.21)
    assert v.sep_min == pytest.approx(min(v.sep))


def test_margin_is_against_the_best_other_prompt() -> None:
    v = Verdict(
        uid="x",
        task="three_view",
        slots=["identity", "rotate_cw", "rotate_ccw"],
        scores=[[0.30, 0.25, 0.28], [0.20, 0.30, 0.10], [0.29, 0.10, 0.28]],
        p=[0.5, 0.5, 0.3],
        j=0.3,
        alignment=0.28,
        concealment=0.4,
    )
    assert v.sep == pytest.approx([0.02, 0.10, -0.01])
    assert v.holds == [True, True, False]
    assert v.diagnose() == "lost:rotate_ccw"


def test_diagnosis_names_every_failed_slot() -> None:
    assert two_view(0.3, 0.2, 0.2, 0.3).diagnose() == "ok"
    assert two_view(0.2, 0.3, 0.3, 0.2).diagnose() == "all_lost"
    assert two_view(0.2, 0.3, 0.2, 0.3).diagnose() == "lost:low"
    assert two_view(0.3, 0.2, 0.3, 0.2).diagnose() == "lost:high"
    v = Verdict(
        uid="x",
        task="three_view",
        slots=["identity", "rotate_cw", "rotate_ccw"],
        scores=[[0.1, 0.3, 0.2], [0.2, 0.3, 0.1], [0.3, 0.1, 0.2]],
        p=[0.2, 0.5, 0.2],
        j=0.2,
        alignment=0.1,
        concealment=0.3,
    )
    assert v.diagnose() == "lost:identity,rotate_ccw"
    assert v.lost_slots() == ["identity", "rotate_ccw"]


def test_a_tie_does_not_count_as_holding() -> None:
    assert two_view(0.3, 0.3, 0.2, 0.3).holds == [False, True]


def test_to_json_persists_the_derived_fields() -> None:
    row = json.loads(two_view(0.30, 0.18, 0.21, 0.30).to_json())
    for key in ("scores", "p", "j", "alignment", "concealment", "sep", "sep_min"):
        assert key in row
    assert row["diagnosis"] == "ok"
    assert row["holds"] == [True, True]
    assert row["slots"] == ["low", "high"]


# -- run state ------------------------------------------------------------------


def test_run_state_round_trips_specs_and_verdicts() -> None:
    spec = CandidateSpec("hybrid", ("a panda", "waterfalls"), "a photo of")
    v = two_view(0.3, 0.2, 0.2, 0.3)
    state = RunState(run_id="r", round_index=2, last_round=[(spec, v)])
    state.record(spec, v)
    again = RunState.from_json(state.to_json())
    assert again.round_index == 2
    assert again.evaluated_uids == {spec.uid()}
    assert again.last_round[0][0] == spec
    assert again.last_round[0][1].sep == v.sep
