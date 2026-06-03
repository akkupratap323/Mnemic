"""Tests for the heuristic signal classifier."""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from mnemic.hybrid.classifier import SignalClassifier
from mnemic.hybrid.types import Tier

clf = SignalClassifier()


def test_plain_chatter_stays_vector_only():
    d = clf.classify('the weather is nice today and i had coffee')
    assert d.tier is Tier.VECTOR_ONLY
    assert d.reasons == ()
    assert d.writes_graph is False


@pytest.mark.parametrize(
    'text,keyword',
    [
        ('we made a decision to ship', 'decision'),
        ('remember to rotate the secret', 'remember'),
        ('found a bug in the parser', 'bug'),
        ('TODO refactor this module', 'todo'),
        ('the api endpoint changed', 'api'),
        ('add a db migration', 'migration'),
    ],
)
def test_signal_keywords_route_to_graph(text, keyword):
    d = clf.classify(text)
    assert d.tier is Tier.VECTOR_AND_GRAPH
    assert f'keyword:{keyword}' in d.reasons


def test_keyword_matching_is_token_based_not_substring():
    # 'api' must not match inside 'therapist'/'rapid'
    d = clf.classify('the therapist gave rapid feedback')
    assert d.tier is Tier.VECTOR_ONLY


def test_case_insensitive():
    assert clf.classify('DECISION made').tier is Tier.VECTOR_AND_GRAPH


def test_phrase_marker():
    d = clf.classify('please make a note of the office address')
    assert d.tier is Tier.VECTOR_AND_GRAPH
    assert 'phrase:make a note' in d.reasons


def test_metadata_force_graph():
    d = clf.classify('just chatter', {'force_graph': True})
    assert d.tier is Tier.VECTOR_AND_GRAPH
    assert 'metadata:force_graph' in d.reasons


@pytest.mark.parametrize('truthy', ['true', 'yes', '1', 'ON'])
def test_metadata_force_graph_accepts_truthy_strings(truthy):
    d = clf.classify('just chatter', {'force_graph': truthy})
    assert d.tier is Tier.VECTOR_AND_GRAPH


@pytest.mark.parametrize('falsy', ['false', 'no', '0', 'maybe', ''])
def test_metadata_force_graph_rejects_falsy_strings(falsy):
    d = clf.classify('just chatter', {'force_graph': falsy})
    assert d.tier is Tier.VECTOR_ONLY


def test_metadata_force_vector_only_overrides_everything():
    d = clf.classify('critical decision bug api', {'force_vector_only': True, 'force_graph': True})
    assert d.tier is Tier.VECTOR_ONLY
    assert d.reasons == ('metadata:force_vector_only',)


def test_metadata_type_triggers_graph():
    d = clf.classify('x', {'type': 'Decision'})
    assert d.tier is Tier.VECTOR_AND_GRAPH
    assert 'type:decision' in d.reasons


def test_importance_threshold():
    assert clf.classify('x', {'importance': 0.9}).tier is Tier.VECTOR_AND_GRAPH
    assert clf.classify('x', {'importance': 0.1}).tier is Tier.VECTOR_ONLY


def test_importance_ignores_bool():
    # bool is not a valid importance score
    assert clf.classify('x', {'importance': True}).tier is Tier.VECTOR_ONLY


def test_reasons_are_deterministic_and_sorted():
    a = clf.classify('bug api decision')
    b = clf.classify('decision api bug')
    assert a.reasons == b.reasons
    keyword_reasons = [r for r in a.reasons if r.startswith('keyword:')]
    assert keyword_reasons == sorted(keyword_reasons)


def test_empty_content_does_not_crash():
    assert clf.classify('').tier is Tier.VECTOR_ONLY
    assert clf.classify('   ').tier is Tier.VECTOR_ONLY


def test_classifier_is_frozen():
    with pytest.raises(FrozenInstanceError):
        clf.importance_threshold = 0.1  # type: ignore[misc]
