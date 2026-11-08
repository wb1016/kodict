"""Tests for streamed-output parsing (JSONL fast path + raw-text fallback)."""

from kodict.parse import JsonlParser, extract_objects


def feed(text: str) -> JsonlParser:
    p = JsonlParser()
    p.feed(text)
    p.flush()
    return p


def test_jsonl_fast_path():
    p = JsonlParser()
    assert p.feed('{"w": "사과", "r": 1}\n{"w": "학') == [{"w": "사과", "r": 1}]
    assert p.feed('교", "r": 2}\n') == [{"w": "학교", "r": 2}]
    assert p.flush() == []
    assert len(p.entries) == 2
    assert p.bad_lines == 0


def test_raw_text_accumulates():
    p = JsonlParser()
    p.feed('{"w": "사과"')
    p.feed(', "r": 1}\n')
    p.flush()
    assert p.raw == '{"w": "사과", "r": 1}\n'


def test_fences_and_prose_ignored():
    p = feed('```json\n{"w": "사과", "r": 1}\n```\nHere are the words:\n')
    assert p.entries == [{"w": "사과", "r": 1}]
    assert p.bad_lines == 0


def test_bad_json_line_counted():
    p = feed('{"w": "사과"}\n{"w": "학교", \n')
    assert p.bad_lines == 1
    assert len(p.entries) == 1


def test_trailing_prose_after_object():
    p = feed('{"w": "사과", "r": 1} (note)\n')
    assert p.entries == [{"w": "사과", "r": 1}]


def test_bullet_prefixed_entry():
    p = feed('- {"w": "사과", "r": 1}\n')
    assert p.entries == [{"w": "사과", "r": 1}]


def test_single_line_array_recovered_at_flush():
    p = feed('[{"w": "사과", "r": 1}, {"w": "학교", "r": 2}]')
    assert p.entries == [{"w": "사과", "r": 1}, {"w": "학교", "r": 2}]


def test_wrapper_object_recovered_at_flush():
    text = '{"words": [{"w": "사과", "r": 1}, {"w": "학교", "r": 2}]}'
    entries = feed(text).entries
    assert {"w": "사과", "r": 1} in entries
    assert {"w": "학교", "r": 2} in entries


def test_pretty_printed_array_recovered():
    text = '[\n  {\n    "w": "사과",\n    "r": 1\n  },\n  {\n    "w": "학교",\n    "r": 2\n  }\n]\n'
    p = feed(text)
    assert {"w": "사과", "r": 1} in p.entries
    assert {"w": "학교", "r": 2} in p.entries


def test_truncated_last_object_dropped_complete_ones_kept():
    text = '{"w": "사과", "r": 1}\n{"w": "학교", "r": 2}\n{"w": "커피", "r'
    p = feed(text)
    assert {"w": "사과", "r": 1} in p.entries
    assert {"w": "학교", "r": 2} in p.entries
    assert len([e for e in p.entries if "커피" in e]) == 0


def test_bare_and_numbered_word_lines():
    p = feed("1. 사과\n2) 학교\ncoffee\n")
    assert {"w": "사과", "r": 1} in p.entries
    assert {"w": "학교", "r": 2} in p.entries
    assert {"w": "coffee"} in p.entries


def test_fallback_does_not_re_add_line_parsed_entries():
    p = feed('{"w": "사과", "r": 1}\n')
    assert p.entries == [{"w": "사과", "r": 1}]


def test_extract_objects_helper():
    text = 'prefix {"a": 1} middle {"b": [2]} suffix'
    assert extract_objects(text) == [{"a": 1}, {"b": [2]}]

def test_pretty_printed_counted_once_via_runner_pattern():
    """flush() then reading entries must not double-count fallback objects."""
    p = JsonlParser()
    p.feed('{\n  "rank": 1,\n  "word": "은"\n}\n{\n  "rank": 2,\n  "word": "는"\n}\n')
    p.flush()
    entries = list(p.entries)          # exactly what the runner does now
    assert len(entries) == 2
    assert p.bad_lines == 0            # lone "{" lines are not errors


def test_lone_braces_are_not_bad_lines():
    p = feed('{\n"rank": 1,\n"word": "은"\n}\n')
    assert p.bad_lines == 0
    assert p.entries == [{"rank": 1, "word": "은"}]
