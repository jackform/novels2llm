"""Tests for the novels2llm pipeline."""

import pytest
from pathlib import Path


class TestChapterSplitter:
    def test_chapter_detection(self):
        from src.novels2llm.chunking.chapter_splitter import (
            detect_chapter_boundaries,
            extract_chapter_number,
        )
        text = "前言内容### 第1章 正文内容### 第2章 更多内容"
        boundaries = detect_chapter_boundaries(text)
        assert len(boundaries) >= 2

    def test_chapter_number_extraction(self):
        from src.novels2llm.chunking.chapter_splitter import extract_chapter_number
        assert extract_chapter_number("第1章") == 1
        assert extract_chapter_number("第一章") == 1
        assert extract_chapter_number("第12章") == 12
        assert extract_chapter_number("前言") is None


class TestPreprocessing:
    def test_yaml_parsing(self):
        from src.novels2llm.pipeline.stage1_preprocess import _parse_yaml_frontmatter
        content = """---
title: Test Novel
author: Test Author
---
Body content"""
        meta, body = _parse_yaml_frontmatter(content)
        assert meta['title'] == 'Test Novel'
        assert body.strip() == 'Body content'

    def test_deduplication(self):
        from src.novels2llm.pipeline.stage1_preprocess import _deduplicate_text
        text = "line 1 content here that is unique and long\nline 1 content here that is unique and long\nline 2 is different content here"
        result, removed = _deduplicate_text(text)
        assert removed == 1


class TestModels:
    def test_character_model(self):
        from src.novels2llm.models.entities import Character
        c = Character(canonical_name="Test", gender="male")
        assert c.canonical_name == "Test"
        assert c.aliases == []

    def test_novel_world(self):
        from src.novels2llm.models.output import NovelWorld, NovelMetadata
        meta = NovelMetadata(novel_id="test", title="Test Novel")
        nw = NovelWorld(metadata=meta)
        assert nw.metadata.novel_id == "test"
        json_str = nw.model_dump_json_pretty()
        assert "test" in json_str


class TestAliasResolution:
    def test_exact_match_merge(self):
        from src.novels2llm.coreference.alias_resolver import resolve_aliases
        chars = [
            {'canonical_name': '小明', 'aliases': []},
            {'canonical_name': '小明', 'aliases': ['明儿']},
        ]
        result = resolve_aliases(chars)
        assert len(result) == 1
        assert result[0]['canonical_name'] == '小明'

    def test_alias_overlap_merge(self):
        from src.novels2llm.coreference.alias_resolver import resolve_aliases
        chars = [
            {'canonical_name': '妈妈', 'aliases': ['母亲']},
            {'canonical_name': '母亲', 'aliases': ['妈妈']},
        ]
        result = resolve_aliases(chars)
        assert len(result) == 1


class TestRelationshipGraph:
    def test_add_relationship(self):
        from src.novels2llm.graph.relationship_graph import RelationshipGraph
        g = RelationshipGraph()
        g.add_relationship('A', 'B', 'friend')
        assert g.graph.number_of_nodes() == 2
        assert g.graph.number_of_edges() == 2  # bidirectional

    def test_get_connected(self):
        from src.novels2llm.graph.relationship_graph import RelationshipGraph
        g = RelationshipGraph()
        g.add_relationship('A', 'B', 'friend')
        g.add_relationship('A', 'C', 'family')
        connected = g.get_connected_characters('A')
        assert 'B' in connected
        assert 'C' in connected
        assert 'A' not in connected


class TestTimelineGraph:
    def test_build_timeline(self):
        from src.novels2llm.graph.timeline_graph import TimelineGraph
        tg = TimelineGraph()
        tg.add_event('evt1', 'Event 1', 1, 1)
        tg.add_event('evt2', 'Event 2', 1, 2)
        timeline = tg.build_timeline()
        assert len(timeline) == 2
        assert timeline[0]['global_order'] == 1
        assert timeline[1]['global_order'] == 2
