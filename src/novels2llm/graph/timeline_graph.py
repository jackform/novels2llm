"""Stage 6: Timeline construction and event ordering."""

from collections import defaultdict
import networkx as nx


class TimelineGraph:
    """DAG of story events ordered along the timeline."""

    def __init__(self):
        self.graph = nx.DiGraph()
        self._events_by_chapter: dict[int, list[dict]] = defaultdict(list)
        self._global_counter = 0

    def add_event(
        self,
        event_id: str,
        description: str,
        chapter: int,
        chapter_order: int,
        participants: list[str] | None = None,
        time_marker: str | None = None,
        is_flashback: bool = False,
    ) -> None:
        """Add an event to the timeline."""
        participants = participants or []

        self.graph.add_node(
            event_id,
            description=description,
            chapter=chapter,
            chapter_order=chapter_order,
            participants=participants,
            time_marker=time_marker,
            is_flashback=is_flashback,
        )

        self._events_by_chapter[chapter].append({
            'event_id': event_id,
            'chapter_order': chapter_order,
        })

    def add_causal_link(self, before_id: str, after_id: str) -> None:
        """Add a causal relationship between events."""
        if self.graph.has_node(before_id) and self.graph.has_node(after_id):
            self.graph.add_edge(before_id, after_id, type='causal')

    def build_timeline(self) -> list[dict]:
        """Build a global timeline by sorting events."""
        # Sort chapters
        sorted_chapters = sorted(self._events_by_chapter.keys())

        events = []
        self._global_counter = 0

        for chapter in sorted_chapters:
            chapter_events = self._events_by_chapter[chapter]
            # Sort events within chapter
            chapter_events.sort(key=lambda e: e['chapter_order'])

            for evt in chapter_events:
                node_data = self.graph.nodes[evt['event_id']]
                self._global_counter += 1

                events.append({
                    'event_id': evt['event_id'],
                    'global_order': self._global_counter,
                    'description': node_data['description'],
                    'chapter': node_data['chapter'],
                    'chapter_order': node_data['chapter_order'],
                    'participants': node_data['participants'],
                    'time_marker': node_data['time_marker'],
                    'is_flashback': node_data['is_flashback'],
                })

        return events

    def to_dict(self) -> dict:
        """Export timeline to dict."""
        events = self.build_timeline()
        edges = [
            {'before': a, 'after': b, 'type': data.get('type', 'causal')}
            for a, b, data in self.graph.edges(data=True)
        ]
        return {'events': events, 'causal_links': edges}
