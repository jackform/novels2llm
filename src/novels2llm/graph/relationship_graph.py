"""Stage 5: Relationship graph construction using NetworkX."""

import networkx as nx
from collections import defaultdict


class RelationshipGraph:
    """Directed graph of character relationships."""

    def __init__(self):
        self.graph = nx.DiGraph()
        self._rel_types = defaultdict(list)  # (a,b) -> list of rel_types observed

    def add_relationship(
        self,
        character_a: str,
        character_b: str,
        rel_type: str,
        direction: str = "bidirectional",
        evidence: list[str] | None = None,
        confidence: float = 0.5,
    ) -> None:
        """Add a relationship edge to the graph."""
        evidence = evidence or []

        # Add nodes if they don't exist
        if not self.graph.has_node(character_a):
            self.graph.add_node(character_a)
        if not self.graph.has_node(character_b):
            self.graph.add_node(character_b)

        # Track relationship types for conflict resolution
        key = tuple(sorted([character_a, character_b]))
        self._rel_types[key].append({
            'type': rel_type,
            'confidence': confidence,
            'direction': direction,
        })

        # Add edges based on direction
        if direction in ("bidirectional", "a_to_b"):
            self.graph.add_edge(
                character_a, character_b,
                rel_type=rel_type,
                evidence=evidence,
                confidence=confidence,
            )
        if direction in ("bidirectional", "b_to_a"):
            self.graph.add_edge(
                character_b, character_a,
                rel_type=rel_type,
                evidence=evidence,
                confidence=confidence,
            )

    def resolve_conflicts(self) -> None:
        """Resolve conflicting relationship types.

        When two characters are reported with different rel_types,
        prefer the one with higher confidence. If confidences are equal,
        keep the most specific type.
        """
        for pair, rels in self._rel_types.items():
            if len(rels) <= 1:
                continue

            # Find the best relationship type
            best = max(rels, key=lambda r: (
                r['confidence'],
                len(r['type']),  # Prefer more specific types
            ))

            # Re-add edges with the best type
            a, b = pair
            for rel in rels:
                if rel == best:
                    continue

                # Remove conflicting edges
                if self.graph.has_edge(a, b):
                    current_type = self.graph[a][b].get('rel_type')
                    if current_type == rel['type']:
                        self.graph[a][b]['rel_type'] = best['type']
                        self.graph[a][b]['confidence'] = max(
                            self.graph[a][b].get('confidence', 0),
                            best['confidence'],
                        )
                if self.graph.has_edge(b, a):
                    current_type = self.graph[b][a].get('rel_type')
                    if current_type == rel['type']:
                        self.graph[b][a]['rel_type'] = best['type']
                        self.graph[b][a]['confidence'] = max(
                            self.graph[b][a].get('confidence', 0),
                            best['confidence'],
                        )

    def get_relationships(self) -> list[dict]:
        """Get all relationships as a list of dicts."""
        seen = set()
        relationships = []

        for a, b, data in self.graph.edges(data=True):
            key = tuple(sorted([a, b]))
            if key in seen:
                continue
            seen.add(key)

            relationships.append({
                'character_a': a,
                'character_b': b,
                'rel_type': data.get('rel_type', 'unknown'),
                'evidence': data.get('evidence', []),
                'confidence': data.get('confidence', 0.5),
            })

        return relationships

    def get_connected_characters(self, name: str, depth: int = 1) -> list[str]:
        """Get characters connected to the given character up to the given depth."""
        if not self.graph.has_node(name):
            return []

        visited = {name}
        frontier = {name}

        for _ in range(depth):
            next_frontier = set()
            for node in frontier:
                next_frontier.update(self.graph.successors(node))
                next_frontier.update(self.graph.predecessors(node))
            frontier = next_frontier - visited
            visited.update(next_frontier)

        visited.discard(name)
        return sorted(visited)

    def to_dict(self) -> dict:
        """Export graph to dict."""
        return {
            'nodes': list(self.graph.nodes()),
            'edges': [
                {'source': a, 'target': b, **data}
                for a, b, data in self.graph.edges(data=True)
            ],
        }
