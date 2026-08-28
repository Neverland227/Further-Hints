"""Factor-graph structure and min-fill diagnostics for exact hints."""

from __future__ import annotations

from typing import Sequence

import networkx as nx


def primal_graph(H: Sequence[Sequence[int]], q: int) -> nx.Graph:
    """Connect secret variables that co-occur in a nonzero hint row."""

    graph = nx.Graph()
    n = len(H[0]) if H else 0
    graph.add_nodes_from(range(n))
    for row in H:
        active = [j for j, value in enumerate(row) if int(value) % q]
        for i, left in enumerate(active):
            for right in active[i + 1 :]:
                graph.add_edge(left, right)
    return graph


def min_fill_order(H: Sequence[Sequence[int]], q: int, max_treewidth: int) -> tuple[list[int], int]:
    """Compute a deterministic greedy min-fill order with a hard width gate."""

    graph = primal_graph(H, q)
    order: list[int] = []
    width = 0
    while graph.nodes:
        def key(node: int) -> tuple[int, int, int]:
            neighbors = list(graph.neighbors(node))
            missing = sum(not graph.has_edge(a, b) for i, a in enumerate(neighbors) for b in neighbors[i + 1 :])
            return missing, len(neighbors), node

        node = min(graph.nodes, key=key)
        neighbors = list(graph.neighbors(node))
        width = max(width, len(neighbors))
        if width > max_treewidth:
            raise RuntimeError("RESOURCE_LIMIT: max_treewidth exceeded")
        for i, left in enumerate(neighbors):
            for right in neighbors[i + 1 :]:
                graph.add_edge(left, right)
        graph.remove_node(node)
        order.append(node)
    return order, width

