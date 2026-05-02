"""
Calibration dependency graph: DAG-based ordering and stale-node detection.
"""

from __future__ import annotations

from typing import Callable


class CalibrationNode:
    """Single node in the calibration DAG."""

    def __init__(self, name: str, run_fn: Callable, provides: list[str], requires: list[str] | None = None):
        self.name = name
        self.run_fn = run_fn
        self.provides: list[str] = provides
        self.requires: list[str] = requires or []

    def __repr__(self) -> str:
        return f"CalibrationNode({self.name!r}, provides={self.provides}, requires={self.requires})"


class CalibrationGraph:
    """
    DAG of calibration steps.

    Each node declares which parameters it *provides* and which it *requires*.
    ``run_stale()`` walks the graph in topological order and re-runs only
    nodes whose outputs are stale (as reported by a CalibrationStore).

    Parameters
    ----------
    store : CalibrationStore
        Used to check staleness of each node's outputs.
    qubit : str
        Qubit label passed to ``store.is_stale()``.

    Example
    -------
    >>> graph = CalibrationGraph(store, "Q1")
    >>> graph.add(CalibrationNode("res_spec", fn, provides=["res_freq_ge"]))
    >>> graph.add(CalibrationNode("qubit_spec", fn2, provides=["qb_freq_ge"], requires=["res_freq_ge"]))
    >>> graph.run_stale()
    """

    def __init__(self, store, qubit: str):
        self._store = store
        self._qubit = qubit
        self._nodes: dict[str, CalibrationNode] = {}

    def add(self, node: CalibrationNode):
        self._nodes[node.name] = node

    def remove(self, name: str):
        self._nodes.pop(name, None)

    # ── Topological sort (Kahn's algorithm) ──────────────────────────────────

    def _topological_order(self) -> list[str]:
        """Return node names in a valid execution order."""
        # Build adjacency: required_param → node_name(s) that provide it
        provided_by: dict[str, str] = {}
        for name, node in self._nodes.items():
            for p in node.provides:
                provided_by[p] = name

        # In-degree = number of prerequisite nodes not yet scheduled
        in_degree: dict[str, int] = {n: 0 for n in self._nodes}
        dep_edges: dict[str, list[str]] = {n: [] for n in self._nodes}

        for name, node in self._nodes.items():
            for req in node.requires:
                provider = provided_by.get(req)
                if provider is not None and provider != name:
                    in_degree[name] += 1
                    dep_edges[provider].append(name)

        queue = [n for n, d in in_degree.items() if d == 0]
        order: list[str] = []
        while queue:
            cur = queue.pop(0)
            order.append(cur)
            for nxt in dep_edges[cur]:
                in_degree[nxt] -= 1
                if in_degree[nxt] == 0:
                    queue.append(nxt)

        if len(order) != len(self._nodes):
            raise RuntimeError("Cycle detected in CalibrationGraph")
        return order

    # ── Stale detection ───────────────────────────────────────────────────────

    def _node_is_stale(self, node: CalibrationNode, max_age_hours: float | None = None) -> bool:
        return any(
            self._store.is_stale(self._qubit, p, max_age_hours)
            for p in node.provides
        )

    # ── Execution ─────────────────────────────────────────────────────────────

    def run_stale(self, max_age_hours: float | None = None, dry_run: bool = False) -> list[str]:
        """
        Run all nodes whose outputs are stale, in dependency order.

        Parameters
        ----------
        max_age_hours : float or None
            Override the store's default staleness threshold.
        dry_run : bool
            If True, print which nodes *would* run without executing them.

        Returns
        -------
        ran : list[str]
            Names of nodes that were (or would be) executed.
        """
        order = self._topological_order()
        ran: list[str] = []
        for name in order:
            node = self._nodes[name]
            if self._node_is_stale(node, max_age_hours):
                if dry_run:
                    print(f"[dry-run] would run: {name}")
                else:
                    print(f"[CalibrationGraph] running: {name}")
                    node.run_fn()
                ran.append(name)
        return ran

    def run_all(self, dry_run: bool = False) -> list[str]:
        """Force-run every node regardless of staleness."""
        order = self._topological_order()
        for name in order:
            if dry_run:
                print(f"[dry-run] would run: {name}")
            else:
                print(f"[CalibrationGraph] running: {name}")
                self._nodes[name].run_fn()
        return order

    def status(self) -> str:
        """Return a human-readable staleness summary for all nodes."""
        order = self._topological_order()
        lines = [f"CalibrationGraph status — qubit {self._qubit}"]
        for name in order:
            node = self._nodes[name]
            stale = self._node_is_stale(node)
            lines.append(f"  {'STALE' if stale else 'OK   '} {name:<24s} provides={node.provides}")
        return "\n".join(lines)

    def __repr__(self) -> str:
        return f"CalibrationGraph(qubit={self._qubit!r}, nodes={list(self._nodes.keys())})"
