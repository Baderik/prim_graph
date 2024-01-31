from collections import namedtuple
from typing import TypeAlias
from pathlib import Path

__all__ = ["get_edges_list", "get_neighbours", "save_graph", "Edge", "Direction", "Neighbors"]

Edge = namedtuple("Edge", ["start", "end", "weight"])
Direction = namedtuple("Direction", ["end", "weight"])
Neighbors: TypeAlias = list[Direction]


def get_edges_list(file_path: str = "pk_model/graph_edges.txt") \
        -> tuple[list[Edge], dict[str, int]]:
    with open(file_path, encoding="utf-8") as inp:
        data = list(map(lambda el: el.split(":"), inp.read().splitlines()))
    towns = {}
    graph: list[Edge] = []
    for edges in data:
        if edges[0] not in towns:
            towns[edges[0]] = len(towns)
        if edges[1] not in towns:
            towns[edges[1]] = len(towns)
        graph.append(Edge(towns[edges[0]], towns[edges[1]], int(edges[2])))

    return graph, towns


def get_neighbours(pre=None) \
        -> tuple[list[Neighbors], dict[str, int]]:
    if pre is None:
        edges, towns = get_edges_list()
    else:
        edges, towns = pre

    graph: list[list[Direction]] = [[] for _ in range(len(towns))]
    for first, second, weight in edges:
        graph[first].append(Direction(second, weight))
        graph[second].append(Direction(first, weight))

    return graph, towns


def save_graph(data: dict, file_name: str = "debug_graph.json", dir_name=Path("")):
    from json import dump
    with open(dir_name / file_name, "w", encoding="utf-8") as out:
        dump(data, out, ensure_ascii=False, indent=True)


if __name__ == '__main__':
    es, t_indexes = get_edges_list()
    ns, t_indexes = get_neighbours(pre=(es, t_indexes))
    print(es)
    print(ns)
    print(t_indexes)
    save_graph({"town_codes": t_indexes, "edges list": es, "neighbours list": ns})
