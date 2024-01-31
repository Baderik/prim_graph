import queue

from parse import get_neighbours, Neighbors

__all__ = ["dijkstra"]


def dijkstra(start, neighbors: list[Neighbors]) -> tuple[list[int | float], list[int]]:
    """
    Считает кратчайшее расстояние от вершины <start>, до всех остальных.
    Возвращает массив кратчайших расстояний и массив предков, для восстановления маршрута.
    """
    next_nodes = queue.PriorityQueue()
    nodes_count = len(neighbors)
    distances = [float("inf") for _ in range(nodes_count)]
    parents = [-1 for _ in range(nodes_count)]
    visited = [False for _ in range(nodes_count)]

    distances[start] = 0
    parents[start] = start
    next_nodes.put((distances[start], start))

    while not next_nodes.empty():
        distance2node, node = next_nodes.get()
        if visited[node]:
            continue
        visited[start] = True

        for neighbour, distance2neighbour in neighbors[node]:
            if visited[neighbour]:
                continue

            distance = distance2node + distance2neighbour
            if distance < distances[neighbour]:
                distances[neighbour] = distance
                next_nodes.put((distance, neighbour))
                parents[neighbour] = node

    return distances, parents


if __name__ == '__main__':
    towns_neighbours: list[Neighbors]
    towns: dict[str, int]
    towns_neighbours, towns = get_neighbours()
    town_count = len(towns)

    ids = sorted(sum([[tuple(sorted((i, e[0]))) for e in towns_neighbours[i]] for i in range(town_count)], []))[::2]
    print(ids)
    used = [0 for _ in range(len(ids))]
    ds = []
    for i in range(town_count):
        d, w = dijkstra(i, towns_neighbours)
        edges = sorted([tuple(sorted((w[j], j))) for j in range(town_count) if j != w[j]])
        j = 0
        for el in edges:
            if j < len(ids):
                while ids[j] < el:
                    j += 1
                if ids[j] == el:
                    used[j] += 1
                    j += 1
        ds.append(d)
    print(*map(lambda line: "|".join(map(str, line)), ds), sep="\n")
    print("Кол-во использований каждого ребра:")
    print(used)
