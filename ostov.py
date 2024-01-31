import queue
from parse import get_neighbours, get_edges_list, Edge


class DisjointSet:
    def __init__(self, size):
        # Инициализация массива предков и рангов
        self.parent = [i for i in range(size)]
        self.rank = [0 for _ in range(size)]

    def find(self, x):
        # Находит корень множества, к которому принадлежит элемент x
        # if self.parent[self.parent[x]] != self.parent[x]:
        if self.parent[x] != x:
            # Сжатие пути: родителем этой вершины становится корень
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, x, y):
        # Объединяет множества, содержащие элементы x и y
        root_x = self.find(x)
        root_y = self.find(y)

        if root_x != root_y:
            # Присоединяем меньшее множество к большему
            if self.rank[root_x] < self.rank[root_y]:
                self.parent[root_x] = root_y
            elif self.rank[root_x] > self.rank[root_y]:
                self.parent[root_y] = root_x
            else:
                # Если ранги одинаковы, увеличиваем ранг одного из множеств
                self.parent[root_y] = root_x
                self.rank[root_x] += 1


def prim(neighbors):
    edges = queue.PriorityQueue()
    n = len(neighbors)
    visited = [False for _ in range(n)]

    start_node = 0
    visited[start_node] = True
    for neighbor, distance in neighbors[start_node]:
        if neighbor:
            edges.put((distance, start_node, neighbor))
    cost = 0
    min_tree = []
    while not edges.empty():
        weight, start, end = edges.get()
        if visited[end]:
            continue
        visited[end] = True
        cost += weight
        min_tree.append(Edge(start, end, weight))

        for neighbor, distance in neighbors[end]:
            if not visited[neighbor]:
                edges.put((distance, end, neighbor))
    return cost, min_tree


def kruskal(edges: list[Edge], nodes_count: int):
    edges = tuple(sorted(edges, key=lambda el: el[::-1]))
    min_tree = []
    cost = 0
    dsu = DisjointSet(nodes_count)

    for start, end, weight in edges:
        if dsu.find(start) != dsu.find(end):
            dsu.union(start, end)
            cost += weight
            min_tree.append(Edge(start, end, weight))

    return cost, min_tree


def beautiful(edges: list[tuple[int, int, int]], towns_names):
    for edge in edges:
        print(towns_names[edge[1]], "-", towns_names[edge[2]], end="; ")
    print()


if __name__ == '__main__':
    es, towns = get_edges_list()
    ns, towns = get_neighbours(pre=(es, towns))
    prim_weight, prim_tree = prim(ns)
    kruskal_weight, kruskal_tree = kruskal(es, len(towns))
    t_names = tuple(towns)
    print("Algorithm Prima")
    beautiful(prim_tree, t_names)
    print("Algorithm Kruskal")
    beautiful(kruskal_tree, t_names)
    print(prim_weight, kruskal_weight)


