from texttable import Texttable
from pathlib import Path

from min_way import *
from parse import *
from ostov import *

DEBUG = False


class Show:
    CONSOLE = True  # Сохранять в файлы
    FILE = True     # Печатать в консоль
    """
    c - Печать в консоль
    w - Сохранять в файл
    """
    FULL_MIN_DISTANCES = "w"
    FULL_MIN_WAYS = "w"
    FULL_EDGES_COUNTS = "w"
    FULL_AVERAGE = "cw"

    OSTOV_MIN_DISTANCES = "w"
    OSTOV_MIN_WAYS = "w"
    OSTOV_EDGES_COUNTS = "w"
    OSTOV_AVERAGE = "cw"

    DELTA_MIN = "cw"

    RES_DIR = Path("result")

    @classmethod
    def _gen_table(cls, mtx, header, col_header=None):
        if col_header is None:
            col_header = header

        t = Texttable(max_width=0)
        t.header(header)
        t.add_rows([[col_header[i + 1]] + mtx[i] for i in range(len(mtx))], header=False)
        return t

    @classmethod
    def _table(cls, title: str, table: Texttable, file_name: str, write):
        if cls.CONSOLE and "c" in write:
            print(title)
            print(table.draw())
        if cls.FILE and "w" in write:
            with open(cls.RES_DIR / file_name, "w", encoding="utf-8") as f_out:
                print(title, file=f_out)
                print(table.draw(), file=f_out)

    @staticmethod
    def _towns4table(corner):
        return [corner] + towns_list

    @staticmethod
    def _short_towns4table(corner):
        return ([corner] + [f"{short_towns[i]}: {towns_list[i]}" for i in range(town_count)],
                [""] + short_towns)

    @classmethod
    def _header_towns_table(cls, short=False, corner="Из\\В"):
        if short:
            return cls._short_towns4table(corner)
        return cls._towns4table(corner),

    @classmethod
    def _base_towns(cls, data, title, file_name, write, short_names=False):
        temp = cls._gen_table(data, *cls._header_towns_table(short_names))
        cls._table(title, temp, file_name, write)

    @classmethod
    def min_dist(cls, sd, file_name="min_distances.txt", short_names=True):
        cls._base_towns(sd,
                        "Таблица дистанций между городами",
                        file_name,
                        cls.FULL_MIN_DISTANCES,
                        short_names)

    @classmethod
    def ostov_dist(cls, sd, file_name="ostov_distances.txt", short_names=True):
        cls._base_towns(sd,
                        "Таблица дистанций между городами только по <Остовным> дорогам",
                        file_name,
                        cls.OSTOV_MIN_DISTANCES,
                        short_names)

    @staticmethod
    def _readable_ways(fw):
        def one(i):
            temp = list(map(lambda e: ("→".join(map(short_towns.__getitem__, e))) + "→", fw[i]))
            temp[i] = "Старт"
            return temp

        return [one(i) for i in range(len(fw))]

    @classmethod
    def min_ways(cls, fw, file_name="min_ways.txt", short_names=False):
        cls._base_towns(cls._readable_ways(fw), "Таблица маршрутов между городами",
                        file_name, cls.FULL_MIN_WAYS, short_names)

    @classmethod
    def ostov_ways(cls, fw, file_name="ostov_ways.txt", short_names=False):
        cls._base_towns(cls._readable_ways(fw), "Таблица маршрутов на остове",
                        file_name, cls.OSTOV_MIN_WAYS, short_names)

    @classmethod
    def edges_counts(cls, table_use, file_name="min_count.txt", short_names=True):
        cls._base_towns(table_use, "Количества поездок через дорогу между городами",
                        file_name, cls.FULL_EDGES_COUNTS, short_names)

    @classmethod
    def ostov_counts(cls, table_use, file_name="ostov_count.txt", short_names=True):
        cls._base_towns(table_use, "Количества поездок через дороги по Остову",
                        file_name, cls.OSTOV_EDGES_COUNTS, short_names)

    @classmethod
    def full_average(cls, td: list[float], fd: float, file_name="min_average.txt",
                     short_towns_names=False):
        temp = cls._gen_table([td],
                              cls._header_towns_table(short_towns_names, "Из города")[0],
                              ["", "Среднее расстояние"])
        cls._table("Средняя протяженность маршрутов", temp, file_name, cls.FULL_AVERAGE)
        if "w" in cls.FULL_AVERAGE and cls.FILE:
            with open(cls.RES_DIR / file_name, "a", encoding="UTF-8") as out:
                print("Всех маршрутов:", fd, "км", file=out)
        if "c" in cls.FULL_AVERAGE and cls.CONSOLE:
            print("Всех маршрутов:", fd, "км")

    @classmethod
    def ostov_average(cls, td: list[float], fd: float, file_name="ostov_average.txt",
                     short_towns_names=False):
        temp = cls._gen_table([td],
                              cls._header_towns_table(short_towns_names, "Из города")[0],
                              ["", "Среднее расстояние"])
        cls._table("Средняя протяженность маршрутов через Остов", temp, file_name, cls.OSTOV_AVERAGE)
        if "w" in cls.OSTOV_AVERAGE and cls.FILE:
            with open(cls.RES_DIR / file_name, "a", encoding="UTF-8") as out:
                print("Всех маршрутов Остова:", fd, "км", file=out)
        if "c" in cls.OSTOV_AVERAGE and cls.CONSOLE:
            print("Всех маршрутов Остова:", fd, "км")

    @classmethod
    def delta_shortest(cls, was_list, was, now_list, now, file_name="delta_distances.txt",
                       short_towns_names=False):
        delta = [f"{(now_list[i] - was_list[i]) / was_list[i] * 100}%"
                 for i in range(town_count)]
        temp = cls._gen_table([delta],
                              cls._header_towns_table(short_towns_names, "Из города")[0],
                              ["", "Выросло на"])
        cls._table("Изменение средней протяженности маршрутов", temp, file_name, cls.DELTA_MIN)
        if "w" in cls.DELTA_MIN and cls.FILE:
            with open(cls.RES_DIR / file_name, "a", encoding="UTF-8") as out:
                print(f"Общая протяженность маршрутов выросла на: {(now - was) / now * 100}%",
                      file=out)
        if "c" in cls.DELTA_MIN and cls.CONSOLE:
            print(f"Общая протяженность маршрутов выросла на {(now - was) / now * 100}%")

    @classmethod
    def delta_distances(cls, was: list[Edge], now):
        was_dist = sum(map(lambda el: el.weight, was))
        now_dist = sum(map(lambda el: el.weight, now))
        print("Суммарная длина дорог:", was_dist)
        print("Суммарная длина остовного дерева:", now_dist)
        print(f"Сократилась на {(was_dist - now_dist) / was_dist * 100}%")


def calc_ways(neighbours):
    mtx_dist = [[] for _ in range(town_count)]
    mtx_parents = [[] for _ in range(town_count)]

    for i in range(town_count):
        mtx_dist[i], mtx_parents[i] = dijkstra(i, neighbours)
    return mtx_dist, mtx_parents


def gen_full_way(distances: list[int], parents: list[int]):
    indexes = sorted(range(len(distances)), key=distances.__getitem__)
    ans: list[tuple] = [() for _ in range(len(distances))]
    for i in range(1, len(distances)):
        ans[indexes[i]] = ans[parents[indexes[i]]] + (parents[indexes[i]],)

    return ans


def gen_short(arr: list[str]):
    uniq = set()

    def short(el: str):
        a = el.split("-")
        r = [i[0] for i in a]
        while r != a:
            for i in range(len(r)):
                if "-".join(r) not in uniq:
                    break
                if len(a[i]) != len(r[i]):
                    r[i] += a[i][len(r[i])]
            else:
                continue
            break
        else:
            print("Проблема, сократить не вышло")
        rs = "-".join(r)
        uniq.add(rs)
        return rs

    return list(map(short, arr))


def calc_edge_use(fw, roads: list[Edge]):
    count = [[-float("inf") for __ in range(town_count)] for _ in range(town_count)]
    for edge in roads:
        count[edge.start][edge.end] = 0
        count[edge.end][edge.start] = 0
    for i in range(len(fw)):
        for j in range(i + 1, len(fw)):
            last = fw[i][j][0]
            for k in range(1, len(fw[i][j])):
                count[last][fw[i][j][k]] += 1
                count[fw[i][j][k]][last] += 1
                last = fw[i][j][k]
            count[last][j] += 1
            count[j][last] += 1
    return count


def calc_dist_all(dist):
    for_town = [sum(dist[i]) / town_count for i in range(town_count)]
    full = sum(for_town) / town_count
    return for_town, full


towns_roads, towns = get_edges_list()
towns_neighbours, towns = get_neighbours(pre=(towns_roads, towns))
town_count = len(towns)
towns_list = list(towns.keys())
short_towns = gen_short(towns_list)

if DEBUG:
    save_graph(
        {"town_codes": towns, "edges list": towns_roads, "neighbours list": towns_neighbours},
        dir_name=Show.RES_DIR)
    print("DEBUG: Номера городов, список ребер и список соседей сохранены")

print("Всего населенных пунктов", town_count)

distances_table, parents_table = calc_ways(towns_neighbours)
Show.min_dist(distances_table)

if DEBUG:
    print("DEBUG: Таблица предков")
    print(*parents_table, sep="\n")

full_ways = [gen_full_way(distances_table[i], parents_table[i]) for i in range(town_count)]
Show.min_ways(full_ways)
ostov_count_use = calc_edge_use(full_ways, towns_roads)
Show.edges_counts(ostov_count_use)

average_towns, average_all = calc_dist_all(distances_table)
Show.full_average(average_towns, average_all)

prim_weight, prim_tree = prim(towns_neighbours)
kruskal_weight, kruskal_tree = kruskal(towns_roads, town_count)
print("...")
if DEBUG:
    print("DEBUG: Минимальные остовные деревья")
    print("Суммарная протяженность Прима:", prim_weight, "Крускала", kruskal_weight)
    print("Дерево Прима:", prim_tree)

print("Дерево Крускала:", kruskal_tree)
print("Кол-во дорог в остове Прима:", len(prim_tree),
      "Крускала:", len(kruskal_tree),
      "Исходной сети", len(towns_roads))
print("Удаленные дороги:", [edge for edge in towns_roads if
       edge not in kruskal_tree and Edge(edge.end, edge.start, edge.weight) not in kruskal_tree])

ostov_neighbours, towns = get_neighbours(pre=(kruskal_tree, towns))
ostov_distances, ostov_parents = calc_ways(ostov_neighbours)
Show.ostov_dist(ostov_distances)

if DEBUG:
    print("DEBUG: Таблица предков остовного дерева")
    print(*ostov_parents, sep="\n")

ostov_ways = [gen_full_way(ostov_distances[i], ostov_parents[i]) for i in range(town_count)]
Show.ostov_ways(ostov_ways)

ostov_count_use = calc_edge_use(ostov_ways, towns_roads)
Show.ostov_counts(ostov_count_use)

ostov_town_average, ostov_average_dist = calc_dist_all(ostov_distances)
Show.ostov_average(ostov_town_average, ostov_average_dist)
Show.delta_shortest(average_towns, average_all, ostov_town_average, ostov_average_dist)
Show.delta_distances(towns_roads, kruskal_tree)
