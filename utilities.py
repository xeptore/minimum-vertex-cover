from algorithm import sorted_vertices_by_degree_then_label
from operator import itemgetter
import numpy as np
import graph_utils
from random import randint
import networkx as nx


def is_vertex_isolated_in_graph(vertex: int, graph: nx.Graph) -> bool:
    return graph.degree(vertex) == 0


def is_graph_disjoint_of_graphs(H: nx.Graph) -> bool:
    def connected_components(graph: nx.Graph) -> list:
        return [graph.subgraph(cc).copy() for cc in nx.connected_components(graph)]

    def is_disjoint_of_complete_graphs(connected_components: list) -> bool:
        return np.alltrue([graph_utils.is_graph_of_type_complete(subgraph) for subgraph in connected_components])

    def is_disjoint_of_circle_graphs(connected_components: list) -> bool:
        return np.alltrue([graph_utils.is_graph_of_type_cycle(subgraph) for subgraph in connected_components])

    def is_disjoint_of_path_graphs(connected_components: list) -> bool:
        return np.alltrue([graph_utils.is_graph_of_type_path(subgraph) for subgraph in connected_components])

    H_connected_components = connected_components(H)
    return is_disjoint_of_complete_graphs(H_connected_components) or is_disjoint_of_path_graphs(H_connected_components) or is_disjoint_of_circle_graphs(H_connected_components)


def select_random_node(L: list) -> int:
    return L[randint(0, len(L) - 1)]


def incorporate_components_of_graph(graph: nx.Graph) -> None:
    for connected_component in list(nx.connected_components(graph)):
        # TODO: rerun the algorithm for `connected_component`...
        pass


def calculate_T(path_lengths: list, d: int) -> list:
    return list(map(itemgetter(0), filter(lambda vertex_distance: vertex_distance[1] == d - 1, path_lengths)))


def select_vertex_with_max_degree_or_min_label(graph: nx.Graph) -> int:
    vertex_degrees = list(graph.degree)
    return sorted_vertices_by_degree_then_label(vertex_degrees)[0][0]
