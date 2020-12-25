from operator import itemgetter
import networkx as nx
import numpy as np


def calculate_H(G: nx.Graph, S: list) -> nx.Graph:
    H = G.copy()
    for v in S:
        H.remove_node(v)
    return H


def H_is_all_isolated(H: nx.Graph) -> bool:
    return np.alltrue([v_degree == 0 for v_degree in list(H.degree)])


def add_leaf_parents_to_S(graph: nx.Graph, S: list) -> list:
    def find_leaf_parent(graph: nx.Graph, vertex: int) -> int:
        neighbors = list(graph.neighbors(vertex))
        return neighbors[0]

    def get_next_leaf(graph: nx.Graph):
        while True:
            leaves = find_leaves(graph)
            if len(leaves) > 0:
                yield leaves[0]
            else:
                return

    def find_leaves(graph: nx.Graph) -> list:
        return list(vertex for (vertex, degree) in dict(graph.degree).items() if degree == 1)

    graph_copy = graph.copy()
    S_cloned = S.copy()
    for next_leaf in get_next_leaf(graph_copy):
        parent = find_leaf_parent(graph_copy, next_leaf)
        last_u = S_cloned.pop()
        S_cloned.append(parent)
        S_cloned.append(last_u)
        graph_copy.remove_node(parent)
    leaves = find_leaves(graph_copy)
    if len(leaves):
        return add_leaf_parents_to_S(graph_copy, S_cloned)
    return S_cloned


def calculate_dijkstra_graph(G: nx.Graph, S: list) -> nx.Graph:
    cloned = G.copy()
    for v in S[:-1]:
        cloned.remove_node(v)
    return cloned


def calculate_L(V: list, S: list, H: nx.Graph) -> list:
    cloned = V.copy()
    [cloned.remove(v) for v in S]
    H_isolated_vertices = [vertex for (vertex, degree) in dict(
        H.degree).items() if degree == 0]
    [cloned.remove(v) for v in H_isolated_vertices]
    return cloned


def find_next_node(H: nx.Graph, T: list) -> int:
    def remove_any_isolated_vertices(graph: nx.Graph) -> None:
        [graph.remove_node(vertex) for (vertex, degree) in dict(
            graph.degree).items() if degree == 0]

    T_connected_components_diff = list()
    for v in T:
        cloned = H.copy()
        remove_any_isolated_vertices(cloned)
        n_before = nx.number_connected_components(cloned)
        cloned.remove_node(v)
        remove_any_isolated_vertices(cloned)
        n_after = nx.number_connected_components(cloned)
        T_connected_components_diff.append((v, n_after - n_before))
    T_connected_components_diff.sort(key=itemgetter(1, 0))

    def no_increase_in_number_of_connected_components(T_connected_components_diff: list) -> bool:
        increase_amount = list(map(itemgetter(1), T_connected_components_diff))
        return min(increase_amount) == max(increase_amount)

    if no_increase_in_number_of_connected_components(T_connected_components_diff):
        vertex_degrees = [(vertex, H.degree(vertex)) for vertex in T]
        return sorted_vertices_by_degree_then_label(vertex_degrees)[0][0]
    else:
        return T_connected_components_diff[0][0]


def execute_dijkstra(graph: nx.Graph, source: int, L: list) -> list:
    return [(target, nx.shortest_path_length(graph, source, target)) for target in L if nx.has_path(graph, source, target)]


def calculate_D(path_lengths: list) -> int:
    return max(path_lengths, key=itemgetter(1))[1]


def sorted_vertices_by_degree_then_label(vertex_degrees: list) -> list:
    return list(sorted(vertex_degrees, key=lambda x: (x[1], -x[0]), reverse=True))
