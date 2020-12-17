import numpy as np
from operator import itemgetter
import matplotlib.pyplot as plt
import networkx as nx
import component


G = nx.Graph()

edges = nx.read_edgelist('./edges.txt')
nodes = []
with open('./nodes.txt') as nodes_file:
    nodes = [int(node.strip()) for node in nodes_file.readlines()]

edges = [(int(n1), int(n2)) for (n1, n2) in edges.edges()]

G.add_edges_from(edges)
G.add_nodes_from(nodes)


def plot_graph(graph):
    plt.plot()
    nx.draw(graph, with_labels=True, font_weight='bold')
    plt.show()


def find_leaves(graph: nx.Graph) -> list:
    degrees = dict(graph.degree)
    leaf_nodes = {node: degree for (
        node, degree) in degrees.items() if degree == 1}
    return list(leaf_nodes.keys())


def test__find_leaves_1():
    fake_graph = G.copy()
    fake_graph.remove_node(6)
    leaves = find_leaves(fake_graph)
    if leaves == [2]:
        print('Works fine :)')
    else:
        print('Test was failed :(')


def test__find_leaves_2():
    cloned = G.copy()
    [cloned.remove_node(v) for v in (22, 5, 3, 17, 23)]
    leaves = find_leaves(cloned)
    if leaves == [1, 24, 25]:
        print('Works fine :)')
    else:
        print('Test was failed :(')


def find_leaf_parent(graph: nx.Graph, node: int) -> int:
    neighbors = list(graph.neighbors(node))
    return neighbors[0]


def test__find_leaf_parent_1():
    cloned = G.copy()
    cloned.remove_node(6)
    leaves = find_leaves(cloned)
    parent = find_leaf_parent(cloned, leaves[0])
    if parent == 8:
        print('Works fine :)')
    else:
        print('Test was failed :(')


def test__find_leaf_parent_2():
    cloned = G.copy()
    [cloned.remove_node(v) for v in (22, 5, 3, 17, 23)]
    leaves = find_leaves(cloned)
    parents = [find_leaf_parent(cloned, node) for node in leaves]
    if list(sorted(parents)) == [6, 20, 21]:
        print('Works fine :)')
    else:
        print('Test was failed :(')


L = list(filter(lambda x: x not in [2, 6, 8], nodes))

S = [6]
L = T = []


def apply_leaves_to_graph(copied: nx.Graph, graph: nx.Graph, leaves: list) -> nx.Graph:
    if len(leaves) > 0:
        leaves_parents = [find_leaf_parent(copied, leaf) for leaf in leaves]
        [S.append(leaf_parent) for leaf_parent in leaves_parents]
        [graph.remove_node(node) for node in (*leaves_parents, *leaves)]
        [copied.remove_node(node) for node in (*leaves_parents, *leaves)]
        leaves = find_leaves(copied)
        return apply_leaves_to_graph(copied, graph, leaves)
    else:
        return graph


def add_path_graph_to_S(graph: nx.Graph):
    for node in list(graph.nodes)[::2]:
        S.append(node)


def collect_leaf_parents_after_delete_node(graph: nx.Graph, node: int) -> nx.Graph:
    adjacency_matrix = nx.to_numpy_array(graph, dtype=np.int)
    if component.is_graph_of_type_cycle(graph):
        graph.remove_node(node)
        add_path_graph_to_S(graph)
        graph.remove_nodes_from(list(graph.nodes))
        return graph

    copied = graph.copy()
    copied.remove_node(node)
    if nx.number_connected_components(copied) > 1:
        for connected_component in nx.connected_components(copied):
            subgraph = graph.subgraph(connected_component)
            adjacency_matrix = nx.to_numpy_array(subgraph, dtype=np.int)
            if component.is_graph_of_type_path(adjacency_matrix):
                add_path_graph_to_S(subgraph)
                graph.remove_nodes_from(subgraph.nodes)
                return graph
            elif component.is_graph_of_type_cycle(connected_component):
                pass
            elif component.is_graph_of_type_complete(connected_component):
                pass
            # else:

    leaves = find_leaves(copied)
    return apply_leaves_to_graph(copied, graph, leaves)


G = collect_leaf_parents_after_delete_node(G, 6)
assert (S == [6, 8]), "S contains unexpected nodes."
assert (6 in G.nodes)
assert (8 not in G.nodes)
assert (2 not in G.nodes)


def calculate_T(graph: nx.Graph, node: int) -> int:
    path_lengths = dict(nx.all_pairs_shortest_path_length(graph)).get(node)
    d = max(path_lengths.values())
    return [key for key in path_lengths.keys() if path_lengths.get(key) == d - 1]


T = calculate_T(G, 6)
G.remove_node(6)
assert (T == [9, 17, 21, 22, 23, 25])
assert (6 not in G.nodes)


def find_next_node(graph: nx.Graph) -> int:
    T_connected_components_diff = list()
    for v in T:
        cloned = graph.copy()
        n_before = nx.number_connected_components(cloned)
        cloned.remove_node(v)
        n_after = nx.number_connected_components(cloned)
        T_connected_components_diff.append((v, n_after - n_before))
    T_connected_components_diff.sort(key=itemgetter(1, 0))

    if min([x[1] for x in T_connected_components_diff]) == max([x[1] for x in T_connected_components_diff]):
        vertex_degrees = [(v, G.degree(v)) for v in T]
        return max(sorted(vertex_degrees, key=lambda x: x[1]), key=lambda x: x[1])[0]
    else:
        return T_connected_components_diff[0][0]


next_node = find_next_node(G)
assert next_node == 17
S.append(next_node)
assert S == [6, 8, 17]

G = collect_leaf_parents_after_delete_node(G, next_node)
assert S == [6, 8, 17, 21]
assert (21 not in G.nodes)
assert (24 not in G.nodes)

T = calculate_T(G, next_node)
G.remove_node(next_node)
assert (T == [1, 5, 11, 14, 19, 20, 23])
assert (next_node not in G.nodes)

next_node = find_next_node(G)
assert next_node == 20
S.append(next_node)
assert sorted(S) == [6, 8, 17, 20, 21]

G = collect_leaf_parents_after_delete_node(G, next_node)
assert sorted(S) == [6, 8, 17, 20, 21]

T = calculate_T(G, next_node)
G.remove_node(next_node)
assert (T == [1, 3, 12])
assert (next_node not in G.nodes)

next_node = find_next_node(G)
assert next_node == 3
S.append(next_node)
assert sorted(S) == [3, 6, 8, 17, 20, 21]

G = collect_leaf_parents_after_delete_node(G, next_node)
assert sorted(S) == [3, 5, 6, 8, 17, 20, 21]
assert (1 not in G.nodes)
assert (5 not in G.nodes)

T = calculate_T(G, next_node)
G.remove_node(next_node)
assert (T == [19, 23])
assert (3 not in G.nodes)

next_node = find_next_node(G)
assert next_node == 19
S.append(next_node)
assert sorted(S) == [3, 5, 6, 8, 17, 19, 20, 21]

G = collect_leaf_parents_after_delete_node(G, next_node)
assert sorted(S) == [3, 5, 6, 8, 17, 18, 19, 20, 21, 25]
assert (18 not in G.nodes)
assert (22 not in G.nodes)
assert (23 not in G.nodes)
assert (25 not in G.nodes)

T = calculate_T(G, next_node)
G.remove_node(next_node)
assert (T == [12])
assert (19 not in G.nodes)

next_node = find_next_node(G)
assert next_node == 12
S.append(next_node)
assert sorted(S) == [3, 5, 6, 8, 12, 17, 18, 19, 20, 21, 25]

G = collect_leaf_parents_after_delete_node(G, next_node)
assert (9 in S) or (4 in S)
assert (4 not in G.nodes)
assert (9 not in G.nodes)

T = calculate_T(G, next_node)
G.remove_node(next_node)

assert (T == [11, 13])
assert (12 not in G.nodes)


next_node = find_next_node(G)
assert next_node == 11
S.append(next_node)
assert (11 in S)

G = collect_leaf_parents_after_delete_node(G, next_node)
assert (16 in S)

T = calculate_T(G, next_node)
G.remove_node(next_node)

assert (T == [10, 13])
assert (11 not in G.nodes)
assert (14 not in G.nodes)
assert (16 not in G.nodes)

# last step❗❗❗
next_node = find_next_node(G)
assert next_node in [7, 13, 10, 15]
S.append(next_node)
assert (7 in S) or (13 in S) or (10 in S) or (15 in S)

G = collect_leaf_parents_after_delete_node(G, next_node)
assert (nx.is_empty(G))

print(sorted(S))
