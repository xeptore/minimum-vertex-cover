from graph_utils import is_there_complete_or_circle_component
from utilities import calculate_T, incorporate_components_of_graph, is_graph_disjoint_of_graphs, select_vertex_with_max_degree_or_min_label
from algorithm import H_is_all_isolated, add_leaf_parents_to_S, calculate_D, calculate_H, calculate_L, calculate_dijkstra_graph, execute_dijkstra, find_next_node
import sys
import networkx as nx


def min_vertex_cover(graph: nx.Graph, initial_node: list) -> list:
    """
    Finds minimum vertex cover of graph starting from initial set S.
    """
    S = []
    current_node = initial_node
    while True:
        S.append(current_node)
        H = calculate_H(graph, S)
        if H_is_all_isolated(H):
            break
        if is_there_complete_or_circle_component(H):
            # TODO: Replace this with real implementation
            incorporate_components_of_graph(H)
        S = add_leaf_parents_to_S(H, S)
        H2 = calculate_dijkstra_graph(graph, S)
        L = calculate_L(list(G.nodes), S, H2)
        shortest_path_lengths = execute_dijkstra(H2, current_node, L)
        d = calculate_D(shortest_path_lengths)
        T = calculate_T(shortest_path_lengths, d)
        next_node = None
        if d > 1:
            next_node = find_next_node(H2, T)
        else:
            H = calculate_H(graph, S)
            if is_graph_disjoint_of_graphs(H):
                break
            next_node = select_vertex_with_max_degree_or_min_label(H)
        current_node = next_node
    return S


G = nx.Graph()

edges = nx.read_edgelist('./edges.txt')

edges = [(int(n1), int(n2)) for (n1, n2) in edges.edges()]

G.add_edges_from(edges)

starting_vertex = int(sys.argv[1])
S = min_vertex_cover(G, starting_vertex)
print(sorted(S))
