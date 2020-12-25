import numpy as np
import networkx as nx


def is_graph_of_type_path(graph: nx.Graph) -> bool:
    adjacency_matrix = nx.to_numpy_array(graph, dtype=np.int)

    # main diagonal MUST be all 0
    diagonal = [element == 0 for element in np.diagonal(adjacency_matrix)]
    if not np.alltrue(diagonal):
        return False

    # adjacency matrix MUST be symmetric. so it MUST be equal with its transposed version.
    if not np.array_equal(adjacency_matrix, np.transpose(adjacency_matrix)):
        return False

    # neighbors of main diagonal MUST be 1
    number_of_nodes = np.shape(adjacency_matrix)[0]
    for i in range(1, number_of_nodes):
        if not adjacency_matrix[i-1][i] == adjacency_matrix[i][i-1]:
            return False

    return False


def is_graph_of_type_cycle(graph: nx.Graph) -> bool:
    graph_cycles = nx.cycle_basis(graph)
    if not (len(graph_cycles) == 1 and len(graph_cycles[0]) == graph.number_of_nodes()):
        return False
    return True


def is_graph_of_type_complete(graph: nx.Graph) -> bool:
    number_of_nodes = graph.number_of_nodes()
    node_degrees = dict(graph.degree).values()
    for node_degree in node_degrees:
        if node_degree != number_of_nodes - 1:
            return False
    return True


def is_there_complete_or_circle_component(graph: nx.Graph) -> bool:
    for cc in nx.connected_components(graph):
        subgraph = graph.subgraph(cc).copy()
        if is_graph_of_type_complete(subgraph):
            return True
        if is_graph_of_type_cycle(subgraph):
            return True
        if is_graph_of_type_path(subgraph):
            return True
    return False
