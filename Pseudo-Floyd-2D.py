from mpi4py import MPI
import numpy as np
import networkx as nx

INF = np.inf


def floyd_2d_block(graph, n):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    sqrt_p = int(np.sqrt(size))
    assert sqrt_p * sqrt_p == size, "Number of processes must be a perfect square"
    block_size = n // sqrt_p

    row = rank // sqrt_p
    col = rank % sqrt_p

    local_block = np.zeros((block_size, block_size), dtype=float)
    for i in range(block_size):
        for j in range(block_size):
            global_i = row * block_size + i
            global_j = col * block_size + j
            if global_i < n and global_j < n:
                local_block[i, j] = graph[global_i, global_j]
            else:
                local_block[i, j] = INF

    for k in range(n):
        k_row_block = k // block_size
        k_col_block = k // block_size

        if col == k_col_block:
            row_segment = local_block[k % block_size, :].copy()
        else:
            row_segment = np.empty(block_size, dtype=graph.dtype)
        comm.Bcast(row_segment, root=k_col_block)

        if row == k_row_block:
            col_segment = local_block[:, k % block_size].copy()
        else:
            col_segment = np.empty(block_size, dtype=graph.dtype)
        comm.Bcast(col_segment, root=k_row_block)

        for i in range(block_size):
            for j in range(block_size):
                if local_block[i, k % block_size] != INF and col_segment[j] != INF:
                    local_block[i, j] = min(local_block[i, j],
                                            local_block[i, k % block_size] + col_segment[j])

    return local_block


def calculate_centralities(shortest_paths):
    n = shortest_paths.shape[0]
    closeness_centrality = np.zeros(n)
    betweenness_centrality = np.zeros(n)

    # Closeness Centrality
    for u in range(n):
        reachable = shortest_paths[u] != INF
        total_distance = np.sum(shortest_paths[u][reachable])
        if total_distance > 0:
            closeness_centrality[u] = (len(reachable) - 1) / total_distance

    # Betweenness Centrality
    for u in range(n):
        for s in range(n):
            if s == u:
                continue
            for t in range(n):
                if t == u or t == s or shortest_paths[s, t] == INF:
                    continue

                # Check if the node `u` lies on the shortest path from `s` to `t`
                if shortest_paths[s, u] + shortest_paths[u, t] == shortest_paths[s, t]:
                    betweenness_centrality[u] += 1

    return closeness_centrality, betweenness_centrality

def extract_component(G): 
    #Extract largest component 
    connected_components = list(nx.connected_components(G))
    largest_comp = max(connected_components, key=len)
    G_largest_comp= G.subgraph(largest_comp)

    #Extract 10% from largest component 
    sorted_nodes = sorted(G_largest_comp.degree, key=lambda x: x[1], reverse=True)
    top_10_percent_count = int(len(sorted_nodes) * 0.015)
    top_10_percent_nodes = [node for node, degree in sorted_nodes[:top_10_percent_count]]
    G_10_subgraph_largest_comp = G_largest_comp.subgraph(top_10_percent_nodes)
  

    #Extract largest component from 10%
    connected_components = list(nx.connected_components(G_10_subgraph_largest_comp))
    largest_comp_10 = max(connected_components, key=len)
    G_largest_comp_10 = G_10_subgraph_largest_comp.subgraph(largest_comp_10)

    return G_largest_comp_10 



def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # Define the NetworkX graph (replace with your graph)
    G = nx.read_edgelist("output.txt", nodetype=int, data=(("weight", float),))
    if rank == 0:
        G = nx.DiGraph()
        G.add_weighted_edges_from([
            (0, 1, 3),
            (0, 3, 7),
            (1, 2, 2),
            (2, 0, 5),
            (2, 3, 1),
            (3, 0, 2)
        ])
        # Convert the graph to an adjacency matrix
        full_graph = nx.to_numpy_array(G, weight="weight", nonedge=INF)
    else:
        full_graph = None

    n = comm.bcast(full_graph.shape[0] if rank == 0 else None, root=0)
    full_graph = comm.bcast(full_graph if rank == 0 else None, root=0)

    local_result = floyd_2d_block(full_graph, n)

    final_result = None
    if rank == 0:
        final_result = np.zeros((n, n), dtype=float)
    gathered_blocks = comm.gather(local_result, root=0)

    if rank == 0:
        sqrt_p = int(np.sqrt(size))
        block_size = n // sqrt_p
        for r in range(sqrt_p):
            for c in range(sqrt_p):
                block = gathered_blocks[r * sqrt_p + c]
                for i in range(block_size):
                    for j in range(block_size):
                        global_i = r * block_size + i
                        global_j = c * block_size + j
                        if global_i < n and global_j < n:
                            final_result[global_i, global_j] = block[i, j]

        closeness_centrality, betweenness_centrality = calculate_centralities(final_result)

        print("\nShortest Paths Matrix:")
        print(final_result)

        print("\nCloseness Centrality:")
        print(closeness_centrality)

        print("\nBetweenness Centrality:")
        print(betweenness_centrality)


if __name__ == "__main__":
    main()
