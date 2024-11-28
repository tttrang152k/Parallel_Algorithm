import numpy as np
import networkx as nx
import time

INF = np.inf

def floyd_warshall(graph):
    """
    Serial implementation of the Floyd-Warshall algorithm.
    
    Parameters:
        graph (numpy.ndarray): Adjacency matrix of the graph.

    Returns:
        numpy.ndarray: Distance matrix with shortest paths.
    """
    n = graph.shape[0]
    dist = graph.copy()

    for k in range(n):
        for i in range(n):
            for j in range(n):
                if dist[i, k] != INF and dist[k, j] != INF:
                    dist[i, j] = min(dist[i, j], dist[i, k] + dist[k, j])

    return dist

def calculate_centralities(shortest_paths):
    n = shortest_paths.shape[0]
    closeness_centrality = np.zeros(n)
    betweenness_centrality = np.zeros(n)

    # Closeness Centrality with normalization
    for u in range(n):
        reachable = shortest_paths[u] != INF
        num_reachable = np.sum(reachable) - 1  # Exclude the node itself
        total_distance = np.sum(shortest_paths[u][reachable])
        if total_distance > 0 and num_reachable > 0:
            closeness_centrality[u] = (num_reachable / total_distance) * ((num_reachable) / (n - 1))

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
    # Extract largest connected component 
    connected_components = list(nx.connected_components(G))
    largest_comp = max(connected_components, key=len)
    G_largest_comp = G.subgraph(largest_comp)

    # Extract top 10% nodes from the largest component 
    sorted_nodes = sorted(G_largest_comp.degree, key=lambda x: x[1], reverse=True)
    top_10_percent_count = int(len(sorted_nodes) * 0.015)
    top_10_percent_nodes = [node for node, degree in sorted_nodes[:top_10_percent_count]]
    G_10_subgraph_largest_comp = G_largest_comp.subgraph(top_10_percent_nodes)

    return G_10_subgraph_largest_comp

def main():
    # Start measuring the total runtime
    total_start_time = time.time()

    # Read the graph and extract the largest component
    G = nx.read_edgelist("fb_output.txt", nodetype=int, data=("weight", float))
    G = extract_component(G)
    N_ori = G.number_of_nodes()
    K_ori = G.number_of_edges()
    print("Graph - Nodes: ", N_ori, " Edges: ", K_ori)

    # Convert to adjacency matrix
    full_graph = nx.to_numpy_array(G, weight="weight", nonedge=INF)

    # Compute all-pairs shortest paths using Floyd-Warshall algorithm
    fw_start_time = time.time()
    shortest_paths = floyd_warshall(full_graph)
    fw_end_time = time.time()

    # Calculate centralities
    centrality_start_time = time.time()
    closeness_centrality, betweenness_centrality = calculate_centralities(shortest_paths)
    centrality_end_time = time.time()

    # Write results to file
    with open("program_output.txt", "w") as f:
        f.write("Closeness Centrality:\n")
        for idx, val in enumerate(closeness_centrality):
            f.write(f"Node {idx}: {val:.4f}\n")

        f.write("\nBetweenness Centrality:\n")
        for idx, val in enumerate(betweenness_centrality):
            f.write(f"Node {idx}: {val:.4f}\n")

    # Print runtime statistics
    total_end_time = time.time()
    print("\nRuntime Statistics:")
    print(f"Floyd-Warshall Computation Time: {fw_end_time - fw_start_time:.4f} seconds")
    print(f"Centrality Calculation Time: {centrality_end_time - centrality_start_time:.4f} seconds")
    print(f"Total Execution Time: {total_end_time - total_start_time:.4f} seconds")

    # Print top 5 nodes with highest centrality values
    top_closeness = np.argsort(-closeness_centrality)[:5]
    top_betweenness = np.argsort(-betweenness_centrality)[:5]

    print("\nTop 5 Nodes by Closeness Centrality:")
    for idx in top_closeness:
        print(f"Node {idx}: {closeness_centrality[idx]:.4f}")

    print("\nTop 5 Nodes by Betweenness Centrality:")
    for idx in top_betweenness:
        print(f"Node {idx}: {betweenness_centrality[idx]:.4f}")

    # Print average centrality values
    avg_closeness = np.mean(closeness_centrality)
    avg_betweenness = np.mean(betweenness_centrality)

    print(f"\nAverage Closeness Centrality: {avg_closeness:.4f}")
    print(f"Average Betweenness Centrality: {avg_betweenness:.4f}")

if __name__ == "__main__":
    main()
