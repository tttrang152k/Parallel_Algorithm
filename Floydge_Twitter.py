import networkx as nx
import numpy as np
import time

def floyd_warshall_from_graph(graph):
    """Applies Floyd-Warshall to a NetworkX graph."""
    nodes = list(graph.nodes)
    n = len(nodes)
    node_idx = {node: i for i, node in enumerate(nodes)}  # Map nodes to indices
    
    # Initialize distance matrix with infinity
    dist = np.full((n, n), float('inf'))
    np.fill_diagonal(dist, 0)  # Distance to self is zero
    
    # Populate distances from edges
    for u, v, data in graph.edges(data=True):
        weight = data.get('weight', 1)  # Default weight is 1
        dist[node_idx[u]][node_idx[v]] = weight
        dist[node_idx[v]][node_idx[u]] = weight  # Undirected graph
    
    # Floyd-Warshall algorithm
    for k in range(n):
        for i in range(n):
            for j in range(n):
                if dist[i][j] > dist[i][k] + dist[k][j]:
                    dist[i][j] = dist[i][k] + dist[k][j]
    
    return dist, nodes

def calculate_closeness_centrality(dist):
    """Calculate closeness centrality using the distance matrix."""
    n = len(dist)
    closeness = []
    for i in range(n):
        total_dist = sum(dist[i][j] for j in range(n) if i != j and dist[i][j] != float('inf'))
        reachable_nodes = sum(1 for j in range(n) if dist[i][j] != float('inf'))
        if total_dist > 0 and reachable_nodes > 1:
            closeness.append((reachable_nodes - 1) / total_dist)
        else:
            closeness.append(0)
    return closeness

def calculate_betweenness_centrality(graph, dist, nodes):
    """Calculate betweenness centrality using shortest paths."""
    n = len(nodes)
    betweenness = {node: 0 for node in nodes}

    # For each pair of nodes, compute shortest paths
    for s in nodes:
        for t in nodes:
            if s != t:
                # Check if there's a path between s and t
                try:
                    paths = list(nx.all_shortest_paths(graph, source=s, target=t, weight="weight"))
                except nx.NetworkXNoPath:
                    # No path exists, skip this pair
                    continue
                
                num_paths = len(paths)
                if num_paths > 0:
                    for path in paths:
                        # Exclude the start and end nodes
                        for node in path[1:-1]:
                            betweenness[node] += 1 / num_paths

    # Normalize by dividing by 2 for undirected graphs
    for node in betweenness:
        betweenness[node] /= 2

    return betweenness

def extract_component(G): 
    #Extract largest component 
    connected_components = list(nx.connected_components(G))
    largest_comp = max(connected_components, key=len)
    G_largest_comp= G.subgraph(largest_comp)

    #Extract 10% from largest component 
    sorted_nodes = sorted(G_largest_comp.degree, key=lambda x: x[1], reverse=True)
    top_10_percent_count = int(len(sorted_nodes) * 0.05)
    top_10_percent_nodes = [node for node, degree in sorted_nodes[:top_10_percent_count]]
    G_10_subgraph_largest_comp = G_largest_comp.subgraph(top_10_percent_nodes)


    return G_10_subgraph_largest_comp 

def write_output_to_file(file_name, closeness, betweenness, nodes):
    """Writes centrality measures to a file."""
    with open(file_name, "w") as f:
        f.write("Closeness Centrality:\n")
        for node, value in zip(nodes, closeness):
            f.write(f"Node {node}: {value:.4f}\n")
        f.write("\nBetweenness Centrality:\n")
        for node, value in betweenness.items():
            f.write(f"Node {node}: {value:.4f}\n")

# Load graph
G = nx.read_edgelist("fb_output.txt", nodetype=int, data=(("weight", float),))

# Start total execution time measurement
total_start_time = time.time()

# Process graph
G1 = extract_component(G)
N_ori = G1.number_of_nodes()
K_ori = G1.number_of_edges()
print("EXTRACTED Graph - Nodes: ", N_ori, " Edges: ", K_ori)


# Floyd-Warshall computation
fw_start_time = time.time()
dist, nodes = floyd_warshall_from_graph(G1)
fw_end_time = time.time()

# Centrality calculations
centrality_start_time = time.time()
closeness = calculate_closeness_centrality(dist)
betweenness = calculate_betweenness_centrality(G1, dist, nodes)
centrality_end_time = time.time()

# End total execution time
total_end_time = time.time()

# Write to output file
write_output_to_file("output.txt", closeness, betweenness, nodes)

# Print top 5 nodes and average centrality values
closeness_dict = dict(zip(nodes, closeness))
average_closeness = np.mean(list(closeness_dict.values()))
average_betweenness = np.mean(list(betweenness.values()))

print("\nRuntime Statistics:")
print(f"Floyd-Warshall Computation Time: {fw_end_time - fw_start_time:.4f} seconds")
print(f"Centrality Calculation Time: {centrality_end_time - centrality_start_time:.4f} seconds")
print(f"Total Execution Time: {total_end_time - total_start_time:.4f} seconds")

print("\nTop 5 Closeness Centrality Nodes:")
for node, value in sorted(closeness_dict.items(), key=lambda x: x[1], reverse=True)[:5]:
    print(f"Node {node}: {value:.4f}")

print("\nTop 5 Betweenness Centrality Nodes:")
for node, value in sorted(betweenness.items(), key=lambda x: x[1], reverse=True)[:5]:
    print(f"Node {node}: {value:.4f}")

print(f"\nAverage Closeness Centrality: {average_closeness:.4f}")
print(f"Average Betweenness Centrality: {average_betweenness:.4f}")
