import numpy as np
import networkx as nx
from heapq import heappop, heappush
import time  # Import the time module for runtime measurement

# Utility: Read and subsample graph
def read_and_sample_graph(file_path, sample_ratio):
    edges = np.loadtxt(file_path, dtype=int)
    G = nx.Graph()
    G.add_edges_from(edges)

    # Extract the largest component
    G_largest_comp = extract_component(G)

    # Extract the top x% (sample_ratio) of nodes by degree from the largest component
    sorted_nodes = sorted(G_largest_comp.degree, key=lambda x: x[1], reverse=True)
    top_percentage_count = int(len(sorted_nodes) * sample_ratio)
    top_percentage_nodes = [node for node, degree in sorted_nodes[:top_percentage_count]]

    # Create the subgraph with the top degree nodes
    G_subgraph = G_largest_comp.subgraph(top_percentage_nodes)

    # Convert to adjacency matrix
    adj_matrix = nx.to_numpy_array(G_subgraph)
    return adj_matrix, len(top_percentage_nodes)

# Function to extract the largest connected component
def extract_component(G):
    connected_components = list(nx.connected_components(G))
    largest_comp = max(connected_components, key=len)
    G_largest_comp = G.subgraph(largest_comp)
    return G_largest_comp

# Dijkstra's algorithm for single-source shortest paths
def dijkstra(adj_matrix, source):
    n = len(adj_matrix)
    distances = [float('inf')] * n
    distances[source] = 0
    pq = [(0, source)]

    while pq:
        current_dist, current_node = heappop(pq)
        if current_dist > distances[current_node]:
            continue

        for neighbor, weight in enumerate(adj_matrix[current_node]):
            if weight != float('inf') and weight > 0:  # Valid edge
                distance = current_dist + weight
                if distance < distances[neighbor]:
                    distances[neighbor] = distance
                    heappush(pq, (distance, neighbor))
    return distances

# Closeness Centrality using Dijkstra's algorithm
def closeness_centrality_dijkstra(adj_matrix):
    n = len(adj_matrix)
    centrality = {}
    for i in range(n):
        distances = dijkstra(adj_matrix, i)
        reachable_distances = [d for d in distances if d < float('inf') and d > 0]  # Exclude unreachable nodes and self-loops
        reachable_count = len(reachable_distances)
        total_dist = sum(reachable_distances)
        if reachable_count > 0 and total_dist > 0:
            centrality[i] = float((reachable_count / (n - 1)) / (total_dist / reachable_count))  # Normalize for graph size
        else:
            centrality[i] = 0.0  # Isolated or unreachable node
    return centrality

# Betweenness Centrality
def betweenness_centrality(A):
    n = len(A)
    centrality = {v: 0.0 for v in range(n)}

    for s in range(n):
        stack = []
        paths = {v: [] for v in range(n)}
        sigma = [0] * n
        sigma[s] = 1
        dist = [-1] * n
        dist[s] = 0
        queue = [s]

        while queue:
            v = queue.pop(0)
            stack.append(v)
            for w in range(n):
                if A[v][w] == 0:
                    continue
                if dist[w] < 0:
                    queue.append(w)
                    dist[w] = dist[v] + 1
                if dist[w] == dist[v] + 1:
                    sigma[w] += sigma[v]
                    paths[w].append(v)

        delta = [0.0] * n
        while stack:
            w = stack.pop()
            for v in paths[w]:
                delta[v] += (sigma[v] / sigma[w]) * (1 + delta[w])
            if w != s:
                centrality[w] += delta[w]

    normalization_factor = 1 / ((n - 1) * (n - 2)) if n > 2 else 1
    for v in centrality:
        centrality[v] *= normalization_factor

    return centrality

# Main execution
file_path = "facebook_combined.txt"
sample_ratio = 0.25  # Sampling ratio

# Measure start time for the overall process
overall_start_time = time.time()

# Read and sample graph
print(f"Reading and sampling graph with sample ratio {sample_ratio}...")
start_time = time.time()
adj_matrix, sampled_nodes = read_and_sample_graph(file_path, sample_ratio)
end_time = time.time()
print(f"Sampled {sampled_nodes} nodes in {end_time - start_time:.2f} seconds.")

# Calculate closeness centrality
print("Calculating closeness centrality...")
start_time = time.time()
closeness = closeness_centrality_dijkstra(adj_matrix)
end_time = time.time()
print(f"Closeness centrality calculated in {end_time - start_time:.2f} seconds.")

# Calculate betweenness centrality
print("Calculating betweenness centrality...")
start_time = time.time()
betweenness = betweenness_centrality(adj_matrix)
end_time = time.time()
print(f"Betweenness centrality calculated in {end_time - start_time:.2f} seconds.")

# Measure end time for the overall process
overall_end_time = time.time()

# Format and display results
top_5_closeness = [(node, float(score)) for node, score in sorted(closeness.items(), key=lambda x: x[1], reverse=True)[:5]]
top_5_betweenness = [(node, float(score)) for node, score in sorted(betweenness.items(), key=lambda x: x[1], reverse=True)[:5]]

print("Top 5 Closeness Centrality Nodes:", top_5_closeness)
print("Top 5 Betweenness Centrality Nodes:", top_5_betweenness)

# Save results to output.txt
print("Saving results to output.txt...")
with open("output.txt", "w") as f:
    f.write("Top 5 Closeness Centrality Nodes:\n")
    for node, score in top_5_closeness:
        f.write(f"{node}: {score}\n")
    f.write("\nTop 5 Betweenness Centrality Nodes:\n")
    for node, score in top_5_betweenness:
        f.write(f"{node}: {score}\n")

# Calculate and display averages
average_closeness = float(np.mean(list(closeness.values())))
average_betweenness = float(np.mean(list(betweenness.values())))
print("Average Closeness Centrality:", average_closeness)
print("Average Betweenness Centrality:", average_betweenness)

# Display total runtime
print(f"Total Runtime: {overall_end_time - overall_start_time:.2f} seconds")