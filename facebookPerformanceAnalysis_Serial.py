import numpy as np
import random
from heapq import heappop, heappush

# Utility: Read and subsample graph
def read_and_sample_graph(file_path, sample_ratio):
    edges = np.loadtxt(file_path, dtype=int)
    nodes = np.unique(edges)
    
    sampled_nodes = set(random.sample(list(nodes), int(sample_ratio * len(nodes))))
    sampled_edges = [edge for edge in edges if edge[0] in sampled_nodes and edge[1] in sampled_nodes]
    
    # Create adjacency matrix
    n = max(max(edge) for edge in sampled_edges) + 1
    adj_matrix = np.full((n, n), float('inf'))
    np.fill_diagonal(adj_matrix, 0)
    for edge in sampled_edges:
        adj_matrix[edge[0], edge[1]] = 1
        adj_matrix[edge[1], edge[0]] = 1 
    return adj_matrix, len(sampled_nodes)

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
        total_dist = np.sum([d for d in distances if d < float('inf')])  # Ignore unreachable nodes
        centrality[i] = (n - 1) / total_dist if total_dist > 0 else 0
    return centrality

# Betweenness Centrality
def betweenness_centrality(A):
    n = len(A)
    centrality = {}

    for s in range(n):
        stack, paths, sigma = [], {}, [0] * n
        sigma[s] = 1
        paths[s] = [-1]
        dist = [-1] * n
        dist[s] = 0
        queue = [s]

        while queue:
            v = queue.pop(0)
            stack.append(v)
            for w in range(n):
                if A[v][w] == float('inf') or v == w:
                    continue
                if dist[w] < 0:
                    queue.append(w)
                    dist[w] = dist[v] + 1
                if dist[w] == dist[v] + 1:
                    sigma[w] += sigma[v]
                    if w not in paths:
                        paths[w] = []
                    paths[w].append(v)

        delta = [0] * n
        while stack:
            w = stack.pop()
            for v in paths.get(w, []):
                delta[v] += (sigma[v] / sigma[w]) * (1 + delta[w])
            if w != s:
                if w not in centrality:
                    centrality[w] = 0
                centrality[w] += delta[w]

    return centrality

sample_ratio = 0.15

# Reading and sampling graph
print(f"Reading and sampling graph with ratio {sample_ratio}...")
adj_matrix, sampled_nodes = read_and_sample_graph("facebook_combined.txt", sample_ratio)
print(f"Sampled {sampled_nodes} nodes.")

# Calculate closeness centrality
print("Calculating closeness centrality...")
closeness = closeness_centrality_dijkstra(adj_matrix)

# Calculate betweenness centrality
print("Calculating betweenness centrality...")
betweenness = betweenness_centrality(adj_matrix)

# Save results to output.txt
print("Saving results to output.txt...")
with open("output.txt", "w") as f:
    for node, score in closeness.items():
        f.write(f"Closeness - Node {node}: {score}\n")
    for node, score in betweenness.items():
        f.write(f"Betweenness - Node {node}: {score}\n")

# Display results
top_5_closeness = sorted(closeness.items(), key=lambda x: x[1], reverse=True)[:5]
top_5_betweenness = sorted(betweenness.items(), key=lambda x: x[1], reverse=True)[:5]
print("Top 5 Closeness Centrality:", top_5_closeness)
print("Top 5 Betweenness Centrality:", top_5_betweenness)
print("Average Closeness Centrality:", np.mean(list(closeness.values())))
print("Average Betweenness Centrality:", np.mean(list(betweenness.values())))
