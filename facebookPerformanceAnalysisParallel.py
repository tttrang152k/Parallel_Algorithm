import sys
from mpi4py import MPI
import numpy as np
from heapq import heappop, heappush

# Read and sample graph
def read_and_sample_graph(file_path, sample_ratio):
    edges = np.loadtxt(file_path, dtype=int)
    nodes = np.unique(edges)
    
    sampled_nodes = set(np.random.choice(list(nodes), int(sample_ratio * len(nodes)), replace=False))
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

# Main function
def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # File and sampling settings
    file_path = "facebook_combined.txt"
    sample_ratio = 0.15  # Sampling ratio

    if rank == 0:
        adj_matrix, sampled_nodes = read_and_sample_graph(file_path, sample_ratio)
        n = len(adj_matrix)
    else:
        adj_matrix = None
        n = None

    # Broadcast adjacency matrix and number of nodes
    adj_matrix = comm.bcast(adj_matrix if rank == 0 else None, root=0)
    n = comm.bcast(n if rank == 0 else None, root=0)

    # Number of nodes per process
    nodes_per_process = n // size
    start = rank * nodes_per_process
    end = n if rank == size - 1 else start + nodes_per_process

    # Synchronize processes
    comm.barrier()

    # Calculate centrality for assigned nodes
    local_centrality = {}
    for node in range(start, end):
        distances = dijkstra(adj_matrix, node)
        reachable_distances = [d for d in distances if d < float('inf')]  # Ignore unreachable nodes
        
        # Calculate total distance and number of reachable nodes
        total_dist = np.sum(reachable_distances)
        reachable_nodes = len(reachable_distances)
        
        # Adjust centrality calculation: normalize by total distance and number of reachable nodes
        if total_dist > 0 and reachable_nodes > 1:
            local_centrality[node] = (reachable_nodes - 1) / total_dist
        else:
            local_centrality[node] = 0  # Node is isolated or disconnected

    # Gather results at root process
    all_centrality = comm.gather(local_centrality, root=0)

    if rank == 0:
        # Combine results from all processes
        centrality = {}
        for sub_centrality in all_centrality:
            centrality.update(sub_centrality)

        # Output 
        top_5 = sorted(centrality.items(), key=lambda x: x[1], reverse=True)[:5]
        average_centrality = np.mean(list(centrality.values()))

        print("Top 5 Closeness Centrality Nodes:", top_5)
        print("Average Closeness Centrality:", average_centrality)
        sys.stdout.flush()

if __name__ == "__main__":
    main()
