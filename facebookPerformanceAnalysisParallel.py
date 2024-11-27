import sys
from mpi4py import MPI
import numpy as np
from heapq import heappop, heappush
import networkx as nx
import time  # Import the time module

# Read and sample graph (modified to use largest component and degree-based sampling)
def read_and_sample_graph(file_path, sample_ratio):
    edges = np.loadtxt(file_path, dtype=int)
    G = nx.Graph()
    G.add_edges_from(edges)

    # Extract the largest component
    G_largest_comp = extract_component(G)

    # Extract the top x% (in this case sample_ratio% of nodes) of the largest component by degree
    sorted_nodes = sorted(G_largest_comp.degree, key=lambda x: x[1], reverse=True)
    top_percentage_count = int(len(sorted_nodes) * sample_ratio)
    top_percentage_nodes = [node for node, degree in sorted_nodes[:top_percentage_count]]

    # Create the subgraph with the top degree nodes
    G_10_subgraph_largest_comp = G_largest_comp.subgraph(top_percentage_nodes)

    # Convert to adjacency matrix
    adj_matrix = nx.to_numpy_array(G_10_subgraph_largest_comp)
    
    return adj_matrix, len(top_percentage_nodes)

# Function to extract the largest component
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

# Main function
def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # File and sampling settings
    file_path = "facebook_combined.txt"
    sample_ratio = 0.3  # Sampling ratio

    # Measure start time
    start_time = time.time()

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
        
        # Measure end time
        end_time = time.time()
        runtime = end_time - start_time

        print("Top 5 Closeness Centrality Nodes:", top_5)
        print("Average Closeness Centrality:", average_centrality)
        print(f"Runtime: {runtime:.2f} seconds")
        sys.stdout.flush()

if __name__ == "__main__":
    main()
