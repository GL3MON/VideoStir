"""Compute graph metrics for VideoStir spatiotemporal graphs."""

import json
import networkx as nx
import torch


def compute_graph_metrics(graph: nx.Graph) -> dict:
    """Compute comprehensive graph metrics.

    Args:
        graph: NetworkX graph to analyze

    Returns:
        Dictionary of computed metrics
    """
    num_nodes = graph.number_of_nodes()
    num_edges = graph.number_of_edges()

    # Edge type breakdown
    temporal_edges = sum(1 for u, v, d in graph.edges(data=True) if d.get('type') == 'temporal')
    spatial_edges = num_edges - temporal_edges

    # 1. Graph Density
    # D = 2|E| / (|V| × (|V| - 1))
    if num_nodes > 1:
        max_edges = num_nodes * (num_nodes - 1)
        density = (2 * num_edges) / max_edges
    else:
        density = 0.0

    # 2. Average Node Degree
    # d̄ = (2 × |E|) / |V|
    avg_degree = (2 * num_edges) / num_nodes if num_nodes > 0 else 0.0

    # 3. Clustering Coefficient
    # C = (1/|V|) × ΣCi
    clustering = nx.average_clustering(graph)

    # 4. Degree Distribution stats
    degrees = [d for n, d in graph.degree()]
    degree_mean = torch.tensor(degrees).float().mean().item()
    degree_std = torch.tensor(degrees).float().std().item()
    degree_min = min(degrees)
    degree_max = max(degrees)

    # 5. Connected Components
    num_components = nx.number_connected_components(graph)
    largest_component = max(nx.connected_components(graph), key=len)
    largest_ratio = len(largest_component) / num_nodes if num_nodes > 0 else 0.0

    # 6. Path metrics
    try:
        avg_path_length = nx.average_shortest_path_length(graph)
    except nx.NetworkXError:
        avg_path_length = float('inf')

    # 7. Graph transitivity (same as clustering coefficient)
    transitivity = nx.transitivity(graph)

    # 8. Edge weight stats (for spatial edges)
    spatial_weights = [d.get('weight', 0.0) for u, v, d in graph.edges(data=True)
                      if d.get('type') == 'spatial']

    if spatial_weights:
        weights_tensor = torch.tensor(spatial_weights)
        spatial_weight_mean = weights_tensor.mean().item()
        spatial_weight_std = weights_tensor.std().item()
        spatial_weight_min = weights_tensor.min().item()
        spatial_weight_max = weights_tensor.max().item()
    else:
        spatial_weight_mean = spatial_weight_std = 0.0
        spatial_weight_min = spatial_weight_max = 0.0

    # 9. Node attribute stats
    node_features = []
    for node, attrs in graph.nodes(data=True):
        if 'feature' in attrs:
            node_features.append(attrs['feature'])

    if node_features:
        features_tensor = torch.stack(node_features)
        feature_mean = features_tensor.mean().item()
        feature_std = features_tensor.std().item()

    return {
        # Basic stats
        'num_nodes': num_nodes,
        'num_edges': num_edges,
        'edge_type_breakdown': {
            'temporal': temporal_edges,
            'spatial': spatial_edges
        },

        # Density metrics
        'density': density,
        'max_possible_edges': num_nodes * (num_nodes - 1) if num_nodes > 1 else 0,
        'average_degree': avg_degree,

        # Clustering
        'clustering_coefficient': clustering,
        'transitivity': transitivity,

        # Degree distribution
        'degree_stats': {
            'mean': degree_mean,
            'std': degree_std,
            'min': degree_min,
            'max': degree_max
        },

        # Connectivity
        'num_connected_components': num_components,
        'largest_component_size': len(largest_component),
        'largest_component_ratio': largest_ratio,

        # Path metrics
        'avg_shortest_path_length': avg_path_length,

        # Edge weights
        'spatial_edge_weights': {
            'mean': spatial_weight_mean,
            'std': spatial_weight_std,
            'min': spatial_weight_min,
            'max': spatial_weight_max
        },

        # Features
        'features_stats': {
            'mean': feature_mean if node_features else 0.0,
            'std': feature_std if node_features else 0.0
        }
    }


def print_metrics_report(metrics: dict) -> str:
    """Format metrics as a readable report."""
    lines = []
    lines.append("# Graph Metrics Analysis")
    lines.append("")
    lines.append("## Basic Statistics")
    lines.append("")
    lines.append(f"| Metric | Value |")
    lines.append(f"|--------|-------|")
    lines.append(f"| Nodes (|V|) | {metrics['num_nodes']} |")
    lines.append(f"| Edges (|E|) | {metrics['num_edges']} |")
    lines.append(f"| Temporal Edges | {metrics['edge_type_breakdown']['temporal']} |")
    lines.append(f"| Spatial Edges | {metrics['edge_type_breakdown']['spatial']} |")
    lines.append("")

    lines.append("## Graph Density")
    lines.append("")
    lines.append(f"- **Density (D)**: {metrics['density']:.4f}")
    lines.append(f"  - Formula: D = 2|E| / (|V| × (|V| - 1))")
    lines.append(f"  - Maximum possible edges: {metrics['max_possible_edges']}")
    lines.append(f"  - **Interpretation**: {'Maximally dense (D=1.0)' if abs(metrics['density'] - 1.0) < 0.001 else 'Partial density'}")
    lines.append("")

    lines.append("## Average Node Degree")
    lines.append("")
    lines.append(f"- **Average Degree (d̄)**: {metrics['average_degree']:.2f}")
    lines.append(f"  - Formula: d̄ = (2 × |E|) / |V|")
    lines.append(f"  - **Interpretation**: Each node connects to ~{metrics['average_degree']:.0f} other nodes on average")
    lines.append("")

    lines.append("## Degree Distribution")
    lines.append("")
    lines.append(f"| Stat | Value |")
    lines.append(f"|------|-------|")
    lines.append(f"| Mean | {metrics['degree_stats']['mean']:.2f} |")
    lines.append(f"| Std Dev | {metrics['degree_stats']['std']:.2f} |")
    lines.append(f"| Min | {metrics['degree_stats']['min']} |")
    lines.append(f"| Max | {metrics['degree_stats']['max']} |")
    lines.append("")

    lines.append("## Clustering Coefficient")
    lines.append("")
    lines.append(f"- **C** (average): {metrics['clustering_coefficient']:.4f}")
    lines.append(f"- **Transitivity**: {metrics['transitivity']:.4f}")
    lines.append(f"  - **Interpretation**: {'High local connectivity' if metrics['clustering_coefficient'] > 0.7 else 'Moderate/low local connectivity'}")
    lines.append("")

    lines.append("## Connectivity")
    lines.append("")
    lines.append(f"| Metric | Value |")
    lines.append(f"|--------|-------|")
    lines.append(f"| Connected Components | {metrics['num_connected_components']} |")
    lines.append(f"| Largest Component Size | {metrics['largest_component_size']} |")
    lines.append(f"| Largest Component Ratio (R) | {metrics['largest_component_ratio']:.4f} |")
    lines.append(f"  - **Interpretation**: {'Fully connected (R=1.0)' if abs(metrics['largest_component_ratio'] - 1.0) < 0.001 else 'Partially connected'}")
    lines.append("")

    lines.append("## Path Metrics")
    lines.append("")
    lines.append(f"- **Avg Shortest Path Length**: {metrics['avg_shortest_path_length']:.4f}")
    lines.append("")

    lines.append("## Spatial Edge Weights")
    lines.append("")
    lines.append(f"| Stat | Value |")
    lines.append(f"|------|-------|")
    lines.append(f"| Mean | {metrics['spatial_edge_weights']['mean']:.4f} |")
    lines.append(f"| Std Dev | {metrics['spatial_edge_weights']['std']:.4f} |")
    lines.append(f"| Min | {metrics['spatial_edge_weights']['min']:.4f} |")
    lines.append(f"| Max | {metrics['spatial_edge_weights']['max']:.4f} |")
    lines.append("")

    lines.append("---")
    lines.append("")
    lines.append("*Metrics computed from VideoStir spatiotemporal graph*")

    return "\n".join(lines)


if __name__ == "__main__":
    # Load segment features from checkpoint
    import sys
    sys.path.insert(0, "/hfcache/harissh/VideoStir")

    from inference.retrieval import SpatiotemporalGraphBuilder

    # Load checkpointed features
    cache_dir = "/hfcache/harissh/VideoStir/artifacts/output/test_medical_video/cache"
    with open(f"{cache_dir}/segment_features.json", "r") as f:
        segment_features = json.load(f)

    # Reconstruct features as tensors
    for seg in segment_features:
        if 'feature' in seg:
            seg['image_features'] = torch.tensor(seg['feature']).unsqueeze(0)

    # Build graph
    builder = SpatiotemporalGraphBuilder(temporal_weight=1.0)
    graph = builder.build(segment_features)

    # Compute metrics
    metrics = compute_graph_metrics(graph)

    # Print metrics
    report = print_metrics_report(metrics)
    print(report)

    # Save to file
    with open("/hfcache/harissh/VideoStir/artifacts/graph_metrics.md", "w") as f:
        f.write(report)

    print("\n\nMetrics saved to artifacts/graph_metrics.md")