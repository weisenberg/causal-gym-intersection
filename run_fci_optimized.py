import pandas as pd
print("Starting script... Imported pandas", flush=True)
import numpy as np
print("Imported numpy", flush=True)
# from causallearn.search.ConstraintBased.FCI import fci
# from causallearn.utils.GraphUtils import GraphUtils
# from causallearn.utils.cit import fisherz
# import matplotlib.pyplot as plt
import io
import sys

def run_fci_discovery(input_file="causal_discovery_dataset.csv", 
                      output_image="causal_graph_fci_optimized.png",
                      max_samples=5000,
                      select_key_vars=True):
    """
    Optimized FCI discovery with options to reduce computational complexity.
    
    Args:
        input_file: Path to the CSV dataset
        output_image: Path to save the output graph
        max_samples: Maximum number of samples to use (reduces computation time)
        select_key_vars: If True, focus on key variables instead of all observations
    """
    print(f"Loading data from {input_file}...")
    
    # Delayed imports to avoid hang at startup
    from causallearn.search.ConstraintBased.FCI import fci
    from causallearn.utils.GraphUtils import GraphUtils
    from causallearn.utils.cit import fisherz
    import matplotlib.pyplot as plt
    print("Imports complete.", flush=True)
    try:
        df = pd.read_csv(input_file)
    except FileNotFoundError:
        print(f"Error: File {input_file} not found.")
        return

    print(f"Original data shape: {df.shape}")
    
    # Exclude non-physical/metadata variables as requested
    exclude_cols = ['episode', 'step', 'reward', 'done', 'action']
    print(f"Excluding requested variables: {exclude_cols}")
    df = df.drop(columns=[c for c in exclude_cols if c in df.columns], errors='ignore')
    
    # Preprocessing
    # Ensure all columns are numeric
    df = df.select_dtypes(include=[np.number])
    
    # Drop any rows with NaN
    df = df.dropna()
    
    # Drop Constant Columns (singularity source)
    print(f"Dropping constant columns...")
    df = df.loc[:, df.nunique() > 1]
    
    # Drop Duplicate Columns (singularity source)
    # Transpose-based drop_duplicates is too slow for 1M rows.
    # We will rely on the correlation check (below) to catch duplicates (corr=1.0).
    print(f"Skipping exact duplicate column check (relying on correlation check)...")
    # df = df.T.drop_duplicates().T
    
    # Drop Highly Correlated Columns (singularity source)
    print(f"Checking for high correlation...")
    # Calculate correlation matrix
    corr_matrix = df.corr().abs()
    # Select upper triangle of correlation matrix
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    # Find features with correlation greater than 0.98
    to_drop = [column for column in upper.columns if any(upper[column] > 0.98)]
    
    if to_drop:
        print(f"Dropping {len(to_drop)} highly correlated features: {to_drop}")
        df = df.drop(columns=to_drop)
    
    # Sample data if too large
    if len(df) > max_samples:
        print(f"Sampling {max_samples} rows from {len(df)} total rows...")
        df = df.sample(n=max_samples, random_state=42)
    
    # Select key variables if requested
    if select_key_vars:
        print("Selecting key variables for analysis...")
        
        # Identify key variables
        key_vars = []
        
        # Add action variables
        if 'action' in df.columns:
            key_vars.append('action')
        
        # Add reward and done
        if 'reward' in df.columns:
            key_vars.append('reward')
        if 'done' in df.columns:
            key_vars.append('done')
        
        # Add hidden/environmental variables
        env_vars = ['temperature', 'traffic_density', 'pedestrian_density', 
                   'driver_impatience', 'npc_color', 'npc_size', 'roughness']
        for var in env_vars:
            if var in df.columns:
                key_vars.append(var)
        
        # Add Simple Env variables (Updated Names)
        simple_vars = [
            'agent_x_position', 'agent_y_position', 'agent_speed_bin', 'agent_angle_bin',
            'obs_lidar_front', 'obs_lidar_back', 'obs_lidar_left', 'obs_lidar_right',
            'obs_tl_state', 'obs_tl_timer'
        ]
        for var in simple_vars:
            if var in df.columns:
                key_vars.append(var)
                
        # NPCs and Peds
        # Check for npc_1_x_position, npc_1_angle_bin, etc.
        for i in range(1, 3): # 1 and 2
            prefix = f'npc_{i}'
            potential_cols = [
                f'{prefix}_x_position', f'{prefix}_y_position', 
                f'{prefix}_angle_bin', f'{prefix}_present',
                f'{prefix}_speed_bin', f'{prefix}_stopping_dist'
            ]
            for col in potential_cols:
                if col in df.columns:
                    key_vars.append(col)
                    
            prefix = f'ped_{i}'
            potential_cols = [
                f'{prefix}_x_position', f'{prefix}_y_position', 
                f'{prefix}_jaywalking', f'{prefix}_present'
            ]
            for col in potential_cols:
                if col in df.columns:
                    key_vars.append(col)

        # Filter to key variables
        df = df[key_vars]
        print(f"Reduced to {len(key_vars)} key variables: {key_vars}")
    
    # Convert to numpy array
    data = df.to_numpy()
    labels = df.columns.tolist()
    
    print(f"\nFinal data shape: {data.shape}")
    print(f"Variables: {labels}")
    
    print("\nRunning FCI algorithm (this may take a moment)...")
    print("Progress will be shown below:")
    
    # Run FCI with slightly relaxed alpha for faster computation
    # alpha=0.05 is standard, but 0.01 can be faster with similar results
    G, edges = fci(data, independence_test_method=fisherz, alpha=0.05, verbose=True)
    
    print("\nCausal Graph Discovered!")
    
    # Print Edges
    print(f"\nDiscovered {len(edges)} edges:")
    for i, edge in enumerate(edges, 1):
        print(f"{i}. {edge}")
        
    # Visualization
    print(f"\nGenerating visualization to {output_image}...")
    try:
        # Try PyDot (needs Graphviz installed on OS)
        pydot_graph = GraphUtils.to_pydot(G, labels=labels)
        pydot_graph.write_png(output_image)
        print(f"✓ Visualization saved successfully to {output_image} (via Graphviz)")
    except Exception as e:
        print(f"Graphviz failed ({e}). Falling back to NetworkX...")
        try:
            import networkx as nx
            import matplotlib.pyplot as plt
            
            # Convert CausalGraph to NetworkX
            # G.graph is the matrix. 
            # -1: -- (Selection Bias?), 1: ->, 2: <->
            # We can also parse 'edges' list.
            
            # Create mapping from X1..Xn to actual labels
            nodes = G.nodes # List of Node objects
            node_map = {}
            for i, node in enumerate(nodes):
                # i matches the index in labels (assuming alignment)
                # node.get_name() returns "X1", "X2", etc.
                if i < len(labels):
                    node_map[node.get_name()] = labels[i]
                else:
                    node_map[node.get_name()] = node.get_name()

            nx_graph = nx.DiGraph()
            # Add nodes (using real names)
            for label in labels:
                nx_graph.add_node(label)
                
            # Parse edges from G
            for edge in edges:
                n1 = edge.get_node1()
                n2 = edge.get_node2()
                name1 = n1.get_name()
                name2 = n2.get_name()
                
                # Map to real names
                label1 = node_map.get(name1, name1)
                label2 = node_map.get(name2, name2)
                
                # Check endpoints
                # end1 goes to node1
                end1 = edge.get_endpoint1() 
                end2 = edge.get_endpoint2()
                
                # Standard conversion for display
                # Tail (3) -- Arrow (2)  =>  n1 -> n2
                # Arrow (2) -- Arrow (2) => n1 <-> n2 (Bi-directed)
                # Circle (1) -- Arrow (2) => n1 o-> n2
                
                # For simplified viz, we just draw arrows if there's an arrow head
                
                if end2 == 2: # Arrow at n2
                    nx_graph.add_edge(label1, label2)
                elif end1 == 2: # Arrow at n1
                    nx_graph.add_edge(label2, label1)
                # If both are circle/tail, maybe undirected line? NetworkX DiGraph handles directed.
                # If Circle (1), strictly it's "o" endpoint.
                
            plt.figure(figsize=(20, 15)) # Increased size for readability
            pos = nx.spring_layout(nx_graph, k=0.8, iterations=100) # Increased spacing
            nx.draw(nx_graph, pos, with_labels=True, node_color='lightblue', 
                    node_size=3000, font_size=8, font_weight='bold', arrows=True)
            plt.title("FCI Causal Graph (NetworkX Fallback)")
            plt.savefig(output_image)
            plt.close()
            print(f"✓ Visualization saved successfully to {output_image} (via NetworkX)")
            
        except Exception as e2:
             print(f"Error generating fallback visualization: {e2}")
    
    # Save edges to a text file for easy review
    edges_file = output_image.replace('.png', '_edges.txt')
    with open(edges_file, 'w') as f:
        f.write(f"FCI Causal Discovery Results\n")
        f.write(f"{'='*50}\n\n")
        f.write(f"Dataset: {input_file}\n")
        f.write(f"Samples used: {data.shape[0]}\n")
        f.write(f"Variables analyzed: {data.shape[1]}\n\n")
        f.write(f"Variables:\n")
        for i, label in enumerate(labels, 1):
            f.write(f"  {i}. {label}\n")
        f.write(f"\nDiscovered Edges ({len(edges)} total):\n")
        f.write(f"{'-'*50}\n")
        for i, edge in enumerate(edges, 1):
            f.write(f"{i}. {edge}\n")
    print(f"✓ Edges saved to {edges_file}")

if __name__ == "__main__":
    # Run optimized version with key variables only
    run_fci_discovery(
        input_file="simple_env_data.csv",
        output_image="causal_graph_fci_optimized.png",
        max_samples=5000,  # Use 5000 samples for faster computation
        select_key_vars=False  # Use all columns as requested
    )
