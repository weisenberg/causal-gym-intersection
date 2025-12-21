import pandas as pd
import numpy as np
from causallearn.search.ConstraintBased.FCI import fci
from causallearn.utils.GraphUtils import GraphUtils
from causallearn.utils.cit import fisherz
import matplotlib.pyplot as plt
import io
import sys

def run_fci_discovery(input_file="causal_discovery_dataset.csv", output_image="causal_graph_fci.png"):
    print(f"Loading data from {input_file}...")
    try:
        df = pd.read_csv(input_file)
    except FileNotFoundError:
        print(f"Error: File {input_file} not found.")
        return

    # Preprocessing
    # Ensure all columns are numeric
    df = df.select_dtypes(include=[np.number])
    
    # Drop any rows with NaN (though our generation shouldn't have any)
    df = df.dropna()
    
    # Convert to numpy array
    data = df.to_numpy()
    labels = df.columns.tolist()
    
    print(f"Data shape: {data.shape}")
    print(f"Variables: {labels}")
    
    print("\nRunning FCI algorithm (this may take a moment)...")
    # Run FCI
    # independence_test_method=fisherz is standard for continuous data
    # alpha=0.05 is standard significance level
    G, edges = fci(data, independence_test_method=fisherz, alpha=0.05)
    
    print("\nCausal Graph Discovered!")
    
    # Print Edges
    print("\nEdges:")
    for edge in edges:
        print(edge)
        
    # Visualization
    print(f"\nGenerating visualization to {output_image}...")
    try:
        pydot_graph = GraphUtils.to_pydot(G, labels=labels)
        pydot_graph.write_png(output_image)
        print("Visualization saved successfully.")
    except Exception as e:
        print(f"Error generating visualization: {e}")
        print("Ensure graphviz is installed on your system (brew install graphviz on Mac).")

if __name__ == "__main__":
    run_fci_discovery()
