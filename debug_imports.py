import sys
print("Starting imports...", flush=True)
try:
    import pandas as pd
    print("Imported pandas", flush=True)
    import numpy as np
    print("Imported numpy", flush=True)
    import matplotlib.pyplot as plt
    print("Imported matplotlib", flush=True)
    from causallearn.search.ConstraintBased.FCI import fci
    print("Imported causallearn.search.ConstraintBased.FCI", flush=True)
    from causallearn.utils.GraphUtils import GraphUtils
    print("Imported causallearn.utils.GraphUtils", flush=True)
    from causallearn.utils.cit import fisherz
    print("Imported causallearn.utils.cit", flush=True)
except Exception as e:
    print(f"Error: {e}", flush=True)
print("Done.", flush=True)
