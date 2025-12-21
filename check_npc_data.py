import pandas as pd
import numpy as np

df = pd.read_csv('simple_env_data.csv')

print("Checking NPC Data...")
if 'npc_1_stopping_dist' in df.columns:
    unique_vals = df['npc_1_stopping_dist'].unique()
    print(f"Unique Stopping Dists: {unique_vals}")
    if len(unique_vals) == 1:
        print("-> Stopping Distance is CONSTANT.")
else:
    print("-> npc_1_stopping_dist column NOT FOUND.")

if 'npc_1_speed_bin' in df.columns and 'temperature' in df.columns:
    corr = df['temperature'].corr(df['npc_1_speed_bin'])
    print(f"Correlation Temp vs NPC_Speed: {corr}")
    
    # Check if there is ANY speed difference by temperature
    # Group by temp and get mean speed
    print("\nMean Speed by Temperature (Sample):")
    print(df.groupby('temperature')['npc_1_speed_bin'].mean().head())
else:
    print("-> Columns for correlation check missing.")
