import pandas as pd
old = pd.read_csv("sim_colors_before.csv").set_index("halo")
new = pd.read_csv("sim_colors.csv").set_index("halo")
for col in ["BK", "BK_nodust"]:
    d = (new[col] - old[col]).abs().max()
    print(f"{col:10s} max |change| = {d:.2e}   {'OK' if d < 1e-9 else 'PROBLEM'}")
print(new[["absmag_B", "absmag_Ks", "absmag_B_nodust", "absmag_Ks_nodust"]].round(3).to_string())