"""
compare colors between ROMULUS intrinsic and SKIRT nodust
"""
import pandas as pd
import numpy as np

# read data
# index_col=0: the ROMULUS file's first column is an unnamed row index that
# pandas wrote out. Without this you inherit a column called "Unnamed: 0".
rom = pd.read_csv("ROMULUS_halo_data.csv", index_col=0)
skirt = pd.read_csv("sim_colors.csv")

# SKIRT names halos "r142" (str); ROMULUS names them 142 (int).
# Build one column, same name and same dtype, on both sides.
skirt["HaloID"] = skirt["halo"].str.removeprefix("r").astype(int)
assert rom["HaloID"].dtype == skirt["HaloID"].dtype, "join key dtypes differ"

# ROMULUS: intrinsic (dust-free) color, from the two magnitude columns.
rom["rom_color"] = rom["Bmag"] - rom["Kmag"]
 
# SKIRT: the dust-free color is ALREADY computed. Do not rebuild it from
# mag_B - mag_Ks -- that subtraction gives the DUST color (the "BK" column).
skirt["skirt_color"] = skirt["BK_nodust"]

# make new frame
# how="outer"  -> keep every halo from BOTH files, matched or not.
# indicator=   -> adds a "_merge" column saying where each row came from.
# validate=    -> raises if either side has a duplicate HaloID, which would
#                 silently multiply rows and corrupt every statistic after.
comp_df = pd.merge(
    rom[["HaloID", "rom_color"]],
    skirt[["HaloID", "skirt_color"]],
    on="HaloID",
    how="outer",
    indicator="source",
    validate="one_to_one",
).sort_values("HaloID")

# --- compare ------------------------------------------------------
 
# Positive delta = SKIRT is redder than ROMULUS.
comp_df["delta"] = comp_df["skirt_color"] - comp_df["rom_color"]
 
matched = comp_df[comp_df["source"] == "both"]
 
print(comp_df.round(3).to_string(index=False))
print()
 
unmatched = comp_df[comp_df["source"] != "both"]
if not unmatched.empty:
    print("UNMATCHED (excluded from statistics):")
    for _, r in unmatched.iterrows():
        side = "ROMULUS only" if r["source"] == "left_only" else "SKIRT only"
        print(f"  halo {int(r['HaloID']):3d}  {side}")
    print()
 
print(f"matched halos : {len(matched)}")
print(f"median delta  : {matched['delta'].median():+.3f} mag")
print(f"mean delta    : {matched['delta'].mean():+.3f} mag")
print(f"std delta     : {matched['delta'].std():.3f} mag")
print(f"range         : {matched['delta'].min():+.3f} to {matched['delta'].max():+.3f}")
 
comp_df.to_csv("rom_vs_skirt_colors.csv", index=False)