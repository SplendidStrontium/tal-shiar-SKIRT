"""
Read in CSV of nearby dwarf catalog and produce a histogram.
"""
import pandas as pd
import matplotlib.pyplot as plt

nearby_catalog = pd.read_csv("/home/pkrsnak/tal-shiar-SKIRT/src/nearby_dwarf_catalog.csv")
gal_color = nearby_catalog['bmag'] - nearby_catalog['ks_mag']

gal_color.hist(bins=10)
plt.savefig("gal_color_histogram.png")
