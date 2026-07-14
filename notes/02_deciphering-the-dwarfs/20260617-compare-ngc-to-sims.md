# Wednesday, June 17, 2026

## working with nearby dwarf catalog
Before a comparison can be made of simulated galaxies to nearby dwarf catalog, entire list must be adjusted for Milky Way foreground extinction. Most galaxies have a B-K of 1.6-4, but 13 objects are much higher than that. These galaxies are in the Zone-of-Avoidance, and therefore are seen through a much higher density of the Milky Way dust. Therefore it is imperative to correct for extinction and thereby deredden the catalog.

## grab some packages
> pip install dustmaps astroquery

## imported dustmaps and moved to data directory

## let's deredden our catalog
> python deredden_catalog.py --in ~/tal-shiar-SKIRT/src/nearby_dwarf_catalog.csv --out catalog_dered.csv

=== before/after on the reddest observed objects (the foreground tail) ===
  Dw2              b=  -0.2  E(B-V)= 1.18  B-K:  7.75 ->  3.83 [SUSPECT]
  HIZSS014         b=   0.6  E(B-V)= 1.31  B-K:  7.58 ->  3.22 [SUSPECT]
  HIPASS J1441-62  b=  -2.5  E(B-V)= 1.34  B-K:  7.57 ->  3.12 [SUSPECT]
  MB1              b=  -0.9  E(B-V)= 0.97  B-K:  7.34 ->  4.11 [SUSPECT]
  HIZSS021         b=  -1.8  E(B-V)= 1.01  B-K:  6.37 ->  3.03 [SUSPECT]
  Cepheus1         b=   8.0  E(B-V)= 0.92  B-K:  6.35 ->  3.31 [SUSPECT]
  KKR59            b=   7.0  E(B-V)= 0.88  B-K:  6.13 ->  3.22 [SUSPECT]
  KKH12            b=  -3.0  E(B-V)= 0.81  B-K:  6.01 ->  3.33 [SUSPECT]
  Cas1             b=   7.1  E(B-V)= 1.02  B-K:  5.95 ->  2.57 [SUSPECT]
  KK49             b=   nan  E(B-V)=  nan  B-K:  5.71 ->   nan
  IC0010           b=  -3.3  E(B-V)= 1.57  B-K:  5.27 ->  0.07 [SUSPECT]
  UGC02773         b=  -6.8  E(B-V)= 0.57  B-K:  5.23 ->  3.35 [SUSPECT]

galaxies with B-K > 4.5: 13 (observed) -> 0 (dereddened)
median B-K: 2.64 (obs) -> 2.41 (dered)

EXCLUDE (SFD unreliable): 22  [low |b|<10: 20, Magellanic: 2]
  SMC, LMC, HIPASS J1441-62, ESO174-001, ESO223-009, RKK1610, HIPASS J0905-36, HIZSS021, CGMW2-3473, ESO495-008, ESO558-011, HIZSS014, IC2171, UGC02773, KKH11, Cepheus1, KKR59, KKH12, Dw2, IC0010, MB1, Cas1

INFO only (review, not excluded):
  large foreground E(B-V)>0.3: 22
  unusually blue B-K<1.5: 4  (SMC, LMC, NGC5408, IC0010)

still unresolved (E(B-V) NaN): 12
  KK55, KK246, KKSG9, KKSG17, KKSG15, KK49, KK127, d0226+3325, KK149, KK251, KK252, A0952+69

clean comparison sample (not suspect, resolved): 221 / 255