# Thursday, July 16, 2026

## produce simple histogram
simple histogram produced demonstrating that ROMULUS intrinsic values are bluer than SKIRT values

## organize
moved some CSVs from Deciphering Dwarfs to data directory, namely catalog_coords, catalog_dered, proposal_census

## census enterprise
before making particles, I would like to run a census to figure out what is in each snapshot.

Wrote 60 rows -> /mnt/data0/pkrsnak/romulus/enterprise/enterprise_census.csv

Most massive (by M*) halo per run:
     run  halo_id     mstar      mgas  n_bh   mbh_max  dm_min_ratio
    r488        1 2.225e+09 8.744e+09     0 0.000e+00     1.000e+00
 r568_BH        1 1.931e+09 5.326e+09     3 2.610e+05     1.000e+00
 r488_BH        1 1.747e+09 8.179e+09     1 1.255e+05     1.000e+00
    r568        1 1.106e+09 2.427e+09     0 0.000e+00     1.000e+00
r613_BH6        1 6.855e+08 5.182e+09     3 3.666e+05     1.000e+00
r568_BH8        1 6.139e+08 2.753e+09     3 1.048e+06     1.000e+00
r568_BH6        1 6.118e+08 2.998e+09     1 1.445e+05     1.000e+00
r488_BH6        1 6.017e+08 6.520e+09     1 1.298e+05     1.000e+00
 r613_BH        1 5.475e+08 4.589e+09     1 1.088e+06     1.000e+00
 r618_BH        1 4.389e+08 3.148e+09     1 7.394e+05     1.000e+00
    r618        1 4.373e+08 3.242e+09     0 0.000e+00     1.000e+00
    r613        1 3.935e+08 3.662e+09     0 0.000e+00     1.000e+00
r488_BH8        1 3.588e+08 5.156e+09     1 1.339e+06     1.000e+00
r618_BH6        1 2.906e+08 2.768e+09     3 3.520e+05     1.000e+00
r741_BH8        1 2.194e+08 2.388e+09     1 3.312e+05     1.000e+00
r613_BH8        1 2.139e+08 2.514e+09     4 8.133e+05     1.000e+00
r618_BH8        1 2.057e+08 2.519e+09     2 1.382e+06     1.000e+00
r741_BH6        1 1.852e+08 2.210e+09     3 2.418e+05     1.000e+00
 r741_BH        1 1.786e+08 1.601e+09     1 2.395e+05     1.000e+00
    r741        1 1.531e+08 2.406e+09     0 0.000e+00     1.000e+00

## a B histogram and a K histogram
somewhat more complicated than it appears; correcting extract_sim_colors.py

## results??

(base) pkrsnak@hamilton:~/tal-shiar-SKIRT/python$ python extract_sim_colors.py --all
halo     mag_B    mag_Ks   absmag_B  absmag_Ks  mag_B_nodust  mag_Ks_nodust  absmag_B_nodust  absmag_Ks_nodust       BK  BK_nodust  internal_reddening flux_style
r107 16.741103 12.953840 -18.258897 -22.046160     16.745153      12.955277       -18.254847        -22.044723 3.787263   3.789876           -0.002613        fnu
r142 15.894752 12.817702 -19.105248 -22.182298     15.456979      12.798887       -19.543021        -22.201113 3.077050   2.658092            0.418958        fnu
r154 16.958213 12.453344 -18.041787 -22.546656     15.272887      12.193532       -19.727113        -22.806468 4.504869   3.079354            1.425514        fnu
r168 16.341910 13.428339 -18.658090 -21.571661     16.201579      13.427679       -18.798421        -21.572321 2.913571   2.773900            0.139671        fnu
r204 16.699880 13.714787 -18.300120 -21.285213     16.409525      13.710031       -18.590475        -21.289969 2.985093   2.699494            0.285598        fnu
r219 16.223342 13.242160 -18.776658 -21.757840     15.947654      13.234941       -19.052346        -21.765059 2.981182   2.712712            0.268470        fnu
r223 16.577605 13.056047 -18.422395 -21.943953     15.603874      12.999013       -19.396126        -22.000987 3.521557   2.604861            0.916696        fnu
r239 16.334336 13.298026 -18.665664 -21.701974     15.667689      13.245418       -19.332311        -21.754582 3.036310   2.422271            0.614039        fnu
r284 17.713220 14.585820 -17.286780 -20.414180     17.666341      14.584259       -17.333659        -20.415741 3.127401   3.082082            0.045319        fnu
r306 16.781949 13.835903 -18.218051 -21.164097     16.468512      13.820081       -18.531488        -21.179919 2.946047   2.648430            0.297616        fnu
r316 16.792944 14.348239 -18.207056 -20.651761     16.731068      14.345955       -18.268932        -20.654045 2.444705   2.385113            0.059591        fnu
r320 16.916196 13.543460 -18.083804 -21.456540     15.901363      13.434149       -19.098637        -21.565851 3.372737   2.467214            0.905523        fnu
r330 17.122790 14.472406 -17.877210 -20.527594     17.075091      14.475367       -17.924909        -20.524633 2.650385   2.599724            0.050661        fnu
r372 17.153430 14.331159 -17.846570 -20.668841     17.002332      14.329026       -17.997668        -20.670974 2.822270   2.673306            0.148964        fnu
r429 17.743750 14.780580 -17.256250 -20.219420     17.656903      14.781082       -17.343097        -20.218918 2.963169   2.875820            0.087349        fnu

wrote sim_colors.csv  (15/15 halos)
dust   B-K: median 2.99  range 2.44-4.50
nodust B-K: median 2.67  range 2.39-3.79
internal reddening from dust: median +0.27 mag in B-K
  -> compare nodust median to the observed median (~2.40):
     if nodust already ~observed, the +offset is the dust model;
     if nodust is still red, it's the stellar populations.
(base) pkrsnak@hamilton:~/tal-shiar-SKIRT/python$ python reg_chk.py 
BK         max |change| = 7.11e-15   OK
BK_nodust  max |change| = 3.55e-15   OK
      absmag_B  absmag_Ks  absmag_B_nodust  absmag_Ks_nodust
halo                                                        
r107   -18.259    -22.046          -18.255           -22.045
r142   -19.105    -22.182          -19.543           -22.201
r154   -18.042    -22.547          -19.727           -22.806
r168   -18.658    -21.572          -18.798           -21.572
r204   -18.300    -21.285          -18.590           -21.290
r219   -18.777    -21.758          -19.052           -21.765
r223   -18.422    -21.944          -19.396           -22.001
r239   -18.666    -21.702          -19.332           -21.755
r284   -17.287    -20.414          -17.334           -20.416
r306   -18.218    -21.164          -18.531           -21.180
r316   -18.207    -20.652          -18.269           -20.654
r320   -18.084    -21.457          -19.099           -21.566
r330   -17.877    -20.528          -17.925           -20.525
r372   -17.847    -20.669          -17.998           -20.671
r429   -17.256    -20.219          -17.343           -20.219

## tweak color histogram to produce three plots
first improve compare_colors, then work on histogram building