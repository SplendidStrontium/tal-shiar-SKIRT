# Thursday, July 9, 2026

## need to compare colors of halos
I have my SKIRT colors and I have colors that ROMULUS expects them to be, now I need to compare the two values.

## results
HaloID  rom_color  skirt_color     source  delta
    107        NaN        3.790 right_only    NaN
    109      1.327          NaN  left_only    NaN
    142      0.799        2.658       both  1.859
    154      1.633        3.079       both  1.446
    168        NaN        2.774 right_only    NaN
    204      1.475        2.699       both  1.224
    219      1.716        2.713       both  0.996
    223      1.634        2.605       both  0.971
    239      1.241        2.422       both  1.181
    284        NaN        3.082 right_only    NaN
    306        NaN        2.648 right_only    NaN
    316        NaN        2.385 right_only    NaN
    320      1.221        2.467       both  1.246
    330      1.696        2.600       both  0.904
    372      1.553        2.673       both  1.120
    429      2.042        2.876       both  0.834

UNMATCHED (excluded from statistics):
  halo 107  SKIRT only
  halo 109  ROMULUS only
  halo 168  SKIRT only
  halo 284  SKIRT only
  halo 306  SKIRT only
  halo 316  SKIRT only

matched halos : 10
median delta  : +1.151 mag
mean delta    : +1.178 mag
std delta     : 0.301 mag
range         : +0.834 to +1.859