# Tuesday, July 14, 2026

## update CSV, rerun
Got an updated CSV from tangos, let's rerun

## updated results
HaloID  rom_color  skirt_color    source  delta
    107      1.422        3.790      both  2.368
    109      1.327          NaN left_only    NaN
    142      0.799        2.658      both  1.859
    154      1.633        3.079      both  1.446
    168      2.066        2.774      both  0.708
    204      1.475        2.699      both  1.224
    219      1.716        2.713      both  0.996
    223      1.634        2.605      both  0.971
    239      1.241        2.422      both  1.181
    284      0.993        3.082      both  2.089
    306      1.362        2.648      both  1.287
    316      1.277        2.385      both  1.108
    320      1.221        2.467      both  1.246
    330      1.696        2.600      both  0.904
    372      1.553        2.673      both  1.120
    429      2.042        2.876      both  0.834

UNMATCHED (excluded from statistics):
  halo 109  ROMULUS only

matched halos : 15
median delta  : +1.181 mag
mean delta    : +1.289 mag
std delta     : 0.472 mag
range         : +0.708 to +2.368

## BH comparison
I have five snapshots, these notably each have more than one halo in each, so I know I need to select the halos within, with the amiga file helps us do.

Each snapshot has four variations, which have either no black hole or three different versions of black hole formation/feedback parameters.

## differences between snapshots
- dEtaDiffusion: timestep accuracy for therm/metal diffusion
  - noBH: 0.3
  - BH:   0.15
  - BH6:  0.3
  - BH8:  0.3

- dBHSinkFeedbackEff: how much energy goes into env
  - BH:  0.005
  - BH6: 0.05
  - BH8: 0.005

- dBHSinkColdDen: density above which BH accretes cold gas
  - BH:  100
  - BH6: 0.2
  - BH8: 0.2

## to summarize

              noBH    BH      BH6   BH8
dEtaDiff      0.3     0.15    0.3   0.3
dBHSinkFeed   --      0.005   0.05  0.005
dBHSinkCld    --      100     0.2   0.2

## control?
it probably makes most sense to compare noBH to BH, and BH6 to BH8

## pipeline gotchas
- naming RESOLVED: on-disk snapshot names are regular (r488_BH6 → ...HsbBH6.004096, etc). The param files' achOutName ("BHT", "BH2") are stale pre-rename working names — never parse achOutName, just glob for *.004096
- iBinaryOutput: noBH/BH = 0 (ASCII aux arrays), BH6/BH8 = 1 (binary)
  — pynbody handles both; hand-rolled readers won't
- dMaxGasMass typo in noBH and BH params: `1.66431-11` (missing e)
