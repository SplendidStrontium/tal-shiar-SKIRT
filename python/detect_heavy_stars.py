import pynbody, numpy as np
base = "/mnt/data0/pkrsnak/romulus"
halos = ["r154","r168","r204","r219","r223","r239","r284","r306","r316","r330","r372","r429"]
for g in halos:
    d = pynbody.load(f"{base}/{g}.007779.tipsy"); d.physical_units()
    m  = np.asarray(d.star['mass'].in_units('Msol'))
    tf = np.asarray(d.star['tform'])
    heavy = m > 1e5
    bh = len(d.bh) if hasattr(d, 'bh') else 'none'
    print(f"{g:5} heavy>1e5:{heavy.sum():>3}  max_M={m.max():.2e}  "
          f"tform<0:{(tf<0).sum():>3}  heavy&tform<0:{int((heavy&(tf<0)).sum()):>3}  bh={bh}")