import numpy as np
# E12 seed3 fold 5 without its feature arrays (the download goes over a phone hotspot): /content/vft/out_seed3/fold_5_slim.npz
z = np.load("/content/vft/out_seed3/fold_5.npz")
np.savez("/content/vft/out_seed3/fold_5_slim.npz", **{k: z[k] for k in z.files if not k.startswith("feat")})
print("slim", sorted(k for k in z.files if not k.startswith("feat")))
