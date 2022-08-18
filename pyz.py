from PIL import Image
import numpy as np

v1 = np.array(Image.open("./submission/pgd_ensemble_s/pgd_ensemble2_epoch0_time0.png")).astype(float)
v2 = np.array(Image.open("./submission/pgd_ensemble_s/pgd_ensemble2_epoch0_time66.png")).astype(float)
print(abs(v1-v2).mean())