"""Generate a coordinate mapping."""

from pathlib import Path
from ssm_model_reduction.utils.map import stretching
import numpy as np

PATH = Path("/Users/njo20/Documents/ERCOFTAC/ssm-model-reduction/data/coordinates.npy")

A, B, C, D = -1.5, -2, 2, 8

s = stretching(192, 0.033, 0.20, int(0.5 / 0.033 + 16), 16, 16, 0.04)
s1 = stretching(256, 0.033, 0.20, int(0.5 / 0.033 + 16), 16, 16, 0.04)
s2 = stretching(128, 0.033, 0.20, int(0.5 / 0.033 + 16), 16, 16, 0.04)

x = np.r_[-s2[::-1], s1[1:]]
y = np.r_[-s[::-1], s[1:]]

indices_x = np.where((x > A) & (x < D))[0]
indices_y = np.where((y > B) & (y < C))[0]

x_low, x_high = indices_x[0], indices_x[-1]
y_low, y_high = indices_y[0], indices_y[-1]

x = x[x_low:x_high]
y = y[y_low:y_high]

X, Y = np.meshgrid(x, y, indexing="ij")
coords = np.stack([X, Y], axis=-1)

print(f"Saving coordinates. Shape = {coords.shape}")
np.save(PATH, coords)
