import pickle

import numpy as np

# open file '/itet-stor/azhuavlev/net_scratch/Projects/Data/HO3D_v3/train/GPMF12/meta/0000.pkl'
with open('/itet-stor/azhuavlev/net_scratch/Projects/Data/HO3D_v3/train/GPMF12/meta/0000.pkl', 'rb') as f:
    meta = pickle.load(f)
print(meta.keys())

# open file /itet-stor/azhuavlev/net_scratch/Projects/Data/HO3D_v3/manual_annotations/ABF1_0093.npy
annotations = np.load('/itet-stor/azhuavlev/net_scratch/Projects/Data/HO3D_v3/manual_annotations/ABF1_0093.npy', allow_pickle=True)
print(annotations)

print(
0.726497948169708 * -0.5865591168403625 + 0.5328468084335327 * 0.8098118305206299+ -0.433906614780426 * 0.01237974688410759

)