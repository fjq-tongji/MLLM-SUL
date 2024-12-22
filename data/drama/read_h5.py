import h5py
import numpy as np
file = h5py.File('./drama_llama_adapter_3_r4_0211.h5', 'r')
print(file.keys())
data = file['labels']
print(data)

file.close()
