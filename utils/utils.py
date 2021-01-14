import torch
import matplotlib.pyplot as plt

def show_interpretation(sample, data, target, shape = (1,28,28)):
  channels = shape[0]
  for i in range(len(sample)):
    print(f"Wanted target category : {target[i]}")
    sample_reshaped = sample[i].reshape(shape)
    for k in range(channels):
        fig = plt.figure()
        plt.subplot(1,2,1)
        plt.imshow(data[i][k], cmap='gray', interpolation='none')
        plt.subplot(1,2,2)
        plt.imshow(sample_reshaped[k], cmap='gray', interpolation='none', vmin=0, vmax=1)
        plt.show()
