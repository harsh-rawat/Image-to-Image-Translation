import numpy as np
import matplotlib.pyplot as plt


def generate_sample(loaders):
    img = next(iter(loaders))[0]
    print('Shape of Sample Image Tensor is : {}'.format(img.shape))
    img = np.transpose((img + 1) / 2, (1, 2, 0))
    plt.imshow(img)
    plt.show()
