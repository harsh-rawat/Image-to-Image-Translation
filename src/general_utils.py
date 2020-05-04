import numpy as np
import matplotlib.pyplot as plt


def generate_sample(loaders):
    loader_X = loaders[0]
    loader_Y = loaders[1]
    img_X = next(iter(loader_X))[0]
    img_Y = next(iter(loader_Y))[0]
    print('Shape of Sample Image Tensor is : {}'.format(img_X.shape))
    img_X = np.transpose((img_X + 1) / 2, (1, 2, 0))
    img_Y = np.transpose((img_Y + 1) / 2, (1, 2, 0))
    fig, axes = plt.subplots(1, 2)
    axes[0].imshow(img_X)
    axes[1].imshow(img_Y)
    plt.show()
