import numpy as np

import matplotlib.pyplot as plt

a = np.ones((32, 32, 3)) * 128 / 255.


plt.subplot(1, 2, 1)
plt.imshow(a)
plt.subplot(1, 2, 2)
plt.imshow(a)
plt.tight_layout(pad=0.1)
plt.show()
