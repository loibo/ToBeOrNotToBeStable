import numpy as np

noise_level = 0.025
n = 256

e = np.random.normal(0, 1, n**2) * noise_level
print(np.linalg.norm(e))


# Sample Normalized Gaussian noise
e = np.random.normal(0, 1, n**2)
e = e / np.linalg.norm(e.flatten())

# Sample a random radius approximately lower than the radius of the Annulus
z = np.random.uniform(0, 1.1*np.sqrt(n**2)*noise_level)

print(np.linalg.norm(e * z))

import matplotlib.pyplot as plt
plt.subplot(1, 2, 1)
plt.imshow((e).reshape((n, n)), cmap='gray')
plt.colorbar()
plt.subplot(1, 2, 2)
plt.imshow((e * z).reshape((n, n)), cmap='gray')
plt.colorbar()
plt.show()