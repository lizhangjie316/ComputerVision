import numpy as np
import matplotlib.pyplot as plt

#image_data
a = np.array([0.313660827978, 0.365348418405, 0.423733120134,
              0.365348418405, 0.439599930621, 0.525083754405,
              0.423733120134, 0.525083754405, 0.651536351379]).reshape(3,3)

plt.imshow(a,interpolation='nearest',cmap='bone',origin='upper') #upper lower
plt.colorbar(shrink=0.9)

plt.xticks(())
plt.yticks(())

plt.show()
