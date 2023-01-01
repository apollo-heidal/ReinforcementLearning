import matplotlib as mpl
from matplotlib import pyplot as plt
import numpy as np

# make values from -5 to 5, for this example
zvals = np.random.rand(100,100)*10-5

# tell imshow about color map so that only set colors are used
img = plt.imshow(zvals,cmap = plt.plasma())

# make a color bar
plt.colorbar(img)

plt.show()
