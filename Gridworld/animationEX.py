import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

fig, ax = plt.subplots()

data = np.random.rand(100,100)*10

im = plt.imshow(data, cmap='plasma')

def init():
    im.set_data(data)
    return im,

def update(frame):
    a=im.get_array()
    a=a*np.exp(-0.001*frame)    # exponential decay of the values
    im.set_array(np.random.rand(100,100)*10)
    return im,

ani = FuncAnimation(fig, update, frames=np.linspace(0, 2*np.pi, 128), interval=20, blit=True)
plt.show()
