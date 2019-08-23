import mymodules as mod
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

nn = 370
x,y= np.loadtxt('./data/ptcls_info_t%3.3d.dat'%(nn),usecols=[0,1],unpack=True)

fig, ax = plt.subplots(figsize=(10,10))
xmin = -8
xmax = 8
ymin = -8
ymax = 8
mod.hist2d_plot(x, y, xmin, xmax, ymin, ymax, ax=ax)
ax.set_xlabel('x',fontsize=18)
ax.set_ylabel('y',fontsize=18)
ax.set_title('$Density$',fontsize=20)
fig.savefig('test.png')
