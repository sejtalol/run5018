import numpy as np
import matplotlib
import matplotlib.pyplot as plt

def readfile(dtime):
    filename = 'bar_ptcls_%d.dat'%(dtime)
    orbit_number, ini_flag, final_flag, cflag= np.loadtxt(filename, usecols = [0, 1, 2, 3], unpack = True)
    return orbit_number, ini_flag, final_flag, cflag

print('Start from T = 7000...')
dtime = 8000
print('When Time Interval is %d:'%(dtime))
orbit_number, ini_flag, final_flag, cflag = readfile(dtime)

n = 0   # initial in-bar particles
m = 0   # always in bar particle

for i in range(len(orbit_number)):
    if(ini_flag[i] == 1):
        n = n +1
    if(final_flag[i] == 1):
#    if(cflag >= 15000):
        m = m + 1

xmin = 7001
xmax = 14999

fig = plt.figure(figsize = (16, 10))
ax = fig.add_subplot(111)
ax.hist(cflag, bins = 200, range=(xmin, xmax), normed = True)
ax.tick_params(direction='out',labelsize=18)
ax.set_xlabel('Time')
ax.set_ylabel('Number of particles')
ax.set_title('The First Time that a bar particle moves out')
figname = 'test'
fig.savefig(figname)

print('%d particles in total'%(len(orbit_number)))
print('%d particles initially inside bar;'%(n))
print('%d particles that is always inside bar;'%(m))
