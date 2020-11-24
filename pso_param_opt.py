import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np

alphas = np.ndarray((400,2))
bests = np.ndarray((400,3))
means = np.ndarray((400,3))

f = open("./output.dat", "r").readlines()
for i in range(0, len(f), 3):
    ii = int(i/3)
    x = f[i][:-1].split(' ')[1:]
    alphas[ii] = np.array(x)
    x = f[i+1][:-1].split(' ')
    bests[ii] = np.array(x)
    x = f[i+2][:-1].split(' ')
    means[ii] = np.array(x)
    
    
fig, ax = plt.subplots()

# Make data.
X = alphas[:,0].reshape((20,20))
Y = alphas[:,1].reshape((20,20))
Z1 = bests[:,2].reshape((20,20))
Z2 = means[:,2].reshape((20,20))

# paint plot
g = ax.pcolormesh(X, Y, Z1, cmap='coolwarm')
#g = ax.pcolormesh(X, Y, Z2, cmap='coolwarm')

print('[Bests] best test loss:',np.round(Z1.min(), 10),
      'achieved with alpha1 =',
      np.round(X[int(Z1.argmin()/20),Z1.argmin()%20],2),
      'and alpha2 =',
      np.round(Y[int(Z1.argmin()/20),Z1.argmin()%20],2),
       )

print('[Means] best test loss:',np.round(Z2.min(), 10),
      'achieved with alpha1 =',
      np.round(X[int(Z2.argmin()/20),Z2.argmin()%20],2),
      'and alpha2 =',
      np.round(Y[int(Z2.argmin()/20),Z2.argmin()%20],2),
       )
fig.colorbar(g,  aspect=10, label='Test loss')
plt.xticks(np.arange(0.5, 2.5, step=0.2))
plt.yticks(np.arange(0.5, 2.5, step=0.2))
ax.set_xlabel('a1')
ax.set_ylabel('a2')

# plot lines of a1 + a2
xs = np.arange(0.5,2.5,step=0.1)
ys2 = 2-xs
ys2_5 = 2.5-xs
ys3 = 3-xs
plt.xlim(0.5,2.4)
plt.ylim(0.5,2.4)
#plt.plot(xs,ys2, label='a1+a2=2', color='orange', ls='--')
plt.plot(xs,ys2_5, label='a1+a2=2.5', color='green', ls='--')
plt.plot(xs,ys3, label='a1+a2=3', color='yellow', ls='--')
plt.legend(framealpha=0.3)

plt.show()