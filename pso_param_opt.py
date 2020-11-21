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
#Z = bests[:,1].reshape((20,20))
Z = bests[:,1].reshape((20,20))

g = ax.pcolormesh(X, Y, Z, cmap='coolwarm')
#g = ax.contourf(X, Y, Z, cmap='coolwarm')


fig.colorbar(g,  aspect=10)
plt.xticks(np.arange(0.5, 2.5, step=0.2))
plt.yticks(np.arange(0.5, 2.5, step=0.2))
ax.set_xlabel('alpha1')
ax.set_ylabel('alpha2')

plt.show()