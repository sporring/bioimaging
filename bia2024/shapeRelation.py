# The following contains demonstrations how some of the shape relation measures can be implemented
# using mathematical morphology.
# 
# Jon Sporring
# Department of Computer Science
# University of Copenhagen
# August 11, 2024

# Given data: An image of an object and a coordinate list of center placements
import matplotlib.pyplot as plt
import numpy as np
import imageio
I = imageio.imread('moon.png')
centers = np.loadtxt("moons.csv",delimiter=",", dtype=float)

# First we analyze the centers
import rpy2
import rpy2.robjects as robjects
from rpy2.robjects.packages import importr
from rpy2.robjects import FloatVector
import math
base = importr('base')
spatstat = importr('spatstat')
ro = robjects.r

def sampleK(x, t, n = 100, bounds = FloatVector([0,1])):
    ppx = ro.ppp(FloatVector(x[:,0]), FloatVector(x[:,1]), window = ro.owin(bounds,bounds))
    f = base.as_function(ro.Kest(ppx))
    ft = [f(i)[0] for i in t] 
    return ft

t = list(range(1,255));
f = sampleK(centers,t,bounds = FloatVector([0,1023]))
plt.plot(t,math.pi*np.array(t)**2.0,'k-')
plt.plot(t,f,'r-')
plt.title('Centers as a point process')
plt.xlabel('r')
plt.ylabel('Kest(r)')
plt.tight_layout()
plt.savefig("moonKest.pdf",bbox_inches='tight')
plt.show()

# Then we perform the same analysis with the shape relation measure
(m,n) = I.shape
topLeft = np.round(centers)
max = np.array(topLeft).max(0)

# produce the image of the shapes
sz = np.ceil(max+[m,n])
X = np.zeros(tuple(sz.astype('uint')))
for i in range(topLeft.shape[0]):
    x0 = int(topLeft[i,0])
    y0 = int(topLeft[i,1])
    X[x0:(x0+m),y0:(y0+n)] = I

# estimate the shape relation measures
import scipy
N = 255
rVals = list(range(1,N+1))
KVals = np.zeros((N,1))
NVals = np.zeros((N,1))
Icount = np.sum(I)
for i in range(topLeft.shape[0]):
    R = np.zeros(X.shape)
    x0 = int(topLeft[i,0])
    y0 = int(topLeft[i,1])
    R[x0:(x0+m),y0:(y0+n)] = I
    S = scipy.ndimage.distance_transform_edt(1-R)
    for j in range(len(rVals)):
        r = rVals[j]
        Yr = (S < r) # the moon's expanded image
        KVals[j] = KVals[j]+np.sum(Yr*X)-Icount # correct for self
        NVals[j] = KVals[j]/np.sum(Yr)

KVals = KVals/(topLeft.shape[0]-1)
NVals = NVals/(topLeft.shape[0]-1)
DVals = (KVals[2:N]-KVals[0:N-2])/2
plt.rcParams["text.usetex"] = True
plt.plot(rVals,KVals,'k-')
plt.title('Shape Relations')
plt.xlabel('r')
plt.ylabel(r"$\mu$(r)")
plt.tight_layout()
plt.savefig("moonShapeRelationK.pdf",bbox_inches='tight')
plt.show()
plt.plot(rVals[1:-1],DVals,'k-')
plt.title('Shape Relations')
plt.xlabel('r')
plt.ylabel('g(r)')
plt.tight_layout()
plt.savefig("moonShapeRelationDK.pdf",bbox_inches='tight')
plt.show()
plt.plot(rVals,NVals,'k-')
plt.title('Shape Relations')
plt.xlabel('r')
plt.ylabel('f(r)')
plt.tight_layout()
plt.savefig("moonShapeRelationN.pdf",bbox_inches='tight')
plt.show()
