import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import pylab
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

s1=np.random.normal(0,1,200)
s2=np.random.normal(0,1,200)
s3=np.random.normal(0,1,200)
x1=s1
x2=s1+0.001*s2
x3=10*s3

data = np.transpose(np.array([x1,x2,x3]))
#plt.scatter(data[:,0],data[:,1],data[:,2])
#ax.scatter(data[:,0],data[:,1],data[:,2])
#data = np.array([[1.,3.],[4.,1.],[3.,3.],[7.,9.],[4.,9.],[7.,8.],[11.,2.],[1.,0.],[5.,8.]])

def doPCA():
    pca = PCA(n_components=1)
    pca.fit(data)
    return pca
    
pca = doPCA()
print(pca.explained_variance_ratio_)
first_pc = pca.components_[0]
#second_pc = pca.components_[1]

transformed_data = pca.transform(data)
kk = transformed_data * first_pc
ax.scatter(kk[:, 0], kk[:, 1], kk[:, 2])
#for i, j in zip(transformed_data, data):
    #plt.scatter(first_pc[0]*i[0], first_pc[1]*i[0], color = 'r')
    #plt.scatter(second_pc[0]*i[1], second_pc[1]*i[1], color = 'g')
    #plt.scatter(j[0], j[1], color = 'b')

print("[x1 x2 x3] = ", first_pc)
print("[x1 x2 x3] = ", kk)
#print(np.matmul(data,np.expand_dims(first_pc,1)))
ax.set_xlabel("x1",fontsize = 20)
ax.set_ylabel("x2",fontsize = 20)
ax.set_zlabel("x3",fontsize = 20)
pylab.xlim([-30,30])
pylab.ylim([-30,30])

plt.show()
