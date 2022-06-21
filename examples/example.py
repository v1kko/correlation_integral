import correlation_integral as ci
import numpy as np
import matplotlib.pyplot as plt

# example Henon map - create & plot data
a=1.4;
b=0.3;

x=np.zeros(shape=(10000+5000),dtype=np.float32)
y=np.zeros(shape=(10000+5000),dtype=np.float32)
x[0]=1
y[0]=0
for i in range(0,len(x)-1):
    x[i+1]=1-a*x[i]**2+y[i]
    y[i+1]=b*x[i]

x=x[5000:]
y=y[5000:]


# correlation integral analysis done only with x-variable
nrsteps=10;
dims=np.arange(1,16) # calculate for d=1 uptil d=16
nrdims=dims.size;
rs=np.logspace(-2.5,0.5,nrsteps)
print(rs)
Cds=np.zeros((nrsteps,nrdims));

print("Start")
for j in range(nrdims):
  Cds[:,j] = ci.manhattan(x,dims[j],rs)
  print("x",end="")
print("\nDONE!")

print(dims[4],Cds[:,4])

# Plot the results
for i in range(len(dims)):
  plt.plot(rs,Cds[:,i])
plt.xscale("log")
plt.yscale("log")
plt.show()
