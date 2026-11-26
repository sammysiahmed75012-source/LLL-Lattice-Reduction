import numpy as np

def GramSchmidt(b):
    n = len(b)
    g = np.zeros((n,n))
    g[0] = b[0]
    mu = np.identity(n)
    for i in range(1,n):
        S = 0
        for j in range(i):
            mu[i,j] = np.dot(b[i],g[j])/np.dot(g[j],g[j])
            S += mu[i,j]*g[j]
        g[i] = b[i] - S
    return g,mu     

def sizeReduction(b,k):
    g,mu = GramSchmidt(b)
    for j in range(k-1,-1,-1):
        if abs(mu[k,j]) > 1/2:
            r = np.round(mu[k,j])
            b[k] = b[k] - r*b[j]
            for i in range(j):
                mu[k,i] = mu[k,i] - r * mu[j,i]
    return b
    
def LLL(b):
    n = len(b)
    k = 1
    while k < n:
        b = sizeReduction(b,k)
        g, mu = GramSchmidt(b)
        if (np.linalg.norm(g[k]+mu[k,k-1]*g[k-1]))**2 >= (3/4)*(np.linalg.norm(g[k-1]))**2:
            k += 1
        else:
            b[[k,k-1]] = b[[k-1,k]]           
            k = max(k-1,1)
    return b