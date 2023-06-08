import numpy as np
import qutip as Q
from qutip import *
def Unitary_trans(rho,n):
    rho.reshape(2**n,2**n)
    I2 = Qobj(np.eye(2))
    para = np.random.randint(-10**8, 10**8, size=(3))
    gama,beta,delta=para[0],para[1],para[2]
    u_mat=np.exp(1j*np.random.randint(-10**8, 10**8))\
        *np.dot(np.diag([np.exp(-1j*beta/2),np.exp(1j*beta/2)]),\
                np.array([[np.cos(gama/2),-np.sin(gama/2)],[np.sin(gama/2),np.cos(gama/2)]]),\
                    np.diag([np.exp(-1j*delta/2),np.exp(1j*delta/2)]))
    for i in range(n):
        if i==0:
            UT=Qobj(u_mat)
        else:
            UT=Q.tensor(UT,I2)
    return UT