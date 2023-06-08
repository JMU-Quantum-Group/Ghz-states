import numpy as np
import random
import qutip as Q
from qutip import *
class generate():#2-可分
    def __init__(self, k,n, number1=None):
        self.k, self.number1 = k, number1
        self.n = n
        self.Beta = [np.pi / 2, 1.231, 1.0115, 0.866, 0.7559, 0.6713, 0.9, 0.6981, 0.6283, 0.5712]  # beta values-->n(2~8)
        self.B_3 = [0.2, 0.2381, 0.2195, 0.2147, 17/81, 61 / 327, 115 / 627, 869/4965]  #4-11
        self.B_4 = [1/9, 5/53, 6/70, 7/71, 11/119, 9/137, 65/1089, 77/1375]     #8-11                                                      #4-10
        if k == 2:
            self.B = (2**(self.n-1)-1)/(2**self.n-1)
        elif k == 3:
            self.B = self.B_3[self.n -4]
        elif k == 4:
            self.B = self.B_4[self.n - 4]
        else:
            self.B = 1 / (1+(2 ** (self.n - 1)))
        self.data = {}
        self.test_data = {}
    def __iter__(self):
        return self
    def __next__(self):
        if self.n > 11:
            raise StopIteration
        else:
            self.n += 1
            if self.k == 2:
                self.B = (2 ** (self.n - 1) - 1) / (2 ** self.n - 1)
            elif self.k == 3:
                self.B = self.B_3[self.n - 4]
            else:
                self.B = 1 / (1 + (2 ** (self.n - 1)))
    def calculate_boundary(self,bpk):  # input prediction of GHZ states
        i, j, bound_value = 0, 1, []
        A = np.argwhere(bpk == 1)
        b_sum10 = 20
        while b_sum10 > 10 and (i < len(A) - 6):
            b_sum10 = A[i + 6] - A[i]
            if b_sum10 > 10:
                i += 1
        if len(A) == 0:
            return None
        else:
            return ((self.B/2000) * A[i])
    def get_data(self):
        GHZ_Features = self.Ghz()
        UNGHZ_Features = self.UNGhz()
        np.savetxt('./data/' + repr(self.n) + '-qubit-' + repr(self.k) + '-0-' + repr(self.number1) +'-features.txt', GHZ_Features[repr(self.n) + '-qubit-0-features'], fmt='%.10f', delimiter='\t')
        np.savetxt('./data/' + repr(self.n) + '-qubit-' + repr(self.k) + '-1-' + repr(self.number1) +'-features.txt', GHZ_Features[repr(self.n) + '-qubit-1-features'], fmt='%.10f', delimiter='\t')
        np.savetxt('./data/' + repr(self.n) + '-qubit-' + repr(self.k) + '-un-' + repr(self.number1) +'-features.txt',
                   UNGHZ_Features[repr(self.n) + '-qubit-un-features'], fmt='%.10f', delimiter='\t')

        bound_states = self.Bound_states()

        np.savetxt('./data/' + repr(self.n) + '-qubit-' + repr(self.k) + '-bound-features.txt',
                   bound_states[repr(self.n) + '-qubit-bound-features'], fmt='%.10f', delimiter='\t')

        '''
        test_data = self.Ghz(t=0)
        np.savetxt('./data/' + repr(self.n) + '-qubit-' + repr(self.k) + '-0-test-' + repr(3000) + '-features.txt',
                   test_data[repr(self.n) + '-qubit-0-features'], fmt='%.10f', delimiter='\t')
        np.savetxt('./data/' + repr(self.n) + '-qubit-' + repr(self.k) + '-1-test-' + repr(3000) + '-features.txt',
            test_data[repr(self.n) + '-qubit-1-features'], fmt='%.10f', delimiter='\t')
        '''
        #return self.data
    def Ghz(self, t=1):
        n = self.n
        if t == 1:
            number1 = self.number1
        else:
            number1 = 1000 # generate test data
        e_0 = basis(2, 0)
        e_1 = basis(2, 1)
        I = sigmax()*sigmax()
        self.GHZ, self.GHZ_Featurev = {}, {}
        B = self.B  # B is a list in which the elements are k-separable boundary values, p ≤ N[i]
        for i in range(1, n):
            e_0 = Q.tensor(e_0, basis(2, 0))
            e_1 = Q.tensor(e_1, basis(2, 1))
            I = Q.tensor(I, sigmax()*sigmax())
        ghz_n = (1 / np.sqrt(2)) * (e_0 + e_1)
        for j in range(2):
            for i in range(int((number1 / 2))):
                if j == 0:
                    g = random.uniform(0, B)
                else :
                    g = random.uniform(B, 2*B)
                ghz = ((1 - g) * I / (2 ** n) + g * ((ghz_n * (ghz_n.conj().trans()))))
                ghza = self.Unitary_trans(ghz)
                #ghz_v = self.Feature_vector(ghz)
                ghz_va = self.Feature_vector(ghza)
                if i == 0:
                    #self.GHZ["{}-qubit-{}-class".format(n, j)] = np.array(ghz).reshape(1, 2**n , 2**n)
                    self.GHZ_Featurev["{}-qubit-{}-features".format(n, j)] = ghz_va
                else:
                    #self.GHZ["{}-qubit-{}-class".format(n, j)] = np.concatenate((self.GHZ["{}-qubit-{}-class".format(n, j)] , np.array(ghz).reshape(1, 2**n , 2**n)), axis=0)
                    self.GHZ_Featurev["{}-qubit-{}-features".format(n, j)] = np.concatenate((self.GHZ_Featurev["{}-qubit-{}-features".format(n, j)], ghz_va), axis=0)
        return self.GHZ_Featurev
    def UNGhz(self):
        n = self.n
        B = self.B
        e_0 = basis(2, 0)
        e_1 = basis(2, 1)
        I = sigmax()*sigmax()
        for i in range(1, n):
            e_0 = Q.tensor(e_0, basis(2, 0))
            e_1 = Q.tensor(e_1, basis(2, 1))
            I = Q.tensor(I,sigmax()*sigmax())
        self.UNGHZ, self.UNGHZ_Featurev = {}, {}
        ghz_n = (1 / np.sqrt(2)) * (e_0 + e_1)
        for i in range(int(2*self.number1)):
            g = random.uniform(0, 3*B)
            ghz = ((1 - g) * I / (2 ** n) + g * ((ghz_n * (ghz_n.conj().trans()))))
            ghza = self.Unitary_trans(ghz)
            ghz_va = self.Feature_vector(ghza)
            if i == 0:
                self.UNGHZ_Featurev["{}-qubit-un-features".format(n)] = ghz_va
            else:
                #self.GHZ["{}-qubit-{}-class".format(n, j)] = np.concatenate((self.GHZ["{}-qubit-{}-class".format(n, j)] , np.array(ghz).reshape(1, 2**n , 2**n)), axis=0)
                self.UNGHZ_Featurev["{}-qubit-un-features".format(n)] = np.concatenate((self.UNGHZ_Featurev["{}-qubit-un-features".format(n)], ghz_va), axis=0)
        return self.UNGHZ_Featurev
    def Bound_states(self):
        B, n = self.B, self.n
        G = np.arange(0, 2*B, B/1000)
        n = self.n
        e_0 = basis(2, 0)
        e_1 = basis(2, 1)
        I = qeye(2)
        for i in range(1, n):
            e_0 = Q.tensor(e_0, basis(2, 0))
            e_1 = Q.tensor(e_1, basis(2, 1))
            I = Q.tensor(I, qeye(2))
        ghz_n = (1 / np.sqrt(2)) * (e_0 + e_1)
        self.bound_states={}
        for (i, g) in zip(range(len(G)), G):
            ghz = ((1 - g) * I / (2 ** n) + g * ((ghz_n * (ghz_n.conj().trans()))))
            ghz_v = self.Feature_vector(ghz, u=False)
            if i == 0:
                self.bound_states["{}-qubit-bound-features".format(n)] = ghz_v
            else:
                # self.GHZ["{}-qubit-{}-class".format(n, j)] = np.concatenate((self.GHZ["{}-qubit-{}-class".format(n, j)] , np.array(ghz).reshape(1, 2**n , 2**n)), axis=0)
                self.bound_states["{}-qubit-bound-features".format(n)] = np.concatenate(
                    (self.bound_states["{}-qubit-bound-features".format(n)], ghz_v), axis=0)
        return self.bound_states
    def Unitary_trans(self, rho):
        rho_a = {}
        UT = Q.tensor(Q.sigmax(), Q.sigmay(), Q.sigmaz())
        UT1 = Q.tensor(Q.sigmax(), Q.sigmax(), Q.sigmax())
        UT2 = Q.tensor(Q.sigmay(), Q.sigmay(), Q.sigmay())
        UT3 = Q.tensor(Q.sigmaz(), Q.sigmaz(), Q.sigmaz())
        for j in range(3, self.n):
            UT = Q.tensor(UT, qeye(2))
            UT1 = Q.tensor(Q.sigmaz(), UT1)
            UT2 = Q.tensor(Q.sigmay(), UT2)
            UT3 = Q.tensor(UT3, Q.sigmax())
        rho_a["{}".format(0)] = rho
        #rho_a["{}".format(1)] = UT * rho * (UT.conj().trans())
        rho_a["{}".format(2)] = UT1 * rho * (UT1.conj().trans())
        rho_a["{}".format(3)] = UT2 * rho * (UT2.conj().trans())
        #rho_a["{}".format(4)] = UT3 * rho * (UT3.conj().trans())
        return rho_a
    def Feature_vector(self, rho_a, u=True):
        X1, X2 = [], []
        n, Beta = self.n, self.Beta
        if n <= 8:
            beta = Beta[n - 2]
        else:
            beta = 2*np.pi/n
        e_0 = basis(2, 0) * (basis(2, 0).trans())
        e_1 = basis(2, 1) * (basis(2, 1).trans())
        A_1 = np.cos((n + 1) / (2 * n) * beta) * Q.sigmax() + np.sin((n + 1) / (2 * n) * beta) * Q.sigmay()  # A+
        A_2 = np.cos(-(n - 1) / (2 * n) * beta) * Q.sigmax() + np.sin(-(n - 1) / (2 * n) * beta) * Q.sigmay()  # A_
        M_x, A_z, A_x = Q.sigmax(), A_1, (A_1 + A_2) / 2
        for i in range(1, n):
            e_0 = Q.tensor(e_0, basis(2, 0) * (basis(2, 0).trans()))
            e_1 = Q.tensor(e_1, basis(2, 1) * (basis(2, 1).trans()))
            M_x = Q.tensor(M_x, Q.sigmax())
            A_z = Q.tensor(A_z, A_1)
            A_x = Q.tensor(A_x, (A_1 + A_2) / 2)
        M_z = e_0 + e_1
        if u is True: #要作酉变换
            for rho in rho_a.values():
                x1 = (M_x*rho).tr()
                x2 = (M_z*rho).tr()
                x3 = (A_x*rho).tr()
                x4 = (A_z*rho).tr()
                xx1 = np.array([x1, x2]).reshape(1, 2).real
                xx2 = np.array([x1, x2, x3, x4]).reshape(1, 4).real
                X1.append(xx1)
                X2.append(xx2)
            X1 = np.array(X1).reshape(3, 2)
            X2 = np.array(X2).reshape(3, 4)
        else:
            x1 = (M_x * rho_a).tr()
            x2 = (M_z * rho_a).tr()
            x3 = (A_x * rho_a).tr()
            x4 = (A_z * rho_a).tr()
            X1 = np.array([x1, x2]).reshape(1, 2).real
            X2 = np.array([x1, x2, x3, x4]).reshape(1, 4).real

        return X1
def Index(index,K):
    index1 = np.zeros((1, ))
    ii = np.ones((len(index1)))
    indexx1 = (K+1)*index
    for k in range(K+1):
        indexk = indexx1+k*ii
        index1 = np.concatenate((index1, indexk), axis=0)
    index1 = np.sort(index1, axis=0)
    index1 = index1.astype(np.int64)
    index1 = index1[1:]
    return index1
def QB(qb,K):
    #qb = to_categorical(qb, num_classes=None)
    Yb_hut = np.zeros(((K+1)*len(qb), 2))
    for k in range(K+1):
        Yb_hut[np.arange(k, len(Yb_hut), K+1), :] = qb
    return Yb_hut

 
