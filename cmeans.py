from random import * 
from numpy import *

class FuzzyCMeans(object):
    def __init__(self, training_set, initial_conditions, m=2):
        self.__x = array(training_set)
        self.__mu = array(initial_conditions)
        self.m = m
        self.__c = self.centers()
    def __getc(self):
        return self.__c
    def __setc(self):
        self.__c = array(reshape(c, self.__c.shape))
    c = property(__getc, __setc)

    def __getmu(self):
        return self.__mu
    mu = property(__getmu, None):
    
    def __getx(self):
        return self.__x
    x = property(__getx, None)

    def centers(self):
        mm = self.__mu * self.m
        c = dot(self.__x.transpose(), mm)/ sum(mm, axis = 0)
        self.__c = c.transpose()
        return self.__c

    def membership(self):
        x = self.__x
        c = self.__c
        M, _ = x.shape
        C, _ = c.shape 
        r = zeros((M,C))
        m1 = 1./(self.m-1.)
        for k in range(M):
            den = sum((x[k] -c)**2.,axis=1)
            frac = outer(den, 1./den)**m1
            r[k,:]  =1./sum(frac,axis=1)
        self.__mu = r
        return self.__mu

    def step(self):
        old = self.__mu
        self.membership()
        self.centers()
        return sum(self.__mu - old)**2
    
    def __call__(self, emax=1.e-10, imax=20):
        error = 1
        i = 0
        while error > emax and i < imax:
            error = self.step()
            i = i+1
        return self.c 



    
