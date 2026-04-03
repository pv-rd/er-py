import numpy as np
from copy import copy, deepcopy
from Value import *

class Data():
    def __init__(self, A = list(), E = 0):
        self.v_read(A)
        self.e_read(E)

    def v_read(self, A):
        if type(A) == type(np.array([0.0])):
            self.v = deepcopy(A)
        else:
            self.v = np.array(A)
    
    def e_read(self, A):
        if type(A) == type(np.array([0.0])):
            self.e = deepcopy(A)
        elif isinstance(A, (int, float)):
            self.e = np.full(len(self.v), A)
        else:
            self.e = np.array(A)
               
    def read(self, file_name, col = 0, 
             e_col = None, e = 0, 
             sep_col = None, sep_dec = '', erase = True):
        
        file = open(file_name, 'r')
        V = list()
        E = list()
        
        for line in file:
            row = line.split(sep_col)
            ceil = row[col]
            for c in sep_dec:
                ceil = ceil.replace(c, '.')
            V.append(float(ceil))

            if e_col != None:
                e_ceil = row[e_col]
                for c in sep_dec:
                    e_ceil = e_ceil.replace(c, '.')
                E.append(float(e_ceil)) 
        file.close()

        if e_col == None:
            E = [e]*len(V)
        self.e = np.array(E)

        if erase: 
            self.v = np.array(V)
        else: 
            self.v = np.concatenate((self.v, np.array(V)))

    def __str__(self):
        L = list()
        for i in range(len(self.v)):
            L.append(str(Value(self.v[i], self.e[i])))
        return(', '.join(L))
    

    def oper(self, other, operator = '+'):
        F = {
            '+' : lambda x, y: x + y,
            '*' : lambda x, y: x * y,
            '-' : lambda x, y: x - y,
            'r-' : lambda x, y: y - x,
            '/' : lambda x, y: x / y,
            'r/' : lambda x, y: y / x,
            'p+': lambda x, y: (x*x + y*y)**0.5
        }

        np_der = {
            'exp': lambda x: np.exp(x),
            'log': lambda x: 1/x,
            'sin': lambda x: np.cos(x),
            'cos': lambda x: -np.sin(x),
            'tan': lambda x: 1/(np.cos(x)**2),
            'sqrt': lambda x: 0.5/np.sqrt(x)
        }

        np_f = {
            'exp': lambda x: np.exp(x),
            'log': lambda x: np.log(x),
            'sin': lambda x: np.sin(x),
            'cos': lambda x: np.cos(x),
            'tan': lambda x: np.tan(x),
            'sqrt': lambda x: np.sqrt(x)
        }

        if operator in F:

            if (type(other) == type(self) or type(other) == type(Value())):
                xv = other.v
                xe = other.e
            elif isinstance(other, (int, float)): 
                xv = other
                xe = 0
            else:
                return NotImplemented

            V = F[operator](self.v, xv)
            if operator in "+-r-":
                E = F['p+'](self.e, xe)
            elif operator in "*/r*r/":
                E = F['p+'](self.e/self.v, xe/xv)
        
        elif operator in np_der:
            V = np_f[operator](self.v)
            E = np.abs(np_der[operator](self.v))*self.e

        elif operator == '**':
            if isinstance(other, (int, float)):
                V = self.v ** other
                E = (self.e / self.v) * V * other
            else:
                return NotImplemented
        
        return Data(V, E)
    
    def __add__(self, other):
        return self.oper(other, '+') 
    
    def __radd__(self, other):
        return self.oper(other, '+') 
    
    def __sub__(self, other):
        return self.oper(other, '-') 
    
    def __rsub__(self, other):
        return self.oper(other, 'r-')
    
    def __neg__(self):
        return Data(-self.v, self.e)
    
    def __mul__(self, other):
        return self.oper(other, '*') 
    
    def __rmul__(self, other):
        return self.oper(other, '*') 
    
    def __truediv__(self, other):
        return self.oper(other, '/') 
    
    def __rtruediv__(self, other):
        return self.oper(other, 'r/')

    def __pow__(self, other):
        return self.oper(other, '**') 
    
    def log(self):
        return self.oper(1, 'log') 

    def exp(self):
        return self.oper(1, 'exp')
    
    def sqrt(self):
        return self.oper(1, 'sqrt')
        
    def mean(self):
        return self.v.mean()
    
    def __invert__(self):
        return self.mean()
    
    def catch(self, d):
        prev = self.v[0]
        i = 1
        while i < len(self.v):
            if abs(self.v[i] - prev) >= d:
                self.v = np.delete(self.v, i)
                self.e = np.delete(self.e, i)
                continue
            prev = self.v[i]
            i += 1

A = Data([1, 2, 3, 8, 5, 6, 9, 8])
B = Data([1, 2, 3, 4, 5 ,6, 7, 8])
print(A.exp())