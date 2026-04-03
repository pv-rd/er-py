import numpy as np
from copy import copy, deepcopy
from Value import *

def add_methods(cls):
    F = {
        '__add__' : lambda x, y: x + y,
        '__radd__' : lambda x, y: x + y,
        '__sub__' : lambda x, y: x - y,
        '__rsub__' : lambda x, y: y - x,
    }
    G = {
        '__mul__' : lambda x, y: x * y,
        '__rmul__' : lambda x, y: x * y,
        '__truediv__' : lambda x, y: x / y,
        '__rtruediv__' : lambda x, y: y / x,
    }
    der = {
        'exp': lambda x: np.exp(x),
        'log': lambda x: 1/x,
        'sin': lambda x: np.cos(x),
        'cos': lambda x: -np.sin(x),
        'tan': lambda x: 1/(np.cos(x)**2),
        'sqrt': lambda x: 0.5/np.sqrt(x)
    }
    f = {
        'exp': lambda x: np.exp(x),
        'log': lambda x: np.log(x),
        'sin': lambda x: np.sin(x),
        'cos': lambda x: np.cos(x),
        'tan': lambda x: np.tan(x),
        'sqrt': lambda x: np.sqrt(x)
    }

    def p(x,y): return (x*x + y*y)**0.5

    for op in F.keys():
        def make_method(op):
            def method(self, other):
                if isinstance(other, (int, float)):
                    v = F[op](self.v, other)
                    e = self.e
                    return Data(v, e)
                elif type(other) == type(self) or type(other) == type(Value()):
                    v = F[op](self.v, other.v)
                    e = p(self.e, other.e)
                    return Data(v, e)
                else:
                    return NotImplemented
            return method
        setattr(cls, op, make_method(op))

    for op in G.keys():
        def make_method(op):
            def method(self, other):
                if isinstance(other, (int, float)):
                    v = G[op](self.v, other)
                    r = abs(self.e/self.v)
                    e = v * r
                    return Data(v, e)
                elif type(other) == type(self) or type(other) == type(Value()):
                    v = G[op](self.v, other.v)
                    r = p(abs(self.e/self.v), abs(other.e/other.v))
                    e = v * r
                    return Data(v, e)
                else:
                    return NotImplemented
            return method
        setattr(cls, op, make_method(op))
    
    for op in der.keys():
        def make_method(op):
            def method(self):
                V = f[op](self.v)
                E = np.abs(der[op](self.v))*self.e
                return Data(V, E)
            return method
        setattr(cls, op, make_method(op))

    return cls   


@add_methods
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
            self.e = np.zeros(len(self.v))
            self.e.fill(A)
        else:
            self.e = np.array(A)
    
    def __len__(self): 
        return self.v.size
               
    def read(self, 
             file_name, 
             col = 0, 
             e_col = None, 
             e = 0,
             ignore_first = False, 
             sep_col = None, 
             sep_dec = '', 
             erase = True):
        
        V = list()
        E = list()

        file = open(file_name, 'r')
        
        if ignore_first:
            s = file.readline()
            del s

        for line in file:
            row = line.split(sep_col)
            cell = row[col]
            for c in sep_dec:
                cell = cell.replace(c, '.')
            V.append(float(cell))

            if e_col != None:
                e_cell = row[e_col]
                for c in sep_dec:
                    e_cell = e_cell.replace(c, '.')
                E.append(float(e_cell))

        file.close()

        if e_col == None:
            E = [e]*len(V)
            

        if erase: 
            self.v = np.array(V)
            self.e = np.array(E)
        else: 
            self.v = np.concatenate((self.v, np.array(V)))
            self.e = np.concatenate((self.e, np.array(E)))

    def __str__(self):
        L = list()
        for i in range(len(self.v)):
            L.append(str(Value(self.v[i], self.e[i])))
        return(', '.join(L))
    
    def mean(self):
        return self.v.mean()
    
    def __invert__(self):
        return self.mean()
    
    def __pow__(self, other):
        if isinstance(other, (int, float)):
            v = self.v ** other
            r = abs(self.e/self.v) * abs(other)
            return Data(v, v * r)
    
    def catch(self, d, rewrite = True):
        V = deepcopy(self.v)
        E = deepcopy(self.e)

        prev = V[0]
        i = 1
        while i < len(V):
            if abs(V[i] - prev) >= d:
                V =  np.delete(V, i)
                E  = np.delete(E, i)
                continue
            prev = V[i]
            i += 1
        
        if rewrite:
            self.v = deepcopy(V)
            self.e = deepcopy(E)

        return Data(V, E)
