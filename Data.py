import numpy as np
from copy import copy, deepcopy
from Value import *
from Functions import *

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
        '__truediv__' : lambda x, y: inf if y == 0 else x / y,
        '__rtruediv__' : lambda x, y: inf if x == 0 else y / x,
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
                    if any(self.v == 0): raise ValueError("Array has zero value(s)")
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

    return cls   


@add_methods
@F_add
@G_add
class Data():

    def __init__(self, A = list(), E = 0):
        self.v_read(A)
        self.e_read(E)

    def v_read(self, A, erase = True):
        if isinstance(A, np.ndarray):
            V = deepcopy(A)

        elif isinstance(A, (list, tuple)):
            V = np.array(A)

        elif isinstance(A, (float, int)):
            V = np.array([A])

        else:
            raise TypeError(f"Unsupported type for reading: {type(A)}")

        if erase:
            self._v = V
        else:
            self._v = np.concatenate((self._v, V))
        
    
    def e_read(self, A, erase = True):
        if isinstance(A, np.ndarray):
            E = deepcopy(A)

        elif isinstance(A, (list,tuple)):
            E = np.array(A)

        elif isinstance(A, (int, float)):
            if hasattr(self, '_v'):
                E = np.zeros(len(self._v))
                E.fill(A)
            else:
                raise AttributeError("Array length is unknown")
            
        else:
            raise TypeError(f"Unsupported type for reading: {type(A)}")

        if erase:
            self._e = E
        else:
            self._e = np.concatenate((self._e, E))
    
    @property
    def v(self): return self._v

    @property
    def e(self): return self._e
    
    def __len__(self): 
        return self._v.size
               
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
        
        with open(file_name, 'r') as file:
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

        if e_col == None:
            E = [e]*len(V)
            
        self.v_read(V, erase)
        self.e_read(E, erase)

    def __str__(self):
        L = list()
        for i in range(len(self.v)):
            L.append(str(Value(self.v[i], self.e[i])))
        return(', '.join(L))
    
    def rand_e(self):
        return np.std(self.v)
    
    def sys_e(self):
        return self.e.max()
    
    def mean(self):
        E = (self.rand_e()**2 + self.sys_e()**2)**0.5
        E_mean = E / len(self)**0.5
        return Value(self.v.mean(), E_mean)
    
    def __invert__(self):
        return self.mean()
    

    def catch(self, d, rewrite = True):
        V = deepcopy(self.v)
        E = deepcopy(self.e)

        if len(self) == 0: return self
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
            self.v_read(V)
            self.e_read(E)

        return Data(V, E)
    
    def __getitem__(self, index):
        if isinstance(index, slice):
            return Data(self.v[index], self.e[index])
        return Value(self.v[index], self.e[index])

    def __setitem__(self, index, value):
        if isinstance(value, Value):
            self.v[index] = value.v
            self.e[index] = value.e
        elif isinstance(value, (int, float)):
            self.v[index] = value
        else:
            raise TypeError(f"Unsupported type: {type(value)}")
        
    def zeroes(self, change = None):
        if change == None:
            return Data(np.nonzero(self.v), np.nonzero(self.e))
        new_v = self.v.copy()
        new_v[new_v == 0] = change
        return Data(new_v, self.e)