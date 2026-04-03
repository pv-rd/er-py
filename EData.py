from Value import *
class EData():
    def __init__(self, A = list()):
        self.data = A[:]

    def read_file(self, file_name, col = 0, e_col = None, e = 0, sep_col = None, sep_dec = '', erase = True):
        file = open(file_name, 'r')
        res = list()
        for line in file:
            row = line.split(sep_col)
            ceil = row[col]
            for c in sep_dec:
                ceil = ceil.replace(c, '.')
            value = float(ceil)
            if e_col != None:
                e_ceil = row[e_col]
                for c in sep_dec:
                    e_ceil = e_ceil.replace(c, '.')
                error = float(e_ceil)
            else:
                error = e
            v = Value(value, error)
            res.append(v)
        
        file.close()

        if erase:
            self.data = res
        else:
            self.data = self.data + res