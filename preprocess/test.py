import numpy as np

a = np.array([[1, 2, 3], [3, 4, 5], [5, 6, 7]])
b = np.array([[1], [2], [3]])
c = a*b
print(c)
print(c.mean(axis=0))

str = ["gd", "o", "u", "l", "i", "g", "u", "o", "j", "i", "a", "s", "h", "e", "n", "s", "i", "y", "i"]
s = set(str)
print(s)
