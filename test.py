from autocorrect import Speller
import re
import numpy as np
import correlation
# spell = Speller(lang='en')
# print(spell("I'm not sleapy and tehre is no place I'm giong to."))

a = np.loadtxt("C:\\Users\\Cecilia\\Desktop\\Untitled.txt")
print(a.shape)
print(correlation.pearson_cor(a[:, 0], a[:, 1]))
