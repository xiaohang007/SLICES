# -*- coding: utf-8 -*-
# Hang Xiao 2023.04
# xiaohang07@live.cn
import json,os
import pickle



with open("formula_reduced_list_unique20.pickle", "rb") as fp:   # Unpickling
    a = pickle.load(fp)
with open("formula_reduced_list_unique.pickle", "rb") as fp:   # Unpickling
    b = pickle.load(fp)
c=a+b
print(len(c))
c=list(set(c))
print(len(c))
with open("formula_reduced_unique_all.pickle", "wb") as fp:   #Pickling
    pickle.dump(c, fp)
#with open('composition_reduced_list_unique.json', 'w') as f:
    #json.dump(composition_reduced_list_unique, f)

