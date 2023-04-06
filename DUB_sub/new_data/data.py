import os
import pandas as pd
f_1 = open(os.path.join("E3_sub_name.txt"), "r")
f_2 = open(os.path.join("id_E3.txt"), "r")
f_3 = open(os.path.join("id_substrate.txt"), "r")

id_e3 = {}
id_sub = {}
for x in f_2:
    x = x.strip().split()
    x1 = int(x[0])
    x2 = x[1].strip('\n')
    id_e3[x2] = x1

for x in f_3:
    x = x.strip().split()
    x1 = int(x[0])
    x2 = x[1].strip('\n')
    id_sub[x2] = x1

print(id_e3)
print(id_sub)
sub_e3 = []
for x in f_1:
    x = x.strip().split()
    x1 = int(id_e3.get(x[0]))
    x2 = int(id_sub.get(x[1].strip('\n')))
    sub_e3.append([x1,x2])
data = pd.DataFrame(sub_e3)
data.to_csv("test.csv")
print(len(sub_e3))