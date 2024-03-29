import numpy as np
from fractions import Fraction
import os

patterns = int(input("Number of patterns: "))
inn = int(input("Number of input neurons: "))
outn = int(input("Number of output neurons: "))
patin = []

aux = True
minus = 0
pesos = []

for i in range(patterns):
	print(f"\nInput vector #{i+1}\n")
	patin.append([])
	for j in range(inn):
		patin[i].append(int(input(f"Data #[{j+1}]: ")))
		if aux and (patin[i][j] == 0):
			minus = 0
			aux = False
		if aux and (patin[i][j] == -1):
			minus = -1
			aux = False
	pesos.append(np.asarray(patin[i]))

patout = []

for i in range(patterns):
	print(f"\nOutput vector #{i+1}\n")
	patout.append([])
	for j in range(outn):
		patout[i].append(int(input(f"Data #[{j+1}]: ")))
		if aux and (patin[i] == 0):
			minus = 0
			aux = False
		if aux and (patin[i] == -1):
			minus = -1
			aux = False

w = []


for i in range(inn):
	w.append([])
	for k in range(outn):
		suma = 0
		for j in range(patterns):
			suma += patin[j][i]*patout[j][k]
		w[i].append(suma)

pesos = np.asarray(pesos)

print("Weight Matrix:\n")
for i in range(len(w)):
	print("[",end="")
	for j in range(len(w[i])):
		if j+1 == len(w[i]):
			print(f"{w[i][j]}", end="")
		else:
			print(f"{w[i][j]}", end=",")
	print("]")
print("\n")

print("X patterns evaluation:\n")

for i in range(len(patin)):
	aux = np.dot(np.transpose(w),np.asarray(patin[i]))
	print(f'f(t){aux} =',end=" ")
	aux[aux >= 0] = 1
	aux[aux < 0] = minus
	if (np.allclose(np.array(aux),np.array(patout[i]))):
		print(f"{np.array(aux)} Correctly associates with {np.array(patout[i])} (Y{i+1})\n")
	else:
		print(f"{np.array(aux)} Does not associate correctly\n")

print("Y patterns evaluation:\n")

for i in range(len(patout)):
	aux = np.dot(w,np.asarray(patout[i]))
	print(f'f(t){aux} =',end=" ")
	aux[aux >= 0] = 1
	aux[aux < 0] = minus
	if (np.allclose(np.array(aux),np.array(patin[i]))):
		print(f"{np.array(aux)} Correctly associates with {np.array(patin[i])} (X{i+1})")
	else:
		print(f"{np.array(aux)} Does not associate correctly")
