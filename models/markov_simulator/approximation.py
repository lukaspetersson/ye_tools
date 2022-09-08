import math
from unicodedata import name
import numpy as np
import matplotlib.pyplot as plt

# get the coeficents in the diophantine approximation of a irrational number
def approximate(irrational_number):
    coefficients = []
    x = irrational_number
    for i in range(3):
        # split number in whole number and decimal part
        x, a = math.modf(x)
        coefficients.append(a)
        # if x == 0, the irrational number was rational
        if (x == 0):
            break
        x = 1/x
    return coefficients

def continued_fraction(coefficients):
    prev_frac = 0
    new_frac = 0
    if (len(coefficients) == 2):
        return (coefficients[0] * coefficients[1] + 1, coefficients[1])
    for i in range(len(coefficients) - 1, 0, -1):
        new_frac = 1/(coefficients[i] + prev_frac)
        prev_frac = new_frac

    return coefficients[0] + new_frac

def evaluate(coefficients): 
    if (len(coefficients) == 1):
        numerator = coefficients[0]
        denominator = 1
        return (numerator, denominator)
    else:
        prev_numerator = coefficients[-1]
        prev_denominator = 1
        numerator = 0
        denominator = 0
        for i in range(len(coefficients) - 2, 0, -1):
            # a + b / c = (ac + b) / c
            numerator = coefficients[i] * prev_numerator + prev_denominator
            denominator = prev_numerator

            prev_numerator = numerator
            prev_denominator = denominator

        # switch numerator and denominator
        return (denominator + numerator, numerator)

res_list = []
for i in range(0, 24):
    
    # a half step has relative frequency 2^(1/12)
    num = 2**(i/12)

    # get value and position of the max coeficent in the diophantine approximation
    res = approximate(num)
    m = max(res)
    ind = res.index(m)

    frac = continued_fraction(res)
    print(f"K = {i} = {res}")
    error_percent = ((num / frac) - 1) * 100
    numerator, denominator = evaluate(res)

    res_list.append([res, m, ind, num, frac, error_percent, i, numerator, denominator])

    
correct_list = [7, 5, 2, 9, 4, 11, 1, 3, 6, 8, 10]

res_list.sort(key=lambda x: 1 / ((abs(x[5] / 100)**2 + x[8])))

print("####################################")
print("####################################")
print("####################################")
print("Evaluating 2^(k/12) for different k.")
for i in range(len(res_list)):
    print("------------------------------")
    #print(f"k = {res_list[i][6]}, Pos: {i+1}({correct_list.index(res_list[i][6]) + 1})")
    print(f"k = {res_list[i][6]}, Pos: {i+1}")
    print(f"Coefficents: {res_list[i][0]}")
    print(f"Biggest coefficient: {res_list[i][1]}, at index: {res_list[i][2]}")
    print(f"Input number: {res_list[i][3]}")
    print(f"Approximation: {res_list[i][4]}")
    print(f"Approximation fraction: {res_list[i][7]} / {res_list[i][8]}")
    print(f"Error percentage: {res_list[i][5]}")
    print(f"KAJGDiUAWHFWo: {1 / (abs(res_list[i][5] / 100)**2 + res_list[i][8])}")
    #print(f"1/Error: {1/abs(res_list[i][5])}")
    print("------------------------------")


# sortera efter storlek på nämnare
res_list.sort(key= lambda x: x[6])

def get_weight(l):
    return 1 / ((abs(l[5] / 100)**2 + l[8]))
    #return l[6]

matrix = []
n = 24
for i in range(n):
    temp_list = [0 for i in range(n)]
    for j in range (n):
        weight = get_weight(res_list[j])
        temp_list[(i+j) % n] = weight
    norm = [e/sum(temp_list) for e in temp_list]
    matrix.append(norm)


with open("matrix.txt", "w") as f:
    for l in matrix:
        f.write("[" + ", ".join(str(e) for e in l) + "]" + "\n")

plt.imshow(matrix, cmap="binary")
plt.colorbar()
plt.savefig("matrixColormap")
plt.show()
