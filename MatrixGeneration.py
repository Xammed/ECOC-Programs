from scipy.linalg import hadamard
import matplotlib.pyplot as hadGraph
import itertools
import random
import numpy as np

def hammingWeight(code):
    weight = 0
    for b in code:
        weight+=b
    return weight

#Produces a Hadamard of 0's and 1's such that n=2^k
def MakeBinaryHadamard(n):
    matrix = []
    for i in range(len(hadamard(n))):
        matrix.append([])
        for entry in hadamard(n)[i]:
            if (entry==-1):
                matrix[i].append(0)
            else:
                matrix[i].append(entry)
    return matrix

#Truncates any given matrix by t rows/columns acting on minor
#Note, minor specifies the vector containing the minor
#(Which by definition gets truncated)
def TruncateMinor(matrix, t, minor):
    a = 0
    w_on = matrix
    while (a<t):
        w_on.remove(w_on[minor])
        temp = transpose(w_on)
        temp.remove(temp[minor])
        w_on = transpose(temp)
        a+=1
    return w_on

#Transposes any given matrix
def transpose(matrix):
    transposed = []
    for i in range(len(matrix[0])):
        transposed.append([])
        for vector in matrix:
            transposed[i].append(vector[i])
    return transposed

#Produces any square matrix of specified length from a Hadamard
def ProduceSquare(length):
    found = 0
    base = 1
    power = 0
    while (found==0):
        base*=2
        power+=1
        if(base>length):
            found = 1
    matrix = MakeBinaryHadamard(base)
    t = base-length
    final = TruncateMinor(matrix, t, 0)
    return final

#Truncates t rows linearly
def truncateRows(matrix, t):
    a = 0
    w_on = matrix
    removalDictionary = []
    while (a<t):
        removalDictionary.append(w_on[0])
        w_on.remove(w_on[0])
        a+=1
    return w_on, removalDictionary

#For Writing matrices to file
def writetoFile(matrix, name):
    file = open(str(name)+".txt", "w")
    for b in matrix:
        file.write(str(b) + "\n")
    file.write("\n")

def generateRandomVector(length):
    vector = []
    onePerm = 0
    if length%2==0:
        onePerm = length/2
    else:
        onePerm = (length+1)/2
    for i in range(int(onePerm)):
        vector.append(1);
    while (len(vector)<length):
        vector.insert(random.randrange(0, length-onePerm),0)
   # for i in range(length):
   #     vector.append(random.choice([0, 1]))
   # if (hammingWeight(vector)==0 or hammingWeight(vector)==length):
   #     generateRandomVector(length)
    return vector

def FilterReplaceMatrices(matrix):

    for i in range (len(matrix)):
        weight = hammingWeight(matrix[i])
        if weight==0 or weight== len(matrix[i]):
            matrix[i]=generateRandomVector(len(matrix[0]))
    matrixColumnOrientation = transpose(matrix)
    for i in range (len(matrixColumnOrientation)):
        weight = hammingWeight(matrixColumnOrientation[i])
        if weight==0 or weight == len(matrixColumnOrientation):
            matrixColumnOrientation[i]=generateRandomVector(len(matrixColumnOrientation[0]))
    return transpose(matrixColumnOrientation)

#Generates a Matrix of your choosing
#Note: codeLength>= numberClasses for all matrices!
def GenerateMatrix(numberClasses, codeLength):
    squareSeedWidth = numberClasses if (numberClasses>codeLength) \
        else codeLength
    squareSeed = ProduceSquare(squareSeedWidth)
    linearRowTruncation = codeLength-numberClasses
    preMatrix, removalDictionary = truncateRows(squareSeed, linearRowTruncation)
    finalMatrix = FilterReplaceMatrices(preMatrix)
    return finalMatrix

#Same as above but writes to file
def GenerateMatrixFILE(numberClasses, codeLength):
    squareSeedWidth = numberClasses if (numberClasses>codeLength) \
        else codeLength
    squareSeed = ProduceSquare(squareSeedWidth)
    linearRowTruncation = codeLength-numberClasses
    preMatrix, removalDictionary = truncateRows(squareSeed, linearRowTruncation)
    finalMatrix = FilterReplaceMatrices(preMatrix)
    writetoFile(finalMatrix, ""+str(numberClasses) + "x"+str(codeLength))
    return finalMatrix

def hammingDistance(c1, c2):
    distance = 0
    for (x, y) in zip(c1, c2):
        distance += (x+y)%2
    return distance