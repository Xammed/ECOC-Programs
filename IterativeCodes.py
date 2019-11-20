
import random
import math
from scipy.linalg import hadamard
import matplotlib.pyplot as hadGraph
import matplotlib.pyplot

def hammingDistance(a, b):  # Finds the hamming distance between two codewords
    distance = 0
    #print(len(a), len(b))
    for i in range(len(a)):
        distance += (a[i] ^ b[i])
    return distance

def hammingWeight(code):
    weight = 0
    for b in code:
        weight+=b
    return weight

def checkWeights(codes, HWmin, HWmax):
    for i in range(len(codes[0])):
        for code in codes:
            weight = hammingWeight(code)
            if (weight>= HWmin and weight<=HWmax):
                return 1
    return 0

def minHammingDistance(code):
    minDistance = len(code[0])
    for i in range(len(code[0])-1):
        for j in range(i+1,len(code[0])):
            tmp = hammingDistance(code[i], code[j])
            if (tmp < minDistance):
                minDistance = tmp
    return minDistance

def complementCheck(code):
    maxd = len(code[0])
    hasComplement = 0
    for a in code:
        for b in code:
            if a != b:
                tmp = hammingDistance(a, b)
                if tmp == maxd:
                    hasComplement = 1
    return hasComplement

def makeIntoCol(Rows):
    Columns = []
    for i in range(len(Rows[0])):
        Columns.append([])
        for row in Rows:
            Columns[i].append(row[i])
    return Columns

def giveMeAnswers(codebook):
    module = str(minHammingDistance(codebook)) + " " +\
             str(minHammingDistance(makeIntoCol(codebook)))
    return module

def makeCodes(l, w, Rd, Cd, HWmin, HWmax):
    columns = []
    rows = []
    while ((len(columns)<w)):
        word = []
        weight = 0
        for i in range(l):
            word.append(random.choice([0,1]))
            weight += word[i]
            #print(weight)
        if (weight >= HWmin and weight <= HWmax):
            if (len(columns)==0):
                columns.append(word)
            else:
                checkAgainst = 0
                for code in columns:
                    distance = 0
                    for i in range(len(word)):
                        distance+= (code[i]+word[i])%2
                    if (distance<Cd or distance==l):
                        checkAgainst=1
                        #print(word, columns, distance)
                if (checkAgainst==0):#and not(columns.__contains__(word))):
                    columns.append(word)
                    #print(columns)
        if (len(columns)==w):
            for i in range(len(columns[0])):
                #print(i)
                rows.append([])
                #print(rows)
                for col in columns:
                    rows[i].append(col[i])
            if(minHammingDistance(rows) >= Rd):
                #print(rows)
                #print (columns)
                return rows
            else:
                columns = []
                rows = []




def makeBinHad(n):
    matrix = []
    for i in range(len(hadamard(n))):
        matrix.append([])
        for entry in hadamard(n)[i]:
            if (entry==-1):
                matrix[i].append(0)
            else:
                matrix[i].append(entry)
    #print(matrix)
    return matrix


def findHadDistances(hadamard):
    xPts = []
    yPts = []
    size = len(hadamard)
    functionSet = []
    for i in range (size-1):
        for j in range((i+1), size):
            yPts.append(hammingDistance(hadamard[i], hadamard[j]))

    for i in range(len(yPts)):
        xPts.append(i)
    functionSet.append(xPts)
    functionSet.append(yPts)
    #print(functionSet)
    #hadGraph.plot(xPts, yPts)
    #hadGraph.xlabel("Lexicographic Postion")
    #hadGraph.ylabel("Hamming Distance")
    #hadGraph.title("Position x Distance")
    #hadGraph.show()
    #print(yPts)
    #print(average(yPts))
    return functionSet

def transpose(matrix):
    transposed = []
    for i in range(len(matrix[0])):
        # print(i)
        transposed.append([])
        # print(rows)
        for vector in matrix:
            transposed[i].append(vector[i])
    return transposed



def truncate(matrix, t, minor):
    #print(matrix)
    a = 0
    w_on = matrix
    while (a<t):
        w_on.remove(w_on[minor])
        temp = transpose(w_on)
        temp.remove(temp[minor])
        w_on = transpose(temp)
        a+=1
    #print(w_on)
    return w_on

def truncateRows(matrix, t):
    #print(matrix)
    a = 0
    w_on = matrix
    while (a<t):
        w_on.remove(w_on[0])
        a+=1
    #print(w_on)
    return w_on


def writetoFile(matrix, name):
    file = open(str(name)+".txt", "a")
    for b in matrix:
        file.write(str(b) + "\n")
    file.write("\n")

def average(values):
    sum = 0
    size = 0
    for i in values:
        sum+=i
    average = sum/len(values)
    return average

def ProduceSquare(length):
    found = 0
    base = 1
    power = 0
    while (found==0):
        base*=2
        power+=1
        if(base>length):
            found = 1
    matrix = makeBinHad(base)
    t = base-length
    final = truncate(matrix, t, 0)
    #writetoFile(final, "3-12 Codes")
    return final

def generateRowTruncationData(n):
    seedMatrix = makeBinHad(n)
    i =0
    distances = []
    xAxis = []
    while (i<=(n/2)):
        temp = transpose(seedMatrix)
        distances.append(generateHammingDistances(temp))
        seedMatrix.remove(seedMatrix[0])
        temp = transpose(temp)
        i+=1
    for i in range (len(distances)):
       xAxis.append(i)
    hadGraph.plot(xAxis, distances)
    hadGraph.xlabel("Number of Rows Truncated")
    hadGraph.ylabel("Column Hamming Distance")
    hadGraph.title("T x Hc")
    hadGraph.show()
    return distances

def generateHammingDistances(matrix):
    size = len(matrix)
    distances = []
    for i in range (size-1):
        for j in range((i+1), size):
            distances.append(hammingDistance(matrix[i], matrix[j]))
    return distances

def findMin(values):
    min = 100
    for i in values:
        if (i<min):
            min = i;
    return min


generateRowTruncationData(64)

#print(generateRowTruncationData(64))
(2, 32)
(1, 21)
(2, 30)
(1, 29)
(3, 28)
(1, 27)
(1, 26)
(1, 25)
(5, 24)
(1, 23)
(1, 22)
(1, 21)
(1, 20)
(1, 19)
(1, 18)
(1, 17)
(9, 16)

(2, 2, 3, 5, 9)

#dataPat = [2, 1, 2, 1, 3, 1, 1, 1, 5, 1, 1, 1, 1, 1, 1, 1, 9]

(1, 1, 3, 7, 15, 31)




#print(truncate(makeBinHad(8), 1, 0))
#findHadDistances(Produce(100))
#print(findHadDistances(makeBinHad(64)))
#print(truncateRows(makeBinHad(32), 2))
#generateRowTruncationData(128)
#print(makeBinHad(8))

#Write transpose, do you already have it?
#Write a code to randomly pick one bit, and change it. (in a hadamard)
#Is it consistent with expectatuibs>
#print(makeCodes(26,26,10,5,8,18))
#makeBinHad(4)

#findHadDistances(makeBinHad(1024))



#makeCodes(4,4,1,1,1,4)
#makeCodes(10,10,1,1,1,10)
#print(giveMeAnswers(makeBinHad(8)))
#Xor on two lists.

#H = 2^(k-1) - the 2-adic valuation of r, where r is the number of rows truncated/
#(H is column distance)
