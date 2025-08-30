import pickle
import numpy as np
import random

TEMPERATURE = 0.6
file = open("10000dp.pkl",'rb')
k1 = pickle.load(file)
file.close()

file = open("100000dp.pkl",'rb')
k2 = pickle.load(file)
file.close()

file = open("400000dp.pkl",'rb')
k3 = pickle.load(file)
file.close()

ws1 = k1[0]
wl1 = k1[1]
b1 = k1[2]
dnnw1 = k1[3]
dnnb1 = k1[4]

ws2 = k2[0]
wl2 = k2[1]
b2 = k2[2]
dnnw2 = k2[3]
dnnb2 = k2[4]

ws3 = k3[0]
wl3 = k3[1]
b3 = k3[2]
dnnw3 = k3[3]
dnnb3 = k3[4]

with open("shakespeare.txt") as f:
    textstring = f.read().lower()

chartonum = dict()
numtochar = dict()

charlist = set(list(textstring))

def onehot(charing):
    nin = np.zeros((len(charlist),1))
    for i in range(len(charlist)):
        if chartonum[charing] == i:
            nin[i,0] = 1
        else:
            nin[i,0] = 0
    return nin

charfrequencies = dict()
for char in textstring:
    if char in charfrequencies:
        charfrequencies[char] +=1
    else:
        charfrequencies[char] = 1

newcf = sorted(charfrequencies.keys(), key=lambda m: charfrequencies[m], reverse=True)

for index, value in enumerate(newcf):
    chartonum[value] = index
    numtochar[index] = value

def hyptan(k):
        return np.tanh(k)

def hyptand(k):
    return 1/((np.cosh(k))**2)

def dnn_p_net(A_vec, w, b, inp):
    #a is a Python list of numpy matrices that will hold the output at each layer
    a = []
    a.append(inp)
    #n = the number of perceptron layers in this network
    n = len(w)
    for i in range(0,n):
        a.append(A_vec(w[i] @ a[i] + b[i]))
    return a[-1]

def softmax(m):
    temp = np.exp(m/TEMPERATURE)
    return temp / np.sum(temp)

seed = "t"
while(len(seed)<400):
    nconst = len(wl1)-1
    fconst = len(seed)
    dim = [len(charlist),40,40]
    a = dict()
    dots = dict()
    for ml in range(1, nconst+1):
        a[(ml,0)] = np.zeros((dim[ml],1))
    for s in range(1, fconst+1):
        a[(0,s)] = onehot(seed[s-1])
        for l in range(1,len(wl1)):
            t = a[(l-1,s)]
            z = a[(l,s-1)]
            dots[(l,s)] = wl1[l]@t + ws1[l]@z + b1[l]
            a[(l,s)] = hyptan(dots[(l,s)])
    a[(nconst+1,fconst)] = dnn_p_net(softmax, dnnw1, dnnb1, a[(nconst,fconst)])
    toret = []
    for i in range(a[(nconst+1,fconst)].shape[0]):
        toret.append(a[(nconst+1,fconst)][i,0])
    mylist = [i for i in range(len(charlist))]
    choice = random.choices(mylist, weights = toret, k=1)
    seed += numtochar[choice[0]]
print("10000 Datapoints Output:")
print(seed)
print()
print()

seed = "t"
while(len(seed)<400):
    nconst = len(wl2)-1
    fconst = len(seed)
    dim = [len(charlist),40,40]
    a = dict()
    dots = dict()
    for ml in range(1, nconst+1):
        a[(ml,0)] = np.zeros((dim[ml],1))
    for s in range(1, fconst+1):
        a[(0,s)] = onehot(seed[s-1])
        for l in range(1,len(wl2)):
            t = a[(l-1,s)]
            z = a[(l,s-1)]
            dots[(l,s)] = wl2[l]@t + ws2[l]@z + b2[l]
            a[(l,s)] = hyptan(dots[(l,s)])
    a[(nconst+1,fconst)] = dnn_p_net(softmax, dnnw2, dnnb2, a[(nconst,fconst)])
    toret = []
    for i in range(a[(nconst+1,fconst)].shape[0]):
        toret.append(a[(nconst+1,fconst)][i,0])
    mylist = [i for i in range(len(charlist))]
    choice = random.choices(mylist, weights = toret, k=1)
    seed += numtochar[choice[0]]
print("100000 Datapoints Output:")
print(seed)
print()
print()

seed = "t"
while(len(seed)<400):
    nconst = len(wl3)-1
    fconst = len(seed)
    dim = [len(charlist),40,40]
    a = dict()
    dots = dict()
    for ml in range(1, nconst+1):
        a[(ml,0)] = np.zeros((dim[ml],1))
    for s in range(1, fconst+1):
        a[(0,s)] = onehot(seed[s-1])
        for l in range(1,len(wl3)):
            t = a[(l-1,s)]
            z = a[(l,s-1)]
            dots[(l,s)] = wl3[l]@t + ws3[l]@z + b3[l]
            a[(l,s)] = hyptan(dots[(l,s)])
    a[(nconst+1,fconst)] = dnn_p_net(softmax, dnnw3, dnnb3, a[(nconst,fconst)])
    toret = []
    for i in range(a[(nconst+1,fconst)].shape[0]):
        toret.append(a[(nconst+1,fconst)][i,0])
    mylist = [i for i in range(len(charlist))]
    choice = random.choices(mylist, weights = toret, k=1)
    seed += numtochar[choice[0]]
print("400000 Datapoints Output:")
print(seed)