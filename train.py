import numpy as np
import random
import sys
import pickle

def generate_time_series(n_steps):
    freq1, freq2, offsets1, offsets2 = np.random.rand(4)
    time = np.linspace(0, 1, n_steps)
    series = 0.5 * np.sin((time - offsets1) * (freq1 * 10 + 10))
    series += 0.2 * np.sin((time - offsets2) * (freq2 * 20 + 20))
    series += 0.1 * (np.random.rand(n_steps) - 0.5)
    return series

def hyptan(k):
        return np.tanh(k)

def hyptand(k):
    return 1/((np.cosh(k))**2)

def softmax(m):
    temp = np.exp(m)
    return temp / np.sum(temp)

def cce(yactual, final_layer_output):
    toret = 0
    for ksimax in range(yactual.shape[0]):
        toret += yactual[(ksimax,0)]*np.log(final_layer_output[(ksimax,0)])
    return -1*toret

def rnn_randgen(dimensions):
        wl = [None]
        ws = [None]
        b = [None]
        for i,n in enumerate(dimensions[1:]):
            temp = (dimensions[i] + 2*dimensions[i+1])/2
            r = (3/temp)**0.5
            wl.append(2*r*np.random.rand(n, dimensions[i])-r)
            ws.append(2*r*np.random.rand(n, n)-r)
            b.append(2*r*np.random.rand(n, 1)-r)
        return wl,ws,b

def dnn_randgen(dimensions):
    w = [None]
    b = [None]
    for i,n in enumerate(dimensions[1:]):
        temp = (dimensions[i] + 2*dimensions[i+1])/2
        r = (3/temp)**0.5
        w.append(2*r*np.random.rand(n, dimensions[i])-r)
        b.append(2*r*np.random.rand(n, 1)-r)
    return w,b



def train(e,lamb,wl,ws,b,tset, dim, dnnw, dnnb):
    nconst = len(wl)-1
    fconst = len(tset[0])-1
    for ks in range(1,e+1):
        for indexy, x in enumerate(tset):
            a = dict()
            dots = dict()
            for ml in range(1, nconst+1):
                a[(ml,0)] = np.zeros((dim[ml],1))
            for s in range(1, fconst+1):
                a[(0,s)] = onehot(x[s-1])
                for l in range(1,len(wl)):
                    t = a[(l-1,s)]
                    z = a[(l,s-1)]
                    dots[(l,s)] = wl[l]@t + ws[l]@z + b[l]
                    a[(l,s)] = hyptan(dots[(l,s)])
            a[(nconst+1,fconst)] = dnn_p_net(softmax, dnnw, dnnb, a[(nconst,fconst)])
            dell = dict() 
            dell[(nconst+1,fconst)] = (np.eye(onehot(x[-1]).shape[0]) + (-1 * a[(nconst+1,fconst)])) @ onehot(x[-1])               
            dell[(nconst,fconst)] = hyptand(dots[(nconst,fconst)])*(dnnw[0].transpose()@dell[(nconst+1,fconst)])
            for s in range(fconst-1,0,-1):
                dell[(nconst,s)] = hyptand(dots[(nconst,s)])*(ws[nconst].transpose()@dell[(nconst,s+1)])
            for l in range(nconst-1,0,-1):
                dell[(l,fconst)] = hyptand(dots[(l,fconst)])*(wl[l+1].transpose()@dell[(l+1,fconst)])
            for s in range(fconst-1,0,-1):
                for l in range(nconst-1,0,-1):
                    dell[(l,s)] = hyptand(dots[(l,s)])*(wl[l+1].transpose()@dell[(l+1,s)]) + hyptand(dots[(l,s)])*(ws[l].transpose()@dell[(l,s+1)])
            for l in range(1,len(wl)):
                for s in range(1,len(x)-1):
                    if s > 1:
                        ws[l] += lamb*(dell[(l,s)]@a[(l,s-1)].transpose())
                    b[l] += lamb*dell[(l,s)]
                    wl[l] += lamb*(dell[(l,s)]@a[(l-1,s)].transpose())
            dnnb[0] = dnnb[0] + lamb*dell[(nconst+1,fconst)]
            dnnw[0] = dnnw[0] + lamb*dell[(nconst+1,fconst)]@(a[(nconst,fconst)].transpose())
                        

            if indexy%10000 == 0 and indexy!= 0:
                print(indexy, "datapoints")
                err = 0
                for m in testset:
                    a = dict()
                    dots = dict()
                    for ml in range(1, nconst+1):
                        a[(ml,0)] = np.zeros((dim[ml],1))
                    for s in range(1, fconst+1):
                        a[(0,s)] = onehot(m[s-1])
                        for l in range(1,len(wl)):
                            t = a[(l-1,s)]
                            z = a[(l,s-1)]
                            dots[(l,s)] = wl[l]@t + ws[l]@z + b[l]
                            a[(l,s)] = hyptan(dots[(l,s)])
                    a[(nconst+1,fconst)] = dnn_p_net(softmax, dnnw, dnnb, a[(nconst,fconst)])
                    err += cce(onehot(m[-1]),a[(nconst+1,fconst)])
                print("Error:", err/len(testset))

                with open("new"+ str(indexy)+"dp.pkl", "wb") as f:
                    pickle.dump([ws,wl,b,dnnw,dnnb], f)
                

def rnn_p_net(b,wl,ws,x,dim):
    nconst = len(wl)-1
    fconst = len(x)
    a = dict()
    dots = dict()
    for ml in range(1, nconst+1):
        a[(ml,0)] = np.zeros((dim[ml],1))
    for s in range(1, fconst+1):
        a[(0,s)] = x[s-1]
        for l in range(1,len(wl)):
            t = a[(l-1,s)]
            z = a[(l,s-1)]
            dotemp = wl[l]@t + ws[l]@z + b[l]
            dots[(l,s)] = dotemp
            a[(l,s)] = hyptan(dots[(l,s)])
    return a[(nconst,fconst)]

def dnn_p_net(A_vec, w, b, inp):
    #a is a Python list of numpy matrices that will hold the output at each layer
    a = []
    a.append(inp)
    #n = the number of perceptron layers in this network
    n = len(w)
    for i in range(0,n):
        a.append(A_vec(w[i] @ a[i] + b[i]))
    return a[-1]

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

sublist = []
for i in range(len(textstring)-100):
    sublist.append(textstring[i:i+101])
random.shuffle(sublist)
trainset = sublist[:int(len(sublist)*.85)]
testset = sublist[int(len(sublist)*.85):]
with open("train_set.txt", "w") as f:
    f.write(str(trainset))
with open("test_set.txt", "w") as f:
    f.write(str(testset))
o = rnn_randgen([len(charlist),40,40])
temp = (40 + 2*len(charlist))/2
r = (3/temp)**0.5
p0 = [2*r*np.random.rand(len(charlist), 40)-r]
p1 = [2*r*np.random.rand(len(charlist), 1)-r]
train(20,0.005, o[0], o[1], o[2], trainset, [len(charlist),40,40], p0, p1)

  