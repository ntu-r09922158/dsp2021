import numpy as np # 1.19.5
import os
from sklearn.datasets import fetch_openml # 1.0.1
import matplotlib.pyplot as plt # 3.2.2

def Q2():

    lbl5 = np.where(mnist['target']=='5')
    ds5 = mnist['data'][lbl5]

    # compute center of '5'
    ds5_mean = np.mean(ds5,0)
    centered_ds5 = np.subtract(ds5, ds5_mean)

    # process PCA
    l,v = PCA(centered_ds5)

    # add back the mean
    img = np.add(v, ds5_mean)

    # show subplot
    fig = plt.figure(figsize=(15,3))
    fig.patch.set_facecolor('white')

    for i in range(3):
        plt.subplot(131+i)
        plt.imshow(img[:,i].astype('float64').reshape(28,28),'gray')
        plt.title('\u03BB='+str(l[i]))
        plt.axis('off')

    fig.savefig(mypath + "/Q2.jpg")

def Q3():
    
    lbl5 = np.where(mnist['target']=='5')
    ds5 = mnist['data'][lbl5]
    topList = np.array([3,10,30,100])

    # compute center of '1,3,6'
    ds5_mean = np.mean(ds5,0)
    centered_ds5 = np.subtract(ds5, ds5_mean)

    # get Top-k eigenvectors
    l,v = PCA(centered_ds5)

    # show subplot
    fig = plt.figure(figsize=(15,3))
    fig.patch.set_facecolor('white')

    # plot Original first '5'
    plt.subplot(151)
    plt.imshow(ds5[0].reshape(28,28),'gray')
    plt.title("Original \'5\'")
    plt.axis('off')

    # plot first '5' by [3,10,30,100] bases
    for i in range(len(topList)):
    
        plt.subplot(152+i)
        
        # shifting x_i by minus mean
        img = np.subtract(ds5[0], ds5_mean)
        
        # compute Coef coresponding to limited-k bases
        coef = np.dot(v[:,0:topList[i]].T, img)
        
        # compute Reconstruct x_i
        img = np.dot(v[:,0:topList[i]], coef)
        
        # adding mean to x_i for visualization
        img = np.add(img, ds5_mean)

        plt.imshow(img.astype('float64').reshape(28,28),'gray')
        plt.title("\'5\' with "+str(topList[i])+" bases")
        plt.axis('off')

    fig.savefig(mypath + "/Q3.jpg")

def PCA(ds):
  
    # compute Scatter matrix
    S = np.dot(ds.T, ds)

    # get unsorted eigenvalues & eigenvectors
    l, v = np.linalg.eig(S)

    # sort eigenvalues & eigenvectors in descending order
    idx = l.argsort()[::-1]
    l_ = l[idx]
    v_ = v[:,idx]

    # return sorted eigenvalues (m,1) & eigenvectors[p1, p2, ..., pm] of size (n,m)
    return l_, v_

def Q4():
    
    # extract dataset of '1,3'6'
    choice = np.array([1,3,6])
    lbl = np.where(mnist['target'][:10000]==str(choice[0]))
    ds136 = mnist['data'][lbl]
    lbl136 = mnist['target'][lbl]

    for i in range(1,len(choice)):
        lbl = np.where(mnist['target'][:10000]==str(choice[i])) 
        ds136 = np.append(ds136, mnist['data'][lbl], axis=0)
        lbl136 = np.append(lbl136, mnist['target'][lbl], axis=0)


    # compute center of '1,3,6'
    ds136_mean = np.mean(ds136,0)
    centered_ds136 = np.subtract(ds136, ds136_mean)

    l, v = PCA(centered_ds136)

    # compute Coef coresponding to limited-k bases
    coef = np.dot(v[:,0:2].T, centered_ds136.T)

    fig = plt.figure()

    # show Scatter Plot , 1127, 2159, 3173
    plt.scatter(coef[0][:1127], coef[1][:1127], c="blue", label="1")
    plt.scatter(coef[0][1127:2159], coef[1][1127:2159], c="red", label="3")
    plt.scatter(coef[0][2159:], coef[1][2159:], c="green", label="6")
    plt.title("Scatter Plot of 3 different images")
    plt.legend()
    plt.show()

    fig.savefig(mypath + "/Q4.jpg")

def OMP(x, dictionary, sparsity): # dict(784,10k)
  
  r = x
  m = dictionary.shape[1]
  selected =[]
  Coef = np.zeros((m))
  temp = dictionary.copy()


  for l in range(sparsity):
    
    # select l-th basis (10k x 784) x (784 x 1) = (10k x 1)
    innerProduct = np.fabs(np.dot(temp.T, r)) # )
    max = np.argmax(innerProduct)
    selected.append(max)
    temp[:,max] = 0
    
    # compute Coefficient by pseudo-inverse Bt
    B = dictionary[:,selected[:]]
    Bt = np.linalg.pinv(B)
    coef = np.dot(Bt, x)
    r = r - np.dot(B, coef)

  Coef[selected[:]] = coef

  return selected, Coef

def Q5():
    ds_10k = mnist['data'][:10000,:] # (10k,784)
    ds_10k = ds_10k.T # (784,10k)
    norm = np.linalg.norm(ds_10k, axis=0, keepdims=True)
    normed_ds_10k = np.divide(ds_10k, norm) # (784,10k)
    #print(np.square(normed_ds_10k).sum(axis=0)) # [1. 1. 1. ... 1. 1. 1.]

    sparsity = 5
    m = normed_ds_10k.shape[1]
    x = mnist['data'][m+0] # '3'

    idx, coef = OMP(x, normed_ds_10k, sparsity)

    fig = plt.figure(figsize=(15,3))
    fig.patch.set_facecolor('white')

    for i in range(len(idx)):
        plt.subplot(151+i)
        plt.imshow(mnist['data'][idx[i]].reshape(28,28),'gray') # normed_ds_10k.T
        plt.title(idx[i])
        plt.axis('off')

    fig.savefig(mypath + "/Q5.jpg")

def l2norm(x, y):

  # x=(1,784), y=(784,1)
  diff = x.T - y
  diag = np.diag(diff)
  dist = np.sqrt(np.square(np.fabs(diag)).sum())
  return dist

def Q6():
    ds_10k = mnist['data'][:10000,:]
    ds_10k = ds_10k.T
    norm = np.linalg.norm(ds_10k, axis=0, keepdims=True)
    normed_ds_10k = np.divide(ds_10k, norm) # (784,10k)
    #print(np.square(normed_ds_10k).sum(axis=0)) # [1. 1. 1. ... 1. 1. 1.]

    sparsity = np.array([5,10,40,200])
    m = normed_ds_10k.shape[1] # 10k
    x = mnist['data'][m+1] # '8'

    idx, Coef = OMP(x, normed_ds_10k, sparsity.max())

    fig = plt.figure(figsize=(15,3))
    fig.patch.set_facecolor('white')

    plt.subplot(151)
    plt.imshow(x.reshape(28,28),'gray')
    dist = l2norm(x,x)
    plt.title("Original \'8\'"+", L2="+"{:.2f}".format(dist))
    plt.axis('off')

    for s in range(len(sparsity)):
    
        B = normed_ds_10k[:,idx[:sparsity[s]]] # (784,l) print(B.shape)
        Bt = np.linalg.pinv(B)
        coef = np.dot(Bt, x)
        reconstruct = np.dot(B, coef)

        dist = l2norm(x,reconstruct)
        
        plt.subplot(152+s)
        plt.imshow(reconstruct.reshape(28,28),'gray')
        plt.title("k="+str(np.count_nonzero(coef))+", L2="+"{:.2f}".format(dist))
        plt.axis('off')
    
    fig.savefig(mypath + "/Q6.jpg")

if __name__ == "__main__":

    mnist = fetch_openml('mnist_784',as_frame=False)
    path = os.path.abspath(__file__)
    file = os.path.dirname(path)
    mypath = file + "/fig"
    
    try:
        os.makedirs(mypath, exist_ok=True)
    except OSError as exc:
        print(exc.errno)
    
    Q2()
    Q3()
    Q4()
    Q5()
    Q6()