import numpy as np
from sklearn.metrics import precision_score,recall_score
import matplotlib.pyplot as plt


ORGAOS = ['artery', 'liver', 'stomach', 'vein']
colors = {'artery':np.array([255,0,0]),
          'liver':np.array([255,0,255]),
          'stomach':np.array([0,255,0]),
          'vein':np.array([0,0,255])}

def evaluate(true,pred):
    res = {}
    for org in ORGAOS:
        p = pred[org].ravel()
        t = true[org].ravel() if org in true else np.zeros(p.shape,'uint8')
        res[org] = (precision_score(t,p,zero_division = 1),recall_score(t,p,zero_division = 1))
    return res

def img_with_labels(img,structures,alpha = .3):
    shape = (img.shape[0],img.shape[1],3)
    final_img = np.zeros(shape,'float32')
    for i in range(3):
        final_img[:,:,i] = img
    for key in structures:
        filt1 = (structures[key] == 0).astype('float32') + structures[key].astype('float32') * alpha
        filt2 = (structures[key] == 0).astype('float32') + structures[key].astype('float32') * (1 - alpha)
        for i in range(3):
            struc_img = colors[key][i] * structures[key]
            final_img[:,:,i] = final_img[:,:,i] * filt2 + struc_img * filt1
    return final_img.astype('uint8')

def plot_pred(img,real,pred,alpha = .3):
    img_real = img_with_labels(img,real,alpha)
    img_pred = img_with_labels(img,pred,alpha)
    res = evaluate(real,pred)
    for key in res:
        res[key] = round(res[key][0],2),round(res[key][1],2)
    rows,index = list(res.values()),list(res.keys())
    colnames = ['Precision','Recall']
    fig,ax = plt.subplots(ncols = 3,nrows = 1,figsize = (15,5))
    ax[0].imshow(img_real)
    ax[0].axis('off')
    ax[0].set_title('Real')
    ax[1].imshow(img_pred)
    ax[1].axis('off')
    ax[1].set_title('Prediction')
    
    table = ax[2].table(cellText=rows,rowLabels=index,colLabels=colnames,loc = 'center',rowColours = [cor/ 255.0 for cor in colors.values()])
    table.scale(.6,3)
    ax[2].axis('off')
    plt.show()

def plot_labels(img,structures,alpha = .3):
    final_img = img_with_labels(img,structures,alpha)
    plt.imshow(final_img)
    plt.xticks([])
    plt.yticks([])
    plt.show()

def plot_all(img,structures,alpha = .3):
    final_img = img_with_labels(img,structures,alpha)
    fig,ax = plt.subplots(ncols = 2,nrows = 1,figsize = (10,5))
    ax[0].imshow(img,'gray')
    ax[0].axis('off')
    ax[1].imshow(final_img)
    ax[1].axis('off')
    plt.show()

def plot(img):
    plt.imshow(img,'gray')
    plt.xticks([])
    plt.yticks([])
    plt.show()


