import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.measure import label,regionprops
import os
from numba import jit
from sklearn.metrics import precision_score,recall_score, confusion_matrix
from scipy.spatial.distance import mahalanobis

colors = {'artery':np.array([255,0,0]),
          'liver':np.array([255,0,255]),
          'stomach':np.array([0,255,0]),
          'vein':np.array([0,0,255])}
ORGAOS = ['artery', 'liver', 'stomach', 'vein']

def load(name):
    data = np.load(name,allow_pickle=True).flatten()[0]
    img = data['image']
    structures = data['structures']
    if len(img.shape) == 3:
        img = img[:,:,0]
    return img,structures

def detect_main_area(img):
    mask = (img > 0).astype('uint8')
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(25,25))
    mask = cv2.erode(mask,kernel,iterations = 1)
    mask = cv2.dilate(mask,kernel,iterations = 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(95,95))
    mask = cv2.dilate(mask,kernel,iterations = 1)
    mask = cv2.erode(mask,kernel,iterations = 1)
    objetos = label(mask)
    info = regionprops(objetos)
    main_label = max(info,key = lambda x: x['area'])['label']
    return objetos == main_label

def clean(img):
    return (img * (detect_main_area(img))).astype('uint8')

def create_reshape(img_clean):
    img = img_clean
    arx = np.arange(img.shape[1])
    ary = np.arange(img.shape[0])
    min_y = min(ary,key = lambda x: np.inf if (img[x] == 0).all() else x)
    max_y = max(ary,key = lambda x: 0 if (img[x] == 0).all() else x)
    min_x = min(arx,key = lambda x: np.inf if (img[:,x] == 0).all() else x)
    max_x = max(arx,key = lambda x: 0 if (img[:,x] == 0).all() else x)
    min_x_top = min(arx,key = lambda x: np.inf if img[min_y + 10,x] == 0 else x)
    max_x_top = max(arx,key = lambda x: 0 if img[min_y + 10,x] == 0 else x)
    dis_left = min_x_top - min_x
    dis_right = max_x - max_x_top
    if dis_left > dis_right:
        max_x = min(max_x_top + dis_left,img.shape[1] - 1)
    if dis_left < dis_right:
        min_x = max(min_x_top - dis_right,0)
    max_x = max_x - min_x
    max_y = max_y - min_y
    return (min_y,max_y,min_x,max_x)

@jit(nopython = True)
def transform(imagem,scaler):
    newshape = 192,256
    newimagem = np.zeros(newshape,'uint8')
    for y in range(newshape[0]):
        ny = int((y / newshape[0]) * scaler[1] + scaler[0])
        for x in range(newshape[1]):
            nx = int((x / newshape[1]) * scaler[3] + scaler[2])
            newimagem[y,x] = imagem[ny,nx]
    return newimagem

@jit(nopython = True)
def reverse_transform(imagem,scaler):
    newshape = 768,1024
    newimagem = np.zeros(newshape,'uint8')
    indexesX = np.zeros(newshape[1],'int32')
    indexesY = np.zeros(newshape[0],'int32')
    for y in range(newshape[0]):
        ny = int(imagem.shape[0] * (y - scaler[0]) / scaler[1])
        indexesY[y] = ny
    for x in range(newshape[1]):
        nx = int(imagem.shape[1] * (x - scaler[2]) / scaler[3])
        indexesX[x] = nx
    indexesX[indexesX < 0] = 0
    indexesX[indexesX >= imagem.shape[1]] = imagem.shape[1] - 1
    indexesY[indexesY < 0] = 0
    indexesY[indexesY >= imagem.shape[0]] = imagem.shape[0] - 1
    for y in range(newshape[0]):
        ny = indexesY[y]
        for x in range(newshape[1]):
            nx = indexesX[x]
            newimagem[y,x] = imagem[ny,nx]
    return newimagem

@jit(nopython = True)
def euclidian(p1,p2):
    return np.abs(p1 - p2)

@jit(nopython = True)
def create_mask(img,trashold_adj,trashold_start,pixel,initial_color):
    img = img.astype('float32')
    ylim,xlim = img.shape[0],img.shape[1]
    imgclass = np.zeros(img.shape[:2],'uint8')
    q0 = list()
    q0.append(pixel)
    imgclass[pixel[1],pixel[0]] = 1
    while q0:
        p = q0.pop()
        x,y = p
        color = img[y,x]
        for X,Y in zip([x,x,x + 1,x - 1],[y + 1,y - 1,y,y]):
            if (not (-1 < X < xlim)) or (not (-1 < Y < ylim)):
                continue
            if imgclass[Y,X]:
                continue
            if euclidian(initial_color,img[Y,X]) > trashold_start:
                continue
            if euclidian(color,img[Y,X]) <= trashold_adj:
                imgclass[Y,X] = 1
                q0.append((X,Y))
    return imgclass * 255

def remove_black(img):
    inicial_points = (0,0),(0,img.shape[0] - 1),(img.shape[1] - 1,0),(img.shape[1] - 1,img.shape[0] - 1)
    for point in inicial_points:
        mask = create_mask(img,5,20,point,0)
        img = img + mask // 3
    return img

def transform_struc(structures,scaler):
    resh = {}
    for key in structures:
        resh[key] = transform(structures[key],scaler)
    return resh

def reduce(img):
    n = 30
    return img[n:-n,n:-n].copy()

def reduce_struc(structures):
    red = {}
    for key in structures:
        red[key] = reduce(structures[key])
    return red

def evaluate(true,pred):
    res = {}
    for org in ORGAOS:
        p = pred[org].ravel()
        t = true[org].ravel() if org in true else np.zeros(p.shape,'uint8')
        res[org] = (precision_score(t,p,zero_division = 1),recall_score(t,p,zero_division = 1))
    return res

def conf_matrix(true,pred):
    conf_matrices = {}
    for org in ORGAOS:
        p = pred[org].ravel()
        t = true[org].ravel() if org in true else np.zeros(p.shape, 'uint8')
        conf_matrices[org] = confusion_matrix(t, p)
    return conf_matrices

_ = transform(np.zeros((10,10)),(1,1,1,1))
_ = reverse_transform(np.zeros((10,10)),(1,1,1,1))
_ = create_mask(np.zeros((5,5)),0,0,(0,0),0)

def cos_sim(img1,img2):
    img1 = img1.astype(np.float32)
    img2 = img2.astype(np.float32)
    
    
    vec1 = img1.flatten()
    vec2 = img2.flatten()
    
    
    dot_product = np.dot(vec1, vec2)
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)
    
    cosine_sim = dot_product / (norm_vec1 * norm_vec2)
    
    return cosine_sim

def find_IOU(y_true, y_pred):

    intersection = np.logical_and(y_true, y_pred).sum()
    union = np.logical_or(y_true, y_pred).sum()

    iou = intersection / union if union != 0 else 0

    return iou



class Model:
    def __init__(self,path,trasholds = {'artery': 5, 'liver': 1, 'stomach': 11, 'vein': 3}):
        self.templates_names = os.listdir(os.path.join(path,'classes')) 
        self.classes = [cv2.imread(os.path.join(path,'classes',name),cv2.IMREAD_GRAYSCALE) for name in sorted(self.templates_names)]
        self.templates = {org:[cv2.imread(os.path.join(path,'templates',org,name),
                                          cv2.IMREAD_GRAYSCALE) for name in self.templates_names] for org in ORGAOS}
        self.trasholds = trasholds
        self.best_class = None
    def predict(self,img):
        pred_structures = self.predict_score(img)
        for org in ORGAOS:
            trash = self.trasholds[org]
            cuted = pred_structures[org]
            cuted[cuted < trash] = 0
            cuted[cuted >= trash] = 1
            pred_structures[org] = cuted
        return pred_structures
    def predict_score(self,img):
        cleaned = clean(img)
        scaler = create_reshape(cleaned)
        timg = transform(cleaned,scaler)
        simg = reduce(timg)
        rimg = remove_black(simg)
        best_score = 0
        for i,classe in enumerate(self.classes):
            subtraction = rimg.astype('int32') - classe.astype('int32')
            subtraction[subtraction < 0] = 0
            score = subtraction.sum() / classe.sum()
            if score > best_score:
                best_score = score
                self.best_class = i
        pred_structures = {}
        for org in ORGAOS:
            template = self.templates[org][self.best_class]
            canvas = np.zeros((192,256),'uint8')
            n = 30
            canvas[n:-n,n:-n] = template
            org_img = reverse_transform(canvas,scaler).astype('int32')
            img32 = img.astype('int32')
            if org in ['artery', 'stomach', 'vein']:
                cuted = org_img - img32
            else:
                cuted = org_img - (245 - img32)
            cuted[cuted < 0] = 0
            cuted[cuted > 255] = 255
            pred_structures[org] = cuted.astype('uint8')
        return pred_structures
    def get_best_class(self,img):
        cleaned = clean(img)
        scaler = create_reshape(cleaned)
        timg = transform(cleaned,scaler)
        simg = reduce(timg)
        rimg = remove_black(simg)
        best_score = 0
        for i,classe in enumerate(self.classes):
            subtraction = rimg.astype('int32') - classe.astype('int32')
            subtraction[subtraction < 0] = 0
            score = subtraction.sum() / classe.sum()
           
            if score > best_score:
                best_score = score
                self.best_class = i
        return self.best_class 
class Model_cos:
    def __init__(self,path,trasholds = {'artery': 5, 'liver': 1, 'stomach': 11, 'vein': 3}):
        self.templates_names = os.listdir(os.path.join(path,'classes')) 
        self.classes = [cv2.imread(os.path.join(path,'classes',name),cv2.IMREAD_GRAYSCALE) for name in sorted(self.templates_names)]
        self.templates = {org:[cv2.imread(os.path.join(path,'templates',org,name),
                                          cv2.IMREAD_GRAYSCALE) for name in self.templates_names] for org in ORGAOS}
        self.trasholds = trasholds
        self.best_class = None
    def predict(self,img):
        pred_structures = self.predict_score(img)
        for org in ORGAOS:
            trash = self.trasholds[org]
            cuted = pred_structures[org]
            cuted[cuted < trash] = 0
            cuted[cuted >= trash] = 1
            pred_structures[org] = cuted
        return pred_structures
    def predict_score(self,img):
        cleaned = clean(img)
        scaler = create_reshape(cleaned)
        timg = transform(cleaned,scaler)
        simg = reduce(timg)
        rimg = remove_black(simg)
        best_score = 0
        for i,classe in enumerate(self.classes):
            # subtraction = rimg.astype('int32') - classe.astype('int32')
            # subtraction[subtraction < 0] = 0
            # score = subtraction.sum() / classe.sum()
            score = cos_sim(rimg,classe)
            if score > best_score:
                best_score = score
                self.best_class = i
        pred_structures = {}
        for org in ORGAOS:
            template = self.templates[org][self.best_class]
            canvas = np.zeros((192,256),'uint8')
            n = 30
            canvas[n:-n,n:-n] = template
            org_img = reverse_transform(canvas,scaler).astype('int32')
            img32 = img.astype('int32')
            if org in ['artery', 'stomach', 'vein']:
                cuted = org_img - img32
            else:
                cuted = org_img - (245 - img32)
            cuted[cuted < 0] = 0
            cuted[cuted > 255] = 255
            pred_structures[org] = cuted.astype('uint8')
        return pred_structures
    def get_best_class(self,img):
        cleaned = clean(img)
        scaler = create_reshape(cleaned)
        timg = transform(cleaned,scaler)
        simg = reduce(timg)
        rimg = remove_black(simg)
        best_score = 0
        for i,classe in enumerate(self.classes):
            # subtraction = rimg.astype('int32') - classe.astype('int32')
            # subtraction[subtraction < 0] = 0
            # score = subtraction.sum() / classe.sum()
            score = cos_sim(rimg,classe)
            if score > best_score:
                best_score = score
                self.best_class = i
        return self.best_class 
    
class Model_2:
    def __init__(self,path,trasholds = {'artery': 5, 'liver': 1, 'stomach': 11, 'vein': 3}):
        self.templates_names = os.listdir(os.path.join(path,'classes')) 
        self.classes = [cv2.imread(os.path.join(path,'classes',name),cv2.IMREAD_GRAYSCALE) for name in sorted(self.templates_names)]
        self.templates = {org:[cv2.imread(os.path.join(path,'templates',org,name),
                                          cv2.IMREAD_GRAYSCALE) for name in self.templates_names] for org in ORGAOS}
        self.trasholds = trasholds
        self.best_class = None
    def predict(self,img):
        pred_structures = self.predict_score(img)
        for org in ORGAOS:
            trash = self.trasholds[org]
            cuted = pred_structures[org]
            cuted[cuted < trash] = 0
            cuted[cuted >= trash] = 1
            pred_structures[org] = cuted
        return pred_structures
    def predict_score(self,img):
        cleaned = clean(img)
        scaler = create_reshape(cleaned)
        timg = transform(cleaned,scaler)
        simg = reduce(timg)
        rimg = remove_black(simg)
        best_score = 0
        for i,classe in enumerate(self.classes):
            subtraction = rimg.astype('int32') - classe.astype('int32')
            subtraction[subtraction < 0] = 0
            score = subtraction.sum() / rimg.sum() 
            if score > best_score: 
                best_score = score
                self.best_class = i
        pred_structures = {}
        for org in ORGAOS:
            template = self.templates[org][self.best_class]
            canvas = np.zeros((192,256),'uint8')
            n = 30
            canvas[n:-n,n:-n] = template
            org_img = reverse_transform(canvas,scaler).astype('int32')
            img32 = img.astype('int32')
            if org in ['artery', 'stomach', 'vein']:
                cuted = org_img - img32
            else:
                cuted = org_img - (245 - img32)
            cuted[cuted < 0] = 0
            cuted[cuted > 255] = 255
            pred_structures[org] = cuted.astype('uint8')
        return pred_structures
    def get_best_class(self,img):
        cleaned = clean(img)
        scaler = create_reshape(cleaned)
        timg = transform(cleaned,scaler)
        simg = reduce(timg)
        rimg = remove_black(simg)
        best_score = 0
        for i,classe in enumerate(self.classes):
            subtraction = rimg.astype('int32') - classe.astype('int32')
            subtraction[subtraction < 0] = 0
            score = subtraction.sum() / rimg.sum()   
           
            if score > best_score:
                best_score = score
                self.best_class = i
        return self.best_class 