import os
import time
import cv2
import pickle
import struct 
import re 
import random
import math
import numpy as np
import matplotlib.pyplot as plt
import keras.backend as K
import tensorflow as tf

from collections import namedtuple
from PIL import Image
from random import shuffle
from keras.preprocessing import image



DEPTH_H, DEPTH_W = 26, 26
BATCH_SIZE = 12 #12

kitti_image_folder = '/home/mjchiu/Documents/darknet-depth/dataset/KITTI/image/'
kitti_depth_folder = '/home/mjchiu/Documents/darknet-depth/dataset/KITTI/depth_interpol/'
airsim_image_folder = '/home/mjchiu/Documents/darknet-depth/dataset/AirSim/Train/'
airsim_depth_folder = '/home/mjchiu/Documents/darknet-depth/dataset/AirSim/Train/'
city_image_folder = '/home/mjchiu/Documents/darknet-depth/dataset/AirSim/CityEnviron/Train/'
city_depth_folder = '/home/mjchiu/Documents/darknet-depth/dataset/AirSim/CityEnviron/Train/'
escenter_image_folder = '/home/mjchiu/Documents/darknet-depth/dataset/NCTU/ES_building/Train/'
escenter_depth_folder = '/home/mjchiu/Documents/darknet-depth/dataset/NCTU/ES_building/Train/'
NYUv1_mat_data_path = '/home/mjchiu/Documents/darknet-depth/dataset/NYU_depth/nyu_depth_data_labeled.mat'
NYUv2_mat_data_path = '/home/mjchiu/Documents/darknet-depth/dataset/NYU_depth/nyu_depth_v2_labeled.mat'
NYUv2_raw_data_path = '/home/mjchiu/Documents/darknet-depth/dataset/NYU_depth/NYUv2_raw_data/'


# 以 list 的方式傳入要顯示的 images，用matplot顯示在螢幕上
# Inputs
#  -(list) imgs: 要顯示的影像們
#  -(tuple) figsize: 整個顯示畫面的大小
#  -(bool) axis_off: 是否要顯示坐標軸。欲設為False
# Output: 顯示畫面
def showResult(imgs, figsize=(20,15), axis_off=True):
    from matplotlib import pyplot as plt
    total_imgs = len(imgs)
    plt.figure(figsize=figsize)
    for i in range(total_imgs):
        plt.subplot(total_imgs,1,i+1)
        plt.imshow(np.uint8(imgs[i]))
        if axis_off:
            plt.axis('off')
        
    plt.tight_layout(pad=0.2, w_pad=0.2, h_pad=0.2)
    plt.show()

# 調整影像的GAMMA值
def adjust_gamma(image, gamma=1.0):
    # build a lookup table mapping the pixel values [0, 255] to
    # their adjusted gamma values
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
        for i in np.arange(0, 256)]).astype("uint8")

    # apply gamma correction using the lookup table
    return cv2.LUT(image, table)

# 計算影像的X方向梯度
def gradient_x(img):
    gx = img[:,:,:-1,:] - img[:,:,1:,:]
    return gx

# 計算影像的y方向梯度
def gradient_y(img):
    gy = img[:,:-1,:,:] - img[:,1:,:,:]
    return gy

def get_disparity_smoothness(depth, image):
    disp_gradients_x = gradient_x(depth)
    disp_gradients_y = gradient_y(depth)

    image_gradients_x = gradient_x(image)
    image_gradients_y = gradient_y(image)

    weights_x = tf.exp(-tf.reduce_mean(tf.abs(image_gradients_x), 3, keep_dims=True))
    weights_y = tf.exp(-tf.reduce_mean(tf.abs(image_gradients_y), 3, keep_dims=True))

    smoothness_x = disp_gradients_x * weights_x
    smoothness_y = disp_gradients_y * weights_y
    return smoothness_x + smoothness_y

# 計算影像的梯度 (包含X方向和Y方向)
def depth_gradient(pred,gt):
    
    pred_grad_dx = gradient_x(pred)
    pred_grad_dy = gradient_y(pred)
    gt_grad_dx = gradient_x(gt)
    gt_grad_dy = gradient_y(gt)

    loss_dx = tf.reduce_mean(tf.abs(pred_grad_dx - gt_grad_dx)) 
    loss_dy = tf.reduce_mean(tf.abs(pred_grad_dy - gt_grad_dy)) 

    return (loss_dx + loss_dy)

# 利用 BerHu 計算 predict depth map 和 ground truth depth map 之間的 loss
def berhu_loss(predict, target):
    d = tf.subtract(predict, target)
    L1_error = tf.abs(d, name='abs_error')
    thr = (0.2*tf.reduce_max(L1_error))+0.000001
    L2_error = (tf.square(d) + tf.square(thr))/(2*thr)
    depth_loss  = tf.where(L1_error<=thr, L1_error, L2_error)
    depth_loss  = tf.reduce_sum(depth_loss ) / (tf.count_nonzero(target, dtype=tf.float32)+0.000001)
    return depth_loss

# 訓練用的loss function
def custom_depth_loss(depth_weight = 0.5, disparity_weight = 0.5):
    def loss(y_true, y_pred):
        
        # depth loss
        y_true_clip = tf.clip_by_value(y_true, -1, 0)
        ones = tf.ones(tf.shape(y_true_clip),tf.float32)
        mask = tf.add(y_true_clip,ones)
        mask_flat = tf.squeeze(mask)

        predict_flat = tf.squeeze(y_pred)
        depth_flat = tf.squeeze(y_true)
        depth_flat = tf.clip_by_value( depth_flat, 0.01, 100)
        predict = tf.multiply(predict_flat, mask_flat)
        target = tf.multiply(depth_flat, mask_flat)

        depth_loss = berhu_loss(predict, target)
        

        # disparity loss
        # predict_flat_1 = tf.squeeze(y_pred)
        # predict_flat_1 = tf.clip_by_value(predict_flat_1, 0.01, 100)
        # depth_flat_1 = tf.squeeze(y_true)
        # depth_flat_1 = tf.clip_by_value(depth_flat_1, 0.1, 100)

        # disparity_predict = tf.div(1.0,predict_flat_1)*200
        # disparity_GT = tf.div(1.0,depth_flat_1)*200

        # predict_1 = tf.multiply(disparity_predict, mask_flat)
        # target_1 = tf.multiply(disparity_GT, mask_flat)
        # disparity_loss = berhu_loss(predict_1, target_1)

        
        # Inverse depth loss
        predict_flat = tf.squeeze(y_pred)
        predict_flat = tf.clip_by_value(predict_flat, 0.01, 100)
        depth_flat = tf.squeeze(y_true)
        depth_flat = tf.clip_by_value(depth_flat, 0.1, 100)
        inverse_predict = 500-predict_flat
        inverse_GT = 500-depth_flat
        predict = tf.multiply(inverse_predict, mask_flat)
        target = tf.multiply(inverse_GT, mask_flat)
        inverse_loss = berhu_loss(predict, target)


        # DEBUG MSG
        depth_loss = K.print_tensor(depth_loss, message='depth = ')
        inverse_loss = K.print_tensor(inverse_loss, message='                   dis = ')

        total_loss = depth_weight*depth_loss + disparity_weight*inverse_loss
        return total_loss

    return loss

# --------------------------------------------------------------------------------
# READ DATA PATH------------------------------------------------------------------
# --------------------------------------------------------------------------------

# 讀取 AirSim Dataset的資料路徑，建成list
def getAirsimPath(path): 
    dataList = []
    for dirname, dirnames, filenames in os.walk(path):
        for filename in filenames:
            imgPath = os.path.join(dirname, filename)
            if 'depthPlanner' in imgPath:
                dataList.append(imgPath)

    print("Total images: "+str(len(dataList)))
    shuffle(dataList)
    return dataList

# 讀取 KITTI Dataset的資料路徑，建成list
def getKittiPath(path): 
    dataList = []
    for dirname, dirnames, filenames in os.walk(path):
        for filename in filenames:
            imgPath = os.path.join(dirname, filename)
            if 'image_02' in imgPath:
                dataList.append(imgPath)

    print("Total images: "+str(len(dataList)))
    shuffle(dataList)
    return dataList[:]

# 讀取 KITTI validation data的資料路徑，建成list
def getValKittiPath(path): 
    dataList = []
    for dirname, dirnames, filenames in os.walk(path):
        for filename in filenames:
            imgPath = os.path.join(dirname, filename)
            if 'image_02' in imgPath or 'image_03' in imgPath:
                dataList.append(imgPath)

    print("Total images: "+str(len(dataList)))
    shuffle(dataList)
    return dataList[:]

# 讀取 CitySdcape Dataset的資料路徑，建成list
def getCityscapesPath(path): 
    dataList = []
    for dirname, dirnames, filenames in os.walk(path):
        for filename in filenames:
            imgPath = os.path.join(dirname, filename)
            if 'labelIds' in imgPath and 'train' in imgPath and 'troisdorf_000000_000073' not in imgPath:
                dataList.append(imgPath)

    print("Total images: "+str(len(dataList)))
    shuffle(dataList)
    return dataList[:]

# 讀取 CityScape validation data的資料路徑，建成list
def getCityscapesValPath(path): 
    dataList = []
    for dirname, dirnames, filenames in os.walk(path):
        for filename in filenames:
            imgPath = os.path.join(dirname, filename)
            if 'labelIds' in imgPath and 'troisdorf_000000_000073' not in imgPath:
                dataList.append(imgPath)

    print("Total images: "+str(len(dataList)))
    shuffle(dataList)
    return dataList[:]

# 混和 KITTI Dataset 和 AirSim Dataset 的路徑 list
def getHybridPath():
    KittiList = getKittiPath(kitti_depth_folder)
    AirSimList = getAirsimPath(airsim_depth_folder)
    shuffle(KittiList)
    dataList = KittiList[:20000] + AirSimList
    shuffle(dataList)
    return dataList

# 混和 KITTI Dataset 和 AirSim Dataset 的路徑 list
def getUltraHybridPath():
    KittiList = getKittiPath(kitti_depth_folder)
    shuffle(KittiList)
    AirSimList = getAirsimPath(airsim_depth_folder)
    shuffle(AirSimList)
    CityList = getAirsimPath(city_depth_folder)
    shuffle(CityList)
    
    dataList = KittiList[:9000] + AirSimList[:500] + CityList[:500]
    shuffle(dataList)
    return dataList

# 讀取 NYU Dataset的資料路徑，建成list
def getNYUPath(path):
    dataList = []

    for i in range(1449):
        dataList.append('NYUv2_labeldata_'+str(i))
    
    for dirname, dirnames, filenames in os.walk(path):
        for filename in filenames:
            imgPath = os.path.join(dirname, filename)
            if 'mat' in imgPath:
                dataList.append(imgPath)
                
    shuffle(dataList)
    return dataList

# 讀取 NYU Dataset 和 ESCenter 的資料路徑，建成list
def getIndoorHybridPath():
    NYUList = getNYUPath(NYUv2_raw_data_path)
    shuffle(NYUList)
    ESCenterList = getAirsimPath(escenter_depth_folder)
    shuffle(ESCenterList)
    
    dataList = NYUList[:12000] + ESCenterList[:]
    shuffle(dataList)
    return dataList


# --------------------------------------------------------------------------------
# READ DATA ----------------------------------------------------------------------
# --------------------------------------------------------------------------------

def depth_read_KITTI(filename, shape=(DEPTH_H, DEPTH_W),rand=-1):
    DEPTH_H, DEPTH_W = shape
    depth_png = np.array(Image.open(filename).resize((DEPTH_W,DEPTH_H),Image.NEAREST), dtype=int)
    # make sure we have a proper 16bit depth map here.. not 8bit!
    assert(np.max(depth_png) > 255)
    
    depth = depth_png.astype(np.float) / 255.
    depth[depth_png == 0] = -1.
    # init_point = int((np.shape(depth)[1] - DEPTH_W)*rand)
    # depth = depth[:,init_point:init_point+DEPTH_W]
    return depth

def read_KITTI_interpol_depth(image_path, shape=(DEPTH_H, DEPTH_W), rand=0.5):
    DEPTH_H, DEPTH_W = shape
    
    depth = image.load_img(image_path, target_size=(DEPTH_H,DEPTH_W), grayscale=True)
    depth = image.img_to_array(depth) 
    
    depth_float = depth.astype(np.float)
    depth_float[depth == 0] = -1.
    
    # init_point = int((np.shape(depth_float)[1] - DEPTH_W)*rand)
    # depth_float = depth_float[:,init_point:init_point+DEPTH_W,:]
    return depth_float

def read_KITTI_PSMNet_depth(image_path, error_mean_var, shape=(DEPTH_H, DEPTH_W), rand=-1, include_sky = False, depth_crrection = False, log_depth = True):
    DEPTH_H, DEPTH_W = shape

    with open(image_path, 'rb') as file:
        depth =pickle.load(file)
        if rand==-1:
            depth = cv2.resize(depth,(DEPTH_W,DEPTH_H))
        else:
            depth = cv2.resize(depth,(int(DEPTH_H*depth.shape[1]/depth.shape[0]),DEPTH_H))
            begin = int((depth.shape[1]-DEPTH_W)*rand)
            depth = depth[:,begin:begin+DEPTH_W]

    if depth_crrection:
        depth_int = np.int32(depth)
        for i in range(80):
            if error_mean_var[i][0]!=0:
                depth =  np.where(depth_int==i,depth-error_mean_var[i][0],depth)


    if include_sky:
        seg_path = image_path.replace('depth_SPNet','segment_deeplabV3')
        seg_path = seg_path.replace('pickle','png')
        seg_map = cv2.imread(seg_path,0)
        seg_map = cv2.resize(seg_map,(DEPTH_W,DEPTH_H),interpolation=cv2.INTER_NEAREST)

        depth[seg_map==10] = 80

    if log_depth:
        depth_min = 4
        depth_max = 80
        depth[depth<depth_min] = depth_min
        depth[depth>depth_max] = depth_max
        depth = (np.log2(depth)-np.log2(depth_min))*depth_max/(np.log2(depth_max)-np.log2(depth_min))
        


    depth_float = depth.astype(np.float)
    
    # init_point = int((np.shape(depth_float)[1] - DEPTH_W)*rand)
    # depth_float = depth_float[:,init_point:init_point+DEPTH_W]
    return np.expand_dims(depth_float,axis=2) 

def read_KITTI_image(image_path, shape, rand=-1):
    
    IMAGE_H, IMAGE_W = shape
    if rand==-1:
        img = image.load_img(image_path, target_size=(IMAGE_H,IMAGE_W))
        img = image.img_to_array(img) 
    else:
        img = image.load_img(image_path)
        img = image.img_to_array(img) 
        img = cv2.resize(img,(int(IMAGE_H*img.shape[1]/img.shape[0]),IMAGE_H))
        begin = int((img.shape[1]-IMAGE_W)*rand)
        img = img[:,begin:begin+IMAGE_W]
    
    ###### Data Augmentation #######
    #gamma_list = [0.75,0.85,1,1.5,2,2.5]
    #gamma = gamma_list[random.randint(0,5)]
    gamma = random.uniform(0.8, 1.2)
    img = adjust_gamma(np.uint8(img), gamma=gamma)

    # blur_list = [1,2,3,5,7]
    # blur = blur_list[random.randint(0,4)]
    # img = cv2.blur(np.uint8(img),(blur,blur))
    ###### Data Augmentation #######

    # init_point = int((np.shape(img)[1] - IMAGE_W)*rand)
    # img = img[:,init_point:init_point+IMAGE_W]
    img = img.astype(np.float)
    return img


def load_seg_data_cityscapes(seg_path,shape, CLASSES=19):    
    seg_map = cv2.imread(seg_path)[:,:,::-1]
    seg_map = cv2.resize(seg_map,(shape[1],shape[0]), interpolation=cv2.INTER_NEAREST)

    ones = np.ones(seg_map[:,:,0].shape)
    seg_one_hot = np.zeros((seg_map.shape[0],seg_map.shape[1],19))
    
    ID = [7,8,11,12,13,17,19,20,21,22,23,24,25,26,27,28,31,32,33]
    
    for i in range (19):
       seg_one_hot[:,:,i] = np.where(seg_map[:,:,0]==ID[i],ones,seg_one_hot[:,:,i])

    return seg_one_hot

def read_Cityscapes_depth(image_path, shape=(DEPTH_H, DEPTH_W)):
    if 'gtFine' in image_path:
        depth_path = image_path.replace('gtFine_labelIds','depth')
        depth_path = depth_path.replace('gtFine','disparity')
    else:
        depth_path = image_path.replace('gtCoarse_labelIds','depth')
        depth_path = depth_path.replace('gtCoarse','disparity')

    depth = cv2.imread(depth_path,0)
    depth = cv2.resize(depth,(shape[1],shape[0]), interpolation=cv2.INTER_NEAREST)
    
    #depth[depth==0] = -1
    depth[depth>=0] = -1
    
    depth = np.expand_dims(depth, axis=2)
    return depth

def read_Cityscapes_depth_from_disparity(image_path, shape=(DEPTH_H, DEPTH_W)):
    if 'gtFine' in image_path:
        disparity_path = image_path.replace('gtFine_labelIds','disparity')
        disparity_path = disparity_path.replace('gtFine','disparity')
    else:
        disparity_path = image_path.replace('gtCoarse_labelIds','disparity')
        disparity_path = disparity_path.replace('gtCoarse','disparity')

    disparity = cv2.imread(disparity_path,0)
    disparity = cv2.resize(disparity,(shape[1],shape[0]), interpolation=cv2.INTER_NEAREST)
    depth = (0.22*2262)/(disparity+0.00001)
    
    # Mask out error depth values
    width = int(shape[1]*183/2048)
    height = int(shape[0]*780/1024)
    zero = np.zeros((shape[0],width))
    depth[:,:width] = np.where(depth[:,:width]>50,zero,depth[:,:width])
    zero = np.zeros((shape[0]-height,shape[1]))
    depth[height:,:] = np.where(depth[height:,:]>50,zero,depth[height:,:])


    # Mask out no-value area
    depth[depth==0] = -1
    depth[disparity==0] = -1
    depth = np.expand_dims(depth, axis=2)
    return depth


# 開啟.pfm檔案
def read_pfm(file):
    """ Read a pfm file """
    file = open(file, 'rb')

    color = None
    width = None
    height = None
    scale = None
    endian = None

    header = file.readline().rstrip()
    header = str(bytes.decode(header, encoding='utf-8'))
    if header == 'PF':
        color = True
    elif header == 'Pf':
        color = False
    else:
        raise Exception('Not a PFM file.')

    temp_str = str(bytes.decode(file.readline(), encoding='utf-8'))
    dim_match = re.match(r'^(\d+)\s(\d+)\s$', temp_str)
    if dim_match:
        width, height = map(int, dim_match.groups())
    else:
        raise Exception('Malformed PFM header.')

    scale = float(file.readline().rstrip())
    if scale < 0: # little-endian
        endian = '<'
        scale = -scale
    else:
        endian = '>' # big-endian

    data = np.fromfile(file, endian + 'f')
    shape = (height, width, 3) if color else (height, width)

    data = np.reshape(data, shape)
    # DEY: I don't know why this was there.
    #data = np.flipud(data)
    file.close()

    return data, scale

# 讀取 AirSim Dataset 中的深度圖 (.pfm)
def read_AirSim_depth(image_path, shape=(DEPTH_H, DEPTH_W)):
    DEPTH_H, DEPTH_W = shape
    depth, _ = read_pfm(image_path)
    depth[depth>100] = 100
    depth = cv2.resize(depth, (DEPTH_W,DEPTH_H),interpolation=cv2.INTER_NEAREST)[::-1,:]

    depth_float = depth.astype(np.float)
    return np.expand_dims(depth_float,axis=2) 

# 讀取 AirSim Dataset 中的影像 (.png)
def read_AirSim_image(image_path, shape):
    #image = cv2.imread(image_path)
    #image = cv2.resize(image/255., (IMAGE_H,IMAGE_W))
    img = image.load_img(image_path, target_size=shape)
    img = image.img_to_array(img) 

    # Data Augmentation
    gamma = random.uniform(0.8, 1.2)
    img = adjust_gamma(np.uint8(img), gamma=gamma)
    
    return img

# 讀取 AirSim Dataset 中的語意分割圖 (.png)
def load_seg_data_airsim(seg_path,shape, CLASSES=19):
    seg_map = cv2.imread(seg_path)[:,:,::-1]
    seg_map = cv2.resize(seg_map,(shape[1],shape[0]), interpolation=cv2.INTER_NEAREST)
    # one-hot-coding
    ones = np.ones(seg_map[:,:,0].shape)
    seg_one_hot = np.zeros((seg_map.shape[0],seg_map.shape[1],CLASSES))

                    
    color_class = [142,249,232,65,187,175,112,88,156,234,211,184,81,105,99,169,3,183,69]
    class_mapping = [1,2,3,6,4,3,5,4,3,3,6,8,7,6,3,6,6,6,6]
    
    # for i in range (CLASSES):
    #     seg_one_hot[:,:,i] = np.where(seg_map[:,:,0]==color_class[i],ones,seg_one_hot[:,:,i])

    for i in range (CLASSES):
        seg_one_hot[:,:,class_mapping[i]-1] = np.where(seg_map[:,:,0]==color_class[i],ones,seg_one_hot[:,:,class_mapping[i]-1])

    return seg_one_hot

# 讀取 KITTI Dataset 中的語意分割圖
def load_seg_data(seg_path,shape, CLASSES=19, rand=-1):
    IMAGE_H, IMAGE_W = shape

    seg_map = cv2.imread(seg_path,0)

    if rand==-1:
        seg_map = cv2.resize(seg_map,(shape[1],shape[0]), interpolation=cv2.INTER_NEAREST)
    else:
        seg_map = cv2.resize(seg_map,(int(IMAGE_H*seg_map.shape[1]/seg_map.shape[0]),IMAGE_H))
        begin = int((seg_map.shape[1]-IMAGE_W)*rand)
        seg_map = seg_map[:,begin:begin+IMAGE_W]


    # one-hot-coding
    ones = np.ones(seg_map[:,:].shape)
    seg_one_hot = np.zeros((seg_map.shape[0],seg_map.shape[1],CLASSES))
    for i in range (CLASSES):
        seg_one_hot[:,:,i] = np.where(seg_map[:,:]==i,ones,seg_one_hot[:,:,i])
    return seg_one_hot

    
# --------------------------------------------------------------------------------
# Data Generator for Training-----------------------------------------------------
# --------------------------------------------------------------------------------


# - 說明：透過這個function，在training時產生所需要的input data和ground truth data。
# - Input：
#       (list) dataList: Dataset中所有資料的路徑，可由getKittiPath這個function得到
#       (tuple) shape: model的input resolution
#       (bool) include_sky: 是否要針對GT depth map做天空的處理。若為True，則會進行前處理把depth map在天空部分的數值設為最大值(80) 
#       (bool) mixGroundTruth: 是否要將PSMNet depth map和sparse depth map混合使用。若為True，則會進行資料前處理，在GT depth map中，sparse depth map有值的地方優先使用，沒有值的地方則用PSMNet depth map代替。
#       (bool) depth_correction: 是否要對PSMNet depth map進行compensation，得到compensated PSMNet depth map
#       (int) CLASSES: semantic segmentation的類別總數量
#       (bool) random_crop: 是否要針對data進行random crop
#       (bool) log_depth: 是否要對depth map進行log transform
#       (bool) train_airsim_with_deeplabv3：若是在AirSim Dataset訓練，設為True會使用deeplabv3的segmentation作training，設為false會使用AirSim本身的ground truth做training
# - Output：每個training batch的input image和training ground truth
def UltraHybridGenerator_multiloss(dataList, shape = [(DEPTH_H, DEPTH_W),(DEPTH_H, DEPTH_W),(DEPTH_H, DEPTH_W),(DEPTH_H, DEPTH_W)], 
                                    batchSize = BATCH_SIZE, 
                                    include_sky = False, 
                                    mixGroundTruth = False,
                                    depth_crrection = False, 
                                    CLASSES = 19, 
                                    random_crop=False, 
                                    log_depth = True, 
                                    train_airsim_with_deeplabv3 = False): 
    BATCH_SIZE = batchSize

    with open('error_mean_var.pickle', 'rb') as file:
        error_mean_var = pickle.load(file)
    for i in range(60,80):
        error_mean_var[i][0] = error_mean_var[60][0] 
    
    while 1:
        for i in range(int(len(dataList)/BATCH_SIZE)): # of batches
            img = np.zeros((BATCH_SIZE,int(shape[0][0]*2),int(shape[0][1]*2),3))
            depth = [np.zeros((BATCH_SIZE,shape[0][0],shape[0][1],1)),
                     np.zeros((BATCH_SIZE,shape[1][0],shape[1][1],1)),
                     np.zeros((BATCH_SIZE,shape[2][0],shape[2][1],1)),
                     np.zeros((BATCH_SIZE,shape[3][0],shape[3][1],1))]

            seg = [np.zeros((BATCH_SIZE,shape[0][0],shape[0][1],CLASSES)),
                     np.zeros((BATCH_SIZE,shape[1][0],shape[1][1],CLASSES)),
                     np.zeros((BATCH_SIZE,shape[2][0],shape[2][1],CLASSES)),
                     np.zeros((BATCH_SIZE,shape[3][0],shape[3][1],CLASSES))]

            #try:
            # Read Images and Depth for a batch
            for j in range(i*BATCH_SIZE,(i+1)*BATCH_SIZE):
                
                depthPath = dataList[j]
                
                rand_init = -1  
                if random_crop == True: 
                    rand_init = np.random.random()
                
                if 'AirSim' in dataList[j]:
                    if 'CityEnviron' in dataList[j]:
                        subPath = depthPath.split(os.sep)
                        imgPath = city_image_folder + subPath[6]+'/image/'+ subPath[8][:-3]+'png'
                        img[j-i*BATCH_SIZE] = read_AirSim_image(imgPath,shape=(int(shape[0][0]*2),int(shape[0][1]*2)))
                    else:
                        subPath = depthPath.split(os.sep)
                        imgPath = airsim_image_folder + subPath[8]+'/image/'+ subPath[10][:-3]+'png'
                        img[j-i*BATCH_SIZE] = read_AirSim_image(imgPath,shape=(int(shape[0][0]*2),int(shape[0][1]*2)))
                elif 'KITTI' in dataList[j]:
                    subPath = depthPath.split(os.sep)
                    imgPath = kitti_image_folder + subPath[6][:10] +'/'+ subPath[6] + '/'+ subPath[9]+'/data/' + subPath[-1]    
                    imgPath = imgPath.replace('pickle','png')
                    img[j-i*BATCH_SIZE] = read_KITTI_image(imgPath, shape=(int(shape[0][0]*2),int(shape[0][1]*2)), rand=rand_init)

                elif 'CityScape' in dataList[j]:
                    if 'gtFine' in depthPath:
                        image_path = depthPath.replace('gtFine_labelIds','leftImg8bit')
                        image_path = image_path.replace('gtFine','leftImg8bit')
                    else:
                        image_path = depthPath.replace('gtCoarse_labelIds','leftImg8bit')
                        image_path = image_path.replace('gtCoarse','leftImg8bit')
                    img[j-i*BATCH_SIZE] = read_KITTI_image(image_path, shape=(int(shape[0][0]*2),int(shape[0][1]*2)))

                    
                # load all the depth with different size
                for size_iter in range(len(depth)):
                    if 'AirSim' in dataList[j]:
                        if 'CityEnviron' in dataList[j]:
                            depth[size_iter][j-i*BATCH_SIZE] = read_AirSim_depth(depthPath, shape=shape[size_iter])
                        else:
                            depth[size_iter][j-i*BATCH_SIZE] = read_AirSim_depth(depthPath, shape=shape[size_iter])
                        
                        if not train_airsim_with_deeplabv3:
                            # Read segmentation map & one-hot-coding
                            seg_path = depthPath.replace('pfm','png')
                            seg_path = seg_path.replace('depthPlanner','seg')
                            seg[size_iter][j-i*BATCH_SIZE] = load_seg_data_airsim(seg_path,shape[size_iter], CLASSES=CLASSES)
                        elif train_airsim_with_deeplabv3:
                            seg_path = depthPath.replace('pfm','png')
                            seg_path = seg_path.replace('depthPlanner','seg_deeplabV3')
                            seg[size_iter][j-i*BATCH_SIZE] = load_seg_data(seg_path,shape[size_iter])


                    elif 'KITTI' in dataList[j]:
                        if 'depth_SPNet' in dataList[j]:
                            if mixGroundTruth:
                                depth_PSMNet = read_KITTI_PSMNet_depth(depthPath, error_mean_var, shape=shape[size_iter], rand=rand_init,include_sky = include_sky, depth_crrection = depth_crrection, log_depth = True)
                                
                                depthPath_interpol = depthPath
                                depthPath_interpol = depthPath_interpol.replace('depth_SPNet','depth')
                                depthPath_interpol = depthPath_interpol.replace('pickle','png')
                                depth_gt = depth_read_KITTI(depthPath_interpol, shape=shape[size_iter], rand=rand_init)
                                depth_gt = np.expand_dims(depth_gt, 2)
                                
                                depth_gt = np.where(depth_gt<0,depth_PSMNet,depth_gt)
                                depth[size_iter][j-i*BATCH_SIZE] = depth_gt
                            else:
                                depth[size_iter][j-i*BATCH_SIZE] = read_KITTI_PSMNet_depth(depthPath, error_mean_var, shape=shape[size_iter], rand=rand_init,include_sky = include_sky, depth_crrection = depth_crrection, log_depth = True)


                            # Read segmentation map & one-hot-coding
                            seg_path = depthPath.replace('pickle','png')
                            seg_path = seg_path.replace('depth_SPNet','segment_deeplabV3')
                            seg[size_iter][j-i*BATCH_SIZE] = load_seg_data(seg_path,shape[size_iter], CLASSES=CLASSES, rand = rand_init)

                        else:
                            depth[size_iter][j-i*BATCH_SIZE] = read_KITTI_interpol_depth(depthPath, shape=shape[size_iter], rand=rand_init)
                    elif 'CityScape' in dataList[j]:
                        depth[size_iter][j-i*BATCH_SIZE] = read_Cityscapes_depth(depthPath, shape=shape[size_iter])
                        seg[size_iter][j-i*BATCH_SIZE] = load_seg_data_cityscapes(depthPath,shape[size_iter], CLASSES=CLASSES)                            
                        
            yield img, depth+seg
            # except:
            #     print('Exception occured.')

# --------------------------------------------------------------------------------
# Variables and Functions for Semantic Segmentation ------------------------------
# --------------------------------------------------------------------------------


# 建立semantic segmentation用的類別label
Label = namedtuple( 'Label' , [

    'name'        , # The identifier of this label, e.g. 'car', 'person', ... .
                    # We use them to uniquely name a class

    'id'          , # An integer ID that is associated with this label.
                    # The IDs are used to represent the label in ground truth images
                    # An ID of -1 means that this label does not have an ID and thus
                    # is ignored when creating ground truth images (e.g. license plate).
                    # Do not modify these IDs, since exactly these IDs are expected by the
                    # evaluation server.

    'trainId'     , # Feel free to modify these IDs as suitable for your method. Then create
                    # ground truth images with train IDs, using the tools provided in the
                    # 'preparation' folder. However, make sure to validate or submit results
                    # to our evaluation server using the regular IDs above!
                    # For trainIds, multiple labels might have the same ID. Then, these labels
                    # are mapped to the same class in the ground truth images. For the inverse
                    # mapping, we use the label that is defined first in the list below.
                    # For example, mapping all void-type classes to the same ID in training,
                    # might make sense for some approaches.
                    # Max value is 255!

    'category'    , # The name of the category that this label belongs to

    'categoryId'  , # The ID of this category. Used to create ground truth images
                    # on category level.

    'hasInstances', # Whether this label distinguishes between single instances or not

    'ignoreInEval', # Whether pixels having this class as ground truth label are ignored
                    # during evaluations or not

    'color'       , # The color of this label
    ] )

# 透過Label來定義每個類別的資訊。labels_19這個物件包含19種類別，用在KITTI Dataset的semantic segmentation上
labels_19 = [
    #       name                     id    trainId   category            catId     hasInstances   ignoreInEval   color
    Label(  'road'                 ,  7 ,        0 , 'flat'            , 1       , False        , False        , (128, 64,128) ),
    Label(  'sidewalk'             ,  8 ,        1 , 'flat'            , 1       , False        , False        , (244, 35,232) ),
    Label(  'building'             , 11 ,        2 , 'construction'    , 2       , False        , False        , ( 70, 70, 70) ),
    Label(  'wall'                 , 12 ,        3 , 'construction'    , 2       , False        , False        , (102,102,156) ),
    Label(  'fence'                , 13 ,        4 , 'construction'    , 2       , False        , False        , (190,153,153) ),
    Label(  'pole'                 , 17 ,        5 , 'object'          , 3       , False        , False        , (153,153,153) ),
    Label(  'traffic light'        , 19 ,        6 , 'object'          , 3       , False        , False        , (250,170, 30) ),
    Label(  'traffic sign'         , 20 ,        7 , 'object'          , 3       , False        , False        , (220,220,  0) ),
    Label(  'vegetation'           , 21 ,        8 , 'nature'          , 4       , False        , False        , (107,142, 35) ),
    Label(  'terrain'              , 22 ,        9 , 'nature'          , 4       , False        , False        , (152,251,152) ),
    Label(  'sky'                  , 23 ,       10 , 'sky'             , 5       , False        , False        , ( 70,130,180) ),
    Label(  'person'               , 24 ,       11 , 'human'           , 6       , True         , False        , (220, 20, 60) ),
    Label(  'rider'                , 25 ,       12 , 'human'           , 6       , True         , False        , (255,  0,  0) ),
    Label(  'car'                  , 26 ,       13 , 'vehicle'         , 7       , True         , False        , (  0,  0,142) ),
    Label(  'truck'                , 27 ,       14 , 'vehicle'         , 7       , True         , False        , (  0,  0, 70) ),
    Label(  'bus'                  , 28 ,       15 , 'vehicle'         , 7       , True         , False        , (  0, 60,100) ),
    Label(  'train'                , 31 ,       16 , 'vehicle'         , 7       , True         , False        , (  0, 80,100) ),
    Label(  'motorcycle'           , 32 ,       17 , 'vehicle'         , 7       , True         , False        , (  0,  0,230) ),
    Label(  'bicycle'              , 33 ,       18 , 'vehicle'         , 7       , True         , False        , (119, 11, 32) ),
    Label(  'license plate'        , -1 ,       -1 , 'vehicle'         , 7       , False        , True         , (  0,  0,142) ),]

# 透過Label來定義每個類別的資訊。labels_6這個物件包含6種類別，用在AirSim Dataset的semantic segmentation上
labels_6 = [
    #       name                     id    trainId   category            catId     hasInstances   ignoreInEval   color
    Label(  'road'                 ,  7 ,        0 , 'flat'            , 1       , False        , False        , (128, 64,128) ),
    Label(  'sky'                  , 23 ,       10 , 'sky'             , 5       , False        , False        , ( 70,130,180) ),
    Label(  'vegetation'           , 21 ,        8 , 'nature'          , 4       , False        , False        , (107,142, 35) ),
    Label(  'building'             , 11 ,        2 , 'construction'    , 2       , False        , False        , ( 70, 70, 70) ),
    Label(  'terrain'              , 22 ,        9 , 'nature'          , 4       , False        , False        , (152,251,152) ),
    Label(  'vegetation'           , 21 ,        8 , 'nature'          , 4       , False        , False        , (107,142, 35) ),
    Label(  'vegetation'           , 21 ,        8 , 'nature'          , 4       , False        , False        , (107,142, 35) ),
    Label(  'terrain'              , 22 ,        9 , 'nature'          , 4       , False        , False        , (152,251,152) ),
    Label(  'truck'                , 27 ,       14 , 'vehicle'         , 7       , True         , False        , (  0,  0, 70) ),
    Label(  'bus'                  , 28 ,       15 , 'vehicle'         , 7       , True         , False        , (  0, 60,100) ),
    Label(  'building'             , 11 ,        2 , 'construction'    , 2       , False        , False        , ( 70, 70, 70) ),
    Label(  'car'                  , 26 ,       13 , 'vehicle'         , 7       , True         , False        , (  0,  0,142) ),
    Label(  'fence'                , 13 ,        4 , 'construction'    , 2       , False        , False        , (190,153,153) ),
    Label(  'train'                , 31 ,       16 , 'vehicle'         , 7       , True         , False        , (  0, 80,100) ),
    Label(  'motorcycle'           , 32 ,       17 , 'vehicle'         , 7       , True         , False        , (  0,  0,230) ),
    Label(  'building'             , 11 ,        2 , 'construction'    , 2       , False        , False        , ( 70, 70, 70) ),
    Label(  'traffic light'        , 19 ,        6 , 'object'          , 3       , False        , False        , (250,170, 30) ),
    Label(  'traffic sign'         , 20 ,        7 , 'object'          , 3       , False        , False        , (220,220,  0) ),
    Label(  'person'               , 24 ,       11 , 'human'           , 6       , True         , False        , (220, 20, 60) ),]

# 會使用到的類別名稱
LABEL_NAMES = np.asarray(['road','sidewalk','building','wall','fence','pole','traffic light','traffic sign','vegetation','terrain','sky','person','rider','car','truck','bus','train','motorcycle','bicycle'])

# 讀取semantic segmentation map，分析上面種共有哪些類別，把這些使用到的類別的顏色、名稱畫出來
def plot_class_sample(seg_pred, segment_type=19):

    if segment_type==19:
        labels = labels_19
    elif segment_type==6:
        labels = labels_6

    seg_pred_vis = np.argmax(seg_pred,axis=3)
    seg_pred_vis = np.squeeze(seg_pred_vis)
    curr_class = []
    for i in range(20): 
        if i in seg_pred_vis:
            curr_class.append(i)
            
    img = np.zeros((int(len(curr_class)/5)*100+100, 5*500, 3), np.uint8)
    img.fill(255)
    counter=0
    for i in curr_class:
        #cv2.rectangle(img, (0, 100*counter), (100, 100*(counter+1)), labels[i].color, -1)
        #cv2.putText(img, labels[i].name, (150, int(100*(counter+0.6))), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.rectangle(img, (400*(counter%5), 100*int(counter/5)), (400*(counter%5)+100, 100*int(counter/5)+100), labels[i].color, -1)
        cv2.putText(img, labels[i].name, (400*(counter%5)+120, int(100*(int(counter/5)+0.6))), cv2.FONT_HERSHEY_DUPLEX, 1.4, (0, 0, 0), 2, cv2.LINE_AA)
        counter+=1
    #showResult([img[:,:,:]],figsize=(20,3))
    img = cv2.resize(img,(5*500//1,(int(len(curr_class)/5)*100+100)//1))   
    return img
    
# 將 model preidct 產生的semantic segmentation做decode，產生視覺化的語意分割圖
def decode_precition(seg_pred, segment_type=19):

    if segment_type==19:
        labels = labels_19
    elif segment_type==6:
        labels = labels_6

    seg_argmax = np.argmax(seg_pred,axis=3)
    seg_argmax = np.expand_dims(np.squeeze(seg_argmax),axis=2) 
    seg_argmax = np.concatenate((seg_argmax,seg_argmax,seg_argmax),axis=2)
    #print(seg_argmax.shape)
    
    seg_vis = np.ones((seg_argmax.shape[0],seg_argmax.shape[1],3))
    
    ones = np.ones((seg_argmax.shape[0],seg_argmax.shape[1],3))
    
    for i in range(19):
        seg_vis = np.where(seg_argmax==[i,i,i],np.array(labels[i].color)*ones,seg_vis)
    #print(seg_vis.shape)
    return seg_vis 

# --------------------------------------------------------------------------------
# Functions for Evaluation -------------------------------------------------------
# --------------------------------------------------------------------------------

# 計算RMSE
# Input
#    -(array) y_predict: 預測的深度圖
#    -(array) y_true: 正確的深度圖
#    -(int) cap: 要計算多少距離以內的RMSE
#    -(int) error_range: 要計算多少距離以外的RMSE。預設為-1，會計算0公尺以外的RMSE
# Output:
#    -(float) RMSE: 計算得到的RMSE數值
def compute_RMSE(y_predict, y_true, cap=1000, error_range=-1):
    mask = np.zeros(y_true.shape)
    if error_range==-1:
        mask[y_true >= 1] = 1
    else:
        mask[y_true >= cap-error_range] = 1
    mask[y_true > cap] = 0
    
    y_predict[y_predict>100] = 100
    y_predict[y_predict<0] = 0
    y_true[y_true>100] = 100
    
    y = mask*(y_predict - y_true)**2
    
    if (mask>0).sum()==0:
        return 0
    else:
        rmse = math.sqrt(y.sum()/(mask>0).sum())
    return rmse

# 計算ARD
# Input
#    -(array) y_predict: 預測的深度圖
#    -(array) y_true: 正確的深度圖
#    -(int) cap: 要計算多少距離以內的RMSE
# Output:
#    -(float) ard: 計算得到的ARD數值
def compute_ARD(y_predict, y_true, cap=1000):
    mask = np.zeros(y_true.shape)
    mask[y_true >= 1] = 1
    mask[y_true > cap] = 0
    
    y_predict[y_predict>100] = 100
    y_predict[y_predict<=0] = 0.01
    y_true[y_true>100] = 100
    y_true[y_true<=0] = 0.01
    
    y = mask*np.absolute(y_predict - y_true)/y_true
    ard = y.sum()/len(mask[mask>0])
        
    return ard

# 計算threshold
# Input
#    -(array) y_predict: 預測的深度圖
#    -(array) y_true: 正確的深度圖
#    -(float) thida: 公式中的參數
# Output:
#    -(float) threshold: 計算得到的threshold數值，單位為百分比
def compute_threshold(y_predict, y_true, thida=1.25):
    mask = np.zeros(y_true.shape)
    mask[y_true >= 1] = 1
    
    y_predict[y_predict>100] = 100
    y_predict[y_predict<0] = 0
    y_true[y_true>100] = 100
    
    y_predict[y_predict>100] = 100
    y_predict[y_predict<0] = 0
    
    maxValue = np.maximum( y_true/y_predict, y_predict/y_true )*mask
    threshold = (len(maxValue[maxValue<thida])-len(maxValue[maxValue==0]))/len(maxValue[maxValue>0])
    
    return threshold

# 計算SRD
# Input
#    -(array) y_predict: 預測的深度圖
#    -(array) y_true: 正確的深度圖
#    -(int) cap: 要計算多少距離以內的RMSE
# Output:
#    -(float) srd: 計算得到的SRD數值
def compute_SRD(y_predict, y_true, cap=1000):
    mask = np.zeros(y_true.shape)
    mask[y_true >= 1] = 1
    mask[y_true > cap] = 0
    
    y_predict[y_predict>100] = 100
    y_predict[y_predict<=0] = 0.01
    y_true[y_true>100] = 100
    y_true[y_true<=0] = 0.01
    
    
    srd = (mask*(y_predict - y_true)**2/y_true).sum()/len(mask[mask>0])
    return srd


# - 說明：用來對model進行一系列的數值評估，並show出數值結果
# - Input：
#     (Keras.models) model: 訓練完成的model
#     (list) dataList: 要用來評估的資料，以list的方式列出所有資料路徑
#     (int) totalData: 總共要測試的資料數量
#     (tuple) image_shape: model的input resolution
#     (bool) log_depth: 在訓練時是否有針對depth map進行log transform
# - Output：印出RMSE、ARD、SRD、threshold等數值分析結果    
def model_evaluation(model, dataList, totalData, image_shape, log_depth = False):
    rmse_all,rmse_50,rmse_30,ARD,SRD, threshold_1,threshold_2,threshold_3 = [],[],[],[],[],[],[],[]

    #data = inputFiles[random.randint(0,len(inputFiles)-1)]
    totalTime =0
    current_RMSE = 0
    for dataIter, data in enumerate(dataList[:totalData]):

        if dataIter%100==1:
            current_RMSE = float(str(sum(rmse_all)/len(rmse_all)))
        print('\r processing '+str(dataIter) + '  RMSE: %.3f' %current_RMSE, end='')

        depth_path = data
        subPath = depth_path.split(os.sep)
        image_path = '../../dataset/KITTI/image/'+subPath[6][:10]+'/'+subPath[6]+'/'+subPath[-2]+'/data/'+subPath[-1]


        # Load depth
        #depth =read_AirSim_blurDepth(depth_path, 7)[::-1,:].reshape(DEPTH_H,DEPTH_W)
        #depth =read_AirSim_maxPoolDepth(depth_path)[::-1,:].reshape(DEPTH_H,DEPTH_W)
        depth =depth_read_KITTI(depth_path, shape=(375,1242))

        # Load image                
        image = read_KITTI_image(image_path, image_shape)
        image_expend = np.expand_dims(image, 0)

        # predict
        start = time.time()
        results = model.predict(image_expend,batch_size=1)[0]
        end = time.time()
        totalTime +=end-start

        results = cv2.resize(results[0,:,:,:],(1242,375))
        
        if log_depth:
            depth_min = 4
            depth_max = 80
            results[results<depth_min] = depth_min
            results[results>depth_max] = depth_max
            results = 2**(results*(np.log2(depth_max)-np.log2(depth_min))/depth_max+np.log2(depth_min))
        

        # evaluate
        rmse_all.append(compute_RMSE(results, depth, cap=100))
        rmse_50.append(compute_RMSE(results, depth, cap=50))
        rmse_30.append(compute_RMSE(results, depth, cap=30))
        ARD.append(compute_ARD(results, depth, cap=1000))
        SRD.append(compute_SRD(results, depth, cap=1000))
        threshold_1.append(compute_threshold(results, depth, thida=1.25*1))
        threshold_2.append(compute_threshold(results, depth, thida=1.25*2))
        threshold_3.append(compute_threshold(results, depth, thida=1.25*3))

    print('\nRMSE all: %.3f' %float(str(sum(rmse_all)/len(rmse_all))))
    print('RMSE 50: %.3f'    %float(str(sum(rmse_50)/len(rmse_50))))
    print('RMSE 30: %.3f'    %float(str(sum(rmse_30)/len(rmse_30))))
    print('\nARD: %.3f'      %float(str(sum(ARD)/len(ARD))))
    print('\nSRD: %.3f'      %float(str(sum(SRD)/len(SRD))))
    print('threshold1: %.3f' %float(str(sum(threshold_1)/len(threshold_1))))
    print('threshold2: %.3f' %float(str(sum(threshold_2)/len(threshold_2))))
    print('threshold3: %.6f' %float(str(sum(threshold_3)/len(threshold_3))))
    print('totalTime: %.2f'  %totalTime)
    print('fps: %.2f'        %float(totalData/totalTime))

def model_evaluation_single_output(model, dataList, totalData, image_shape, log_depth = False):
    rmse_all,rmse_50,rmse_30,ARD,SRD, threshold_1,threshold_2,threshold_3 = [],[],[],[],[],[],[],[]

    #data = inputFiles[random.randint(0,len(inputFiles)-1)]
    totalTime =0
    current_RMSE = 0
    for dataIter, data in enumerate(dataList[:totalData]):

        if dataIter%100==1:
            current_RMSE = float(str(sum(rmse_all)/len(rmse_all)))
        print('\r processing '+str(dataIter) + '  RMSE: %.3f' %current_RMSE, end='')

        depth_path = data
        subPath = depth_path.split(os.sep)
        image_path = '../../dataset/KITTI/image/'+subPath[6][:10]+'/'+subPath[6]+'/'+subPath[-2]+'/data/'+subPath[-1]


        # Load depth
        #depth =read_AirSim_blurDepth(depth_path, 7)[::-1,:].reshape(DEPTH_H,DEPTH_W)
        #depth =read_AirSim_maxPoolDepth(depth_path)[::-1,:].reshape(DEPTH_H,DEPTH_W)
        depth =depth_read_KITTI(depth_path, shape=(375,1242))

        # Load image                
        image = read_KITTI_image(image_path, image_shape)
        image_expend = np.expand_dims(image, 0)

        # predict
        start = time.time()
        results = model.predict(image_expend,batch_size=1)
        end = time.time()
        totalTime +=end-start

        results = cv2.resize(results[0,:,:,:],(1242,375))
        
        if log_depth:
            depth_min = 4
            depth_max = 80
            results[results<depth_min] = depth_min
            results[results>depth_max] = depth_max
            results = 2**(results*(np.log2(depth_max)-np.log2(depth_min))/depth_max+np.log2(depth_min))
        

        # evaluate
        rmse_all.append(compute_RMSE(results, depth, cap=100))
        rmse_50.append(compute_RMSE(results, depth, cap=50))
        rmse_30.append(compute_RMSE(results, depth, cap=30))
        ARD.append(compute_ARD(results, depth, cap=1000))
        SRD.append(compute_SRD(results, depth, cap=1000))
        threshold_1.append(compute_threshold(results, depth, thida=1.25*1))
        threshold_2.append(compute_threshold(results, depth, thida=1.25*2))
        threshold_3.append(compute_threshold(results, depth, thida=1.25*3))

    print('\nRMSE all: %.3f' %float(str(sum(rmse_all)/len(rmse_all))))
    print('RMSE 50: %.3f'    %float(str(sum(rmse_50)/len(rmse_50))))
    print('RMSE 30: %.3f'    %float(str(sum(rmse_30)/len(rmse_30))))
    print('\nARD: %.3f'      %float(str(sum(ARD)/len(ARD))))
    print('\nSRD: %.3f'      %float(str(sum(SRD)/len(SRD))))
    print('threshold1: %.3f' %float(str(sum(threshold_1)/len(threshold_1))))
    print('threshold2: %.3f' %float(str(sum(threshold_2)/len(threshold_2))))
    print('threshold3: %.6f' %float(str(sum(threshold_3)/len(threshold_3))))
    print('totalTime: %.2f'  %totalTime)
    print('fps: %.2f'        %float(totalData/totalTime))

def model_evaluation_TX2(model, dataList, totalData, image_shape, log_depth = False):
    rmse_all,rmse_50,rmse_30,ARD,threshold_1,threshold_2,threshold_3 = [],[],[],[],[],[],[]

    totalTime =0
    current_RMSE = 0
    for dataIter, data in enumerate(dataList[:totalData]):

        if dataIter%100==1:
            current_RMSE = float(str(sum(rmse_all)/len(rmse_all)))
        print('\r processing '+str(dataIter) + '  RMSE: %.3f' %current_RMSE, end='')

        depth_path = data
        subPath = depth_path.split(os.sep)
        image_path = '../../dataset/KITTI/image/'+subPath[6][:10]+'/'+subPath[6]+'/'+subPath[-2]+'/data/'+subPath[-1]
        

        # Load depth
        depth_gt =depth_read_KITTI(depth_path, shape=(375,1242))
        
        
        # Load image                
        image = read_KITTI_image(image_path,shape=image_shape,rand=0)
        image_expend = np.expand_dims(image, 0)
        # predict
        start = time.time()
        depth = model.predict(image_expend,batch_size=1)[0]
        end = time.time()
        totalTime +=end-start
        
        depth = cv2.resize(depth[:,:,0],(563,375))
        results = depth
        
        
        # Load image                
        image = read_KITTI_image(image_path,shape=image_shape,rand=0.82916)
        image_expend = np.expand_dims(image, 0)
        # predict
        start = time.time()
        depth = model.predict(image_expend,batch_size=1)[0]
        end = time.time()
        totalTime +=end-start

        depth = cv2.resize(depth[:,:,0],(563,375))
        results = np.concatenate((results,depth[:,:116]),axis=1)
        
        # Load image                
        image = read_KITTI_image(image_path,shape=image_shape,rand=1)
        image_expend = np.expand_dims(image, 0)
        # predict
        start = time.time()
        depth = model.predict(image_expend,batch_size=1)[0]
        end = time.time()
        totalTime +=end-start

        depth = cv2.resize(depth[:,:,0],(563,375))
        results = np.concatenate((results,depth),axis=1)

        if log_depth:
            depth_min = 4
            depth_max = 80
            results[results<depth_min] = depth_min
            results[results>depth_max] = depth_max
            results = 2**(results*(np.log2(depth_max)-np.log2(depth_min))/depth_max+np.log2(depth_min))

        # evaluate
        rmse_all.append(compute_RMSE(results, depth_gt, cap=100))
        rmse_50.append(compute_RMSE(results, depth_gt, cap=50))
        rmse_30.append(compute_RMSE(results, depth_gt, cap=30))
        ARD.append(compute_ARD(results, depth_gt, cap=1000))
        threshold_1.append(compute_threshold(results, depth_gt, thida=1.25*1))
        threshold_2.append(compute_threshold(results, depth_gt, thida=1.25*2))
        threshold_3.append(compute_threshold(results, depth_gt, thida=1.25*3))

    print('\nRMSE all: %.3f' %float(str(sum(rmse_all)/len(rmse_all))))
    print('RMSE 50: %.3f'    %float(str(sum(rmse_50)/len(rmse_50))))
    print('RMSE 30: %.3f'    %float(str(sum(rmse_30)/len(rmse_30))))
    print('\nARD: %.3f'      %float(str(sum(ARD)/len(ARD))))
    print('threshold1: %.3f' %float(str(sum(threshold_1)/len(threshold_1))))
    print('threshold2: %.3f' %float(str(sum(threshold_2)/len(threshold_2))))
    print('threshold3: %.6f' %float(str(sum(threshold_3)/len(threshold_3))))
    print('totalTime: %.2f'  %totalTime)
    print('fps: %.2f'        %float(totalData/totalTime))

def model_evaluation_AirSim(model, dataList, totalData, image_shape, log_depth = False):
    rmse_all,rmse_50,rmse_30,ARD,threshold_1,threshold_2,threshold_3 = [],[],[],[],[],[],[]

    totalTime =0
    current_RMSE = 0
    for dataIter, data in enumerate(dataList[:totalData]):

        if dataIter%100==1:
            current_RMSE = float(str(sum(rmse_all)/len(rmse_all)))
        print('\r processing '+str(dataIter) + '  RMSE: %.3f' %current_RMSE, end='')

        
        # Load depth
        depth_path = data
        depth = np.squeeze(read_AirSim_depth(depth_path, shape=image_shape))

        # Load image                
        image_path = depth_path.replace('depthPlanner','image')
        image_path = image_path.replace('pfm','png')
        image = read_KITTI_image(image_path, shape=image_shape)
        image_expend = np.expand_dims(image, 0)

        # predict
        start = time.time()
        results = model.predict(image_expend,batch_size=1)[0]
        end = time.time()
        totalTime +=end-start

        results = cv2.resize(results[0,:,:,:],(image_shape[1],image_shape[0]))

        if log_depth:
            depth_min = 0
            depth_max = 100
            results[results<depth_min] = depth_min
            results[results>depth_max] = depth_max
            results = 2**(results*(np.log2(depth_max)-np.log2(depth_min))/depth_max+np.log2(depth_min))
        
        # evaluate
        rmse_all.append(compute_RMSE(results, depth, cap=100))
        rmse_50.append(compute_RMSE(results, depth, cap=50))
        rmse_30.append(compute_RMSE(results, depth, cap=30))
        ARD.append(compute_ARD(results, depth, cap=1000))
        threshold_1.append(compute_threshold(results, depth, thida=1.25*1))
        threshold_2.append(compute_threshold(results, depth, thida=1.25*2))
        threshold_3.append(compute_threshold(results, depth, thida=1.25*3))
        
        
    print('\nRMSE all: %.3f' %float(str(sum(rmse_all)/len(rmse_all))))
    print('RMSE 50: %.3f'    %float(str(sum(rmse_50)/len(rmse_50))))
    print('RMSE 30: %.3f'    %float(str(sum(rmse_30)/len(rmse_30))))
    print('\nARD: %.3f'      %float(str(sum(ARD)/len(ARD))))
    print('threshold1: %.3f' %float(str(sum(threshold_1)/len(threshold_1))))
    print('threshold2: %.3f' %float(str(sum(threshold_2)/len(threshold_2))))
    print('threshold3: %.6f' %float(str(sum(threshold_3)/len(threshold_3))))
    print('totalTime: %.2f'  %totalTime)
    print('fps: %.2f'        %float(totalData/totalTime))