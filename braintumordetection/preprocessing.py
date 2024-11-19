#---------objective
#---1. convert bgr to gra
#---2. Gaussian blur
#---3. threshold
#---4. Erode
#---5. Dilate
#---6. find contours
import imutils
import cv2
import os
import re
import uuid
import matplotlib.pyplot as plt

def preprocess_data(path):
    
    savedir = re.sub(r'interim','processed',path)
    for files in os.listdir(path):
        if not files=='.gitkeep':
            dir = os.path.join(savedir,files)
            os.makedirs(dir,exist_ok=True)
        print('Directory Created Successfully')
            
    for filename in os.listdir(path):
        if not filename=='.gitkeep':
            paths = os.path.join(path,filename)
            imgfile = os.listdir(paths)
            for img in imgfile:
                image_path = os.path.join(paths,img)
                print(image_path)
                image = cv2.imread(image_path)
                imagegy = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
                imgblur = cv2.GaussianBlur(imagegy,(5,5),0)
                _,thres = cv2.threshold(imgblur,45,255,cv2.THRESH_BINARY)
                erode = cv2.erode(thres, None,iterations=2)
                dilate = cv2.dilate(erode, None,iterations=2)
                
                cnts = cv2.findContours(dilate.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
                cnts = imutils.grab_contours(cnts=cnts)
                c = max(cnts,key = cv2.contourArea)
                
                extleft = tuple(c[c[:,:,0].argmin()][0])
                extright = tuple(c[c[:,:,0].argmax()][0])
                exttop = tuple(c[c[:,:,1].argmin()][0])
                extbot = tuple(c[c[:,:,1].argmax()][0])
                
                new_image = image[exttop[1]:extbot[1],extleft[0]:extright[0]]
                save_dir = re.sub(r'interim','processed',paths)
                save_path = os.path.join(save_dir,f'{uuid.uuid1()}.jpg')
                cv2.imwrite(save_path,new_image)

def plot_beforeAfter(file):
     path = re.sub(r'interim','processed',file)
     interim=os.listdir(file)[0]
     processed = os.listdir(path)[0]
     
     imgpro = cv2.imread(os.path.join(path,processed))
     imgin = cv2.imread(os.path.join(file,interim))
     
     fig,ax = plt.subplots(1,2,figsize=(15,5))
     ax[0].imshow(imgin)
     ax[0].set_title('Before')
     ax[1].imshow(imgpro)
     ax[1].set_title('After')
     
     plt.suptitle('Image showing the removal of whitespace in augmented picture')
     plt.savefig(os.path.join(r'C:\Users\gupta\OneDrive\Desktop\Projects\brain_tumor\Brain-tumor-detection\reports\figures',f"{uuid.uuid1()}.png"))
     plt.show()
