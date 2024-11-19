import os,re
import cv2
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def data_augmentation(file_dir, n_generation):
    data_gen = ImageDataGenerator(rotation_range=10, 
                      width_shift_range=0.1,
                      height_shift_range=0.1,
                      shear_range=0.1,
                      brightness_range=(0.3, 1.0),
                      horizontal_flip=True,
                      vertical_flip=True,
                      fill_mode='nearest')
    
#-Created Processed Directory to store preprocessed data---------
    filedir = re.sub(r'raw','interim',file_dir)
    for file_ in os.listdir(file_dir):
        if not file_=='.gitkeep':
            os.makedirs(os.path.join(filedir,file_),exist_ok=True)
        print('path created successfully')

#data already stored directory    
    for filename in os.listdir(file_dir):
        if not filename=='.gitkeep':
            paths = os.path.join(file_dir,filename)
            
            for img in os.listdir(paths):
                image = cv2.imread(paths+'/'+img)
                image = image.reshape((1,) + image.shape)
                save_prefix = 'aug_' + filename[:-4]
                i=0
                savepath = re.sub(r'raw','interim',paths)
                for batch in data_gen.flow(x = image, batch_size = 1, save_to_dir = savepath, save_prefix = save_prefix, save_format = "jpg"):
                    i+=1
                    if i>n_generation:
                        break
            
def list_file_dir(path_):
    path = []
    for folder in os.listdir(path_):
        join = os.path.join(path_,folder)
        path.append(join)
    return path
    