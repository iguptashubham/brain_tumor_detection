import os
import random
import shutil

def TrainTestSplit(data_path,exist_data_dir,proportion=(0.7,0.15,15)):
    #-----------------------directory creation---------------------
    folderpath = os.path.join(data_path,'traintestval')
    os.makedirs(os.path.join(data_path,'traintestval'),exist_ok=True)
    
    for folder in ['train','test','val']:
        os.makedirs(os.path.join(folderpath,folder),exist_ok=True)
    
    datafl = os.listdir(os.path.join(data_path,'interim'))
    for fl in datafl:
        if not fl == '.gitkeep':
            print(fl)
            for dir in os.listdir(folderpath):
                os.makedirs(os.path.join(folderpath,dir,fl),exist_ok=True)
                print(f'{dir}-created successfully')
    #-------------------directory creation complete-----------------
    
    #--------------------move data and split in created directorty----------
    existdata = os.listdir(exist_data_dir)
    for exist in existdata:
         if not exist == '.gitkeep':
            imagefile = os.listdir(os.path.join(exist_data_dir, exist))
            random.shuffle(imagefile)
            total = len(imagefile)   
            split1 = int(total*proportion[0])    
            split2 = split1+int(total*proportion[1])
            
            train = imagefile[:split1]
            test = imagefile[split1:split2]
            val = imagefile[split2:]

            for folder, dataset in zip(['train', 'test', 'val'], [train, test, val]):
                savedir = os.path.join(folderpath, folder, exist)
                os.makedirs(savedir, exist_ok=True)
                
                for filename in dataset:
                    source = os.path.join(exist_data_dir, exist, filename)
                    destination = os.path.join(savedir, filename)
                    shutil.copy(src=source, dst=destination)

                print(f'{len(dataset)} files saved in {folder}')

# Example usage
path = r'C:\Users\gupta\OneDrive\Desktop\Projects\brain_tumor\Brain-tumor-detection\data'
data = r'C:\Users\gupta\OneDrive\Desktop\Projects\brain_tumor\Brain-tumor-detection\data\processed'
TrainTestSplit(data_path=path, exist_data_dir=data)

    