import os
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import warnings
import uuid
warnings.filterwarnings('ignore')


def DataDF(file):
    path = r'C:\Users\gupta\OneDrive\Desktop\Projects\brain_tumor\Brain-tumor-detection\data'
    datapath = os.path.join(path,file)
    data = os.listdir(datapath)
    full_paths = [os.path.join(datapath, file) for file in data]

    result = {}

    for folder in full_paths:
        if os.path.isdir(folder):
            # Filter out .gitkeep and count images
            images = [file for file in os.listdir(folder) if file != '.gitkeep' and file.lower().endswith(('.png', '.jpg', '.jpeg'))]
            result[os.path.basename(folder)] = len(images)
    
    df = pd.DataFrame(list(result.items()), columns=['name', 'count'])
    return df

def braintumordist(data, plot=True):
    fig, ax = plt.subplots(figsize=(15, 5), nrows=1, ncols=2)
    sns.barplot(x=data['name'], y=data['count'], ax=ax[0])
    ax[0].set_title('Total Images Available')
    ax[0].set_xticklabels(ax[0].get_xticklabels(), rotation=45, ha='right')
    sizes = data['count']
    labels = data['name']
    ax[1].pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140)
    ax[1].set_title('Image Distribution by Folder')
    
    plt.tight_layout()  # Adjust layout to prevent overlapping
    if plot:
        plt.savefig(os.path.join(r'C:\Users\gupta\OneDrive\Desktop\Projects\brain_tumor\Brain-tumor-detection\reports\figures',f'{uuid.uuid1()}.png'))
        plt.show()

def CompareDist(raw, processed, fig=(15,5)):
    fig, ax = plt.subplots(1, 2, figsize=fig)
    
    sns.barplot(x=raw['name'], y=raw['count'], ax=ax[0])
    ax[0].set_title('Available Data')
    ax[0].set_xticklabels(ax[0].get_xticklabels(), rotation=45, ha='right')
    for p in ax[0].patches:
        ax[0].annotate(format(p.get_height(), '.2f'),
                       (p.get_x() + p.get_width() / 2., p.get_height()),
                       ha = 'center', va = 'center',
                       xytext = (0, 9),
                       textcoords = 'offset points')
    sns.despine(ax=ax[0], top=True, right=True)  # Remove upper and right border

    sns.barplot(x=processed['name'], y=processed['count'], ax=ax[1])
    ax[1].set_title('Processed Data')
    ax[1].set_xticklabels(ax[1].get_xticklabels(), rotation=45, ha='right')
    for p in ax[1].patches:
        ax[1].annotate(format(p.get_height(), '.2f'),
                       (p.get_x() + p.get_width() / 2., p.get_height()),
                       ha = 'center', va = 'center',
                       xytext = (0, 9),
                       textcoords = 'offset points')
    sns.despine(ax=ax[1], top=True, right=True)  # Remove upper and right border
    
    plt.suptitle('Data', fontsize=20)
    plt.tight_layout()
    plt.savefig(os.path.join(r'C:\Users\gupta\OneDrive\Desktop\Projects\brain_tumor\Brain-tumor-detection\reports\figures',f'{uuid.uuid1()}.png'))
    plt.show()