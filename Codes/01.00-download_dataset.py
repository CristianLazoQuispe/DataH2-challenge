import opendatasets as od
import shutil
import os
import glob
url_data = "https://www.kaggle.com/c/data-h-m1-challenge-final/data"
original = 'data-h-m1-challenge-final'
target = '../Dataset/'




if os.path.isdir(original):
    shutil.move(original,target)

    print("Exists")
else:
    print("Doesn't exists",original)
    if os.path.isdir(os.path.join(target,'datah-m1-challange')):
        print("Exists",original)
        pass
    else:
        print('downloading..')
        od.download(url_data)
        print('downloaded !')
        
        shutil.move(original,target)
        print('move to',target)
        


