import os
import shutil
import glob
import math
import random
print(os.listdir())

lis = os.listdir()
lis.remove('.DS_Store')
lis.remove('random_sample.py')

for i in lis :
    path_ = f"{i}/keyframes/*"
    dir_path= f"{i}/keyframes/"
    
    Lenn = (len([entry for entry in os.listdir(dir_path) if os.path.isfile(os.path.join(dir_path, entry))]))

    print(i,Lenn)
    to_be_moved = random.sample(glob.glob(path_), Lenn-1)
    for f in enumerate(to_be_moved[: math.ceil(0.20* Lenn)], 1):
        dest = os.path.join("data/train")
        if not os.path.exists(dest):
            os.makedirs(dest)
        shutil.copy(f[1], dest)

    for f in enumerate(to_be_moved[math.ceil(0.20* Lenn):], 1):
        dest = os.path.join("data/val/")
        if not os.path.exists(dest):
            os.makedirs(dest)
        shutil.copy(f[1], dest)