import os
import shutil
import glob
import random

to_be_moved = random.sample(glob.glob("hands/hand_dataset/training_dataset/training_data/images/*"), 1000)

for f in enumerate(to_be_moved[:700], 1):
    dest = os.path.join("data/train")
    if not os.path.exists(dest):
        os.makedirs(dest)
    shutil.copy(f[1], dest)

for f in enumerate(to_be_moved[700:], 1):
    dest = os.path.join("data/val/")
    if not os.path.exists(dest):
        os.makedirs(dest)
    shutil.copy(f[1], dest)