import os
import cv2
from PIL import Image

DirList = [
    'Pic/city'
]

for path in DirList:
    for filename in os.listdir(path):
        fullName = os.path.join(path, filename)
        if fullName.endswith('.jpeg'):
            mainName, ext = os.path.splitext(fullName)
            # print(fullName, mainName + '.jpeg')
            im = Image.open(fullName).convert("RGB")
            im.save(mainName + '.jpeg', quality=80)
            os.remove(fullName)
