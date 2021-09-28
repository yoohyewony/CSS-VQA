import os

fpath = 'Resized_Images/train2014'
files = os.listdir(fpath)

for name in files:
    src = os.path.join(fpath, name)
    new = str(int(name[-16:-4])) + '.jpg'
    new = os.path.join(fpath, new)
    os.rename(src, new)