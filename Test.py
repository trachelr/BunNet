import os

import numpy as np

from difflib import SequenceMatcher

img_path = "/media/BunNet/pictures/"

img_names = sorted(os.listdir(img_path))

thres = 0.8
di_class = {}
#For each images
for idImg, img in enumerate(img_names):
    #Rank Dictionnary keys
    di_keyRank = {}
    for key in di_class:
        avg_score = []
        for i in di_class[key]:
            avg_score = SequenceMatcher(None, i, img).ratio()
        di_keyRank[key] = np.mean(avg_score)
    #Store img under the key with highest avg_score, if it's above ratio
    if len(di_keyRank) != 0:
        maxKey = max(di_keyRank)
        if di_keyRank[maxKey] > thres:
            di_class[maxKey].append(img)
            print('img: {} stored under key {} with avg ratio of {}'\
                  .format(img, maxKey, di_keyRank[maxKey]))
            continue
    
    # If none exist, create a new one.
    di_class[img] = [img]
    print('img {} stored under a new key'.format(img))
    if idImg % (int(len(img_names)/100)) == 0:
        print('Done {} img over {} ({}%)'.format())

#Gather some stats
keySize = [len(di_class[key]) for key in di_class]
print('Total number of key: {}, with average size:{} std:{}, min:{}, max:{}'\
      .format(len(di_class), np.mean(keySize), np.std(keySize), min(keySize), max(keySize)))







