import os
import operator

import numpy as np

from difflib import SequenceMatcher

img_path = "/media/BunNet/pictures/"
verbose=False

img_names = sorted(os.listdir(img_path))

#Replace digits with 'X' becasue gitis are part of the pattern but their values are irrelevent.
#Also remove file extensions
#Original filename can be found from the position of the item in the list (this is a ugly hack... and rely on sorted listdir)
img_names_filt = []
for idImg in range(len(img_names)):
    img_names_filt.append(''.join(i if not i.isdigit() else 'X' for i in img_names[idImg])\
                              .split('.')[0])

thres = 0.7 #Set empirically
di_class = {}
#For each images
for idImg, img in enumerate(img_names_filt):
    #Rank Dictionnary keys
    di_keyRank = {}
    for key in di_class:
        #Speedup: skip key with low match ratio as they won't probably met the required avg ratio.
        if SequenceMatcher(None, key, img).ratio() < 0.5*thres:
            di_keyRank[key] = SequenceMatcher(None, key, img).ratio()
            if verbose: print('Early skip, key:{}, img: {}, ratio: {}'.format(key, img, di_keyRank[key]))
        else:
            avg_score = []
            for i in di_class[key]:
                avg_score = SequenceMatcher(None, i[0], img).ratio()
            di_keyRank[key] = np.mean(avg_score)
            
    #Store img under the key with highest avg_score, if it's above ratio
    if len(di_keyRank) != 0:
        maxKey = max(di_keyRank.items(), key=operator.itemgetter(1))[0]
        if di_keyRank[maxKey] > thres:
            di_class[maxKey].append((img, idImg))
            if verbose: print('img: {} stored under key {} with avg ratio of {}'\
                              .format(img, maxKey, di_keyRank[maxKey]))
            continue
    # If none exist, create a new one.
    di_class[img] = [(img, idImg)]
    print('img {} stored under a new key'.format(img))
    if idImg % (int(len(img_names)/100)) == 0:
        print('Done {} img over {} ({:.1f}%)'.format(idImg, len(img_names), 100*idImg/len(img_names)))

#Gather some stats
keySize = [len(di_class[key]) for key in di_class]
print('Total number of key: {}, with average size:{} std:{}, min:{}, max:{}'\
      .format(len(di_class), np.mean(keySize), np.std(keySize), min(keySize), max(keySize)))







