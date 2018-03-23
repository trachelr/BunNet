##Check duplicates classes in BunNet data

import os
import cv2
import datetime
import time
import joblib

import numpy as np
import pylab as plt
import pickle as pkl

from PIL import Image

#Check for Similarities.
#This is done by maintaining a imageXimage similarity matrix
#Each entry is a bitFlag as follow:
# -b6: images are the same
# -b5: same model/constructor exif data
# -b4: close DateTime exif data (less tahn 2H)
# -b3: large gap in DateTime exif data (more than 1M)
# -b2: No exif model data
# -b1: No exif time data
# -b0: exif data not loadable 
def checkDuplicate(idF1, file1, idF2, file2, simValue, 
                   defaultDate, defaultModel):
    #exif tags of relevance
    e_tags_date = 36867
    e_tag_make = 271
    e_tag_model = 272

    if simValue != 0:
        return simValue
    if idF2 <= idF1: #skip pairs already checked
        return simValue

    similarity = 0
    
    #Load exif data
    img1 = Image.open(file1)
    info1 = img1._getexif() if '_getexif()' in dir(img1) else None
    date1 = defaultDate
    model1 = defaultModel
    if info1 is not None:
        if e_tags_date in info1.keys():
            try: #sometimes there's nonsense in here
                date1 = datetime.datetime.strptime(info1[e_tags_date], '%Y:%m:%d %H:%M:%S')
            except ValueError:
                print('Invalid date detected: {} for image: {}'.format(info1[e_tags_date], file2))
        if e_tag_make in info1.keys() and e_tag_model in info1.keys():
            model1 = info1[e_tag_make]+"_"+info1[e_tag_model]
       
    img2 = Image.open(file2)
    info2 = img2._getexif() if '_getexif()' in dir(img2) else None
    date2 = defaultDate
    model2 = defaultModel
    if info2 is not None:
        if e_tags_date in info2.keys():
            try: #sometimes there's nonsense in here
                date2 = datetime.datetime.strptime(info2[e_tags_date], '%Y:%m:%d %H:%M:%S')
            except ValueError:
                print('Invalid date detected: {} for image: {}'.format(info2[e_tags_date], file2))
        if e_tag_make in info2.keys() and e_tag_model in info2.keys():
            model2 = info2[e_tag_make]+"_"+info2[e_tag_model]
 
    datediff = np.abs((date1 - date2).total_seconds())
    
    #exif present
    if info1 is None and info2 is None:
        similarity += 1 << 0
    else:
        if date1 == defaultDate and date2 == defaultDate:
            similarity += 1 << 1
        if model1 == defaultModel and model2 == defaultModel:
            similarity += 1 << 2
        
    #Date check
    if datediff > 30*24*3600: #Take more than 1M apart
        similarity += 1 << 3
    elif datediff < 2*3600: #taken less than 2H apart
        similarity += 1 << 4
        
    #Model check
    if model1 == model2:
        similarity += 1 << 5
    
    #Identity check
    cvImg1 = cv2.imread(file1)
    cvImg2 = cv2.imread(file2)
    if cvImg1.shape == cvImg2.shape:
        if not np.any(cv2.subtract(cvImg1, cvImg2)):
            similarity += 1 << 6
            
    return similarity
    

if __name__ == '__main__':
    #Windows stuff, multiprocessing ?
    __spec__ = None
    
    fastMode  = False
    
    #Path
    root_path = 'D:\BunNet/'
    img_path = root_path+'data2'
    log_path = root_path+'classFusion.log'
    sim_path = root_path+'simMat.pkl'
    log_file = open(log_path, 'w')
    
    
    #First pass, compute some stats
    print('root_path is: {}'.format(root_path))
    print('log file is: {}'.format(log_path))
    all_files = []
    classSize = []
    all_folders = sorted(os.listdir(img_path))
    for fold in all_folders:
        for file in sorted(os.listdir(img_path+'/'+fold)):
            all_files.append(img_path+'/'+fold+'/'+file)      
        classSize.append(len(os.listdir(img_path+'/'+fold)))
    all_files = sorted(all_files)
    
    if fastMode:
        all_files = all_files[0:25]
    
    print('found {} folder and {} files'.format(len(all_folders), len(all_files)))
    
    #Stats
    classSize = np.array(classSize)
    classSize.sort()
    
    meanSize = classSize.mean()
    stdSize = classSize.std()
    minSize = classSize.min()
    maxSize = classSize.max()
    print('Class size report -- mean: {}, std: {}, min:{}, max:{}'.format(meanSize, stdSize, minSize, maxSize))
    
    ratioClass = np.array([(classSize>i).sum() for i in np.arange(minSize, maxSize, 1)])
    
    skip = True
    if not skip:
        plt.ion()
        plt.figure()
        plt.bar(np.arange(classSize.shape[0]), classSize)
        plt.xlabel('classes')
        plt.ylabel('# of elements')
                 
        plt.figure()
        plt.bar(np.arange(minSize, maxSize, 1), ratioClass)
        plt.xlabel('Smallest class size')
        plt.ylabel('Number of class')
    
    #Check for existing simMatrix and load if needed
    if sim_path.split('/')[-1] in os.listdir(root_path):
        simMatrix = pkl.load(open(sim_path, 'rb'))
        if simMatrix.shape != (len(all_files), len(all_files)):
            simMatrix = np.zeros((len(all_files), len(all_files)), dtype='int')
    else:
        simMatrix = np.zeros((len(all_files), len(all_files)), dtype='int')
    
    #Start checking for similarities
    print('checking for similarities')
    defaultDate = datetime.datetime.today().replace(second=0, microsecond=0)
    defaultModel = 'default'
    for idF1, file1 in enumerate(all_files):
        t_jlib = time.time()
        simRow = joblib.Parallel(n_jobs=-1)(
                joblib.delayed(checkDuplicate)
                (idF1, file1, idF2, file2, simMatrix[idF1, idF2], defaultDate, defaultModel)
                for idF2, file2 in enumerate(all_files))
        t_jlib = time.time() - t_jlib
        
        t_copy = time.time()
        simMatrix[idF1] = simRow
        simMatrix[:, idF1] = simRow
        pkl.dump(simMatrix, open(sim_path, 'wb'))
        t_copy = time.time() - t_copy
            
        
        if idF1 % 1 == 0:
            print('Done {} img over {} {:.2f}% -- Time:: total: {:.2f}, Jlib:{:.2f}, copy:{:.2f}'\
                  .format(idF1, len(all_files), 100*(idF1/len(all_files)), t_jlib + t_copy, t_jlib, t_copy))


    #Check similarityMatrix and log anomalies
    #Many cases can appears
    # -Images are copy
    # -- Make a DUPLI entry
    # -- if in different folder: make a STRONG_MATCH entry
    #
    # -Images are taken from the same camera AND model exif valid AND exif valid
    # -- If in different folder: make a MEDIUM_MATCH entry
    # -Images aren't from the same camera
    # -- If in same folder: make an MEDIUM_ANO entry
    #
    # -Close DateTime exif AND date exif valid AND exif valid
    # -- If different folder: WEAK MATCH      
    #
    # - Distant time AND date exif valid AND exif valid
    # -- same folder: WEAK ANO
    nbImg = len(all_files)
    duplicate = []
    match_strong = []
    match_medium = []
    match_weak = []
    anomaly_medium = []
    anomaly_weak = []
    for i in range(nbImg):
        for j in range(nbImg):
            #only work on the upper diag
            if j <= i:
                continue
            
            name1 =  all_files[i].split('/')[-1]
            folder1 = all_files[i].split('/')[-2]
            fname1 = folder1+'/'+name1
            
            name2 =  all_files[j].split('/')[-1]
            folder2 = all_files[j].split('/')[-2]
            fname2 = folder2+'/'+name2
            
            #check for duplicate
            if (simMatrix[i,j] >> 6) & 1:
                duplicate.append((fname1, fname2))
                if folder1 != folder2:
                    match_strong.append((fname1, fname2))
                    continue #Do not check for exif, everything else will match
            
            #Check for valid exif
            if not (simMatrix[i,j] >> 0) & 1:
                #check for valid model
                if not (simMatrix[i,j] >> 2) & 1:
                    #check for same model
                    if (simMatrix[i, j] >> 5) & 1:
                        if folder1 != folder2:
                            match_medium.append((fname1, fname2))
                    else:
                        if folder1 == folder2:
                            anomaly_medium.append((fname1, fname2))
                    
                #check for valid time data
                if not (simMatrix[i,j] >> 1) & 1:
                    #Check for small time gap
                    if (simMatrix[i,j] >> 4) & 1:
                        if folder1 != folder2:
                            match_weak.append((fname1, fname2))
                    #check for large gap
                    if (simMatrix[i,j] >> 3) & 1:
                        if folder1 == folder2:
                            anomaly_weak.append((fname1, fname2))
    
    print('{} duplicate detected'.format(len(duplicate)))
    print('{} strong matches detected'.format(len(match_strong)))
    print('{} medium matches detected'.format(len(match_medium)))
    print('{} weak matches detected'.format(len(match_weak)))
    print('{} medium anomalies detected'.format(len(anomaly_medium)))
    print('{} weak anomalies detected'.format(len(anomaly_weak)))
    
    
    
    ##Logging
    log_file.writelines('{} duplicate entries\n'.format(len(duplicate)))
    log_file.writelines('{} strong match entries\n'.format(len(match_strong)))
    log_file.writelines('{} medium match entries\n'.format(len(match_medium)))
    log_file.writelines('{} weak match entries\n'.format(len(match_weak)))
    log_file.writelines('{} medium anomalies entries\n'.format(len(anomaly_medium)))
    log_file.writelines('{} weak anomalies entries\n'.format(len(anomaly_weak)))
    log_file.writelines('\n')
    log_file.writelines('======== DUPLICATE ========\n')
    log_file.writelines('(exactly the same file)\n')
    for e in duplicate:
        log_file.writelines('{} and {}\n'.format(e[0], e[1]))
    log_file.writelines('\n')
    log_file.writelines('\n')
    log_file.writelines('======== STRONG MATCH ========\n')
    log_file.writelines('(Duplicate file in different folder. Folders should likely be merged)\n')
    for e in match_strong:
        log_file.writelines('{} and {}\n'.format(e[0], e[1]))
    log_file.writelines('\n')
    log_file.writelines('\n')
    log_file.writelines('======== MEDIUM MATCH ========\n')
    log_file.writelines('(Same camera, different folder. Might be different bunnies but the same photographer)\n')
    for e in match_medium:
        log_file.writelines('{} and {}\n'.format(e[0], e[1]))
    log_file.writelines('\n')
    log_file.writelines('\n')
    log_file.writelines('======== WEAK MATCH ========\n')
    log_file.writelines('(Pictures taken within a short time frame across different folders. Might be just luck)\n')
    for e in match_weak:
        log_file.writelines('{} and {}\n'.format(e[0], e[1]))
    log_file.writelines('\n')
    log_file.writelines('\n')
    log_file.writelines('======== MEDIUM ANOMALY ========\n')
    log_file.writelines('(Different camera, same folder. Owner might have changed camera. If shots are close in time it is probably a true error)\n')
    for e in anomaly_medium:
        log_file.writelines('{} and {}\n'.format(e[0], e[1]))
    log_file.writelines('\n')
    log_file.writelines('\n')
    log_file.writelines('======== WEAK ANOMALY ========\n')
    log_file.writelines('(Long time between shot. Might just be different shooting sessions)\n')
    for e in anomaly_weak:
        log_file.writelines('{} and {}\n'.format(e[0], e[1]))
    
    log_file.close()