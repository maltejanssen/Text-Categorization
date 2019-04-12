import os
import glob
import numpy as np

class SFU_Spanish_Reviews_Loader:
    def __init__(self,path_to_dataset,restrictTo = []):
        self.path_to_dataset = path_to_dataset
        
        categories = [name for name in os.listdir(path_to_dataset) if os.path.isdir(os.path.join(path_to_dataset, name))] 
        self.paths = dict()

        for cat in categories: 
            current_paths=sorted(glob.glob(os.path.join(path_to_dataset,cat,"*.txt")))
            self.paths[cat]=current_paths
            
        if len(restrictTo)>0: 
            new_paths = {cat:self.paths[cat] for cat in restrictTo}
            self.paths=new_paths
            
    def getCategories(self):
        return sorted(self.paths.keys())
    
    
    def getData(self):
        X = []
        C = []
        Y = []
        for cat in self.getCategories(): # por cada categoria
            paths=self.paths[cat]
            paths = sorted(paths)
            for path in paths:
                clase = 0 if path.find("no_") > 0 else 1
                f = open(path, 'r', encoding='latin1')
                X.append(f.read())
                Y.append(clase)
                C.append(cat)
        #return  np.array(X), np.array(Y),np.array(C)
        return  X,Y,C
