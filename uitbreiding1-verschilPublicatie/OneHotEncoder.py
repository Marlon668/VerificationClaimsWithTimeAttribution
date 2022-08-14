import numpy
import numpy as np
import torch
from numpy import array

class oneHotEncoder:

    def __init__(self,path):
        self.elementToIndex = dict()
        index = 0
        with open(path, 'r', encoding='utf-8') as file:
            for element in file:
                #remove /n at end of element
                self.elementToIndex[element[:-1]] = index
                index = index+1
            file.close()

    def getLength(self):
        return len(self.elementToIndex.keys())+4

    def encode(self,metadata,device):
        categories = metadata.split('\n')
        encodingAll = []
        for j in range(len(categories)):
            data = categories[j].split('\t')
            for i in range(len(data)):
                encoding = np.zeros(self.getLength())
                encoding[self.getLength()-(4-j)] = 1.0
                encoding[self.elementToIndex[data[i]]] = 1.0
                encodingAll.append(encoding)
        return torch.tensor(numpy.array(encodingAll),dtype=torch.float).to(device)





