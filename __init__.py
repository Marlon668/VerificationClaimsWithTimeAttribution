import csv
import os
import pickle

from allennlp.predictors import Predictor

import Claim

import torch
from torch.utils.data import Dataset, DataLoader


class NUS(Dataset):
    #change root to your root
    def __init__(self, mode):
        super().__init__()
        assert mode in ['Train', 'Dev', 'Test']
        print('Loading {} set...'.format(mode))

        self.mode = mode
        print('Reading the claims')
        self.filenames = self.getClaims(mode)
        print('Reading the snippets')
        self.captions = self.get_snippets("../snippets/")
        print('Reading ground-truth descriptions...')
        self.groundtruth = self.get_groundtruth()
        print('Reading the image features...')
        self.image_features = self.get_image_features(self.root + 'images/image_features.csv')
        print('')
        self.clean_filenames = self.get_clean_names()

    def getClaims(self,mode):
        path = ""
        if mode == "Test":
            path = "test.tsv"
        else:
            if mode == "Train":
                path = "train.tsv"
            else:
                path = "dev.tsv"
        with open(path,'r') as file:
            for claim in file:
                print(claim)


def getClaims(mode):
    claims = []
    speaker = set()
    category = set()
    tags = set()
    entities = set()
    labels = set()
    path = ""
    if mode == "Test":
        path += "test.tsv"
    else:
        if mode == "Train":
            path += "train.tsv"
        else:
            path += "dev.tsv"
    predictor = Predictor.from_path(
        "https://storage.googleapis.com/allennlp-public-models/openie-model.2020.03.26.tar.gz")
    print(mode)
    with open(path,'r',encoding='utf-8') as file:
        for claim in file:
            elements = claim.split('\t')
            claim = Claim.claim(elements[0],elements[1],elements[2],elements[3],elements[4],elements[5],elements[6],elements[7],elements[8],elements[9],elements[10],elements[11],elements[12],"snippets/",predictor)
            speaker.add(claim.speaker)
            labels.add(elements[2])
            for cat in claim.categories:
                if (cat[-1] == ','):
                    print(cat)
                category.add(cat)
            for tag in claim.tags:
                tags.add(tag)
            for ent in claim.entities:
                entities.add(ent)

            claims.append(claim)
    return claims
    '''
    print('train')
    with open("../train.tsv",'r',encoding='utf-8') as file:
        for claim in file:
            elements = claim.split('\t')
            claim = Claim.claim(elements[0],elements[1],elements[2],elements[3],elements[4],elements[5],elements[6],elements[7],elements[8],elements[9],elements[10],elements[11],elements[12],"../snippets/")
            speaker.add(claim.speaker)
            labels.add(elements[2])
            for cat in claim.categories:
                category.add(cat)
            for tag in claim.tags:
                tags.add(tag)
            for ent in claim.entities:
                entities.add(ent)

            claims.append(claim)

    print(len(claims))
    print("length category : " + str(len(category)))
    print("length speaker : " + str(len(speaker)))
    print("length tags : " + str(len(tags)))
    print("length entities : " + str(len(entities)))
    print("Number of labels : " + str(len(labels)))
    print(speaker)

    print('test')
    with open("../test.tsv",'r',encoding='utf-8') as file:
        for claim in file:
            elements = claim.split('\t')
            claim = Claim.claim(elements[0],elements[1],0,elements[2],elements[3],elements[4],elements[5],elements[6],elements[7],elements[8],elements[9],elements[10],elements[11],"../snippets/")
            speaker.add(claim.speaker)
            for cat in claim.categories:
                if (cat[-1] == ','):
                    print(cat)
                category.add(cat)
            for tag in claim.tags:
                if (tag[-1] == ','):
                    print('rrr')
                tags.add(tag)
            for ent in claim.entities:
                if (ent[-1] == ','):
                    print('rrr')
                entities.add(ent)

            claims.append(claim)
    
    print(len(claims))
    numberOfDates = 0
    for claim in claims:
        if(claim.claimDate!="None"):
            numberOfDates+=1
    print("Number of dates: " + str(numberOfDates))
    print("length category : " + str(len(category)))
    print("length speaker : " + str(len(speaker)))
    print("length tags : " + str(len(tags)))
    print("length entities : " + str(len(entities)))
    '''
    '''
    f = open("times2.txt", "w")
    for claim in claims:
        if(claim.claimDate!="None"):
            f.write(claim.claimID + "\t" + claim.claimDate+"\n")
    f.close()
    '''

claims = getClaims("Dev")
timeOK = 0
timeNOK = 0
for claim in claims:
    timeOK += claim.TimeOK
    timeNOK += claim.TimeNOK

print('Number OK : '+str(timeOK))
print('Number NOK : ' + str(timeNOK))
'''
# Dump dataset into file
def dump_write(dataset, name):
    with open(name, 'wb') as f:
        pickle.dump(dataset, f)


# retrieve dataset from file
def dump_load(name):
    with open(name, 'rb') as f:
        dataset = pickle.load(f)
    return dataset


# Check of the dataloader
# train_set = NUS(mode='Little')
# dump_write(train_set, "little_dataset")
train_set = NUS(mode='Little')
train_loader = DataLoader(train_set,
                          batch_size=4,
                          shuffle=True)
for c in train_loader:
    print('Image features:')
    print(c[0])
    print('--------------')
    print('Caption embeddings:')
    print(c[1])
    print('--------------')
    print('Concept descriptions:')
    print(c[2])
    print('--------------')
    print('Indexes:')
    print(c[3])
    break
'''