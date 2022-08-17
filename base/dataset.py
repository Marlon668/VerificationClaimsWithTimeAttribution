import pickle
import os

import spacy

import Claim

import torch
from torch.utils.data import Dataset, DataLoader


class NUS(Dataset):
    # change root to your root
    def __init__(self, mode,path,domain):
        super().__init__()
        assert mode in ['Train', 'Dev', 'Test']
        print('Loading {} set...'.format(mode))
        self.domain = domain
        self.mode = mode
        print('Reading the claims')
        self.getClaims(mode,path)
        print('Done')

    def getClaims(self, mode,path):
        self.claimIds = []
        self.documents = []
        self.labels = []
        self.snippetDocs = []
        self.metadata = []
        self.metadataSet = set()
        self.labelsAll = set()

        nlp = "None"
        predictorOIE = "None"
        predictorNER = "None"
        coreference = "None"
        if mode != "Test":
            with open(path, 'r', encoding='utf-8') as file:
                for claim in file:
                    elements = claim.split('\t')
                    claim = Claim.claim(elements[0], elements[1], elements[2], elements[3], elements[4], elements[5],
                                        elements[6],
                                        elements[7], elements[8], elements[9], elements[10], elements[11], elements[12],
                                        os.pardir+"/snippets/", predictorOIE, predictorNER, nlp, coreference)

                    metadata = ''
                    # speaker
                    metadata += str(claim.speaker)
                    self.metadataSet.add(claim.speaker)

                    metadata +='\n'
                    # category
                    for category in claim.categories:
                        self.metadataSet.add(category)
                        metadata += category +'\t'
                    metadata = metadata[:-1]
                    metadata +="\n"
                    # tags
                    for tag in claim.tags:
                        self.metadataSet.add(tag)
                        metadata += tag +'\t'
                    metadata = metadata[:-1]
                    metadata += '\n'
                    # entities
                    for entitie in claim.entities:
                        self.metadataSet.add(entitie)
                        metadata += entitie +'\t'
                    metadata = metadata[:-1]
                    self.metadata.append(metadata)

                    # label
                    self.labels.append(claim.label)
                    self.labelsAll.add(claim.label)
                    self.claimIds.append(claim.claimID)
                    self.documents.append(self.getClaimText(claim))
                    self.snippetDocs.append(self.getSnippets(claim.snippets))
                print(len(self.snippetDocs))
        else:
            labelsTest = dict()
            pathLabels = os.pardir + '/test/test-'+ self.domain + '-labels.tsv'
            with open(pathLabels, 'r', encoding='utf-8') as fileL:
                for line in fileL:
                    elements = line.split('\t')
                    labelsTest[elements[0]] = elements[1].replace('\n','')
            with open(path, 'r', encoding='utf-8') as file:
                for claim in file:
                    elements = claim.split('\t')

                    claim = Claim.claim(elements[0], elements[1], labelsTest[elements[0]], elements[2], elements[3], elements[4], elements[5],
                                        elements[6],
                                        elements[7], elements[8], elements[9], elements[10], elements[11],
                                        os.pardir+"/snippets/", predictorOIE, predictorNER, nlp, coreference)

                    metadata = ''
                    # speaker
                    metadata += str(claim.speaker)
                    self.metadataSet.add(claim.speaker)

                    metadata += '\n'
                    # category
                    for category in claim.categories:
                        self.metadataSet.add(category)
                        metadata += category + '\t'
                    metadata = metadata[:-1]
                    metadata += "\n"
                    # tags
                    for tag in claim.tags:
                        self.metadataSet.add(tag)
                        metadata += tag + '\t'
                    metadata = metadata[:-1]
                    metadata += '\n'
                    # entities
                    for entitie in claim.entities:
                        self.metadataSet.add(entitie)
                        metadata += entitie + '\t'
                    metadata = metadata[:-1]
                    self.metadata.append(metadata)
                    # label
                    self.labels.append(claim.label)

                    self.claimIds.append(claim.claimID)
                    self.documents.append(self.getClaimText(claim))
                    self.snippetDocs.append(self.getSnippets(claim.snippets))
                print(len(self.snippetDocs))

    def getClaimText(self,claim):
        return claim.claim

    def getSnippets(self, snippets):
        dataSnippets = ''
        for snippet in snippets:
            dataSnippets += snippet.article
            dataSnippets += ' 0123456789 '
        return dataSnippets

    def getMetaDataSet(self):
        return self.metadataSet

    def __len__(self):
        return len(self.claimIds)

    def __getitem__(self, index):
        # Take the filename.


        return self.claimIds[index], self.documents[index], self.snippetDocs[index],self.metadata[index], self.labels[index]




# Dump dataset into file
def dump_write(dataset, name):
    with open(name, 'wb') as f:
        pickle.dump(dataset, f)


# retrieve dataset from file
def dump_load(name):
    with open(name, 'rb') as f:
        dataset = pickle.load(f)
    return dataset

'''
domain = "abbc"
# Check of the dataloader
# train_set = NUS(mode='Little')
# dump_write(train_set, "little_dataset")
train_set = NUS(mode='Train', path=os.pardir + '/train/train-' + domain + '.tsv', domain=domain)
#train_set.writeMetadata()
#dump_write(train_set, "trainLoader")
#train_set = dump_load( "trainLoader")
train_loader = DataLoader(train_set,
                          batch_size=4,
                          shuffle=True)
for c in train_loader:
    print('Claim-ID')
    print(c[0])
    print('--------------')
    print('Claim:')
    print(c[1])
    print('--------------')
    print('Snippets')
    print(c[2])
    print('Speaker:')
    print(c[3])
    print('metadata:')
    print('Label:')
    print(c[4])
    break
'''
