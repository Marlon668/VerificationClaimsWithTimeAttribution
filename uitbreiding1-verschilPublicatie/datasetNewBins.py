from allennlp.predictors.predictor import Predictor as pred
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
        self.domain = domain
        print('Loading {} set...'.format(mode))
        self.mode = mode
        print('Get buckets of differences')
        self.getDifferences('newBinsVerschilDays.txt')
        print('Reading the claims')
        self.getClaims(mode,path)
        print('Done')

    def getDifferences(self, path):
        self.buckets = list()
        with open(path, 'r', encoding='utf-8') as file:
            for line in file:
                bucket = line.split('\t')
                bucket[1] = bucket[1].replace("\n","")
                if bucket[0]=="-infinity":
                    self.buckets.append(tuple([float('-inf'),int(bucket[1])]))
                else:
                    if bucket[1] == "infinity":
                        self.buckets.append(tuple([int(bucket[0]), float('inf')]))
                    else:
                        self.buckets.append(tuple([int(bucket[0]), int(bucket[1])]))

    def matchBucket(self, difference):
        for i in range(len(self.buckets)):
            if self.buckets[i][0]<=difference<=self.buckets[i][1]:
                return i

    def getClaims(self, mode,path):
        self.claimIds = []
        self.documents = []
        self.labels = []
        self.snippetDocs = []
        self.metadata = []
        self.metadataSet = set()
        self.labelsAll = set()
        self.bucketsSnippets = []
        self.claimDateAvaialble = []

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
                                        "snippets/", predictorOIE, predictorNER, nlp, coreference)
                    claim.processDate()
                    bucketsSnippetClaim = ''

                    if not type(claim.claimDate) is tuple:
                        if claim.claimDate != None:
                            self.claimDateAvaialble.append(1)
                        else:
                            self.claimDateAvaialble.append(0)

                        for snippet in claim.getSnippets():
                            snippet.processDate()
                            if (snippet.publishTime != None and claim.claimDate != None):
                                if type(snippet.publishTime) is tuple:
                                    if ((snippet.publishTime[0] - claim.claimDate).days <= 0 and (
                                            snippet.publishTime[1] - claim.claimDate).days >= 0):
                                        verschil = 0
                                    else:
                                        if (snippet.publishTime[1] - claim.claimDate).days <= 0:
                                            verschil = (snippet.publishTime[1] - claim.claimDate).days
                                        else:
                                            verschil = (snippet.publishTime[0] - claim.claimDate).days
                                    bucketsSnippetClaim +=  '\t' + str(self.matchBucket(verschil))
                                else:
                                    verschil = (snippet.publishTime-claim.claimDate).days
                                    bucketsSnippetClaim += '\t' + str(self.matchBucket(verschil))
                            else:
                                if snippet.publishTime != None:
                                    bucketsSnippetClaim += '\t' + str(63)
                                else:
                                    bucketsSnippetClaim += '\t' + str(64)
                    else:
                        if claim.claimDate != None:
                            self.claimDateAvaialble.append(1)
                        else:
                            self.claimDateAvaialble.append(0)
                        for snippet in claim.getSnippets():
                            snippet.processDate()
                            if (snippet.publishTime != None and claim.claimDate != None):
                                if type(snippet.publishTime) is tuple:
                                    if (snippet.publishTime[0] - claim.claimDate[1]).days >= 0:
                                        verschil = (snippet.publishTime[0] - claim.claimDate[1]).days
                                        bucketsSnippetClaim += '\t' + str(self.matchBucket(verschil))
                                    else:
                                        if (snippet.publishTime[1] - claim.claimDate[0]).days <= 0:
                                            verschil = (snippet.publishTime[1] - claim.claimDate[0]).days
                                            bucketsSnippetClaim +=  '\t' + str(self.matchBucket(verschil))
                                        else:
                                            verschil = 0
                                            bucketsSnippetClaim +=  '\t' + str(self.matchBucket(verschil))
                                else:
                                    if ((snippet.publishTime - claim.claimDate[0]).days >= 0 and (
                                            snippet.publishTime - claim.claimDate[1]).days <= 0):
                                        verschil = 0
                                        bucketsSnippetClaim += '\t' + str(self.matchBucket(verschil))
                                    else:
                                        if (snippet.publishTime - claim.claimDate[0]).days <= 0:
                                            verschil = (snippet.publishTime - claim.claimDate[0]).days
                                        else:
                                            verschil = (snippet.publishTime - claim.claimDate[1]).days
                                        bucketsSnippetClaim +=  '\t' + str(self.matchBucket(verschil))
                            else:
                                if snippet.publishTime != None:
                                    bucketsSnippetClaim += '\t' + str(63)
                                else:
                                    bucketsSnippetClaim += '\t' + str(64)

                    self.bucketsSnippets.append(bucketsSnippetClaim)
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
                print('lengtes')
                print(len(self.snippetDocs))
                print(len(self.claimDateAvaialble))
                print(len(self.claimIds))
                print(len(self.bucketsSnippets))
        else:
            labelsTest = dict()
            pathLabels = 'test/test-' + self.domain + '-labels.tsv'
            with open(pathLabels, 'r', encoding='utf-8') as fileL:
                for line in fileL:
                    elements = line.split('\t')
                    labelsTest[elements[0]] = elements[1].replace('\n', '')
            with open(path, 'r', encoding='utf-8') as file:
                for claim in file:
                    elements = claim.split('\t')

                    claim = Claim.claim(elements[0], elements[1], labelsTest[elements[0]], elements[2], elements[3],
                                        elements[4], elements[5],
                                        elements[6],
                                        elements[7], elements[8], elements[9], elements[10], elements[11],
                                        "snippets/", predictorOIE, predictorNER, nlp, coreference)
                    claim.processDate()
                    bucketsSnippetClaim = ''

                    if not type(claim.claimDate) is tuple:
                        if claim.claimDate != None:
                            self.claimDateAvaialble.append(1)
                        else:
                            self.claimDateAvaialble.append(0)

                        for snippet in claim.getSnippets():
                            snippet.processDate()
                            if (snippet.publishTime != None and claim.claimDate != None):
                                if type(snippet.publishTime) is tuple:
                                    if ((snippet.publishTime[0] - claim.claimDate).days <= 0 and (
                                            snippet.publishTime[1] - claim.claimDate).days >= 0):
                                        verschil = 0
                                    else:
                                        if (snippet.publishTime[1] - claim.claimDate).days <= 0:
                                            verschil = (snippet.publishTime[1] - claim.claimDate).days
                                        else:
                                            verschil = (snippet.publishTime[0] - claim.claimDate).days
                                    bucketsSnippetClaim +=  '\t' + str(self.matchBucket(verschil))
                                else:
                                    verschil = (snippet.publishTime-claim.claimDate).days
                                    bucketsSnippetClaim += '\t' + str(self.matchBucket(verschil))
                            else:
                                if snippet.publishTime != None:
                                    bucketsSnippetClaim += '\t' + str(63)
                                else:
                                    bucketsSnippetClaim += '\t' + str(64)
                    else:
                        if claim.claimDate != None:
                            self.claimDateAvaialble.append(1)
                        else:
                            self.claimDateAvaialble.append(0)
                        for snippet in claim.getSnippets():
                            snippet.processDate()
                            if (snippet.publishTime != None and claim.claimDate != None):
                                if type(snippet.publishTime) is tuple:
                                    if (snippet.publishTime[0] - claim.claimDate[1]).days >= 0:
                                        verschil = (snippet.publishTime[0] - claim.claimDate[1]).days
                                        bucketsSnippetClaim += '\t' + str(self.matchBucket(verschil))
                                    else:
                                        if (snippet.publishTime[1] - claim.claimDate[0]).days <= 0:
                                            verschil = (snippet.publishTime[1] - claim.claimDate[0]).days
                                            bucketsSnippetClaim +=  '\t' + str(self.matchBucket(verschil))
                                        else:
                                            verschil = 0
                                            bucketsSnippetClaim +=  '\t' + str(self.matchBucket(verschil))
                                else:
                                    if ((snippet.publishTime - claim.claimDate[0]).days >= 0 and (
                                            snippet.publishTime - claim.claimDate[1]).days <= 0):
                                        verschil = 0
                                        bucketsSnippetClaim += '\t' + str(self.matchBucket(verschil))
                                    else:
                                        if (snippet.publishTime - claim.claimDate[0]).days <= 0:
                                            verschil = (snippet.publishTime - claim.claimDate[0]).days
                                        else:
                                            verschil = (snippet.publishTime - claim.claimDate[1]).days
                                        bucketsSnippetClaim +=  '\t' + str(self.matchBucket(verschil))
                            else:
                                if snippet.publishTime != None:
                                    bucketsSnippetClaim += '\t' + str(63)
                                else:
                                    bucketsSnippetClaim += '\t' + str(64)

                    self.bucketsSnippets.append(bucketsSnippetClaim)
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
        basepath = os.path.normpath(os.getcwd() + os.sep + os.pardir)
        file = open(basepath + "/text" + "/" + claim.claimID + "/" + "claim",encoding="utf-8")
        return file.read()

    def getMetaDataSet(self):
        return self.metadataSet

    def __len__(self):
        return len(self.claimIds)

    def __getitem__(self, index):
        # Take the filename.


        return self.claimIds[index], self.documents[index], self.snippetDocs[index],self.metadata[index], self.labels[index],self.claimDateAvaialble[index],self.bucketsSnippets[index]

    def getSnippets(self, snippets):
        dataSnippets = ''
        basepath = os.path.normpath(os.getcwd() + os.sep + os.pardir)
        for snippet in snippets:
            file = open(basepath + "/text" + "/" + snippet.claimID + "/" + snippet.number,encoding="utf-8")
            dataSnippets += file.read()
            dataSnippets += ' 0123456789 '
            file.close()
        return dataSnippets


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
# Check of the dataloader
# train_set = NUS(mode='Little')
# dump_write(train_set, "little_dataset")
#train_set = NUS(mode='Dev',path='dev/dev-huca.tsv')
#train_set.writeMetadata()
#dump_write(train_set, "trainLoader")
#train_set = dump_load( "trainLoader")
train_loader = DataLoader(train_set,
                          batch_size=1,
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
    print('Claim date available')
    print(c[5])
    print('Snippet buckets')
    print(c[6])
'''