import sys

from allennlp.predictors.predictor import Predictor as pred
import pickle
import os

import spacy

import Claim

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
import spacy


class NUS(Dataset):
    # change root to your root
    def __init__(self, mode,path,pathToSave,domain,number):
        super().__init__()
        assert mode in ['Train', 'Dev', 'Test']
        self.domain = domain
        print('Loading {} set...'.format(mode))
        self.mode = mode
        print('Get buckets of differences')
        self.getDifferences('verschilDays-buckets.txt')
        print('Get buckets of absolute differences')
        self.getDifferencesAbsolute('../uitbreiding2-verschilTijdTekst/verschilDays-buckets-tekst.txt')
        print('Get buckets of durations')
        self.getDurations('verschilDays-buckets-durations.txt')
        print('Reading the claims')
        self.getClaims(mode,path)
        print('Done')
        pathToSave = pathToSave + domain
        if not os.path.exists(pathToSave):
            os.mkdir(pathToSave)
        pathToSave = pathToSave + "/" + number
        if not os.path.exists(pathToSave):
            os.mkdir(pathToSave)
        pathToSave += "/"
        with open(pathToSave+"claimIds", 'wb') as f:
            pickle.dump(self.claimIds, f)
        with open(pathToSave+"documents", 'wb') as f:
            pickle.dump(self.documents, f)
        with open(pathToSave+"snippetDocs", 'wb') as f:
            pickle.dump(self.snippetDocs, f)
        with open(pathToSave+"metadata", 'wb') as f:
            pickle.dump(self.metadata, f)
        with open(pathToSave+"labels", 'wb') as f:
            pickle.dump(self.labels, f)
        with open(pathToSave+"claimDateAvaialble", 'wb') as f:
            pickle.dump(self.claimDateAvaialble, f)
        with open(pathToSave+"bucketsSnippets", 'wb') as f:
            pickle.dump(self.bucketsSnippets, f)
        with open(pathToSave+"verbIndices", 'wb') as f:
            pickle.dump(self.verbIndices, f)
        with open(pathToSave+"timeIndices", 'wb') as f:
            pickle.dump(self.timeIndices, f)
        with open(pathToSave+"positions", 'wb') as f:
            pickle.dump(self.positions, f)
        with open(pathToSave+"refsIndices", 'wb') as f:
            pickle.dump(self.refsIndices, f)
        with open(pathToSave+"time", 'wb') as f:
            pickle.dump(self.time, f)
        with open(pathToSave+"verbIndicesSnippets", 'wb') as f:
            pickle.dump(self.verbIndicesSnippets, f)
        with open(pathToSave+"timeIndicesSnippets", 'wb') as f:
            pickle.dump(self.timeIndicesSnippets, f)
        with open(pathToSave+"positionsSnippets", 'wb') as f:
            pickle.dump(self.positionsSnippets, f)
        with open(pathToSave+"refsIndicesSnippets", 'wb') as f:
            pickle.dump(self.refsIndicesSnippets, f)
        with open(pathToSave+"timeSnippets", 'wb') as f:
            pickle.dump(self.timeSnippets, f)

    def getDifferences(self, path):
        self.buckets = list()
        with open(path, 'r', encoding='utf-8') as file:
            for line in file:
                bucket = line.split('\t')
                if bucket[0]=="-infinity":
                    self.buckets.append(tuple([float('-inf'),int(bucket[1])]))
                else:
                    if bucket[1] == "infinity":
                        self.buckets.append(tuple([int(bucket[0]), float('inf')]))
                    else:
                        self.buckets.append(tuple([int(bucket[0]), int(bucket[1])]))

    def getDurations(self, path):
        self.durations = list()
        with open(path, 'r', encoding='utf-8') as file:
            for line in file:
                bucket = line.split('\t')
                if bucket[1] == "infinity":
                    self.durations.append(tuple([int(bucket[0]), float('inf')]))
                else:
                    self.durations.append(tuple([int(bucket[0]), int(bucket[1])]))

    def getDifferencesAbsolute(self, path):
        self.bucketsAbsolute = list()
        with open(path, 'r', encoding='utf-8') as file:
            for line in file:
                bucket = line.split('\t')
                if bucket[0]=="-infinity":
                    self.bucketsAbsolute.append(tuple([float('-inf'),int(bucket[1])]))
                else:
                    if bucket[1] == "infinity":
                        self.bucketsAbsolute.append(tuple([int(bucket[0]), float('inf')]))
                    else:
                        self.bucketsAbsolute.append(tuple([int(bucket[0]), int(bucket[1])]))

    def matchBucket(self, difference):
        for i in range(len(self.buckets)):
            if self.buckets[i][0]<=difference<=self.buckets[i][1]:
                return i

    def matchBucketAbsoluteTime(self, difference):
        for i in range(len(self.bucketsAbsolute)):
            if self.bucketsAbsolute[i][0]<=difference<=self.bucketsAbsolute[i][1]:
                return str(i)

    def matchBucketRelativeTime(self, relativeTime,difference):
        '''

        past = 0
        present = 1
        future = 2
        no = -1

        '''
        if relativeTime == "PRESENT_REF":
            if difference <0:
                return str(0)
            else:
                if difference == 0:
                    return str(1)
                else:
                    return str(2)
        if relativeTime == "PAST_REF":
            if difference <0:
                return str(0)
            else:
                if difference == 0:
                    return str(0)
                else:
                    return str(-1)
        if relativeTime == "FUTURE_REF":
            if difference <0:
                return str(-1)
            else:
                if difference == 0:
                    return str(1)
                else:
                    return str(1)
        return relativeTime

    def matchBucketDuur(self, duur):
        calculation = {"Y": 31556926,
                       "M": 2629746,
                       "W": 604800,
                       "D": 86400,
                       "H": 3600,
                       "Min": 60,
                       "WE": 172800,
                       "Q": 15778476,
                       "DE": 315569260,
                       "S": 1,
                       "CE": 3155692600}
        result = 0
        if duur[0:2] == "PT":
            if duur[-1] == "M":
                symbol = "Min"
            else:
                symbol = duur[-1]
            if duur[2:-1] == "X":
                value = 1
            else:
                value = int(duur[2:-1])
            result = value * calculation[symbol]
        else:
            if duur[0] == "P":
                if duur[-2:] == "WE" or duur[-2:] == "DE" or duur[-2:] == "CE":
                    symbol = duur[-2:]
                    if duur[1:-2] == "X":
                        value = 1
                    else:
                        value = int(duur[1:-2])
                    result = value * calculation[symbol]
                else:
                    symbol = duur[-1]
                    if duur[1:-1] == "X":
                        value = 1
                    else:
                        value = int(duur[1:-1])
                    result = value * calculation[symbol]
        if result >0:
            for i in range(len(self.durations)):
                if self.durations[i][0]<=result<=self.durations[i][1]:
                    return str(i)
        else:
            return "None"

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
        self.verbIndices = []
        self.timeIndices = []
        self.positions = []
        self.refsIndices = []
        self.time = []
        self.verbIndicesSnippets = []
        self.timeIndicesSnippets = []
        self.positionsSnippets = []
        self.refsIndicesSnippets = []
        self.timeSnippets = []

        predictorNER = pred.from_path(
            "https://storage.googleapis.com/allennlp-public-models/ner-elmo.2021-02-12.tar.gz", cuda_device=-1)
        nlp = spacy.load("en_core_web_sm")
        nlpOIE = spacy.load("en_core_web_trf")
        predictorOIE = "None"
        coreference = "None"
        tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-distilroberta-v1')

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
                    verbsSnippetIndices =''
                    timeSnippetIndices = ''
                    positionSnippetIndices =''
                    refsSnippetIndices =''
                    timeSnippetElements =''
                    verschilSnippets = []

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
                                    verschilSnippets.append(verschil)
                                else:
                                    verschil = (snippet.publishTime-claim.claimDate).days
                                    bucketsSnippetClaim += '\t' + str(self.matchBucket(verschil))
                                    verschilSnippets.append(verschil)
                            else:
                                if snippet.publishTime != None:
                                    verschilSnippets.append("None")
                                    bucketsSnippetClaim += '\t' + str(20)
                                else:
                                    bucketsSnippetClaim += '\t' + str(21)
                                    verschilSnippets.append("None")
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
                                        verschilSnippets.append(verschil)
                                    else:
                                        if (snippet.publishTime[1] - claim.claimDate[0]).days <= 0:
                                            verschil = (snippet.publishTime[1] - claim.claimDate[0]).days
                                            bucketsSnippetClaim +=  '\t' + str(self.matchBucket(verschil))
                                            verschilSnippets.append(verschil)
                                        else:
                                            verschil = 0
                                            bucketsSnippetClaim +=  '\t' + str(self.matchBucket(verschil))
                                            verschilSnippets.append(verschil)
                                else:
                                    if ((snippet.publishTime - claim.claimDate[0]).days >= 0 and (
                                            snippet.publishTime - claim.claimDate[1]).days <= 0):
                                        verschil = 0
                                        bucketsSnippetClaim += '\t' + str(self.matchBucket(verschil))
                                        verschilSnippets.append(verschil)
                                    else:
                                        if (snippet.publishTime - claim.claimDate[0]).days <= 0:
                                            verschil = (snippet.publishTime - claim.claimDate[0]).days
                                        else:
                                            verschil = (snippet.publishTime - claim.claimDate[1]).days
                                        bucketsSnippetClaim +=  '\t' + str(self.matchBucket(verschil))
                                        verschilSnippets.append(verschil)
                            else:
                                if snippet.publishTime != None:
                                    bucketsSnippetClaim += '\t' + str(20)
                                    verschilSnippets.append("None")
                                else:
                                    bucketsSnippetClaim += '\t' + str(21)
                                    verschilSnippets.append("None")

                    # print(claim.claimID + '/' + 'claim')
                    input_ids = [i for i in tokenizer.encode(
                        text=getClaimText(claim))]
                    #print("Tekst claim")
                    #print(getClaimText(claim))
                    decoding = [tokenizer.decode([i]) for i in input_ids]
                    #print("Dedoding")
                    #print(decoding)
                    # print(len(input_ids))
                    startIndex = claim.getIndex()
                    OIE = claim.readOpenInformationExtraction()
                    indicesAll, tagsAll, positionsAll, allWords, zinnenIndices = ProcessOpenInformation(
                        OIE, decoding, startIndex, nlpOIE)
                    verbClaim = ""
                    for verb in tagsAll['Verbs']:
                        for part in verb:
                            verbClaim += str(part) +"\t"
                    verbClaim = verbClaim[:-1]
                    self.verbIndices.append(verbClaim)
                    '''
                    positionsClaim = ""
                    for position in positionsAll:
                        positionsClaim+= str(position[0])+','+str(position[1]) + "\t"
                    '''
                    positionsClaim = ""
                    timeClaim = ""
                    if not type(claim.claimDate) is tuple:
                        datasM, durenM, refsM, setsM = claim.readTime()
                        refsIndices = []
                        tokenisation = []
                        startIndex = claim.getIndexHeidel()
                        for data in datasM:
                            if type(data) != int:
                                if type(data) == str:
                                    tokenisation += [token for token in nlp.tokenizer(str(data))]
                                else:
                                    startItem = len(tokenisation) - startIndex
                                    tokenisation += [token for token in nlp.tokenizer(str(data[1]))]
                                    endItem = len(tokenisation) - startIndex - 1
                                    refsIndices.append([startItem, endItem])
                        tokenisation = tokenisation[startIndex:]
                        indicesRefs = match(indicesAll, allWords, tokenisation, startIndex == 6,
                                            refsIndices)
                        index = 0
                        tijdClaim = ""
                        indicesRefsClaim = ""
                        for time in tagsAll['Time']:
                            if len(indicesRefs) <= index:
                                for position in positionsAll:
                                    if time[0] == position[0] + position[1]:
                                        positionsClaim += str(position[0]) + ',' + str(
                                            position[1]) + "\t"
                                for part in time:
                                    tijdClaim += str(part) + "\t"
                            else:
                                if time[-1] < indicesRefs[index][0]:
                                    for position in positionsAll:
                                        if time[0] == position[0] + position[1]:
                                            positionsClaim += str(position[0]) + ',' + str(
                                                position[1]) + "\t"
                                    for part in time:
                                        tijdClaim += str(part) + "\t"
                                else:
                                    if time[0] > indicesRefs[index][-1]:
                                        while time[0] > indicesRefs[index][-1]:
                                            indicesRefsClaim += str(indicesRefs[index][0]) + "\t"
                                            for timeI in range(indicesRefs[index][0],indicesRefs[index][-1]+1):
                                                tijdClaim += str(timeI) + "\t"
                                            index += 1
                                            if len(indicesRefs) <= index:
                                                break
                                        if len(indicesRefs) <= index:
                                            for position in positionsAll:
                                                if time[0] == position[0] + position[1]:
                                                    positionsClaim += str(position[0]) + ',' + str(
                                                        position[1]) + "\t"
                                            for part in time:
                                                tijdClaim += str(part) + "\t"
                                        else:
                                            if time[-1] < indicesRefs[index][0]:
                                                for position in positionsAll:
                                                    if time[0] == position[0] + position[1]:
                                                        positionsClaim += str(position[0]) + ',' + str(
                                                            position[1]) + "\t"
                                                for part in time:
                                                    tijdClaim += str(part) + "\t"
                                            else:
                                                indicesRefsClaim += str(min(indicesRefs[index][0], time[0])) + "\t"
                                                for position in positionsAll:
                                                    if time[0]== position[0] + position[1]:
                                                        second =  min(indicesRefs[index][0], time[0])-position[0]
                                                        positionsClaim += str(position[0]) + ',' + str(second) + "\t"
                                                for timeI in range(min(indicesRefs[index][0], time[0]),
                                                                   max(indicesRefs[index][1]+1, time[-1]+1)):
                                                    tijdClaim += str(timeI) + "\t"
                                                index += 1
                                    else:
                                        for position in positionsAll:
                                            if time[0] == position[0] + position[1]:
                                                second = min(indicesRefs[index][0], time[0]) - position[0]
                                                positionsClaim += str(position[0]) + ',' + str(second) + "\t"
                                        indicesRefsClaim += str(min(indicesRefs[index][0],time[0])) + "\t"
                                        for timeI in range(min(indicesRefs[index][0],time[0]),max(indicesRefs[index][1]+1,time[-1]+1)):
                                            tijdClaim += str(timeI) + "\t"
                                        index += 1
                        while index<len(indicesRefs):
                            indicesRefsClaim += str(indicesRefs[index][0]) + "\t"
                            for timeI in range(indicesRefs[index][0], indicesRefs[index][-1]+1):
                                tijdClaim += str(timeI) + "\t"
                            index += 1
                        positionsClaim = positionsClaim[:-1]
                        self.positions.append(positionsClaim)
                        tijdClaim = tijdClaim[:-1]
                        self.timeIndices.append(tijdClaim)
                        '''
                        for indices in indicesRefs:
                            indicesRefsClaim += str(indices[0]) +"\t"
                        '''
                        indicesRefsClaim = indicesRefsClaim[:-1]
                        #print(indicesRefsClaim)
                        self.refsIndices.append(indicesRefsClaim)

                        for data in datasM:
                            if type(data) == list:
                                if data[0] in durenM:
                                    if self.matchBucketDuur(data[0]) != "None":
                                        timeClaim += "Duur-"+self.matchBucketDuur(data[0]) + "\t"
                                else:
                                    if data[0] in refsM:
                                        timeClaim += "Refs-"+self.matchBucketRelativeTime(data[0],difference=0) + "\t"
                                    else:
                                        if data[0] in setsM:
                                            if self.matchBucketDuur(data[0]) != "None":
                                                timeClaim += "Duur-" + self.matchBucketDuur(data[0]) + "\t"
                                        else:
                                            if (claim.claimDate != None):
                                                if type(data[0]) is tuple:
                                                    if ((data[0][
                                                             0] - claim.claimDate).days <= 0 and (
                                                            data[0][
                                                                1] - claim.claimDate).days >= 0):
                                                        verschil = 0
                                                    else:
                                                        if (data[0][1] - claim.claimDate).days <= 0:
                                                            verschil = (data[0][
                                                                            1] - claim.claimDate).days
                                                        else:
                                                            verschil = (data[0][
                                                                            0] - claim.claimDate).days
                                                    timeClaim += str(self.matchBucketAbsoluteTime(verschil)) + "\t"
                                                else:
                                                    verschil = (data[0]-claim.claimDate).days
                                                    timeClaim += str(self.matchBucketAbsoluteTime(verschil)) + "\t"
                        timeClaim = timeClaim[:-1]
                        self.time.append(timeClaim)
                    else:
                        datasM, durenM, refsM, setsM = claim.readTime()
                        refsIndices = []
                        tokenisation = []
                        startIndex = claim.getIndexHeidel()
                        for data in datasM:
                            if type(data) == str:
                                tokenisation += [token for token in nlp.tokenizer(str(data))]
                            else:
                                startItem = len(tokenisation) - startIndex
                                tokenisation += [token for token in nlp.tokenizer(str(data[1]))]
                                endItem = len(tokenisation) - startIndex - 1
                                refsIndices.append([startItem, endItem])
                        tokenisation = tokenisation[startIndex:]
                        indicesRefs = match(indicesAll, allWords, tokenisation, startIndex == 6,
                                            refsIndices)
                        index = 0
                        tijdClaim = ""
                        indicesRefsClaim = ""
                        for time in tagsAll['Time']:
                            if len(indicesRefs) <= index:
                                for position in positionsAll:
                                    if time[0] == position[0] + position[1]:
                                        positionsClaim += str(position[0]) + ',' + str(
                                            position[1]) + "\t"
                                for part in time:
                                    tijdClaim += str(part) + "\t"
                            else:
                                if time[-1] < indicesRefs[index][0]:
                                    for position in positionsAll:
                                        if time[0] == position[0] + position[1]:
                                            positionsClaim += str(position[0]) + ',' + str(
                                                position[1]) + "\t"
                                    for part in time:
                                        tijdClaim += str(part) + "\t"
                                else:
                                    if time[0] > indicesRefs[index][-1]:
                                        while time[0] > indicesRefs[index][-1]:
                                            indicesRefsClaim += str(indicesRefs[index][0]) + "\t"
                                            for timeI in range(indicesRefs[index][0], indicesRefs[index][-1] + 1):
                                                tijdClaim += str(timeI) + "\t"
                                            index += 1
                                            if len(indicesRefs) <= index:
                                                break
                                        if len(indicesRefs) <= index:
                                            for position in positionsAll:
                                                if time[0] == position[0] + position[1]:
                                                    positionsClaim += str(position[0]) + ',' + str(
                                                        position[1]) + "\t"
                                            for part in time:
                                                tijdClaim += str(part) + "\t"
                                        else:
                                            if time[-1] < indicesRefs[index][0]:
                                                for position in positionsAll:
                                                    if time[0] == position[0] + position[1]:
                                                        positionsClaim += str(position[0]) + ',' + str(
                                                            position[1]) + "\t"
                                                for part in time:
                                                    tijdClaim += str(part) + "\t"
                                            else:
                                                indicesRefsClaim += str(min(indicesRefs[index][0], time[0])) + "\t"
                                                for position in positionsAll:
                                                    if time[0] == position[0] + position[1]:
                                                        second = min(indicesRefs[index][0], time[0]) - position[0]
                                                        positionsClaim += str(position[0]) + ',' + str(
                                                            second) + "\t"
                                                for timeI in range(min(indicesRefs[index][0], time[0]),
                                                                   max(indicesRefs[index][1] + 1, time[-1] + 1)):
                                                    tijdClaim += str(timeI) + "\t"
                                                index += 1
                                    else:
                                        for position in positionsAll:
                                            if time[0] == position[0] + position[1]:
                                                second = min(indicesRefs[index][0], time[0]) - position[0]
                                                positionsClaim += str(position[0]) + ',' + str(second) + "\t"
                                        indicesRefsClaim += str(min(indicesRefs[index][0], time[0])) + "\t"
                                        for timeI in range(min(indicesRefs[index][0], time[0]),
                                                           max(indicesRefs[index][1] + 1, time[-1] + 1)):
                                            tijdClaim += str(timeI) + "\t"
                                        index += 1
                        while index < len(indicesRefs):
                            indicesRefsClaim += str(indicesRefs[index][0]) + "\t"
                            for timeI in range(indicesRefs[index][0], indicesRefs[index][-1] + 1):
                                tijdClaim += str(timeI) + "\t"
                            index += 1
                        positionsClaim = positionsClaim[:-1]
                        self.positions.append(positionsClaim)
                        tijdClaim = tijdClaim[:-1]
                        self.timeIndices.append(tijdClaim)
                        '''
                        for indices in indicesRefs:
                            indicesRefsClaim += str(indices[0]) +"\t"
                        '''
                        indicesRefsClaim = indicesRefsClaim[:-1]
                        # print(indicesRefsClaim)
                        self.refsIndices.append(indicesRefsClaim)
                        for data in datasM:
                            if type(data) == list:
                                if data[0] in durenM:
                                    if self.matchBucketDuur(data[0]) != "None":
                                        timeClaim += "Duur-"+self.matchBucketDuur(data[0]) + "\t"
                                else:
                                    if data[0] in refsM:
                                        timeClaim += "Refs-"+self.matchBucketRelativeTime(data[0],difference=0) + "\t"
                                    else:
                                        if data[0] in setsM:
                                            timeClaim += "Duur-" + self.matchBucketRelativeTime(data[0],
                                                                                                difference=0) + "\t"
                                        else:
                                            if type(data[0]) is tuple:
                                                if (data[0][0] - claim.claimDate[1]).days >= 0:
                                                    verschil = (data[0][0] - claim.claimDate[
                                                        1]).days
                                                else:
                                                    if (data[0][1] - claim.claimDate[0]).days <= 0:
                                                        verschil = (data[0][1] - claim.claimDate[
                                                            0]).days
                                                    else:
                                                        verschil = 0
                                                timeClaim += str(self.matchBucketAbsoluteTime(verschil)) + "\t"
                                            else:
                                                if ((data[0] - claim.claimDate[0]).days >= 0 and (
                                                        data[0] - claim.claimDate[1]).days <= 0):
                                                    verschil = 0
                                                else:
                                                    if (data[0] - claim.claimDate[0]).days <= 0:
                                                        verschil = (data[0] - claim.claimDate[
                                                            0]).days
                                                    else:
                                                        verschil = (data[0] - claim.claimDate[
                                                            1]).days
                                                timeClaim += str(self.matchBucketAbsoluteTime(verschil)) + "\t"
                        timeClaim = timeClaim[:-1]
                        self.time.append(timeClaim)
                    for snippet in claim.snippets:
                        input_ids = [i for i in tokenizer.encode(
                            text=getSnippetText(snippet))]
                        # print(len(input_ids))
                        decoding = [tokenizer.decode([i]) for i in input_ids]
                        OIE = snippet.readOpenInformationExtraction()
                        startIndex = snippet.getIndex()
                        indicesAll, tagsAll, positionsAll, allWords, zinnenIndices = ProcessOpenInformation(
                            OIE, decoding, startIndex, nlpOIE)
                        if len(tagsAll['Verbs']) > 0:
                            for verb in tagsAll['Verbs']:
                                for part in verb:
                                    verbsSnippetIndices += str(part) + "\t"
                            verbsSnippetIndices = verbsSnippetIndices[:-1]
                        verbsSnippetIndices += ' 0123456789 '
                        '''
                        if len(positionsAll) > 0:
                            for position in positionsAll:
                                positionSnippetIndices += str(position[0]) + ',' + str(position[1]) + "\t"
                            positionSnippetIndices = positionSnippetIndices[:-1]
                        positionSnippetIndices += ' 0123456789 '
                        '''
                        if not type(claim.claimDate) is tuple:
                            datasM, durenM, refsM, setsM = snippet.readTime()
                            tokenisation = []
                            refsIndices = []
                            startIndex = snippet.getIndexHeidel()
                            for data in datasM:
                                # print(data)
                                if type(data) != int:
                                    if type(data) == str:
                                        tokenisation += [token for token in nlp.tokenizer(str(data))]
                                    else:
                                        startItem = len(tokenisation) - startIndex
                                        tokenisation += [token for token in nlp.tokenizer(str(data[1]))]
                                        endItem = len(tokenisation) - startIndex - 1
                                        refsIndices.append([startItem, endItem])
                            tokenisation = tokenisation[startIndex:]
                            # print(refsIndices)
                            indicesRefs = match(indicesAll, allWords, tokenisation, startIndex == 6,
                                                refsIndices)
                            index = 0
                            #print(tagsAll['Time'])
                            for time in tagsAll['Time']:
                                if len(indicesRefs) <= index:
                                    for position in positionsAll:
                                        if time[0] == position[0] + position[1]:
                                            positionSnippetIndices += str(position[0]) + ',' + str(
                                                position[1]) + "\t"
                                    for part in time:
                                        timeSnippetIndices += str(part) + "\t"
                                else:
                                    if time[-1] < indicesRefs[index][0]:
                                        for position in positionsAll:
                                            if time[0] == position[0] + position[1]:
                                                positionSnippetIndices += str(position[0]) + ',' + str(
                                                    position[1]) + "\t"
                                        for part in time:
                                            timeSnippetIndices += str(part) + "\t"
                                    else:
                                        if time[0] > indicesRefs[index][-1]:
                                            while time[0] > indicesRefs[index][-1]:
                                                refsSnippetIndices += str(indicesRefs[index][0]) + "\t"
                                                for timeI in range(indicesRefs[index][0], indicesRefs[index][-1] + 1):
                                                    timeSnippetIndices += str(timeI) + "\t"
                                                index += 1
                                                if len(indicesRefs) <= index:
                                                    break
                                            if len(indicesRefs) <= index:
                                                for position in positionsAll:
                                                    if time[0] == position[0] + position[1]:
                                                        positionSnippetIndices += str(position[0]) + ',' + str(
                                                            position[1]) + "\t"
                                                for part in time:
                                                    timeSnippetIndices += str(part) + "\t"
                                            else:
                                                if time[-1] < indicesRefs[index][0]:
                                                    for position in positionsAll:
                                                        if time[0] == position[0] + position[1]:
                                                            positionSnippetIndices += str(position[0]) + ',' + str(
                                                                position[1]) + "\t"
                                                    for part in time:
                                                        timeSnippetIndices += str(part) + "\t"
                                                else:
                                                    refsSnippetIndices += str(
                                                        min(indicesRefs[index][0], time[0])) + "\t"
                                                    for position in positionsAll:
                                                        if time[0] == position[0] + position[1]:
                                                            second = min(indicesRefs[index][0], time[0]) - position[
                                                                0]
                                                            positionSnippetIndices += str(position[0]) + ',' + str(
                                                                second) + "\t"
                                                    for timeI in range(min(indicesRefs[index][0], time[0]),
                                                                       max(indicesRefs[index][1] + 1, time[-1] + 1)):
                                                        timeSnippetIndices += str(timeI) + "\t"
                                                    index += 1
                                        else:
                                            for position in positionsAll:
                                                if time[0] == position[0] + position[1]:
                                                    second = min(indicesRefs[index][0], time[0]) - position[0]
                                                    positionSnippetIndices += str(position[0]) + ',' + str(second) + "\t"
                                            refsSnippetIndices += str(min(indicesRefs[index][0], time[0])) + "\t"
                                            for timeI in range(min(indicesRefs[index][0], time[0]),
                                                               max(indicesRefs[index][1] + 1, time[-1] + 1)):
                                                timeSnippetIndices += str(timeI) + "\t"
                                            index += 1
                            while index < len(indicesRefs):
                                refsSnippetIndices += str(indicesRefs[index][0]) + "\t"
                                for timeI in range(indicesRefs[index][0], indicesRefs[index][-1] + 1):
                                    timeSnippetIndices += str(timeI) + "\t"
                                index += 1
                            if (len(positionSnippetIndices) > 0 and positionSnippetIndices[-1] == "\t"):
                                positionSnippetIndices = positionSnippetIndices[:-1]
                            positionSnippetIndices += ' 0123456789 '
                            if (len(timeSnippetIndices) > 0 and timeSnippetIndices[-1] == "\t"):
                                timeSnippetIndices = timeSnippetIndices[:-1]
                            timeSnippetIndices += ' 0123456789 '
                            if (len(refsSnippetIndices) > 0 and refsSnippetIndices[-1] == "\t"):
                                refsSnippetIndices = refsSnippetIndices[:-1]
                            refsSnippetIndices += ' 0123456789 '
                            # print('IndicesRefs')
                            # print(indicesRefs)
                            #verschilSnippet = verschilSnippets[0]
                            #verschilSnippets = verschilSnippets[1:]
                            '''
                            index = 0

                            for time in tagsAll['Time']:
                                for part in time:
                                    if len(indicesRefs) <= index:
                                        timeSnippetIndices += str(part) + "\t"
                                    else:
                                        # print('Part : ' + str(part))
                                        # print('IndicesRefs : ' + str(indicesRefs[index][0]))
                                        if part < indicesRefs[index][0]:
                                            timeSnippetIndices += str(part) + "\t"
                                        else:
                                            if part == indicesRefs[index][0]:
                                                timeSnippetIndices += str(part) + "\t"
                                                index += 1
                                            else:
                                                for timeI in range(indicesRefs[index][0], indicesRefs[index][1]):
                                                    timeSnippetIndices += str(timeI) + "\t"
                                                    index += 1
                            if (len(timeSnippetIndices) > 0 and timeSnippetIndices[-1] == "\t"):
                                timeSnippetIndices = timeSnippetIndices[:-1]
                            timeSnippetIndices += ' 0123456789 '
                            for indices in indicesRefs:
                                refsSnippetIndices += str(indices[0]) + "\t"
                            if (len(refsSnippetIndices) > 0 and refsSnippetIndices[-1] == "\t"):
                                refsSnippetIndices = refsSnippetIndices[:-1]
                            refsSnippetIndices += ' 0123456789 '
                            # print('IndicesRefs')
                            # print(indicesRefs)
                            '''
                            verschilSnippet = verschilSnippets[0]
                            verschilSnippets = verschilSnippets[1:]
                            for data in datasM:
                                if type(data) == list:
                                    if data[0] in durenM:
                                        if self.matchBucketDuur(data[0]) != "None":
                                            timeSnippetElements += "Duur-" + self.matchBucketDuur(data[0]) + "\t"
                                    else:
                                        if data[0] in refsM:
                                            if verschilSnippet != "None":
                                                timeSnippetElements += "Refs-" + self.matchBucketRelativeTime(
                                                    data[0], difference=verschilSnippet) + "\t"
                                        else:
                                            if data[0] in setsM:
                                                if self.matchBucketDuur(data[0]) != "None":
                                                    timeSnippetElements += "Duur-" + self.matchBucketDuur(
                                                        data[0]) + "\t"
                                            else:
                                                if (claim.claimDate != None):
                                                    if type(data[0]) is tuple:
                                                        if ((data[0][
                                                                 0] - claim.claimDate).days <= 0 and (
                                                                data[0][
                                                                    1] - claim.claimDate).days >= 0):
                                                            verschil = 0
                                                        else:
                                                            if (data[0][
                                                                    1] - claim.claimDate).days <= 0:
                                                                verschil = (data[0][
                                                                                1] - claim.claimDate).days
                                                            else:
                                                                verschil = (data[0][
                                                                                0] - claim.claimDate).days
                                                        timeSnippetElements += str(
                                                            self.matchBucketAbsoluteTime(verschil)) + "\t"
                                                    else:
                                                        verschil = (data[0] - claim.claimDate).days
                                                        timeSnippetElements += str(
                                                            self.matchBucketAbsoluteTime(verschil)) + "\t"
                            if len(timeSnippetElements) > 0 and timeSnippetElements[-1] == "\t":
                                timeSnippetElements = timeSnippetElements[:-1]
                            timeSnippetElements += ' 0123456789 '
                        else:
                            datasM, durenM, refsM, setsM = snippet.readTime()
                            # print(snippet.getSnippetText())
                            tokenisation = []
                            refsIndices = []
                            startIndex = snippet.getIndexHeidel()
                            for data in datasM:
                                # print(data)
                                if type(data) != int:
                                    if type(data) == str:
                                        tokenisation += [token for token in nlp.tokenizer(str(data))]
                                    else:
                                        startItem = len(tokenisation) - startIndex
                                        tokenisation += [token for token in nlp.tokenizer(str(data[1]))]
                                        endItem = len(tokenisation) - startIndex - 1
                                        refsIndices.append([startItem, endItem])
                            tokenisation = tokenisation[startIndex:]
                            # print(refsIndices)
                            indicesRefs = match(indicesAll, allWords, tokenisation, startIndex == 6,
                                                refsIndices)
                            index = 0
                            for time in tagsAll['Time']:
                                if len(indicesRefs) <= index:
                                    for position in positionsAll:
                                        if time[0] == position[0] + position[1]:
                                            positionSnippetIndices += str(position[0]) + ',' + str(
                                                position[1]) + "\t"
                                    for part in time:
                                        timeSnippetIndices += str(part) + "\t"
                                else:
                                    if time[-1] < indicesRefs[index][0]:
                                        for position in positionsAll:
                                            if time[0] == position[0] + position[1]:
                                                positionSnippetIndices += str(position[0]) + ',' + str(
                                                    position[1]) + "\t"
                                        for part in time:
                                            timeSnippetIndices += str(part) + "\t"
                                    else:
                                        if time[0] > indicesRefs[index][-1]:
                                            while time[0] > indicesRefs[index][-1]:
                                                refsSnippetIndices += str(indicesRefs[index][0]) + "\t"
                                                for timeI in range(indicesRefs[index][0],
                                                                   indicesRefs[index][-1] + 1):
                                                    timeSnippetIndices += str(timeI) + "\t"
                                                index += 1
                                                if len(indicesRefs) <= index:
                                                    break
                                            if len(indicesRefs) <= index:
                                                for position in positionsAll:
                                                    if time[0] == position[0] + position[1]:
                                                        positionSnippetIndices += str(position[0]) + ',' + str(
                                                            position[1]) + "\t"
                                                for part in time:
                                                    timeSnippetIndices += str(part) + "\t"
                                            else:
                                                if time[-1] < indicesRefs[index][0]:
                                                    for position in positionsAll:
                                                        if time[0] == position[0] + position[1]:
                                                            positionSnippetIndices += str(position[0]) + ',' + str(
                                                                position[1]) + "\t"
                                                    for part in time:
                                                        timeSnippetIndices += str(part) + "\t"
                                                else:
                                                    refsSnippetIndices += str(
                                                        min(indicesRefs[index][0], time[0])) + "\t"
                                                    for position in positionsAll:
                                                        if time[0] == position[0] + position[1]:
                                                            second = min(indicesRefs[index][0], time[0]) - position[
                                                                0]
                                                            positionSnippetIndices += str(position[0]) + ',' + str(
                                                                second) + "\t"
                                                    for timeI in range(min(indicesRefs[index][0], time[0]),
                                                                       max(indicesRefs[index][1] + 1, time[-1] + 1)):
                                                        timeSnippetIndices += str(timeI) + "\t"
                                                    index += 1
                                        else:
                                            for position in positionsAll:
                                                if time[0] == position[0] + position[1]:
                                                    second = min(indicesRefs[index][0], time[0]) - position[0]
                                                    positionSnippetIndices += str(position[0]) + ',' + str(
                                                        second) + "\t"
                                            refsSnippetIndices += str(min(indicesRefs[index][0], time[0])) + "\t"
                                            for timeI in range(min(indicesRefs[index][0], time[0]),
                                                               max(indicesRefs[index][1] + 1, time[-1] + 1)):
                                                timeSnippetIndices += str(timeI) + "\t"
                                            index += 1
                            while index < len(indicesRefs):
                                refsSnippetIndices += str(indicesRefs[index][0]) + "\t"
                                for timeI in range(indicesRefs[index][0], indicesRefs[index][-1] + 1):
                                    timeSnippetIndices += str(timeI) + "\t"
                                index += 1
                            if (len(positionSnippetIndices) > 0 and positionSnippetIndices[-1] == "\t"):
                                positionSnippetIndices = positionSnippetIndices[:-1]
                            positionSnippetIndices += ' 0123456789 '
                            if (len(timeSnippetIndices) > 0 and timeSnippetIndices[-1] == "\t"):
                                timeSnippetIndices = timeSnippetIndices[:-1]
                            timeSnippetIndices += ' 0123456789 '
                            if (len(refsSnippetIndices) > 0 and refsSnippetIndices[-1] == "\t"):
                                refsSnippetIndices = refsSnippetIndices[:-1]
                            refsSnippetIndices += ' 0123456789 '
                            verschilSnippet = verschilSnippets[0]
                            verschilSnippets = verschilSnippets[1:]
                            for data in datasM:
                                if type(data) != int:
                                    if type(data[0]) != str:
                                        if data[0] in durenM:
                                            if self.matchBucketDuur(data[0]) != "None":
                                                timeSnippetElements += "Duur-" + self.matchBucketDuur(
                                                    data[0]) + "\t"
                                        else:
                                            if data[0] in refsM:
                                                if verschilSnippet != "None":
                                                    timeSnippetElements += "Refs-" + self.matchBucketRelativeTime(
                                                        data[0], difference=verschilSnippet) + "\t"
                                            else:
                                                if data[0] in setsM:
                                                    if self.matchBucketDuur(data[0]) != "None":
                                                        timeSnippetElements += "Duur-" + self.matchBucketRelativeTime(
                                                            data[0],
                                                            difference=0) + "\t"
                                                else:
                                                    if type(data[0]) is tuple:
                                                        if (data[0][0] - claim.claimDate[1]).days >= 0:
                                                            verschil = (data[0][0] - claim.claimDate[
                                                                1]).days
                                                            timeSnippetElements += str(
                                                                self.matchBucketAbsoluteTime(verschil)) + "\t"
                                                        else:
                                                            if (data[0][1] - claim.claimDate[
                                                                0]).days <= 0:
                                                                verschil = (data[0][1] -
                                                                            claim.claimDate[0]).days
                                                                timeSnippetElements += str(
                                                                    self.matchBucketAbsoluteTime(verschil)) + "\t"
                                                            else:
                                                                verschil = 0
                                                                timeSnippetElements += str(
                                                                    self.matchBucketAbsoluteTime(verschil)) + "\t"
                                                    else:
                                                        if ((data[0] - claim.claimDate[
                                                            0]).days >= 0 and (
                                                                data[0] - claim.claimDate[
                                                            1]).days <= 0):
                                                            verschil = 0
                                                        else:
                                                            if (data[0] - claim.claimDate[0]).days <= 0:
                                                                verschil = (data[0] - claim.claimDate[
                                                                    0]).days
                                                            else:
                                                                verschil = (data[0] - claim.claimDate[
                                                                    1]).days
                                                        timeSnippetElements += str(
                                                            self.matchBucketAbsoluteTime(verschil)) + "\t"
                            if len(timeSnippetElements) > 0 and timeSnippetElements[-1] == "\t":
                                timeSnippetElements = timeSnippetElements[:-1]
                            timeSnippetElements += ' 0123456789 '
                    self.timeSnippets.append(timeSnippetElements)
                    self.refsIndicesSnippets.append(refsSnippetIndices)
                    self.verbIndicesSnippets.append(verbsSnippetIndices)
                    self.positionsSnippets.append(positionSnippetIndices)
                    self.timeIndicesSnippets.append(timeSnippetIndices)
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
                    '''
                    except:
                        f = open("faultDataSetIteratie2.txt", "a", encoding="utf-8")
                        f.write(claim.claimID + "\n")
                        f.close()
                    '''

                print('lengtes')
                print(len(self.snippetDocs))
                print(len(self.claimDateAvaialble))
                print(len(self.claimIds))
                print(len(self.bucketsSnippets))
                print(len(self.verbIndices))
                print(len(self.timeIndices))
                print(len(self.positions))
                print(len(self.refsIndices))
                print(len(self.time))
                print(len(self.verbIndicesSnippets))
                print(len(self.timeIndicesSnippets))
                print(len(self.positionsSnippets))
                print(len(self.refsIndicesSnippets))
                print(len(self.timeSnippets))
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
                    verbsSnippetIndices = ''
                    timeSnippetIndices = ''
                    positionSnippetIndices = ''
                    refsSnippetIndices = ''
                    timeSnippetElements = ''
                    verschilSnippets = []

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
                                    bucketsSnippetClaim += '\t' + str(self.matchBucket(verschil))
                                    verschilSnippets.append(verschil)
                                else:
                                    verschil = (snippet.publishTime - claim.claimDate).days
                                    bucketsSnippetClaim += '\t' + str(self.matchBucket(verschil))
                                    verschilSnippets.append(verschil)
                            else:
                                if snippet.publishTime != None:
                                    verschilSnippets.append("None")
                                    bucketsSnippetClaim += '\t' + str(20)
                                else:
                                    bucketsSnippetClaim += '\t' + str(21)
                                    verschilSnippets.append("None")
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
                                        verschilSnippets.append(verschil)
                                    else:
                                        if (snippet.publishTime[1] - claim.claimDate[0]).days <= 0:
                                            verschil = (snippet.publishTime[1] - claim.claimDate[0]).days
                                            bucketsSnippetClaim += '\t' + str(self.matchBucket(verschil))
                                            verschilSnippets.append(verschil)
                                        else:
                                            verschil = 0
                                            bucketsSnippetClaim += '\t' + str(self.matchBucket(verschil))
                                            verschilSnippets.append(verschil)
                                else:
                                    if ((snippet.publishTime - claim.claimDate[0]).days >= 0 and (
                                            snippet.publishTime - claim.claimDate[1]).days <= 0):
                                        verschil = 0
                                        bucketsSnippetClaim += '\t' + str(self.matchBucket(verschil))
                                        verschilSnippets.append(verschil)
                                    else:
                                        if (snippet.publishTime - claim.claimDate[0]).days <= 0:
                                            verschil = (snippet.publishTime - claim.claimDate[0]).days
                                        else:
                                            verschil = (snippet.publishTime - claim.claimDate[1]).days
                                        bucketsSnippetClaim += '\t' + str(self.matchBucket(verschil))
                                        verschilSnippets.append(verschil)
                            else:
                                if snippet.publishTime != None:
                                    bucketsSnippetClaim += '\t' + str(20)
                                    verschilSnippets.append("None")
                                else:
                                    bucketsSnippetClaim += '\t' + str(21)
                                    verschilSnippets.append("None")

                    # print(claim.claimID + '/' + 'claim')
                    input_ids = [i for i in tokenizer.encode(
                        text=getClaimText(claim))]
                    # print("Tekst claim")
                    # print(getClaimText(claim))
                    decoding = [tokenizer.decode([i]) for i in input_ids]
                    # print("Dedoding")
                    # print(decoding)
                    # print(len(input_ids))
                    startIndex = claim.getIndex()
                    OIE = claim.readOpenInformationExtraction()
                    indicesAll, tagsAll, positionsAll, allWords, zinnenIndices = ProcessOpenInformation(
                        OIE, decoding, startIndex, nlpOIE)
                    verbClaim = ""
                    for verb in tagsAll['Verbs']:
                        for part in verb:
                            verbClaim += str(part) + "\t"
                    verbClaim = verbClaim[:-1]
                    self.verbIndices.append(verbClaim)
                    '''
                    positionsClaim = ""
                    for position in positionsAll:
                        positionsClaim+= str(position[0])+','+str(position[1]) + "\t"
                    '''
                    positionsClaim = ""
                    timeClaim = ""
                    if not type(claim.claimDate) is tuple:
                        datasM, durenM, refsM, setsM = claim.readTime()
                        refsIndices = []
                        tokenisation = []
                        startIndex = claim.getIndexHeidel()
                        for data in datasM:
                            if type(data) != int:
                                if type(data) == str:
                                    tokenisation += [token for token in nlp.tokenizer(str(data))]
                                else:
                                    startItem = len(tokenisation) - startIndex
                                    tokenisation += [token for token in nlp.tokenizer(str(data[1]))]
                                    endItem = len(tokenisation) - startIndex - 1
                                    refsIndices.append([startItem, endItem])
                        tokenisation = tokenisation[startIndex:]
                        indicesRefs = match(indicesAll, allWords, tokenisation, startIndex == 6,
                                            refsIndices)
                        index = 0
                        tijdClaim = ""
                        indicesRefsClaim = ""
                        for time in tagsAll['Time']:
                            if len(indicesRefs) <= index:
                                for position in positionsAll:
                                    if time[0] == position[0] + position[1]:
                                        positionsClaim += str(position[0]) + ',' + str(
                                            position[1]) + "\t"
                                for part in time:
                                    tijdClaim += str(part) + "\t"
                            else:
                                if time[-1] < indicesRefs[index][0]:
                                    for position in positionsAll:
                                        if time[0] == position[0] + position[1]:
                                            positionsClaim += str(position[0]) + ',' + str(
                                                position[1]) + "\t"
                                    for part in time:
                                        tijdClaim += str(part) + "\t"
                                else:
                                    if time[0] > indicesRefs[index][-1]:
                                        while time[0] > indicesRefs[index][-1]:
                                            indicesRefsClaim += str(indicesRefs[index][0]) + "\t"
                                            for timeI in range(indicesRefs[index][0], indicesRefs[index][-1] + 1):
                                                tijdClaim += str(timeI) + "\t"
                                            index += 1
                                            if len(indicesRefs) <= index:
                                                break
                                        if len(indicesRefs) <= index:
                                            for position in positionsAll:
                                                if time[0] == position[0] + position[1]:
                                                    positionsClaim += str(position[0]) + ',' + str(
                                                        position[1]) + "\t"
                                            for part in time:
                                                tijdClaim += str(part) + "\t"
                                        else:
                                            if time[-1] < indicesRefs[index][0]:
                                                for position in positionsAll:
                                                    if time[0] == position[0] + position[1]:
                                                        positionsClaim += str(position[0]) + ',' + str(
                                                            position[1]) + "\t"
                                                for part in time:
                                                    tijdClaim += str(part) + "\t"
                                            else:
                                                indicesRefsClaim += str(min(indicesRefs[index][0], time[0])) + "\t"
                                                for position in positionsAll:
                                                    if time[0] == position[0] + position[1]:
                                                        second = min(indicesRefs[index][0], time[0]) - position[0]
                                                        positionsClaim += str(position[0]) + ',' + str(second) + "\t"
                                                for timeI in range(min(indicesRefs[index][0], time[0]),
                                                                   max(indicesRefs[index][1] + 1, time[-1] + 1)):
                                                    tijdClaim += str(timeI) + "\t"
                                                index += 1
                                    else:
                                        for position in positionsAll:
                                            if time[0] == position[0] + position[1]:
                                                second = min(indicesRefs[index][0], time[0]) - position[0]
                                                positionsClaim += str(position[0]) + ',' + str(second) + "\t"
                                        indicesRefsClaim += str(min(indicesRefs[index][0], time[0])) + "\t"
                                        for timeI in range(min(indicesRefs[index][0], time[0]),
                                                           max(indicesRefs[index][1] + 1, time[-1] + 1)):
                                            tijdClaim += str(timeI) + "\t"
                                        index += 1
                        while index < len(indicesRefs):
                            indicesRefsClaim += str(indicesRefs[index][0]) + "\t"
                            for timeI in range(indicesRefs[index][0], indicesRefs[index][-1] + 1):
                                tijdClaim += str(timeI) + "\t"
                            index += 1
                        positionsClaim = positionsClaim[:-1]
                        self.positions.append(positionsClaim)
                        tijdClaim = tijdClaim[:-1]
                        self.timeIndices.append(tijdClaim)
                        '''
                        for indices in indicesRefs:
                            indicesRefsClaim += str(indices[0]) +"\t"
                        '''
                        indicesRefsClaim = indicesRefsClaim[:-1]
                        # print(indicesRefsClaim)
                        self.refsIndices.append(indicesRefsClaim)

                        for data in datasM:
                            if type(data) == list:
                                if data[0] in durenM:
                                    if self.matchBucketDuur(data[0]) != "None":
                                        timeClaim += "Duur-" + self.matchBucketDuur(data[0]) + "\t"
                                else:
                                    if data[0] in refsM:
                                        timeClaim += "Refs-" + self.matchBucketRelativeTime(data[0],
                                                                                            difference=0) + "\t"
                                    else:
                                        if data[0] in setsM:
                                            if self.matchBucketDuur(data[0]) != "None":
                                                timeClaim += "Duur-" + self.matchBucketDuur(data[0]) + "\t"
                                        else:
                                            if (claim.claimDate != None):
                                                if type(data[0]) is tuple:
                                                    if ((data[0][
                                                             0] - claim.claimDate).days <= 0 and (
                                                            data[0][
                                                                1] - claim.claimDate).days >= 0):
                                                        verschil = 0
                                                    else:
                                                        if (data[0][1] - claim.claimDate).days <= 0:
                                                            verschil = (data[0][
                                                                            1] - claim.claimDate).days
                                                        else:
                                                            verschil = (data[0][
                                                                            0] - claim.claimDate).days
                                                    timeClaim += str(self.matchBucketAbsoluteTime(verschil)) + "\t"
                                                else:
                                                    verschil = (data[0] - claim.claimDate).days
                                                    timeClaim += str(self.matchBucketAbsoluteTime(verschil)) + "\t"
                        timeClaim = timeClaim[:-1]
                        self.time.append(timeClaim)
                    else:
                        datasM, durenM, refsM, setsM = claim.readTime()
                        refsIndices = []
                        tokenisation = []
                        startIndex = claim.getIndexHeidel()
                        for data in datasM:
                            if type(data) == str:
                                tokenisation += [token for token in nlp.tokenizer(str(data))]
                            else:
                                startItem = len(tokenisation) - startIndex
                                tokenisation += [token for token in nlp.tokenizer(str(data[1]))]
                                endItem = len(tokenisation) - startIndex - 1
                                refsIndices.append([startItem, endItem])
                        tokenisation = tokenisation[startIndex:]
                        indicesRefs = match(indicesAll, allWords, tokenisation, startIndex == 6,
                                            refsIndices)
                        index = 0
                        tijdClaim = ""
                        indicesRefsClaim = ""
                        for time in tagsAll['Time']:
                            if len(indicesRefs) <= index:
                                for position in positionsAll:
                                    if time[0] == position[0] + position[1]:
                                        positionsClaim += str(position[0]) + ',' + str(
                                            position[1]) + "\t"
                                for part in time:
                                    tijdClaim += str(part) + "\t"
                            else:
                                if time[-1] < indicesRefs[index][0]:
                                    for position in positionsAll:
                                        if time[0] == position[0] + position[1]:
                                            positionsClaim += str(position[0]) + ',' + str(
                                                position[1]) + "\t"
                                    for part in time:
                                        tijdClaim += str(part) + "\t"
                                else:
                                    if time[0] > indicesRefs[index][-1]:
                                        while time[0] > indicesRefs[index][-1]:
                                            indicesRefsClaim += str(indicesRefs[index][0]) + "\t"
                                            for timeI in range(indicesRefs[index][0], indicesRefs[index][-1] + 1):
                                                tijdClaim += str(timeI) + "\t"
                                            index += 1
                                            if len(indicesRefs) <= index:
                                                break
                                        if len(indicesRefs) <= index:
                                            for position in positionsAll:
                                                if time[0] == position[0] + position[1]:
                                                    positionsClaim += str(position[0]) + ',' + str(
                                                        position[1]) + "\t"
                                            for part in time:
                                                tijdClaim += str(part) + "\t"
                                        else:
                                            if time[-1] < indicesRefs[index][0]:
                                                for position in positionsAll:
                                                    if time[0] == position[0] + position[1]:
                                                        positionsClaim += str(position[0]) + ',' + str(
                                                            position[1]) + "\t"
                                                for part in time:
                                                    tijdClaim += str(part) + "\t"
                                            else:
                                                indicesRefsClaim += str(min(indicesRefs[index][0], time[0])) + "\t"
                                                for position in positionsAll:
                                                    if time[0] == position[0] + position[1]:
                                                        second = min(indicesRefs[index][0], time[0]) - position[0]
                                                        positionsClaim += str(position[0]) + ',' + str(
                                                            second) + "\t"
                                                for timeI in range(min(indicesRefs[index][0], time[0]),
                                                                   max(indicesRefs[index][1] + 1, time[-1] + 1)):
                                                    tijdClaim += str(timeI) + "\t"
                                                index += 1
                                    else:
                                        for position in positionsAll:
                                            if time[0] == position[0] + position[1]:
                                                second = min(indicesRefs[index][0], time[0]) - position[0]
                                                positionsClaim += str(position[0]) + ',' + str(second) + "\t"
                                        indicesRefsClaim += str(min(indicesRefs[index][0], time[0])) + "\t"
                                        for timeI in range(min(indicesRefs[index][0], time[0]),
                                                           max(indicesRefs[index][1] + 1, time[-1] + 1)):
                                            tijdClaim += str(timeI) + "\t"
                                        index += 1
                        while index < len(indicesRefs):
                            indicesRefsClaim += str(indicesRefs[index][0]) + "\t"
                            for timeI in range(indicesRefs[index][0], indicesRefs[index][-1] + 1):
                                tijdClaim += str(timeI) + "\t"
                            index += 1
                        positionsClaim = positionsClaim[:-1]
                        self.positions.append(positionsClaim)
                        tijdClaim = tijdClaim[:-1]
                        self.timeIndices.append(tijdClaim)
                        '''
                        for indices in indicesRefs:
                            indicesRefsClaim += str(indices[0]) +"\t"
                        '''
                        indicesRefsClaim = indicesRefsClaim[:-1]
                        # print(indicesRefsClaim)
                        self.refsIndices.append(indicesRefsClaim)
                        for data in datasM:
                            if type(data) == list:
                                if data[0] in durenM:
                                    if self.matchBucketDuur(data[0]) != "None":
                                        timeClaim += "Duur-" + self.matchBucketDuur(data[0]) + "\t"
                                else:
                                    if data[0] in refsM:
                                        timeClaim += "Refs-" + self.matchBucketRelativeTime(data[0],
                                                                                            difference=0) + "\t"
                                    else:
                                        if data[0] in setsM:
                                            timeClaim += "Duur-" + self.matchBucketRelativeTime(data[0],
                                                                                                difference=0) + "\t"
                                        else:
                                            if type(data[0]) is tuple:
                                                if (data[0][0] - claim.claimDate[1]).days >= 0:
                                                    verschil = (data[0][0] - claim.claimDate[
                                                        1]).days
                                                else:
                                                    if (data[0][1] - claim.claimDate[0]).days <= 0:
                                                        verschil = (data[0][1] - claim.claimDate[
                                                            0]).days
                                                    else:
                                                        verschil = 0
                                                timeClaim += str(self.matchBucketAbsoluteTime(verschil)) + "\t"
                                            else:
                                                if ((data[0] - claim.claimDate[0]).days >= 0 and (
                                                        data[0] - claim.claimDate[1]).days <= 0):
                                                    verschil = 0
                                                else:
                                                    if (data[0] - claim.claimDate[0]).days <= 0:
                                                        verschil = (data[0] - claim.claimDate[
                                                            0]).days
                                                    else:
                                                        verschil = (data[0] - claim.claimDate[
                                                            1]).days
                                                timeClaim += str(self.matchBucketAbsoluteTime(verschil)) + "\t"
                        timeClaim = timeClaim[:-1]
                        self.time.append(timeClaim)
                    for snippet in claim.snippets:
                        input_ids = [i for i in tokenizer.encode(
                            text=getSnippetText(snippet))]
                        # print(len(input_ids))
                        decoding = [tokenizer.decode([i]) for i in input_ids]
                        OIE = snippet.readOpenInformationExtraction()
                        startIndex = snippet.getIndex()
                        indicesAll, tagsAll, positionsAll, allWords, zinnenIndices = ProcessOpenInformation(
                            OIE, decoding, startIndex, nlpOIE)
                        if len(tagsAll['Verbs']) > 0:
                            for verb in tagsAll['Verbs']:
                                for part in verb:
                                    verbsSnippetIndices += str(part) + "\t"
                            verbsSnippetIndices = verbsSnippetIndices[:-1]
                        verbsSnippetIndices += ' 0123456789 '
                        '''
                        if len(positionsAll) > 0:
                            for position in positionsAll:
                                positionSnippetIndices += str(position[0]) + ',' + str(position[1]) + "\t"
                            positionSnippetIndices = positionSnippetIndices[:-1]
                        positionSnippetIndices += ' 0123456789 '
                        '''
                        if not type(claim.claimDate) is tuple:
                            datasM, durenM, refsM, setsM = snippet.readTime()
                            tokenisation = []
                            refsIndices = []
                            startIndex = snippet.getIndexHeidel()
                            for data in datasM:
                                # print(data)
                                if type(data) != int:
                                    if type(data) == str:
                                        tokenisation += [token for token in nlp.tokenizer(str(data))]
                                    else:
                                        startItem = len(tokenisation) - startIndex
                                        tokenisation += [token for token in nlp.tokenizer(str(data[1]))]
                                        endItem = len(tokenisation) - startIndex - 1
                                        refsIndices.append([startItem, endItem])
                            tokenisation = tokenisation[startIndex:]
                            # print(refsIndices)
                            indicesRefs = match(indicesAll, allWords, tokenisation, startIndex == 6,
                                                refsIndices)
                            index = 0
                            # print(tagsAll['Time'])
                            for time in tagsAll['Time']:
                                if len(indicesRefs) <= index:
                                    for position in positionsAll:
                                        if time[0] == position[0] + position[1]:
                                            positionSnippetIndices += str(position[0]) + ',' + str(
                                                position[1]) + "\t"
                                    for part in time:
                                        timeSnippetIndices += str(part) + "\t"
                                else:
                                    if time[-1] < indicesRefs[index][0]:
                                        for position in positionsAll:
                                            if time[0] == position[0] + position[1]:
                                                positionSnippetIndices += str(position[0]) + ',' + str(
                                                    position[1]) + "\t"
                                        for part in time:
                                            timeSnippetIndices += str(part) + "\t"
                                    else:
                                        if time[0] > indicesRefs[index][-1]:
                                            while time[0] > indicesRefs[index][-1]:
                                                refsSnippetIndices += str(indicesRefs[index][0]) + "\t"
                                                for timeI in range(indicesRefs[index][0], indicesRefs[index][-1] + 1):
                                                    timeSnippetIndices += str(timeI) + "\t"
                                                index += 1
                                                if len(indicesRefs) <= index:
                                                    break
                                            if len(indicesRefs) <= index:
                                                for position in positionsAll:
                                                    if time[0] == position[0] + position[1]:
                                                        positionSnippetIndices += str(position[0]) + ',' + str(
                                                            position[1]) + "\t"
                                                for part in time:
                                                    timeSnippetIndices += str(part) + "\t"
                                            else:
                                                if time[-1] < indicesRefs[index][0]:
                                                    for position in positionsAll:
                                                        if time[0] == position[0] + position[1]:
                                                            positionSnippetIndices += str(position[0]) + ',' + str(
                                                                position[1]) + "\t"
                                                    for part in time:
                                                        timeSnippetIndices += str(part) + "\t"
                                                else:
                                                    refsSnippetIndices += str(
                                                        min(indicesRefs[index][0], time[0])) + "\t"
                                                    for position in positionsAll:
                                                        if time[0] == position[0] + position[1]:
                                                            second = min(indicesRefs[index][0], time[0]) - position[
                                                                0]
                                                            positionSnippetIndices += str(position[0]) + ',' + str(
                                                                second) + "\t"
                                                    for timeI in range(min(indicesRefs[index][0], time[0]),
                                                                       max(indicesRefs[index][1] + 1, time[-1] + 1)):
                                                        timeSnippetIndices += str(timeI) + "\t"
                                                    index += 1
                                        else:
                                            for position in positionsAll:
                                                if time[0] == position[0] + position[1]:
                                                    second = min(indicesRefs[index][0], time[0]) - position[0]
                                                    positionSnippetIndices += str(position[0]) + ',' + str(
                                                        second) + "\t"
                                            refsSnippetIndices += str(min(indicesRefs[index][0], time[0])) + "\t"
                                            for timeI in range(min(indicesRefs[index][0], time[0]),
                                                               max(indicesRefs[index][1] + 1, time[-1] + 1)):
                                                timeSnippetIndices += str(timeI) + "\t"
                                            index += 1
                            while index < len(indicesRefs):
                                refsSnippetIndices += str(indicesRefs[index][0]) + "\t"
                                for timeI in range(indicesRefs[index][0], indicesRefs[index][-1] + 1):
                                    timeSnippetIndices += str(timeI) + "\t"
                                index += 1
                            if (len(positionSnippetIndices) > 0 and positionSnippetIndices[-1] == "\t"):
                                positionSnippetIndices = positionSnippetIndices[:-1]
                            positionSnippetIndices += ' 0123456789 '
                            if (len(timeSnippetIndices) > 0 and timeSnippetIndices[-1] == "\t"):
                                timeSnippetIndices = timeSnippetIndices[:-1]
                            timeSnippetIndices += ' 0123456789 '
                            if (len(refsSnippetIndices) > 0 and refsSnippetIndices[-1] == "\t"):
                                refsSnippetIndices = refsSnippetIndices[:-1]
                            refsSnippetIndices += ' 0123456789 '
                            # print('IndicesRefs')
                            # print(indicesRefs)
                            # verschilSnippet = verschilSnippets[0]
                            # verschilSnippets = verschilSnippets[1:]
                            '''
                            index = 0

                            for time in tagsAll['Time']:
                                for part in time:
                                    if len(indicesRefs) <= index:
                                        timeSnippetIndices += str(part) + "\t"
                                    else:
                                        # print('Part : ' + str(part))
                                        # print('IndicesRefs : ' + str(indicesRefs[index][0]))
                                        if part < indicesRefs[index][0]:
                                            timeSnippetIndices += str(part) + "\t"
                                        else:
                                            if part == indicesRefs[index][0]:
                                                timeSnippetIndices += str(part) + "\t"
                                                index += 1
                                            else:
                                                for timeI in range(indicesRefs[index][0], indicesRefs[index][1]):
                                                    timeSnippetIndices += str(timeI) + "\t"
                                                    index += 1
                            if (len(timeSnippetIndices) > 0 and timeSnippetIndices[-1] == "\t"):
                                timeSnippetIndices = timeSnippetIndices[:-1]
                            timeSnippetIndices += ' 0123456789 '
                            for indices in indicesRefs:
                                refsSnippetIndices += str(indices[0]) + "\t"
                            if (len(refsSnippetIndices) > 0 and refsSnippetIndices[-1] == "\t"):
                                refsSnippetIndices = refsSnippetIndices[:-1]
                            refsSnippetIndices += ' 0123456789 '
                            # print('IndicesRefs')
                            # print(indicesRefs)
                            '''
                            verschilSnippet = verschilSnippets[0]
                            verschilSnippets = verschilSnippets[1:]
                            for data in datasM:
                                if type(data) == list:
                                    if data[0] in durenM:
                                        if self.matchBucketDuur(data[0]) != "None":
                                            timeSnippetElements += "Duur-" + self.matchBucketDuur(data[0]) + "\t"
                                    else:
                                        if data[0] in refsM:
                                            if verschilSnippet != "None":
                                                timeSnippetElements += "Refs-" + self.matchBucketRelativeTime(
                                                    data[0], difference=verschilSnippet) + "\t"
                                        else:
                                            if data[0] in setsM:
                                                if self.matchBucketDuur(data[0]) != "None":
                                                    timeSnippetElements += "Duur-" + self.matchBucketDuur(
                                                        data[0]) + "\t"
                                            else:
                                                if (claim.claimDate != None):
                                                    if type(data[0]) is tuple:
                                                        if ((data[0][
                                                                 0] - claim.claimDate).days <= 0 and (
                                                                data[0][
                                                                    1] - claim.claimDate).days >= 0):
                                                            verschil = 0
                                                        else:
                                                            if (data[0][
                                                                    1] - claim.claimDate).days <= 0:
                                                                verschil = (data[0][
                                                                                1] - claim.claimDate).days
                                                            else:
                                                                verschil = (data[0][
                                                                                0] - claim.claimDate).days
                                                        timeSnippetElements += str(
                                                            self.matchBucketAbsoluteTime(verschil)) + "\t"
                                                    else:
                                                        verschil = (data[0] - claim.claimDate).days
                                                        timeSnippetElements += str(
                                                            self.matchBucketAbsoluteTime(verschil)) + "\t"
                            if len(timeSnippetElements) > 0 and timeSnippetElements[-1] == "\t":
                                timeSnippetElements = timeSnippetElements[:-1]
                            timeSnippetElements += ' 0123456789 '
                        else:
                            datasM, durenM, refsM, setsM = snippet.readTime()
                            # print(snippet.getSnippetText())
                            tokenisation = []
                            refsIndices = []
                            startIndex = snippet.getIndexHeidel()
                            for data in datasM:
                                # print(data)
                                if type(data) != int:
                                    if type(data) == str:
                                        tokenisation += [token for token in nlp.tokenizer(str(data))]
                                    else:
                                        startItem = len(tokenisation) - startIndex
                                        tokenisation += [token for token in nlp.tokenizer(str(data[1]))]
                                        endItem = len(tokenisation) - startIndex - 1
                                        refsIndices.append([startItem, endItem])
                            tokenisation = tokenisation[startIndex:]
                            # print(refsIndices)
                            indicesRefs = match(indicesAll, allWords, tokenisation, startIndex == 6,
                                                refsIndices)
                            index = 0
                            for time in tagsAll['Time']:
                                if len(indicesRefs) <= index:
                                    for position in positionsAll:
                                        if time[0] == position[0] + position[1]:
                                            positionSnippetIndices += str(position[0]) + ',' + str(
                                                position[1]) + "\t"
                                    for part in time:
                                        timeSnippetIndices += str(part) + "\t"
                                else:
                                    if time[-1] < indicesRefs[index][0]:
                                        for position in positionsAll:
                                            if time[0] == position[0] + position[1]:
                                                positionSnippetIndices += str(position[0]) + ',' + str(
                                                    position[1]) + "\t"
                                        for part in time:
                                            timeSnippetIndices += str(part) + "\t"
                                    else:
                                        if time[0] > indicesRefs[index][-1]:
                                            while time[0] > indicesRefs[index][-1]:
                                                refsSnippetIndices += str(indicesRefs[index][0]) + "\t"
                                                for timeI in range(indicesRefs[index][0],
                                                                   indicesRefs[index][-1] + 1):
                                                    timeSnippetIndices += str(timeI) + "\t"
                                                index += 1
                                                if len(indicesRefs) <= index:
                                                    break
                                            if len(indicesRefs) <= index:
                                                for position in positionsAll:
                                                    if time[0] == position[0] + position[1]:
                                                        positionSnippetIndices += str(position[0]) + ',' + str(
                                                            position[1]) + "\t"
                                                for part in time:
                                                    timeSnippetIndices += str(part) + "\t"
                                            else:
                                                if time[-1] < indicesRefs[index][0]:
                                                    for position in positionsAll:
                                                        if time[0] == position[0] + position[1]:
                                                            positionSnippetIndices += str(position[0]) + ',' + str(
                                                                position[1]) + "\t"
                                                    for part in time:
                                                        timeSnippetIndices += str(part) + "\t"
                                                else:
                                                    refsSnippetIndices += str(
                                                        min(indicesRefs[index][0], time[0])) + "\t"
                                                    for position in positionsAll:
                                                        if time[0] == position[0] + position[1]:
                                                            second = min(indicesRefs[index][0], time[0]) - position[
                                                                0]
                                                            positionSnippetIndices += str(position[0]) + ',' + str(
                                                                second) + "\t"
                                                    for timeI in range(min(indicesRefs[index][0], time[0]),
                                                                       max(indicesRefs[index][1] + 1, time[-1] + 1)):
                                                        timeSnippetIndices += str(timeI) + "\t"
                                                    index += 1
                                        else:
                                            for position in positionsAll:
                                                if time[0] == position[0] + position[1]:
                                                    second = min(indicesRefs[index][0], time[0]) - position[0]
                                                    positionSnippetIndices += str(position[0]) + ',' + str(
                                                        second) + "\t"
                                            refsSnippetIndices += str(min(indicesRefs[index][0], time[0])) + "\t"
                                            for timeI in range(min(indicesRefs[index][0], time[0]),
                                                               max(indicesRefs[index][1] + 1, time[-1] + 1)):
                                                timeSnippetIndices += str(timeI) + "\t"
                                            index += 1
                            while index < len(indicesRefs):
                                refsSnippetIndices += str(indicesRefs[index][0]) + "\t"
                                for timeI in range(indicesRefs[index][0], indicesRefs[index][-1] + 1):
                                    timeSnippetIndices += str(timeI) + "\t"
                                index += 1
                            if (len(positionSnippetIndices) > 0 and positionSnippetIndices[-1] == "\t"):
                                positionSnippetIndices = positionSnippetIndices[:-1]
                            positionSnippetIndices += ' 0123456789 '
                            if (len(timeSnippetIndices) > 0 and timeSnippetIndices[-1] == "\t"):
                                timeSnippetIndices = timeSnippetIndices[:-1]
                            timeSnippetIndices += ' 0123456789 '
                            if (len(refsSnippetIndices) > 0 and refsSnippetIndices[-1] == "\t"):
                                refsSnippetIndices = refsSnippetIndices[:-1]
                            refsSnippetIndices += ' 0123456789 '
                            verschilSnippet = verschilSnippets[0]
                            verschilSnippets = verschilSnippets[1:]
                            for data in datasM:
                                if type(data) == list:
                                    if data[0] in durenM:
                                        if self.matchBucketDuur(data[0]) != "None":
                                            timeSnippetElements += "Duur-" + self.matchBucketDuur(
                                                data[0]) + "\t"
                                    else:
                                        if data[0] in refsM:
                                            if verschilSnippet != "None":
                                                timeSnippetElements += "Refs-" + self.matchBucketRelativeTime(
                                                    data[0], difference=verschilSnippet) + "\t"
                                        else:
                                            if data[0] in setsM:
                                                if self.matchBucketDuur(data[0]) != "None":
                                                    timeSnippetElements += "Duur-" + self.matchBucketRelativeTime(
                                                        data[0],
                                                        difference=0) + "\t"
                                            else:
                                                if type(data[0]) is tuple:
                                                    if (data[0][0] - claim.claimDate[1]).days >= 0:
                                                        verschil = (data[0][0] - claim.claimDate[
                                                            1]).days
                                                        timeSnippetElements += str(
                                                            self.matchBucketAbsoluteTime(verschil)) + "\t"
                                                    else:
                                                        if (data[0][1] - claim.claimDate[
                                                            0]).days <= 0:
                                                            verschil = (data[0][1] -
                                                                        claim.claimDate[0]).days
                                                            timeSnippetElements += str(
                                                                self.matchBucketAbsoluteTime(verschil)) + "\t"
                                                        else:
                                                            verschil = 0
                                                            timeSnippetElements += str(
                                                                self.matchBucketAbsoluteTime(verschil)) + "\t"
                                                else:
                                                    if ((data[0] - claim.claimDate[
                                                        0]).days >= 0 and (
                                                            data[0] - claim.claimDate[
                                                        1]).days <= 0):
                                                        verschil = 0
                                                    else:
                                                        if (data[0] - claim.claimDate[0]).days <= 0:
                                                            verschil = (data[0] - claim.claimDate[
                                                                0]).days
                                                        else:
                                                            verschil = (data[0] - claim.claimDate[
                                                                1]).days
                                                    timeSnippetElements += str(
                                                        self.matchBucketAbsoluteTime(verschil)) + "\t"
                            if len(timeSnippetElements) > 0 and timeSnippetElements[-1] == "\t":
                                timeSnippetElements = timeSnippetElements[:-1]
                            timeSnippetElements += ' 0123456789 '
                    self.timeSnippets.append(timeSnippetElements)
                    self.refsIndicesSnippets.append(refsSnippetIndices)
                    self.verbIndicesSnippets.append(verbsSnippetIndices)
                    self.positionsSnippets.append(positionSnippetIndices)
                    self.timeIndicesSnippets.append(timeSnippetIndices)
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
                    self.labelsAll.add(claim.label)
                    self.claimIds.append(claim.claimID)
                    self.documents.append(self.getClaimText(claim))
                    self.snippetDocs.append(self.getSnippets(claim.snippets))
                    '''
                    except:
                        f = open("faultDataSetIteratie2.txt", "a", encoding="utf-8")
                        f.write(claim.claimID + "\n")
                        f.close()
                    '''

                print('lengtes')
                print(len(self.snippetDocs))
                print(len(self.claimDateAvaialble))
                print(len(self.claimIds))
                print(len(self.bucketsSnippets))
                print(len(self.verbIndices))
                print(len(self.timeIndices))
                print(len(self.positions))
                print(len(self.refsIndices))
                print(len(self.time))
                print(len(self.verbIndicesSnippets))
                print(len(self.timeIndicesSnippets))
                print(len(self.positionsSnippets))
                print(len(self.refsIndicesSnippets))
                print(len(self.timeSnippets))

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


        return self.claimIds[index], self.documents[index], self.snippetDocs[index],self.metadata[index], self.labels[index],\
               self.claimDateAvaialble[index],self.bucketsSnippets[index],self.verbIndices[index], self.timeIndices[index],\
               self.positions[index],self.refsIndices[index],self.time[index],self.verbIndicesSnippets[index],\
               self.timeIndicesSnippets[index],self.positionsSnippets[index],self.refsIndicesSnippets[index],self.timeSnippets[index]

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

def getClaimText(claim):
    basepath = os.path.normpath(os.getcwd() + os.sep + os.pardir)
    file = open(basepath + "/text" + "/" + claim.claimID + "/" + "claim", encoding="utf-8")
    return file.read()

def getSnippetText(snippet):
    basepath = os.path.normpath(os.getcwd() + os.sep + os.pardir)
    file = open(basepath + "/text" + "/" + snippet.claimID + "/" + snippet.number, encoding="utf-8")
    return file.read()

def getSentence(words):
    sentence = ""
    for word in words:
        sentence += word + " "
    return sentence

def ProcessOpenInformation(OIE,decoding,startIndex,nlp):
    tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-distilroberta-v1')
    indicesAll = []
    #print(decoding)
    tagsAll = {"Verbs":[],"Time":[],"Relative":[]}
    positionsAll = []
    index = startIndex
    lastIndex = index
    hasTitle = False
    if startIndex == 7:
        firstSentence = True
        hasTitle = True
    else:
        firstSentence = False
    wait = 0
    #print("decoding OIE")
    #print(decoding)
    #print(len(decoding))
    allWords = []
    zinnenIndices = []
    nextWordD = None
    nextWordE = None
    for items in OIE:
        indices = []
        words = items['words']
        #print(words)
        allWords += words
        #print('uuu : '+str(lastIndex))
        lastIndex = index
        #print(lastIndex)
        lastPart = None
        stillToDo = []
        start = 0
        window = 1
        if len(words)==0:
            continue
        word = str(words[start]).replace('!','').replace('İ','').replace('@','').replace(',','').replace('•','').replace('_','').replace('[','').replace(']','').replace('*','').replace('�','').replace('.','').replace(chr(8220),'').replace(' ', '').replace("'", "").replace('"', '').replace('<', '').replace('/',
                                                                                                                 '').replace(
            '>', '').replace("\\", '').replace('(','').replace(')','').lower().encode("ascii", "ignore").decode()

        #word = str(words[start]).replace(' ', '').lower().encode("ascii", "ignore").decode()
        if nextWordE != None:
            word = nextWordE + word
        while len(word) == 0:
            #print("Empty")
            indices.append([lastIndex])
            start += 1
            if start < len(words):

                word = str(words[start]).replace('!','').replace('İ','').replace('@','').replace(',','').replace('•','').replace('_','').replace('[','').replace(']','').replace('*','').replace('�','').replace('.','').replace(chr(8220),'').replace(' ', '').replace("'", "").replace('"', '').replace('<', '').replace('/',
                                                                                                                     '').replace(
                    '>', '').replace("\\", '').replace('(','').replace(')','').lower().encode("ascii", "ignore").decode()

                #word = str(words[start]).replace(' ', '').lower().encode("ascii", "ignore").decode()
            else:
                start -=1
                break

        if len(word) == 0:
            nextWordD = None
            nextWordE = None
            indicesAll += indices
            zinnenIndices.append(len(indicesAll))
            continue
        #print(word)
        #print(word == '"')
        #print(ord(word[0]))
        #print(ord('"'))

        '''
        if lastIndex + 1 < len(decoding) and start + 1 <len(words):
            word2 = str(decoding[lastIndex]).replace(' ', '').lower()
            nextWord = str(decoding[lastIndex + 1]).replace(' ', '').lower()
            word2Decoding = str(words[start + 1 ]).replace(' ', '').lower()
            print("Next : " + nextWord)
            print("Word 2 decoding : " + word2Decoding)

            if not(word2Decoding == nextWord) and len(word2)>0 and word2[-1] == "'" and nextWord == 'says' and firstSentence:
                firstSentence = False
                lastIndex = lastIndex + 2
        '''
        #print(word)
        #print(indicesAll)
        #print(lastIndex)
        if nextWordD == None:
            wordDecoding = str(decoding[lastIndex]).replace('!','').replace('İ','').replace('@','').replace(',','').replace('•','').replace('_','').replace('[','').replace(']','').replace('*','').replace(chr(8220),'').replace('.','').replace('�','').replace(' ', '').replace("'", "").replace('"', '').replace('<',
                                                                                                               '').replace(
                '/', '').replace(
                '>', '').replace("\\", '').replace('(','').replace(')','').lower().encode("ascii", "ignore").decode()

            #wordDecoding5 = str(words[start]).replace(' ', '').lower().encode("ascii", "ignore").decode()
            #print(wordDecoding5)
            while (len(wordDecoding) == 0):
                if lastIndex +1 < len(decoding):
                    lastIndex += 1

                    wordDecoding = str(decoding[lastIndex]).replace('!','').replace('İ','').replace('@','').replace(',','').replace('•','').replace('_','').replace('[','').replace(']','').replace('*','').replace('.','').replace(chr(8220),'').replace('�','').replace(' ', '').replace("'", "").replace('"', '').replace('<',
                                                                                                                       '').replace(
                        '/', '').replace(
                        '>', '').replace("\\", '').replace('(','').replace(')','').lower().encode("ascii", "ignore").decode()

                    #wordDecoding5 = str(words[start]).replace(' ', '').lower().encode("ascii", "ignore").decode()
                    #print(wordDecoding5)
                else:
                    break
        else:
            wordDecoding = nextWordD
        if lastIndex < len(decoding) and start < len(words):
            word2 = str(decoding[lastIndex]).replace(' ', '').lower()
            #nextWord = str(decoding[lastIndex + 1]).replace(' ', '').lower()
            word2Decoding = str(words[start]).replace(' ', '').lower()
            #print("Next : " + word2)
            #print("Word 2 decoding : " + word2Decoding)

            if not (word2Decoding == word2) and len(word2) > 0  and word2 == 'says' and firstSentence:
                firstSentence = False
                lastIndex = lastIndex + 1
                wordDecoding = str(decoding[lastIndex]).replace('!','').replace('İ','').replace('@','').replace(',', '').replace('•', '').replace('_', '').replace('[',
                                                                                                                   '').replace(
                    ']', '').replace('*', '').replace(chr(8220), '').replace('.', '').replace('�', '').replace(' ',
                                                                                                               '').replace(
                    "'", "").replace('"', '').replace('<',
                                                      '').replace(
                    '/', '').replace(
                    '>', '').replace("\\", '').replace('(', '').replace(')', '').lower().encode("ascii",
                                                                                                "ignore").decode()

                # wordDecoding5 = str(words[start]).replace(' ', '').lower().encode("ascii", "ignore").decode()
                # print(wordDecoding5)
                while (len(wordDecoding) == 0):
                    if lastIndex + 1 < len(decoding):
                        lastIndex += 1

                        wordDecoding = str(decoding[lastIndex]).replace('!','').replace('İ','').replace('@','').replace(',', '').replace('•', '').replace('_',
                                                                                                          '').replace(
                            '[', '').replace(']', '').replace('*', '').replace('.', '').replace(chr(8220), '').replace(
                            '�', '').replace(' ', '').replace("'", "").replace('"', '').replace('<',
                                                                                                '').replace(
                            '/', '').replace(
                            '>', '').replace("\\", '').replace('(', '').replace(')', '').lower().encode("ascii",
                                                                                                        "ignore").decode()

                        # wordDecoding5 = str(words[start]).replace(' ', '').lower().encode("ascii", "ignore").decode()
                        # print(wordDecoding5)
                    else:
                        break
        #print("aaaaa : " +str(wordDecoding))
        #print(lastIndex)
        #print("Word : " + str(word))
        #print(nextWordD)
        #print(wordDecoding)
        #print(firstSentence)
        #print(wordDecoding.split(word))
        #print(word.split(wordDecoding))
        #print('yyyyy : '+str(lastIndex))
        while (len(wordDecoding.split(word)[0]) != 0 and
               len(word.split(wordDecoding)[0]) != 0):
            if lastIndex +1 < len(decoding):
                lastIndex += 1

                wordDecoding = str(decoding[lastIndex]).replace('!','').replace('İ','').replace('@','').replace(',','').replace('•','').replace('_','').replace('[','').replace(']','').replace('*','').replace('.','').replace(chr(8220),'').replace('�','').replace(' ', '').replace("'", "").replace('"', '').replace(
                    '<',
                    '').replace(
                    '/', '').replace(
                    '>', '').replace("\\", '').replace('(','').replace(')','').lower().encode("ascii", "ignore").decode()

                #wordDecoding = str(words[start]).replace(' ', '').lower().encode("ascii", "ignore").decode()
            else:
                break
            #print("eeeeee " + str(wordDecoding))
            while (len(wordDecoding) == 0):
                if lastIndex +1 < len(decoding):
                    lastIndex += 1

                    wordDecoding = str(decoding[lastIndex]).replace('!','').replace('İ','').replace('@','').replace(',','').replace('•','').replace('_','').replace('[','').replace(']','').replace('*','').replace('.','').replace(chr(8220),'').replace('�','').replace(' ', '').replace("'", "").replace('"', '').replace(
                        '<',
                        '').replace(
                        '/', '').replace(
                        '>', '').replace("\\", '').replace('(','').replace(')','').lower().encode("ascii", "ignore").decode()

                    #wordDecoding = str(words[start]).replace(' ', '').lower().encode("ascii", "ignore").decode()

                else:
                    break
            #print("rrrrr " + str(wordDecoding))
        #print(lastIndex)
        '''
        print(str(decoding[lastIndex]).replace(' ', '').replace("'", "").replace('"', '').replace(
                        '<',
                        '').replace(
                        '/', '').replace(
                        '>', '').replace("\\", '').replace(".", '').lower())
        '''
        lastIndexWord = start
        while(lastIndexWord<len(words)):
            indexWord = lastIndexWord
            #print(lastIndexWord)
            done = False
            word = str(words[indexWord]).replace('!','').replace('İ','').replace('@','').replace(',','').replace('•','').replace('_','').replace('[','').replace(']','').replace('*','').replace('.','').replace(chr(8220),'').replace('�','').replace(' ', '').replace("'","").replace('"','').replace('<','').replace('/','').replace(
                '>','').replace("\\",'').replace(".",'').replace('(','').replace(')','').lower().encode("ascii", "ignore").decode()
            if nextWordE != None:
                word = nextWordE + word

            #word = str(words[start]).replace(' ', '').lower().encode("ascii", "ignore").decode()
            #print("Word 5 : " +str(word))
            if len(word)==0:
                #print("Empty 2")
                for index in range(len(stillToDo)):
                    tuple = [lastIndex - len(stillToDo) + index]
                    indices.append(tuple)
                stillToDo = []
                window+=1
                indices.append([lastIndex])
                #print("LastIndex : " + str(lastIndex))
                nextWordE = None
                done = True
            else:
                for decod in range(lastIndex, len(decoding)):
                    if nextWordD == None:
                        wordDecoding = str(decoding[decod]).replace('!','').replace('İ','').replace('@','').replace(',','').replace('•','').replace('_','').replace('[','').replace(']','').replace('*','').replace('.','').replace(chr(8220),'').replace('�','').replace(' ', '').replace("'", "").replace('"', '').replace(
                            '<', '').replace('/', '').replace(
                            '>', '').replace("\\", '').replace('(','').replace('.','').replace(')','').lower().encode("ascii", "ignore").decode()

                        #wordDecoding = str(words[start]).replace(' ', '').lower().encode("ascii", "ignore").decode()
                        #print(wordDecoding)
                        while (len(wordDecoding) == 0):
                            decod += 1
                            if decod < len(decoding):

                                wordDecoding = str(decoding[decod]).replace('!','').replace('İ','').replace('@','').replace(',','').replace('•','').replace('_','').replace('[','').replace(']','').replace('*','').replace('.','').replace(chr(8220),'').replace('�','').replace(' ', '').replace("'", "").replace('"',
                                                                                                              '').replace(
                                    '<', '').replace('/', '').replace(
                                    '>', '').replace("\\", '').replace('(','').replace(')','').lower().encode("ascii", "ignore").decode()

                                #wordDecoding = str(words[start]).replace(' ', '').lower().encode("ascii", "ignore").decode()
                            else:
                                #print('Break')
                                break
                        if len(wordDecoding)==0:
                            break
                    else:
                        wordDecoding = nextWordD
                    #print(word)
                    #print(wordDecoding)
                    #print(word==wordDecoding)
                    if word == wordDecoding:
                        nextWordD = None
                        nextWordE = None
                        for index in range(len(stillToDo)):
                            tuple = [decod-len(stillToDo)+index]
                            indices.append(tuple)
                        stillToDo= []
                        window = 1
                        tuple = [decod]
                        indices.append(tuple)
                        done = True
                        lastIndex = decod + 1
                        break
                    else:
                        #print(word)
                        #print(word[0]=="'")
                        #print(wordDecoding)
                        parts = word.split(wordDecoding)
                        #print(parts)
                        index = decod
                        #print(index)
                        if len(parts) >= 2 and index +1 <len(decoding) and len(parts[0])==0:
                            nextWordD = None
                            nextWordE = None
                            lengte = len(wordDecoding)
                            newIndices = []
                            newIndices.append(index)
                            index += 1
                            wordDecoding = str(decoding[index]).replace('!','').replace('İ','').replace('@','').replace(',','').replace('•','').replace('_','').replace('[','').replace(']','').replace('*','').replace('.','').replace(chr(8220),'').replace('�','').replace(' ', '').replace("'", "").replace('"',
                                                                                                          '').replace(
                                '<', '').replace('/', '').replace(
                                '>', '').replace("\\", '').replace('(','').replace(')','').lower().encode("ascii", "ignore").decode()
                            while (len(wordDecoding) == 0):
                                index += 1
                                if index < len(decoding):

                                    wordDecoding = str(decoding[index]).replace('!','').replace('İ','').replace('@','').replace(',','').replace('•','').replace('_','').replace('[','').replace(']','').replace('*','').replace('.','').replace(chr(8220), '').replace('�', '').replace(
                                        ' ', '').replace("'", "").replace('"',
                                                                          '').replace(
                                        '<', '').replace('/', '').replace(
                                        '>', '').replace("\\", '').replace('(', '').replace(')', '').lower().encode(
                                        "ascii", "ignore").decode()

                                    # wordDecoding = str(words[start]).replace(' ', '').lower().encode("ascii", "ignore").decode()
                                else:
                                    # print('Break')
                                    break
                            if len(wordDecoding)!= 0:
                                nextWord = word[lengte:]
                                parts = nextWord.lower().split(wordDecoding)
                                if len(parts) >= 2 and len(parts[0]) == 0 and not(nextWord==wordDecoding):
                                    nextWordD = None
                                    nextWordE = None
                                    #print("Lengte : " + str(len(stillToDo) ))
                                    for index2 in range(len(stillToDo)):
                                        tuple = [index - len(stillToDo) + index2]
                                        indices.append(tuple)
                                    stillToDo = []
                                    window = 1
                                    newIndices.append(index)
                                    index += 1
                                    done = True
                                    while len(parts) >= 2 and len(parts[0]) == 0 and not(nextWord==wordDecoding):
                                        lengte= len(wordDecoding)
                                        wordDecoding = str(decoding[index]).replace('!','').replace('İ','').replace('@','').replace(',','').replace('•','').replace('_','').replace('[','').replace(']','').replace('*','').replace('.','').replace(chr(8220),'').replace('�','').replace(' ', '').replace("'", "").replace(
                                            '"',
                                            '').replace(
                                            '<', '').replace('/', '').replace(
                                            '>', '').replace("\\", '').replace('(','').replace(')','').lower().encode("ascii", "ignore").decode()
                                        while (len(wordDecoding) == 0):
                                            index += 1
                                            if index < len(decoding):

                                                wordDecoding = str(decoding[index]).replace('!','').replace('İ','').replace('@','').replace(',','').replace('•','').replace('_','').replace('[','').replace(']','').replace('*','').replace('.','').replace(chr(8220), '').replace('�',
                                                                                                                   '').replace(
                                                    ' ', '').replace("'", "").replace('"',
                                                                                      '').replace(
                                                    '<', '').replace('/', '').replace(
                                                    '>', '').replace("\\", '').replace('(', '').replace(')',
                                                                                                        '').lower().encode(
                                                    "ascii", "ignore").decode()

                                                # wordDecoding = str(words[start]).replace(' ', '').lower().encode("ascii", "ignore").decode()
                                            else:
                                                index -= 1
                                                break
                                        if len(wordDecoding) != 0:
                                            nextWord = nextWord[lengte:]
                                            parts = nextWord.lower().split(wordDecoding)
                                            if len(parts)>=2 and len(parts[0]) == 0 and not(nextWord==wordDecoding):
                                                newIndices.append(index)
                                                index += 1
                                                if index >= len(decoding):
                                                    parts = []
                                        else:
                                            parts = []
                                    if nextWord == wordDecoding:
                                        nextWordD = None
                                        nextWordE = None
                                        newIndices.append(index)
                                        index += 1
                                        indices.append(newIndices)
                                        lastIndex = index
                                    else:
                                        nextWordD = None
                                        nextWordE = nextWord
                                        indices.append(newIndices)
                                        lastIndex = index
                                    break
                                else:
                                    if nextWord == wordDecoding:
                                        nextWordD = None
                                        nextWordE = None
                                        newIndices.append(index)
                                        index += 1
                                        indices.append(newIndices)
                                        lastIndex = index
                                        break
                                    else:
                                        nextWordD = None
                                        nextWordE = nextWord
                                        indices.append(newIndices)
                                        lastIndex = index
                                        break

                        else:
                            #print("WordDecoding : " + str(wordDecoding))
                            #print("Word : " +str(word))
                            #print(str(word)=="?")
                            parts = wordDecoding.split(word)
                            index = indexWord
                            #print("Index begin " + str(index))
                            if len(parts) >= 2 and index + 1 < len(words) and len(parts[0]) == 0:
                                nextWordD = None
                                nextWordE = None
                                indices.append([decod])
                                lengte = len(word)
                                index += 1
                                word = str(words[index]).replace('!','').replace('İ','').replace('@','').replace(',','').replace('•','').replace('_','').replace('[','').replace(']','').replace('*','').replace('.','').replace(chr(8220),'').replace('�','').replace(' ', '').replace("'", "").replace('"',
                                                                                                       '').replace(
                                    '<', '').replace('/', '').replace(
                                    '>', '').replace("\\", '').replace('(','').replace(')','').lower().encode("ascii", "ignore").decode()
                                #print("Word in "  + str(word))
                                while len(word) == 0:
                                    # print("Empty")
                                    indices.append([decod])
                                    index += 1
                                    if index < len(words):

                                        word = str(words[index]).replace('!','').replace('İ','').replace('@','').replace(',','').replace('•','').replace('_','').replace('[','').replace(']','').replace('*','').replace('.','').replace('�', '').replace(chr(8220), '').replace(' ',
                                                                                                                 '').replace(
                                            "'", "").replace('"', '').replace('<', '').replace('/',
                                                                                               '').replace(
                                            '>', '').replace("\\", '').replace('(', '').replace(')', '').lower().encode(
                                            "ascii", "ignore").decode()

                                        # word = str(words[start]).replace(' ', '').lower().encode("ascii", "ignore").decode()
                                    else:
                                        index -= 1
                                        break
                                if len(word)>0:
                                    nextWord = wordDecoding[lengte:]
                                    #print(nextWord)
                                    parts = nextWord.lower().split(word.lower())
                                    #print("Pars in " + str(parts))
                                    if len(parts) >= 2 and len(parts[0]) == 0 and not (nextWord == word):
                                        nextWordD = None
                                        nextWordE = None
                                        for index2 in range(len(stillToDo)):
                                            tuple = [decod - len(stillToDo) + index2]
                                            indices.append(tuple)
                                        stillToDo = []
                                        nextWordD = None
                                        window = 1
                                        indices.append([decod])
                                        done = True
                                        index += 1
                                        while len(parts) >= 2 and len(parts[0])==0 and not(nextWord==word):
                                            lengte = len(word)
                                            word = str(words[index]).replace('!','').replace('İ','').replace('@','').replace('@','').replace(',','').replace('•','').replace('_','').replace('[','').replace(']','').replace('*','').replace('.','').replace(chr(8220),'').replace('�','').replace(' ', '').replace("'", "").replace('"',
                                                                                                               '').replace(
                                                '<', '').replace('/', '').replace(
                                                '>', '').replace("\\", '').replace('(','').replace(')','').lower().encode("ascii", "ignore").decode()
                                            while len(word) == 0:
                                                # print("Empty")
                                                indices.append([decod])
                                                index += 1
                                                if index < len(words):
                                                    word = str(words[index]).replace('!','').replace('İ','').replace('@','').replace(',','').replace('•','').replace('_','').replace('[','').replace(']','').replace('*','').replace('.','').replace('�', '').replace(chr(8220),
                                                                                                      '').replace(
                                                        ' ',
                                                        '').replace(
                                                        "'", "").replace('"', '').replace('<', '').replace('/',
                                                                                                           '').replace(
                                                        '>', '').replace("\\", '').replace('(', '').replace(')',
                                                                                                            '').lower().encode(
                                                        "ascii", "ignore").decode()

                                                    # word = str(words[start]).replace(' ', '').lower().encode("ascii", "ignore").decode()
                                                else:
                                                    index -= 1
                                                    break

                                            #print("Word in 2 : "+str(word))
                                            if len(word)>0 :
                                                nextWord = nextWord[lengte:]
                                                #print("NextWord 5 : " + nextWord)
                                                #print("Word 5 : " + word)
                                                parts = nextWord.lower().split(word.lower())
                                                if len(parts) >= 2 and len(parts[0]) == 0 and not(nextWord==word):
                                                    indices.append([decod])
                                                    nextWordE = None
                                                    nextWordD = None
                                                    index += 1
                                                    if index >= len(words):
                                                        parts = []
                                            else:
                                                parts = []
                                        #print("NextWordA : " + nextWord)
                                        #print(nextWord[0]=="?")
                                        #print(word == "?")
                                        #print("WordA : " + word)
                                        if nextWord == word:
                                            nextWordD = None
                                            nextWordE = None
                                            for index2 in range(len(stillToDo)):
                                                tuple = [decod - len(stillToDo) + index2]
                                                indices.append(tuple)
                                            stillToDo = []
                                            window = 1
                                            #indices.append([decod])
                                            indices.append([decod])
                                            done = True
                                            lastIndexWord = index
                                            lastIndex = decod + 1
                                        else:
                                            nextWordD = nextWord
                                            stillToDo = []
                                            window = 1
                                            done = True
                                            lastIndexWord = index - 1
                                            lastIndex = decod
                                            #print("Index out " + str(lastIndexWord))

                                        break
                                    else:
                                        if nextWord == word:
                                            nextWordD = None
                                            nextWordE = None
                                            for index2 in range(len(stillToDo)):
                                                tuple = [decod - len(stillToDo) + index2]
                                                indices.append(tuple)
                                            stillToDo = []
                                            window = 1
                                            indices.append([decod])
                                            done = True
                                            lastIndexWord = index
                                            lastIndex = decod + 1
                                            break
                                        else:
                                            nextWordD = nextWord
                                            stillToDo = []
                                            window = 1
                                            done = True
                                            lastIndexWord = index - 1
                                            lastIndex = decod
                                            #print("Index out " + str(lastIndexWord))
                                            break

            lastIndexWord += 1
            '''
            if not done:
                stillToDo.append(indexWord)
                window += 1
                # indices.append([lastIndex])
                # lastIndex + 1
            '''
        '''
        for index in range(len(stillToDo)):
            tuple = [lastIndex]
            indices.append(tuple)
            lastIndex += 1
        '''
        #print("Indices : " + str(indices))
        tagsSentence= {"Verbs":[],"Time":[],"Relative":[]}
        positionsSentence = []
        sentence = getSentence(words)
        sentenceNLP = nlp(sentence)
        indexNLPCurrent = 0
        for item in items['verbs']:
            positionsItem = False
            tags = item['tags']
            verbIndex = -1
            lastVerbs = []
            timeIndex = -1
            timeVerb = []
            args = set()
            #print('Lengte tags: ' + str(len(tags)))
            #print('Lengte indices: ' + str(len(indices)))
            for index in range(len(tags)):
                if tags[index][2:]=="V":
                    '''
                    indexV = indices[index]
                    startIndex = index
                    endIndex = index
                    while len(indices[startIndex])==0 and startIndex
                    '''
                    tagsSentence["Verbs"].append(indices[index])
                    if verbIndex == -1:
                        verbIndex = indices[index][0]
                        done = False
                        for indexNLP in range(indexNLPCurrent,len(sentenceNLP)):
                            if(str.lower(words[index]).find(str.lower(str(sentenceNLP[indexNLP])))!=-1):
                                '''
                                    0 = verleden
                                    1 = heden
                                    2 = toekomst
                                    -1 = vaag
                                '''
                                if sentenceNLP[indexNLP].pos_ == "VERB" or sentenceNLP[indexNLP].pos_ =="AUX":
                                    indexNLPCurrent = indexNLP
                                    if sentenceNLP[indexNLP].tag_ == "VBD":
                                        tagsSentence["Relative"].append("PAST_REF")
                                        done = True
                                    else:
                                        if sentenceNLP[indexNLP].tag_ == "VBP" or sentenceNLP[indexNLP].tag_ == "VBZ":
                                            tagsSentence["Relative"].append("PRESENT_REF")
                                            done = True
                                        else:
                                            if sentenceNLP[indexNLP].pos_ == "AUX" and (sentenceNLP[indexNLP].lemma_ == "shall" or sentenceNLP[indexNLP].lemma_ == "will"):
                                                tagsSentence["Relative"].append("FUTURE_REF")
                                                done = True
                                            else:
                                                for child in sentenceNLP[indexNLP].children:
                                                    if child.dep_ == "aux" and (child.lemma_ == "shall" or child.lemma_ == "will"):
                                                        tagsSentence["Relative"].append("FUTURE_REF")
                                                        done = True
                                                        break
                                    if not done:
                                        tagsSentence["Relative"].append(-1)
                                        done = True
                                    break
                        if not done:
                            tagsSentence["Relative"].append(-1)
                    lastVerbs.append(indices[index])
                else:
                    if tags[index][2:] == "ARGM-TMP":
                        timeVerb += indices[index]
                        #tagsSentence["Time"].append(indices[index])
                        if timeIndex == -1:
                            timeIndex = indices[index][0]
                            positionsItem = True
                        args.add(tags[index][2:])
                    else:
                        if tags[index] != "O":
                            args.add(tags[index][2:])
            if len(timeVerb)>0:
                tagsSentence["Time"].append(list(dict.fromkeys(timeVerb)))
            if len(args)>1:
                if positionsItem:
                    positionsItem = ([verbIndex, timeIndex - verbIndex])
                else:
                    positionsItem = []
                positionsSentence.append(positionsItem)
            else:
                for index in lastVerbs:
                    tagsSentence["Verbs"].remove(index)
                if len(tagsSentence["Relative"])>0:
                    tagsSentence["Relative"].pop()

        indicesAll += indices
        zinnenIndices.append(len(indicesAll))
        index = lastIndex
        #print(tagsSentence["Verbs"])
        #print(tagsSentence["Time"])
        #print(tagsSentence["Relative"])
        tagsAll["Verbs"] += tagsSentence["Verbs"]
        tagsAll["Time"] += tagsSentence["Time"]
        tagsAll["Relative"] += tagsSentence["Relative"]
        for pos in positionsSentence:
            if len(pos)>0:
                positionsAll.append(pos)
    #print(positionsAll)
    #print(indicesAll)
    #print("Last indices : " + str(indicesAll[-1]))
    #print("Lengte decoding : " + str(len(decoding)))
    #print(tagsAll)
    #print(positionsAll)
    #print(allWords)
    return indicesAll,tagsAll,positionsAll,allWords,zinnenIndices


def linkToVerb(OIE, sentenceIndices, indicesTimexes, positionsAll, tagsTime):
    nlp = spacy.load("en_core_web_trf")
    tagsIndices = []
    newPostions = []
    for index in indicesTimexes:
        done = False
        for position in positionsAll:
            if len(position) == 2:
                startIndex = position[0] + position[1]
                times = tagsTime['Time']
                endIndex = startIndex
                for time in times:
                    if endIndex == time[0]:
                        endIndex = time[-1]
                if bool(set(range(startIndex, endIndex)) & set(range(index[0], index[1]))):
                    done = True
                    if index[0] < startIndex:
                        tagsIndices.append(index[0])
                        newPostions.append([position[0], index[0]])
                    else:
                        tagsIndices.append(startIndex)
                        newPostions.append([position[0], startIndex])
'''
def match(indicesAll, allWordsOIE, allWordsHeidel, hasTitle, indicesRefs):
    print(indicesAll)
    print(allWordsOIE)
    print(allWordsHeidel)
    print(hasTitle)
    print(indicesRefs)
    indicesAllHeidel = []
    index = 0
    wait = 0
    indicesTimexes = []
    firstSentence = True
    timexes = dict()
    for indexHeidel in range(len(allWordsHeidel)):
        lastIndex = index
        word = str(allWordsHeidel[indexHeidel]).replace(' ', '')
        if len(word) != 0:
            if indexHeidel + 1 < len(allWordsHeidel):
                nextWord = str(allWordsHeidel[indexHeidel + 1]).replace(' ', '')
                if hasTitle and word == "'" and nextWord == 'says' and firstSentence:
                    firstSentence = False
                    wait = 2
            if wait == 0:
                for indexWord in range(lastIndex, len(allWordsOIE)):
                    # print('Next')
                    # print(word)
                    # print(allWordsOIE[indexWord])
                    if (allWordsOIE[indexWord] != '...'):
                        # print('Heidel')
                        # print(word.lower())
                        # print('OIE')
                        # print(allWordsOIE[indexWord].lower())
                        if word.lower() == allWordsOIE[indexWord].lower():
                            indicesAllHeidel.append(indicesAll[indexWord])
                            for indices in indicesRefs:
                                if indices[0] == indexHeidel:
                                    timexes[indices[0]] = indicesAll[indexWord][0]
                                    print("timex 1 :" +  str(timexes[indices[0]]))
                                if indices[1] == indexHeidel:
                                    startIndex = timexes.get(indices[0])
                                    endIndex = indicesAll[indexWord][-1]
                                    print("StartIndex 1 : " + str(startIndex))
                                    indicesTimexes.append([startIndex, endIndex])
                            index = indexWord + 1
                            break
                        else:
                            parts = word.lower().split(allWordsOIE[indexWord].lower())
                            index = indexWord
                            if len(parts) > 1:
                                for indices in indicesRefs:
                                    print(indexHeidel)
                                    if indices[0] == indexHeidel:
                                        timexes[indices[0]] = indicesAll[indexWord][0]
                                        print("timex 2 :" +  str(timexes[indices[0]]))
                                    if indices[1] == indexHeidel:
                                        startIndex = timexes.get(indices[0])
                                        endIndex = indicesAll[indexWord][-1]
                                        print("StartIndex 2 : " + str(startIndex))
                                        print(indices[0])
                                        print(indices[1])
                                        print(indexWord)
                                        indicesTimexes.append([startIndex, endIndex])
                            while len(parts) > 1:
                                # print('Parts')
                                # print(parts)
                                if len(parts[0]) == 0:
                                    indicesAllHeidel.append(indicesAll[index])
                                    index += 1
                                    parts = parts[1].lower().split(allWordsOIE[index].lower())
                                    # print(parts)
                                else:
                                    for indices in indicesRefs:
                                        if indices[1] == indexHeidel:
                                            startIndex = timexes.get(indices[0])
                                            endIndex = indicesAll[index][-1]
                                            print("StartIndex 3 : " + str(startIndex))
                                            indicesTimexes.append([startIndex, endIndex])
                                    parts = parts[0]
                            break

            wait = max(0, wait - 1)
            # print('Wait ---')
            # print(word)
            # print(wait)
    print(indicesAll)
    print(allWordsOIE)
    print(allWordsHeidel)
    print(indicesAllHeidel)
    print("IndicesTimexes : ")
    print(indicesTimexes)
    return indicesTimexes
'''
'''
def match(indicesAll, allWordsOIE, allWordsHeidel, hasTitle, indicesRefs):
    print(indicesAll)
    print("Lengte : " + str(len(indicesAll)))
    print(allWordsOIE)
    print("Lengte Words : " + str(len(allWordsOIE)))
    print(allWordsHeidel)
    #print(hasTitle)
    print(indicesRefs)
    indicesAllHeidel = []
    indexO = 0
    wait = 0
    indicesTimexes = []
    if hasTitle:
        firstSentence = True
    else:
        firstSentence = False
    timexes = dict()
    lastIndexHeidel = 0
    indicesRefs2 = indicesRefs.copy()
    while lastIndexHeidel<len(allWordsHeidel):
        indexHeidel = lastIndexHeidel
        #print('First : ' + str(indexHeidel))
        lastIndex = indexO
        if firstSentence:
            if indexHeidel + 1 < len(allWordsHeidel):
                word2 = str(allWordsHeidel[indexHeidel]).replace(' ', '').lower()
                nextWord = str(allWordsHeidel[indexHeidel + 1]).replace(' ', '').lower()
                if (word2.find("'")!=-1 and nextWord == 'says' and firstSentence):
                    #print('token : ' + word)
                    indicesAllHeidel.append([])
                    indicesAllHeidel.append([])
                    lastIndexHeidel += 2
                    firstSentence = False
                    continue

        word = str(allWordsHeidel[indexHeidel]).replace('.','').replace(chr(8220),'').replace('�','').replace(' ', '').replace("'", "").replace('"', '').replace('<', '').replace('/',
                                                                                                                 '').replace(
            '>', '').replace("\\", '').replace('(','').replace(')','').lower().encode("ascii", "ignore").decode()
        while len(word) == 0:
            #print("Empty")
            indicesAllHeidel.append([lastIndex])
            if indexHeidel +1 < len(allWordsHeidel):
                indexHeidel += 1
                word = str(allWordsHeidel[indexHeidel]).replace('.','').replace(chr(8220),'').replace('�','').replace(' ', '').replace("'", "").replace('"', '').replace('<',
                                                                                                                   '').replace(
                    '/',
                    '').replace(
                    '>', '').replace("\\", '').replace('(','').replace(')','').lower().encode("ascii", "ignore").decode()
            else:
                break
        if len(word) == 0:
            break
        #print('Word : ' + word)
        #print('Second : ' + str(indexHeidel))
        lastIndexHeidel =indexHeidel + 1
        for indexWord in range(lastIndex, len(allWordsOIE)):
            wordDecoding = str(allWordsOIE[indexWord]).replace('.','').replace(chr(8220),'').replace('�','').replace(' ', '').replace("'", "").replace('"', '').replace(
                '<', '').replace('/', '').replace(
                '>', '').replace("\\", '').replace('(','').replace(')','').lower().encode("ascii", "ignore").decode()
            while (len(wordDecoding) == 0):
                indexWord += 1
                if indexWord < len(allWordsOIE):
                    wordDecoding = str(allWordsOIE[indexWord]).replace('.','').replace(chr(8220),'').replace('�','').replace(' ', '').replace("'", "").replace('"',
                                                                                                  '').replace(
                        '<', '').replace('/', '').replace(
                        '>', '').replace("\\", '').replace('(','').replace(')','').lower().encode("ascii", "ignore").decode()
                else:
                    # print('Break')
                    break
            if len(wordDecoding) == 0:
                break

            # print('Next')

            #print(word)
            #print(allWordsOIE[indexWord])

            #print('Heidel')
            #print(word.lower())
            #print('OIE')
            #print(wordDecoding)
            if word == wordDecoding:
                #print(indexWord)
                indicesAllHeidel.append(indicesAll[indexWord])
                for indices in indicesRefs:
                    if indices[0] == indexHeidel:
                        timexes[indices[0]] = indicesAll[indexWord][0]
                        #print("timex 1 :" +  str(timexes[indices[0]]))
                    if indices[-1] == indexHeidel:
                        startIndex = timexes.get(indices[0])
                        if startIndex != None:
                            endIndex = indicesAll[indexWord][-1]
                            #print("StartIndex 1 : " + str(startIndex))
                            indicesTimexes.append([startIndex, endIndex])
                            indicesRefs2.remove(indices)
                indexO = indexWord + 1
                break
            else:
                #print(parts)
                index = indexWord
                #print(index)
                wordDecoding = str(allWordsOIE[index]).replace('.','').replace(chr(8220),'').replace('�','').replace(' ', '').replace("'", "").replace('"',
                                                                                                 '').replace(
                    '<', '').replace('/', '').replace(
                    '>', '').replace("\\", '').replace('(','').replace(')','').lower().encode("ascii", "ignore").decode()
                #print(wordDecoding)
                #print(index)
                parts = word.lower().split(wordDecoding)
                if len(parts) >=2 and index +1 <len(allWordsOIE) and len(parts[0])==0:
                    lengte = len(wordDecoding)
                    newIndices = []
                    index += 1
                    wordDecoding = str(allWordsOIE[index]).replace('.','').replace(chr(8220),'').replace('�','').replace(' ', '').replace("'", "").replace('"',
                                                                                                  '').replace(
                        '<', '').replace('/', '').replace(
                        '>', '').replace("\\", '').replace('(','').replace(')','').lower().encode("ascii", "ignore").decode()
                    #print(wordDecoding)
                    #print(index)
                    while (len(wordDecoding) == 0):
                        index += 1
                        if index < len(allWordsOIE):
                            wordDecoding = str(allWordsOIE[indexWord]).replace('.', '').replace(chr(8220), '').replace(
                                '�',
                                '').replace(
                                ' ', '').replace("'", "").replace('"',
                                                                  '').replace(
                                '<', '').replace('/', '').replace(
                                '>', '').replace("\\", '').replace('(', '').replace(')', '').lower().encode(
                                "ascii", "ignore").decode()
                        else:
                            # print('Break')
                            break
                    if len(wordDecoding) != 0:
                        nextWord = word[lengte:]
                        parts = nextWord.split(wordDecoding)
                    #print(parts)
                        if len(parts)>=2 and len(parts[0]) == 0:
                            newIndices += indicesAll[index-1]
                            newIndices += indicesAll[index]
                            for indices in indicesRefs:
                                startIndex = timexes.get(indices[0])
                                if indices[0] == indexHeidel:
                                    timexes[indices[0]] = indicesAll[indexWord][0]
                                    #print("timex 1 :" + str(timexes[indices[0]]))
                                if indices[1] == indexHeidel:
                                    if startIndex != None:
                                        endIndex = indicesAll[indexWord][-1]
                                        #print("StartIndex 1 : " + str(startIndex))
                                        indicesTimexes.append([startIndex, endIndex])
                                        indicesRefs2.remove(indices)
                            index += 1
                            while len(parts) >= 2 and len(parts[0]) == 0:
                                lengte = len(wordDecoding)
                                wordDecoding = str(allWordsOIE[index]).replace('.','').replace(chr(8220),'').replace('�','').replace(' ', '').replace("'", "").replace('"',
                                                                                                                 '').replace(
                                    '<', '').replace('/', '').replace(
                                    '>', '').replace("\\", '').replace('(','').replace(')','').lower().encode("ascii", "ignore").decode()
                                while (len(wordDecoding) == 0):
                                    index += 1
                                    if index < len(allWordsOIE):
                                        wordDecoding = str(allWordsOIE[indexWord]).replace('.','').replace(chr(8220), '').replace('�',
                                                                                                                  '').replace(
                                            ' ', '').replace("'", "").replace('"',
                                                                              '').replace(
                                            '<', '').replace('/', '').replace(
                                            '>', '').replace("\\", '').replace('(', '').replace(')', '').lower().encode(
                                            "ascii", "ignore").decode()
                                    else:
                                        # print('Break')
                                        break
                                #print(wordDecoding)
                                #print(index)
                                if len(wordDecoding) != 0:
                                    nextWord = nextWord[lengte:]
                                    parts = nextWord.split(wordDecoding)
                                    #print('Parts in ')
                                    #print(parts)
                                    if len(parts)>=2 and len(parts[0]) == 0:
                                        newIndices += indicesAll[index]
                                        index += 1
                                        #print('Index 2 : ' + str(index))
                                        if index >= len(allWordsOIE):
                                            parts = []
                                        for indices in indicesRefs:
                                            if indices[0] == indexHeidel:
                                                timexes[indices[0]] = indicesAll[indexWord][0]
                                                #print("timex 1 :" + str(timexes[indices[0]]))
                                            startIndex = timexes.get(indices[0])
                                            if indices[-1] == indexHeidel:
                                                if startIndex != None:
                                                    endIndex = indicesAll[indexWord][-1]
                                                    #print("StartIndex 1 : " + str(startIndex))
                                                    indicesTimexes.append([startIndex, endIndex])
                                                    indicesRefs2.remove(indices)
                                    else:
                                        parts = []
                                else:
                                    parts = []
                            if nextWord == wordDecoding:
                                newIndices += indicesAll[index]
                                index += 1
                            #print(newIndices)
                            indexO = index
                            #print('Index O : '+ str(indexO))
                            indicesAllHeidel.append(newIndices)
                            break
                        else:
                            if nextWord == wordDecoding:
                                newIndices += indicesAll[index - 1]
                                newIndices += indicesAll[index]
                                indicesAllHeidel.append(newIndices)
                                index += 1
                                indexO = index
                            else:
                                indicesAllHeidel.append([])
                                indexO = index
                                break
                    else:
                        indicesAllHeidel.append([index])
                        indexO = index
                        break

                else:
                    parts = wordDecoding.split(word)
                    #print(parts)
                    if len(parts) >= 2 and indexHeidel +1 <len(allWordsHeidel) and len(parts[0])==0:
                        lengte = len(word)
                        indexHeidel += 1
                        word = str(allWordsHeidel[indexHeidel]).replace('.','').replace(chr(8220),'').replace('�','').replace(' ', '').replace("'", "").replace('"',
                                                                                                       '').replace(
                                    '<', '').replace('/', '').replace(
                                    '>', '').replace("\\", '').replace('(','').replace(')','').lower().encode("ascii", "ignore").decode()
                        while len(word) == 0:
                            # print("Empty")
                            indicesAllHeidel.append([])
                            if indexHeidel + 1 < len(allWordsHeidel):
                                indexHeidel += 1
                                word = str(allWordsHeidel[indexHeidel]).replace('.','').replace(chr(8220), '').replace('�', '').replace(
                                    ' ', '').replace("'", "").replace('"', '').replace('<',
                                                                                       '').replace(
                                    '/',
                                    '').replace(
                                    '>', '').replace("\\", '').replace('(', '').replace(')', '').lower().encode("ascii",
                                                                                                                "ignore").decode()
                            else:
                                break
                        if len(word) > 0:
                            #print('Word 2 : ' + word)
                            nextWord = wordDecoding[lengte:]
                            parts = nextWord.lower().split(word.lower())
                            #print(parts)
                            if len(parts) > 1:
                                if len(parts)>=2 and len(parts[0]) == 0:
                                    indicesAllHeidel.append(indicesAll[index])
                                    indicesAllHeidel.append(indicesAll[index])
                                    indexHeidel += 1
                                    for indices in indicesRefs:
                                        startIndex = timexes.get(indices[0])
                                        if indices[0] == indexHeidel-2:
                                            timexes[indices[0]] = indicesAll[indexWord][0]
                                            #print("timex 1 :" + str(timexes[indices[0]]))
                                        if indices[1] == indexHeidel:
                                            if startIndex != None:
                                                endIndex = indicesAll[indexWord][-1]
                                                #print("StartIndex 1 : " + str(startIndex))
                                                indicesTimexes.append([startIndex, endIndex])
                                                indicesRefs2.remove(indices)
                                    while len(parts) >=2 and len(parts[0])==0:
                                        #print('Word 2 : ' + word)
                                        lengte = len(word)
                                        word = str(allWordsHeidel[indexHeidel]).replace('.','').replace(chr(8220),'').replace('�','').replace(' ', '').replace("'", "").replace('"',
                                                                                                                   '').replace(
                                                    '<', '').replace('/', '').replace(
                                                    '>', '').replace("\\", '').replace('(','').replace(')','').lower().encode("ascii", "ignore").decode()
                                        while len(word) == 0:
                                            # print("Empty")
                                            indicesAllHeidel.append([])
                                            if indexHeidel + 1 < len(allWordsHeidel):
                                                indexHeidel += 1
                                                word = str(allWordsHeidel[indexHeidel]).replace('.','').replace(chr(8220), '').replace(
                                                    '�', '').replace(
                                                    ' ', '').replace("'", "").replace('"', '').replace('<',
                                                                                                       '').replace(
                                                    '/',
                                                    '').replace(
                                                    '>', '').replace("\\", '').replace('(', '').replace(')',
                                                                                                        '').lower().encode(
                                                    "ascii",
                                                    "ignore").decode()
                                            else:
                                                break
                                        if len(word)>0:
                                            nextWord = nextWord[lengte:]
                                            parts = nextWord.lower().split(word.lower())
                                            #print('Parts in 2')
                                            #print(parts)
                                            if len(parts)>=2 and len(parts[0]) == 0:
                                                indicesAllHeidel.append(indicesAll[index])
                                                indexHeidel += 1
                                                if indexHeidel >= len(allWordsHeidel):
                                                    parts = []
                                                for indices in indicesRefs:
                                                    startIndex = timexes.get(indices[0])
                                                    if indices[-1] == indexHeidel:
                                                        if startIndex != None:
                                                            endIndex = indicesAll[indexWord][-1]
                                                            #print("StartIndex 1 : " + str(startIndex))
                                                            indicesTimexes.append([startIndex, endIndex])
                                                            indicesRefs2.remove(indices)
                                            else:
                                                parts = []
                                        else:
                                            parts = []
                                    if nextWord == word:
                                        indicesAllHeidel.append(indicesAll[index])
                                        indexHeidel += 1
                                        indexO += 1
                                        for indices in indicesRefs:
                                            startIndex = timexes.get(indices[0])
                                            if indices[-1] == indexHeidel:
                                                if startIndex != None:
                                                    endIndex = indicesAll[indexWord][-1]
                                                    # print("StartIndex 1 : " + str(startIndex))
                                                    indicesTimexes.append([startIndex, endIndex])
                                                    indicesRefs2.remove(indices)
                                    lastIndexHeidel = indexHeidel
                                    break
                                else:
                                    if nextWord == word:
                                        indicesAllHeidel.append(indicesAll[index])
                                        indexHeidel += 1
                                        indexO += 1
                                        for indices in indicesRefs:
                                            startIndex = timexes.get(indices[0])
                                            if indices[-1] == indexHeidel:
                                                if startIndex != None:
                                                    endIndex = indicesAll[indexWord][-1]
                                                    # print("StartIndex 1 : " + str(startIndex))
                                                    indicesTimexes.append([startIndex, endIndex])
                                                    indicesRefs2.remove(indices)
    #print("indicesRefs over")
    #print(indicesRefs2)
    #print(indicesAllHeidel)
    for indices in indicesRefs2:
        print("Indices : " + str(indices))
        startIndex = indicesAllHeidel[indices[0]][-1]
        endIndex = indicesAllHeidel[indices[-1]][-1]
        #print("StartIndex 1 : " + str(startIndex))
        indicesTimexes.append([startIndex, endIndex])
            # print('Wait ---')
            # print(word)
            # print(wait)
    #print(indicesAll)
    #print(allWordsOIE)
    #print(allWordsHeidel)

    #print("IndicesTimexes : ")
    #print(indicesTimexes)
    return indicesTimexes
'''
def match(indicesAll, allWordsOIE, allWordsHeidel, hasTitle, indicesRefs):
    #print(indicesAll)
    #print("Lengte : " + str(len(indicesAll)))
    #print(allWordsOIE)
    #print("Lengte Words : " + str(len(allWordsOIE)))
    #print(allWordsHeidel)
    #print(hasTitle)
    #print(indicesRefs)
    indicesAllHeidel = []
    indexO = 0
    wait = 0
    indicesTimexes = []
    if hasTitle:
        firstSentence = True
    else:
        firstSentence = False
    timexes = dict()
    lastIndexHeidel = 0
    indicesRefs2 = indicesRefs.copy()
    nextWordD = None
    nextWordE = None
    while lastIndexHeidel<len(allWordsHeidel):
        goToNext = False
        indexHeidel = lastIndexHeidel
        #print('First : ' + str(indexHeidel))
        lastIndex = indexO
        #print("First : " + str(allWordsHeidel[indexHeidel]))
        if firstSentence:
            if indexHeidel + 1 < len(allWordsHeidel) and lastIndex + 1 < len(allWordsOIE):
                #word2 = str(allWordsHeidel[indexHeidel]).replace('•','').replace('_','').replace('[','').replace(']','').replace(' ', '').lower()
                #nextWord = str(allWordsHeidel[indexHeidel + 1]).replace(' ', '').lower()
                word2 = str(allWordsHeidel[indexHeidel]).replace(' ', '').lower()
                nextWord = str(allWordsHeidel[indexHeidel + 1]).replace(' ', '').lower()
                word2D = str(allWordsOIE[lastIndex]).replace(' ', '').lower()
                nextWordDecoding = str(allWordsOIE[lastIndex + 1]).replace(' ', '').lower()
                #print("Word2 : " + word2)
                #print("NextWord : " +nextWord)
                #print("Word2D : " + word2D)
                #print("NextWordD : " + nextWordDecoding)
                if not (word2D == word2 and nextWord == nextWordDecoding):
                    if (len(word2)>0 and word2[-1] == "'" and nextWord == 'says'):
                        #print('token : ' + word)
                        indicesAllHeidel.append(indicesAll[min(lastIndex,len(indicesAll)-1)])
                        indicesAllHeidel.append(indicesAll[min(lastIndex,len(indicesAll)-1)])
                        lastIndexHeidel = indexHeidel + 2
                        firstSentence = False
                        continue

        word = str(allWordsHeidel[indexHeidel]).replace('!','').replace('İ','').replace('@','').replace(',','').replace('•','').replace('_','').replace('[','').replace(']','').replace('*','').replace('.','').replace(chr(8220),'').replace('�','').replace(' ', '').replace("'", "").replace('"', '').replace('<', '').replace('/',
                                                                                                                 '').replace(
            '>', '').replace("\\", '').replace('(','').replace(')','').lower().encode("ascii", "ignore").decode()
        if nextWordE != None:
            word = nextWordE + word
        indexDecoding = lastIndex
        if indexDecoding < len(allWordsOIE):
            word2D = str(allWordsOIE[indexDecoding]).replace('!','').replace('İ','').replace('@','').replace(',','').replace('•','').replace('_','').replace('[','').replace(']','').replace('*','').replace('.','').replace(chr(8220),'').replace('�','').replace(' ', '').replace('"', '').replace('<', '').replace('/',
                                                                                                                     '').replace(
                '>', '').replace("\\", '').replace('(','').replace(')','').lower().encode("ascii", "ignore").decode()
        while(len(word2D)==0) and indexDecoding + 1 < len(allWordsOIE):
            indexDecoding += 1
            word2D = str(allWordsOIE[indexDecoding]).replace('!', '').replace('İ', '').replace('@', '').replace(',',
                                                                                                                '').replace(
                '•', '').replace('_', '').replace('[', '').replace(']', '').replace('*', '').replace('.', '').replace(
                chr(8220), '').replace('�', '').replace(' ', '').replace('"', '').replace('<', '').replace('/',
                                                                                                           '').replace(
                '>', '').replace("\\", '').replace('(', '').replace(')', '').lower().encode("ascii", "ignore").decode()
        while len(word) == 0:
            #print("Empty")
            nextWordD = None
            indicesAllHeidel.append(indicesAll[min(lastIndex,len(indicesAll)-1)])
            if indexHeidel +1 < len(allWordsHeidel):
                if firstSentence and lastIndex + 1 < len(allWordsOIE):
                    word2 = str(allWordsHeidel[indexHeidel]).replace(' ', '').lower()
                    nextWord = str(allWordsHeidel[indexHeidel + 1]).replace(' ', '').lower()
                    word2D = str(allWordsOIE[indexDecoding]).replace(' ', '').lower()
                    nextWordDecoding = str(allWordsOIE[indexDecoding + 1]).replace(' ', '').lower()
                    #print("Word2 : " + word2)
                    #print("NextWord : " + nextWord)
                    #print("Word2D : " + word2D)
                    #print("NextWordD : " + nextWordDecoding)

                    if not (word2D == word2 and nextWord == nextWordDecoding):
                        if (len(word2)>0 and word2[-1] == "'" and nextWord == 'says'):
                            indicesAllHeidel.append(indicesAll[min(lastIndex,len(indicesAll)-1)])
                            indicesAllHeidel.append(indicesAll[min(lastIndex, len(indicesAll) - 1)])
                            lastIndexHeidel =  indexHeidel + 2
                            firstSentence = False
                            goToNext = True
                            break
                indexHeidel += 1
                word = str(allWordsHeidel[indexHeidel]).replace('!','').replace('İ','').replace('@','').replace(',','').replace('•','').replace('_','').replace('[','').replace(']','').replace('*','').replace('.','').replace(chr(8220),'').replace('�','').replace(' ', '').replace("'", "").replace('"', '').replace('<',
                                                                                                                   '').replace(
                    '/',
                    '').replace(
                    '>', '').replace("\\", '').replace('(','').replace(')','').lower().encode("ascii", "ignore").decode()
            else:
                break
        if goToNext:
            continue
        if len(word) == 0:
            break
        #print("Second: " + str(allWordsHeidel[indexHeidel]))
        #print("Index: " + str(indexHeidel))
        #print('Word : ' + word)
        #print('Second : ' + str(indexHeidel))
        #print(firstSentence)
        lastIndexHeidel = indexHeidel + 1
        #print("LastIndexHeidel: " + str(lastIndexHeidel))
        for indexWord in range(lastIndex, len(allWordsOIE)):
            if nextWordD == None:
                wordDecoding = str(allWordsOIE[indexWord]).replace('!','').replace('İ','').replace('@','').replace(',','').replace('•','').replace('_','').replace('[','').replace(']','').replace('*','').replace('.','').replace(chr(8220),'').replace('�','').replace(' ', '').replace("'", "").replace('"', '').replace(
                    '<', '').replace('/', '').replace(
                    '>', '').replace("\\", '').replace('(','').replace(')','').lower().encode("ascii", "ignore").decode()
                while (len(wordDecoding) == 0):
                    indexWord += 1
                    if indexWord < len(allWordsOIE):
                        wordDecoding = str(allWordsOIE[indexWord]).replace('!','').replace('İ','').replace('@','').replace(',','').replace('•','').replace('_','').replace('[','').replace(']','').replace('*','').replace('.','').replace(chr(8220),'').replace('�','').replace(' ', '').replace("'", "").replace('"',
                                                                                                      '').replace(
                            '<', '').replace('/', '').replace(
                            '>', '').replace("\\", '').replace('(','').replace(')','').lower().encode("ascii", "ignore").decode()
                    else:
                        # print('Break')
                        break
                if len(wordDecoding) == 0:
                    break
            else:
                wordDecoding = nextWordD

            #print('Next')

            #print(word)
            #print(allWordsOIE[indexWord])

            #print('Heidel')
            #print(word.lower())
            #print('OIE')
            #print(wordDecoding)
            #print("FirstSentence : " + str(firstSentence))
            if word == wordDecoding:
                #print(indexWord)
                nextWordD = None
                nextWordE = None
                indicesAllHeidel.append(indicesAll[indexWord])
                '''
                for indices in indicesRefs:
                    if indices[0] == indexHeidel:
                        timexes[indices[0]] = indicesAll[indexWord][0]
                        #print("timex 1 :" +  str(timexes[indices[0]]))
                    if indices[-1] == indexHeidel:
                        startIndex = timexes.get(indices[0])
                        if startIndex != None:
                            endIndex = indicesAll[indexWord][-1]
                            #print("StartIndex 1 : " + str(startIndex))
                            indicesTimexes.append([startIndex, endIndex])
                            indicesRefs2.remove(indices)
                '''
                indexO = indexWord + 1
                break
            else:
                index = indexWord
                #print(index)
                '''
                wordDecoding = str(allWordsOIE[index]).replace('.','').replace(chr(8220),'').replace('�','').replace(' ', '').replace("'", "").replace('"',
                                                                                                 '').replace(
                    '<', '').replace('/', '').replace(
                    '>', '').replace("\\", '').replace('(','').replace(')','').lower().encode("ascii", "ignore").decode()
                '''
                #print(wordDecoding)
                #print(index)
                parts = word.lower().split(wordDecoding)
                if len(parts) >=2 and index +1 <len(allWordsOIE) and len(parts[0])==0:
                    nextWordD = None
                    nextWordE = None
                    lengte = len(wordDecoding)
                    newIndices = []
                    newIndices += indicesAll[index]
                    index += 1
                    '''
                    for indices in indicesRefs:
                        startIndex = timexes.get(indices[0])
                        if indices[0] == indexHeidel:
                            timexes[indices[0]] = indicesAll[index][0]
                            # print("timex 1 :" + str(timexes[indices[0]]))
                        if indices[1] == indexHeidel:
                            if startIndex != None:
                                endIndex = indicesAll[index][-1]
                                # print("StartIndex 1 : " + str(startIndex))
                                indicesTimexes.append([startIndex, endIndex])
                                indicesRefs2.remove(indices)
                    '''
                    wordDecoding = str(allWordsOIE[index]).replace('!','').replace('İ','').replace('@','').replace(',','').replace('•','').replace('_','').replace('[','').replace(']','').replace('*','').replace('.','').replace(chr(8220),'').replace('�','').replace(' ', '').replace("'", "").replace('"',
                                                                                                  '').replace(
                        '<', '').replace('/', '').replace(
                        '>', '').replace("\\", '').replace('(','').replace(')','').lower().encode("ascii", "ignore").decode()
                    #print(wordDecoding)
                    #print(index)
                    while (len(wordDecoding) == 0):
                        index += 1
                        if index < len(allWordsOIE):
                            wordDecoding = str(allWordsOIE[index]).replace('!','').replace('İ','').replace('@','').replace(',','').replace('•','').replace('_','').replace('[','').replace(']','').replace('*','').replace('.', '').replace(chr(8220), '').replace(
                                '�',
                                '').replace(
                                ' ', '').replace("'", "").replace('"',
                                                                  '').replace(
                                '<', '').replace('/', '').replace(
                                '>', '').replace("\\", '').replace('(', '').replace(')', '').lower().encode(
                                "ascii", "ignore").decode()
                        else:
                            # print('Break')
                            break
                    if len(wordDecoding) != 0:
                        nextWord = word[lengte:]
                        parts = nextWord.split(wordDecoding)
                        if len(parts) >= 2 and len(parts[0]) == 0 and not(nextWord==wordDecoding):
                            nextWordD = None
                            nextWordE = None
                            newIndices += indicesAll[index]
                            '''
                            for indices in indicesRefs:
                                startIndex = timexes.get(indices[0])
                                if indices[0] == indexHeidel:
                                    timexes[indices[0]] = indicesAll[index][0]
                                    #print("timex 1 :" + str(timexes[indices[0]]))
                                if indices[1] == indexHeidel:
                                    if startIndex != None:
                                        endIndex = indicesAll[index][-1]
                                        #print("StartIndex 1 : " + str(startIndex))
                                        indicesTimexes.append([startIndex, endIndex])
                                        indicesRefs2.remove(indices)
                            '''
                            index += 1
                            while len(parts) >= 2 and len(parts[0]) == 0 and not(nextWord==wordDecoding):
                                lengte = len(wordDecoding)
                                wordDecoding = str(allWordsOIE[index]).replace('!','').replace('İ','').replace('@','').replace(',','').replace('•','').replace('_','').replace('[','').replace(']','').replace('*','').replace('.','').replace(chr(8220),'').replace('�','').replace(' ', '').replace("'", "").replace('"',
                                                                                                                 '').replace(
                                    '<', '').replace('/', '').replace(
                                    '>', '').replace("\\", '').replace('(','').replace(')','').lower().encode("ascii", "ignore").decode()
                                while (len(wordDecoding) == 0):
                                    index += 1
                                    if index < len(allWordsOIE):
                                        wordDecoding = str(allWordsOIE[index]).replace('!','').replace('İ','').replace('@','').replace(',','').replace('•','').replace('_','').replace('[','').replace(']','').replace('*','').replace('.','').replace(chr(8220), '').replace('�',
                                                                                                                  '').replace(
                                            ' ', '').replace("'", "").replace('"',
                                                                              '').replace(
                                            '<', '').replace('/', '').replace(
                                            '>', '').replace("\\", '').replace('(', '').replace(')', '').lower().encode(
                                            "ascii", "ignore").decode()
                                    else:
                                        index -=1
                                        # print('Break')
                                        break
                                #print(wordDecoding)
                                #print(index)
                                if len(wordDecoding) != 0:
                                    nextWord = nextWord[lengte:]
                                    parts = nextWord.split(wordDecoding)
                                    #print('Parts in ')
                                    #print(parts)
                                    if len(parts)>=2 and len(parts[0]) == 0 and not(nextWord==wordDecoding):
                                        newIndices += indicesAll[index]
                                        index += 1
                                        #print('Index 2 : ' + str(index))
                                        if index >= len(allWordsOIE):
                                            parts = []
                                        '''
                                        for indices in indicesRefs:
                                            if indices[0] == indexHeidel:
                                                timexes[indices[0]] = indicesAll[index][0]
                                                #print("timex 1 :" + str(timexes[indices[0]]))
                                            startIndex = timexes.get(indices[0])
                                            if indices[-1] == indexHeidel:
                                                if startIndex != None:
                                                    endIndex = indicesAll[index][-1]
                                                    #print("StartIndex 1 : " + str(startIndex))
                                                    indicesTimexes.append([startIndex, endIndex])
                                                    indicesRefs2.remove(indices)
                                        '''
                                else:
                                    parts = []
                            if nextWord == wordDecoding:
                                nextWordD = None
                                nextWordE = None
                                newIndices += indicesAll[index]
                                index += 1
                                indexO = index
                                #print('Index O : '+ str(indexO))
                                indicesAllHeidel.append(newIndices)
                                '''
                                for indices in indicesRefs:
                                    if indices[0] == indexHeidel:
                                        timexes[indices[0]] = indicesAll[indexWord][0]
                                        # print("timex 1 :" + str(timexes[indices[0]]))
                                    startIndex = timexes.get(indices[0])
                                    if indices[-1] == indexHeidel:
                                        if startIndex != None:
                                            endIndex = indicesAll[indexWord][-1]
                                            # print("StartIndex 1 : " + str(startIndex))
                                            indicesTimexes.append([startIndex, endIndex])
                                            indicesRefs2.remove(indices)
                                '''
                                break
                            else:
                                nextWordD = None
                                nextWordE = nextWord
                                indicesAllHeidel.append(newIndices)
                                indexO = index
                            break
                        else:
                            #print('Next word after')
                            #print(nextWord)
                            #print("Next wordDecoding after")
                            #print(wordDecoding)
                            if nextWord == wordDecoding:
                                nextWordD = None
                                nextWordE = None
                                newIndices += indicesAll[index]
                                index += 1
                                indexO = index
                                # print('Index O : '+ str(indexO))
                                indicesAllHeidel.append(newIndices)
                                '''
                                for indices in indicesRefs:
                                    if indices[0] == indexHeidel:
                                        timexes[indices[0]] = indicesAll[indexWord][0]
                                        # print("timex 1 :" + str(timexes[indices[0]]))
                                    startIndex = timexes.get(indices[0])
                                    if indices[-1] == indexHeidel:
                                        if startIndex != None:
                                            endIndex = indicesAll[indexWord][-1]
                                            # print("StartIndex 1 : " + str(startIndex))
                                            indicesTimexes.append([startIndex, endIndex])
                                            indicesRefs2.remove(indices)
                                '''
                                break
                            else:
                                nextWordD = None
                                nextWordE = nextWord
                                indicesAllHeidel.append(newIndices)
                                indexO = index
                            break
                else:
                    parts = wordDecoding.split(word)
                    #print(parts)
                    index = indexHeidel
                    if len(parts) >= 2 and index +1 <len(allWordsHeidel) and len(parts[0])==0:
                        nextWordD = None
                        nextWordE = None
                        indicesAllHeidel.append(indicesAll[indexWord])
                        '''
                        for indices in indicesRefs:
                            startIndex = timexes.get(indices[0])
                            if indices[0] == index:
                                timexes[indices[0]] = indicesAll[indexWord][0]
                                # print("timex 1 :" + str(timexes[indices[0]]))
                            if indices[1] == index:
                                if startIndex != None:
                                    endIndex = indicesAll[indexWord][-1]
                                    # print("StartIndex 1 : " + str(startIndex))
                                    indicesTimexes.append([startIndex, endIndex])
                                    indicesRefs2.remove(indices)
                        '''
                        lengte = len(word)
                        index += 1
                        word = str(allWordsHeidel[index]).replace('!','').replace('İ','').replace('@','').replace(',','').replace('•','').replace('_','').replace('[','').replace(']','').replace('*','').replace('.','').replace(chr(8220),'').replace('�','').replace(' ', '').replace("'", "").replace('"',
                                                                                                       '').replace(
                                    '<', '').replace('/', '').replace(
                                    '>', '').replace("\\", '').replace('(','').replace(')','').lower().encode("ascii", "ignore").decode()
                        while len(word) == 0:
                            # print("Empty")
                            indicesAllHeidel.append(indicesAll[indexWord])
                            index += 1
                            if index < len(allWordsHeidel):
                                word = str(allWordsHeidel[index]).replace('!','').replace('İ','').replace('@','').replace(',','').replace('•','').replace('_','').replace('[','').replace(']','').replace('*','').replace('.','').replace(chr(8220), '').replace('�', '').replace(
                                    ' ', '').replace("'", "").replace('"', '').replace('<',
                                                                                       '').replace(
                                    '/',
                                    '').replace(
                                    '>', '').replace("\\", '').replace('(', '').replace(')', '').lower().encode("ascii",
                                                                                                                "ignore").decode()
                            else:
                                break
                        if len(word) > 0:
                            #print('Word 2 : ' + word)
                            nextWord = wordDecoding[lengte:]
                            parts = nextWord.lower().split(word.lower())
                            #print(parts)
                            if len(parts) >= 2 and len(parts[0]) == 0 and not(nextWord==word):
                                nextWordD = None
                                nextWordE = None
                                indicesAllHeidel.append(indicesAll[indexWord])
                                '''
                                for indices in indicesRefs:
                                    startIndex = timexes.get(indices[0])
                                    if indices[0] == index:
                                        timexes[indices[0]] = indicesAll[indexWord][0]
                                        # print("timex 1 :" + str(timexes[indices[0]]))
                                    if indices[1] == index:
                                        if startIndex != None:
                                            endIndex = indicesAll[indexWord][-1]
                                            # print("StartIndex 1 : " + str(startIndex))
                                            indicesTimexes.append([startIndex, endIndex])
                                            indicesRefs2.remove(indices)
                                '''
                                index += 1
                                while len(parts) >= 2 and len(parts[0]) == 0 and not(nextWord==word):
                                    #print('Word 2 : ' + word)
                                    lengte = len(word)
                                    word = str(allWordsHeidel[index]).replace('!','').replace('İ','').replace('@','').replace(',','').replace('•','').replace('_','').replace('[','').replace(']','').replace('*','').replace('.','').replace(chr(8220),'').replace('�','').replace(' ', '').replace("'", "").replace('"',
                                                                                                               '').replace(
                                                '<', '').replace('/', '').replace(
                                                '>', '').replace("\\", '').replace('(','').replace(')','').lower().encode("ascii", "ignore").decode()
                                    while len(word) == 0:
                                        # print("Empty")
                                        indicesAllHeidel.append(indicesAll[indexWord])
                                        index += 1
                                        if index < len(allWordsHeidel):
                                            word = str(allWordsHeidel[index]).replace('!','').replace('İ','').replace('@','').replace(',','').replace('•','').replace('_','').replace('[','').replace(']','').replace('*','').replace('.','').replace(chr(8220), '').replace(
                                                '�', '').replace(
                                                ' ', '').replace("'", "").replace('"', '').replace('<',
                                                                                                   '').replace(
                                                '/',
                                                '').replace(
                                                '>', '').replace("\\", '').replace('(', '').replace(')',
                                                                                                    '').lower().encode(
                                                "ascii",
                                                "ignore").decode()
                                        else:
                                            index -= 1
                                            break
                                    if len(word)>0:
                                        nextWord = nextWord[lengte:]
                                        parts = nextWord.lower().split(word.lower())
                                        #print('Parts in 2')
                                        #print(parts)
                                        if len(parts) >= 2 and len(parts[0]) == 0 and not(nextWord==word):
                                            indicesAllHeidel.append(indicesAll[indexWord])
                                            nextWordD = None
                                            nextWordE = None
                                            index += 1
                                            if index>= len(allWordsHeidel):
                                                parts = []
                                            '''
                                            for indices in indicesRefs:
                                                startIndex = timexes.get(indices[0])
                                                if indices[-1] == index:
                                                    if startIndex != None:
                                                        endIndex = indicesAll[indexWord][-1]
                                                        #print("StartIndex 1 : " + str(startIndex))
                                                        indicesTimexes.append([startIndex, endIndex])
                                                        indicesRefs2.remove(indices)
                                            '''

                                    else:
                                        parts = []
                                if nextWord == word:
                                    nextWordD = None
                                    nextWordE = None
                                    indicesAllHeidel.append(indicesAll[indexWord])
                                    #indicesAllHeidel.append(indicesAll[lastIndex])
                                    indexO = indexWord + 1
                                    '''
                                    for indices in indicesRefs:
                                        startIndex = timexes.get(indices[0])
                                        if indices[-1] == index:
                                            if startIndex != None:
                                                endIndex = indicesAll[indexWord][-1]
                                                # print("StartIndex 1 : " + str(startIndex))
                                                indicesTimexes.append([startIndex, endIndex])
                                                indicesRefs2.remove(indices)
                                    '''
                                    lastIndexHeidel = index + 1
                                    break
                                else:
                                    nextWordD = nextWord
                                    indexO = indexWord
                                    lastIndexHeidel = index
                                    nextWordE = None
                                    break
                            else:
                                if nextWord == word:
                                    nextWordD = None
                                    nextWordE = None
                                    indicesAllHeidel.append(indicesAll[indexWord])
                                    lastIndexHeidel = index + 1
                                    indexO = indexWord + 1
                                    '''
                                    for indices in indicesRefs:
                                        startIndex = timexes.get(indices[0])
                                        if indices[-1] == indexHeidel:
                                            if startIndex != None:
                                                endIndex = indicesAll[indexWord][-1]
                                                # print("StartIndex 1 : " + str(startIndex))
                                                indicesTimexes.append([startIndex, endIndex])
                                                indicesRefs2.remove(indices)
                                    '''
                                else:
                                    nextWordD = nextWord
                                    nextWordE = None
                                    indexO = indexWord
                                    lastIndexHeidel = index
                                break
    for indices in indicesRefs2:
        if indices[0] < len(indicesAllHeidel) and indices[-1] < len(indicesAllHeidel):
            startIndex = indicesAllHeidel[indices[0]][0]
            endIndex = indicesAllHeidel[indices[-1]][-1]
            #print("StartIndex 1 : " + str(startIndex))
            indicesTimexes.append([startIndex, endIndex])
        else:
            print("*********Error**********")
            print("Indices : " + str(indices))
            print("IndicesAll")
            print(indicesAll)
            print("IndicesHeidel")
            print(indicesAllHeidel)
            print("Lengte : " + str(len(indicesAll)))
            print("AllWordsOIE")
            print(allWordsOIE)
            print("Lengte Words : " + str(len(allWordsOIE)))
            print("AllWordsHeidel")
            print(allWordsHeidel)
            print("Lengte Words : " + str(len(allWordsHeidel)))
            print("IndicesRefs : " )
            print(indicesRefs2)
            print("***********************")

            # print('Wait ---')
            # print(word)
            # print(wait)
    #print(indicesAll)
    #print(allWordsOIE)
    #print(allWordsHeidel)

    #print("IndicesTimexes : ")
    #print(indicesTimexes)
    return indicesTimexes
'''
def match(indicesAll, allWordsOIE, allWordsHeidel, hasTitle, indicesRefs):
    print(indicesAll)
    print("Lengte : " + str(len(indicesAll)))
    print(allWordsOIE)
    print("Lengte Words : " + str(len(allWordsOIE)))
    print(allWordsHeidel)
    #print(hasTitle)
    print(indicesRefs)
    indicesAllHeidel = []
    indexO = 0
    wait = 0
    indicesTimexes = []
    if hasTitle:
        firstSentence = True
    else:
        firstSentence = False
    timexes = dict()
    lastIndexHeidel = 0
    indicesRefs2 = indicesRefs.copy()
    nextWordD = None
    nextWordE = None
    while lastIndexHeidel<len(allWordsHeidel):
        indexHeidel = lastIndexHeidel
        #print('First : ' + str(indexHeidel))
        lastIndex = indexO
        if firstSentence:
            if indexHeidel + 1 < len(allWordsHeidel):
                word2 = str(allWordsHeidel[indexHeidel]).replace(' ', '').lower()
                nextWord = str(allWordsHeidel[indexHeidel + 1]).replace(' ', '').lower()
                if (word2.find("'")!=-1 and nextWord == 'says' and firstSentence):
                    #print('token : ' + word)
                    indicesAllHeidel.append([])
                    indicesAllHeidel.append([])
                    lastIndexHeidel += 2
                    firstSentence = False
                    continue

        word = str(allWordsHeidel[indexHeidel]).replace('.','').replace(chr(8220),'').replace('�','').replace(' ', '').replace("'", "").replace('"', '').replace('<', '').replace('/',
                                                                                                                 '').replace(
            '>', '').replace("\\", '').replace('(','').replace(')','').lower().encode("ascii", "ignore").decode()
        while len(word) == 0:
            #print("Empty")
            nextWordD = None
            indicesAllHeidel.append([lastIndex])
            if indexHeidel +1 < len(allWordsHeidel):
                indexHeidel += 1
                word = str(allWordsHeidel[indexHeidel]).replace('.','').replace(chr(8220),'').replace('�','').replace(' ', '').replace("'", "").replace('"', '').replace('<',
                                                                                                                   '').replace(
                    '/',
                    '').replace(
                    '>', '').replace("\\", '').replace('(','').replace(')','').lower().encode("ascii", "ignore").decode()
            else:
                break
        if len(word) == 0:
            break
        if nextWordE != None:
            word = nextWordE + word
        #print('Word : ' + word)
        #print('Second : ' + str(indexHeidel))
        lastIndexHeidel =indexHeidel + 1
        for indexWord in range(lastIndex, len(allWordsOIE)):
            if nextWordD == None:
                wordDecoding = str(allWordsOIE[indexWord]).replace('.','').replace(chr(8220),'').replace('�','').replace(' ', '').replace("'", "").replace('"', '').replace(
                    '<', '').replace('/', '').replace(
                    '>', '').replace("\\", '').replace('(','').replace(')','').lower().encode("ascii", "ignore").decode()
                while (len(wordDecoding) == 0):
                    indexWord += 1
                    if indexWord < len(allWordsOIE):
                        wordDecoding = str(allWordsOIE[indexWord]).replace('.','').replace(chr(8220),'').replace('�','').replace(' ', '').replace("'", "").replace('"',
                                                                                                      '').replace(
                            '<', '').replace('/', '').replace(
                            '>', '').replace("\\", '').replace('(','').replace(')','').lower().encode("ascii", "ignore").decode()
                    else:
                        # print('Break')
                        break
                if len(wordDecoding) == 0:
                    break
            else:
                wordDecoding = nextWordD

            # print('Next')

            #print(word)
            #print(allWordsOIE[indexWord])

            #print('Heidel')
            #print(word.lower())
            #print('OIE')
            #print(wordDecoding)
            if word == wordDecoding:
                #print(indexWord)
                nextWordD = None
                nextWordE = None
                indicesAllHeidel.append(indicesAll[indexWord])
                for indices in indicesRefs:
                    if indices[0] == indexHeidel:
                        timexes[indices[0]] = indicesAll[indexWord][0]
                        #print("timex 1 :" +  str(timexes[indices[0]]))
                    if indices[-1] == indexHeidel:
                        startIndex = timexes.get(indices[0])
                        if startIndex != None:
                            endIndex = indicesAll[indexWord][-1]
                            #print("StartIndex 1 : " + str(startIndex))
                            indicesTimexes.append([startIndex, endIndex])
                            indicesRefs2.remove(indices)
                indexO = indexWord + 1
                break
            else:
                #print(parts)
                index = indexWord
                #print(index)
                wordDecoding = str(allWordsOIE[index]).replace('.','').replace(chr(8220),'').replace('�','').replace(' ', '').replace("'", "").replace('"',
                                                                                                 '').replace(
                    '<', '').replace('/', '').replace(
                    '>', '').replace("\\", '').replace('(','').replace(')','').lower().encode("ascii", "ignore").decode()
                #print(wordDecoding)
                #print(index)
                parts = word.lower().split(wordDecoding)
                if len(parts) >=2 and index +1 <len(allWordsOIE) and len(parts[0])==0:
                    lengte = len(wordDecoding)
                    newIndices = []
                    index += 1
                    wordDecoding = str(allWordsOIE[index]).replace('.','').replace(chr(8220),'').replace('�','').replace(' ', '').replace("'", "").replace('"',
                                                                                                  '').replace(
                        '<', '').replace('/', '').replace(
                        '>', '').replace("\\", '').replace('(','').replace(')','').lower().encode("ascii", "ignore").decode()
                    #print(wordDecoding)
                    #print(index)
                    while (len(wordDecoding) == 0):
                        index += 1
                        if index < len(allWordsOIE):
                            wordDecoding = str(allWordsOIE[indexWord]).replace('.', '').replace(chr(8220), '').replace(
                                '�',
                                '').replace(
                                ' ', '').replace("'", "").replace('"',
                                                                  '').replace(
                                '<', '').replace('/', '').replace(
                                '>', '').replace("\\", '').replace('(', '').replace(')', '').lower().encode(
                                "ascii", "ignore").decode()
                        else:
                            # print('Break')
                            break
                    if len(wordDecoding) != 0:
                        nextWord = word[lengte:]
                        parts = nextWord.split(wordDecoding)
                    #print(parts)
                        if len(parts[0]) == 0:
                            nextWordD = None
                            nextWordE = None
                            newIndices += indicesAll[index-1]
                            newIndices += indicesAll[index]
                            for indices in indicesRefs:
                                startIndex = timexes.get(indices[0])
                                if indices[0] == indexHeidel:
                                    timexes[indices[0]] = indicesAll[indexWord][0]
                                    #print("timex 1 :" + str(timexes[indices[0]]))
                                if indices[1] == indexHeidel:
                                    if startIndex != None:
                                        endIndex = indicesAll[indexWord][-1]
                                        #print("StartIndex 1 : " + str(startIndex))
                                        indicesTimexes.append([startIndex, endIndex])
                                        indicesRefs2.remove(indices)
                            index += 1
                            while len(parts) >= 2 and len(parts[0]) == 0 and not(nextWord==wordDecoding):
                                lengte = len(wordDecoding)
                                wordDecoding = str(allWordsOIE[index]).replace('.','').replace(chr(8220),'').replace('�','').replace(' ', '').replace("'", "").replace('"',
                                                                                                                 '').replace(
                                    '<', '').replace('/', '').replace(
                                    '>', '').replace("\\", '').replace('(','').replace(')','').lower().encode("ascii", "ignore").decode()
                                while (len(wordDecoding) == 0):
                                    index += 1
                                    if index < len(allWordsOIE):
                                        wordDecoding = str(allWordsOIE[indexWord]).replace('.','').replace(chr(8220), '').replace('�',
                                                                                                                  '').replace(
                                            ' ', '').replace("'", "").replace('"',
                                                                              '').replace(
                                            '<', '').replace('/', '').replace(
                                            '>', '').replace("\\", '').replace('(', '').replace(')', '').lower().encode(
                                            "ascii", "ignore").decode()
                                    else:
                                        index -=1
                                        # print('Break')
                                        break
                                #print(wordDecoding)
                                #print(index)
                                if len(wordDecoding) != 0:
                                    nextWord = nextWord[lengte:]
                                    parts = nextWord.split(wordDecoding)
                                    #print('Parts in ')
                                    #print(parts)
                                    if len(parts)>=2 and len(parts[0]) == 0 and not(nextWord==wordDecoding):
                                        newIndices += indicesAll[index]
                                        index += 1
                                        #print('Index 2 : ' + str(index))
                                        if index >= len(allWordsOIE):
                                            parts = []
                                        for indices in indicesRefs:
                                            if indices[0] == indexHeidel:
                                                timexes[indices[0]] = indicesAll[indexWord][0]
                                                #print("timex 1 :" + str(timexes[indices[0]]))
                                            startIndex = timexes.get(indices[0])
                                            if indices[-1] == indexHeidel:
                                                if startIndex != None:
                                                    endIndex = indicesAll[indexWord][-1]
                                                    #print("StartIndex 1 : " + str(startIndex))
                                                    indicesTimexes.append([startIndex, endIndex])
                                                    indicesRefs2.remove(indices)
                                else:
                                    parts = []
                            if nextWord == wordDecoding:
                                nextWordD = None
                                nextWordE = None
                                newIndices += indicesAll[index]
                                index += 1
                                indexO = index
                                #print('Index O : '+ str(indexO))
                                indicesAllHeidel.append(newIndices)
                                for indices in indicesRefs:
                                    if indices[0] == indexHeidel:
                                        timexes[indices[0]] = indicesAll[indexWord][0]
                                        # print("timex 1 :" + str(timexes[indices[0]]))
                                    startIndex = timexes.get(indices[0])
                                    if indices[-1] == indexHeidel:
                                        if startIndex != None:
                                            endIndex = indicesAll[indexWord][-1]
                                            # print("StartIndex 1 : " + str(startIndex))
                                            indicesTimexes.append([startIndex, endIndex])
                                            indicesRefs2.remove(indices)
                                break
                            else:
                                nextWordD = None
                                nextWordE = nextWord
                                indicesAllHeidel.append(newIndices)
                                indexO = index
                            break
                else:
                    parts = wordDecoding.split(word)
                    #print(parts)
                    index = indexHeidel
                    if len(parts) >= 2 and index +1 <len(allWordsHeidel) and len(parts[0])==0:
                        lengte = len(word)
                        index += 1
                        word = str(allWordsHeidel[index]).replace('.','').replace(chr(8220),'').replace('�','').replace(' ', '').replace("'", "").replace('"',
                                                                                                       '').replace(
                                    '<', '').replace('/', '').replace(
                                    '>', '').replace("\\", '').replace('(','').replace(')','').lower().encode("ascii", "ignore").decode()
                        while len(word) == 0:
                            # print("Empty")
                            indicesAllHeidel.append(indicesAll[lastIndex])
                            index += 1
                            if index < len(allWordsHeidel):
                                word = str(allWordsHeidel[index]).replace('.','').replace(chr(8220), '').replace('�', '').replace(
                                    ' ', '').replace("'", "").replace('"', '').replace('<',
                                                                                       '').replace(
                                    '/',
                                    '').replace(
                                    '>', '').replace("\\", '').replace('(', '').replace(')', '').lower().encode("ascii",
                                                                                                                "ignore").decode()
                            else:
                                index -= 1
                                break
                        if len(word) > 0:
                            #print('Word 2 : ' + word)
                            nextWord = wordDecoding[lengte:]
                            parts = nextWord.lower().split(word.lower())
                            #print(parts)
                            if len(parts) >1:
                                if len(parts[0]) == 0 and len(parts)>=2:
                                    indicesAllHeidel.append(indicesAll[lastIndex])
                                    indicesAllHeidel.append(indicesAll[lastIndex])
                                    index += 1
                                    nextWordD = None
                                    for indices in indicesRefs:
                                        startIndex = timexes.get(indices[0])
                                        if indices[0] == index-2:
                                            timexes[indices[0]] = indicesAll[indexWord][0]
                                            #print("timex 1 :" + str(timexes[indices[0]]))
                                        if indices[1] == index:
                                            if startIndex != None:
                                                endIndex = indicesAll[indexWord][-1]
                                                #print("StartIndex 1 : " + str(startIndex))
                                                indicesTimexes.append([startIndex, endIndex])
                                                indicesRefs2.remove(indices)
                                    while len(parts) >=2 and len(parts[0])==0 and index<len(allWordsHeidel):
                                        #print('Word 2 : ' + word)
                                        lengte = len(word)
                                        word = str(allWordsHeidel[index]).replace('.','').replace(chr(8220),'').replace('�','').replace(' ', '').replace("'", "").replace('"',
                                                                                                                   '').replace(
                                                    '<', '').replace('/', '').replace(
                                                    '>', '').replace("\\", '').replace('(','').replace(')','').lower().encode("ascii", "ignore").decode()
                                        while len(word) == 0:
                                            # print("Empty")
                                            indicesAllHeidel.append(indicesAll[lastIndex])
                                            index += 1
                                            if index < len(allWordsHeidel):
                                                index +=1
                                                word = str(allWordsHeidel[index]).replace('.','').replace(chr(8220), '').replace(
                                                    '�', '').replace(
                                                    ' ', '').replace("'", "").replace('"', '').replace('<',
                                                                                                       '').replace(
                                                    '/',
                                                    '').replace(
                                                    '>', '').replace("\\", '').replace('(', '').replace(')',
                                                                                                        '').lower().encode(
                                                    "ascii",
                                                    "ignore").decode()
                                            else:
                                                index -= 1
                                                break
                                        if len(word)>0:
                                            nextWord = nextWord[lengte:]
                                            parts = nextWord.lower().split(word.lower())
                                            #print('Parts in 2')
                                            #print(parts)
                                            if len(parts)>=2 and len(parts[0]) == 0:
                                                indicesAllHeidel.append(indicesAll[lastIndex])
                                                index += 1
                                                nextWordD = None
                                                if index>= len(allWordsHeidel):
                                                    parts = []
                                                for indices in indicesRefs:
                                                    startIndex = timexes.get(indices[0])
                                                    if indices[-1] == indexHeidel:
                                                        if startIndex != None:
                                                            endIndex = indicesAll[indexWord][-1]
                                                            #print("StartIndex 1 : " + str(startIndex))
                                                            indicesTimexes.append([startIndex, endIndex])
                                                            indicesRefs2.remove(indices)
                                        else:
                                            index += 1
                                            parts = []
                                    if nextWord == word:
                                        nextWordD = None
                                        indicesAllHeidel.append(indicesAll[lastIndex])
                                        #indicesAllHeidel.append(indicesAll[lastIndex])
                                        indexO = lastIndex + 1
                                        for indices in indicesRefs:
                                            startIndex = timexes.get(indices[0])
                                            if indices[-1] == index:
                                                if startIndex != None:
                                                    endIndex = indicesAll[indexWord][-1]
                                                    # print("StartIndex 1 : " + str(startIndex))
                                                    indicesTimexes.append([startIndex, endIndex])
                                                    indicesRefs2.remove(indices)
                                        lastIndexHeidel = index
                                    else:
                                        if len(nextWord) == 0:
                                            nextWordD = None
                                            lastIndexHeidel = index
                                        else:
                                            if len(word.split(nextWord)[0]) == 0:
                                                nextWordD = nextWord
                                            indicesAllHeidel.append(indicesAll[lastIndex])
                                            # indicesAllHeidel.append(indicesAll[lastIndex])
                                            indexO = lastIndex + 1
                                            for indices in indicesRefs:
                                                startIndex = timexes.get(indices[0])
                                                if indices[-1] == index:
                                                    if startIndex != None:
                                                        endIndex = indicesAll[indexWord][-1]
                                                        # print("StartIndex 1 : " + str(startIndex))
                                                        indicesTimexes.append([startIndex, endIndex])
                                                        indicesRefs2.remove(indices)
                                            lastIndexHeidel = index
                                        break

                            else:
                                nextWordD = None
                                if nextWord == word or len(nextWord)==0:
                                    indicesAllHeidel.append(indicesAll[index])
                                    indicesAllHeidel.append(indicesAll[index])
                                    indexHeidel += 1
                                    indexO += 1
                                    for indices in indicesRefs:
                                        startIndex = timexes.get(indices[0])
                                        if indices[-1] == indexHeidel:
                                            if startIndex != None:
                                                endIndex = indicesAll[indexWord][-1]
                                                # print("StartIndex 1 : " + str(startIndex))
                                                indicesTimexes.append([startIndex, endIndex])
                                                indicesRefs2.remove(indices)
                                else:
                                    if len(word.split(nextWord)[0]) == 0:
                                        nextWordD = nextWord
                                    indicesAllHeidel.append(indicesAll[lastIndex])
                                    # indicesAllHeidel.append(indicesAll[lastIndex])
                                    indexO = lastIndex + 1
                                    for indices in indicesRefs:
                                        startIndex = timexes.get(indices[0])
                                        if indices[-1] == index:
                                            if startIndex != None:
                                                endIndex = indicesAll[indexWord][-1]
                                                # print("StartIndex 1 : " + str(startIndex))
                                                indicesTimexes.append([startIndex, endIndex])
                                                indicesRefs2.remove(indices)
                                    lastIndexHeidel = index
                                break
    "print("indicesRefs over")
    #print(indicesRefs2)
    print(indicesAllHeidel)
    for indices in indicesRefs2:
        print("Indices : " + str(indices))
        startIndex = indicesAllHeidel[indices[0]][-1]
        endIndex = indicesAllHeidel[indices[-1]][-1]
        #print("StartIndex 1 : " + str(startIndex))
        indicesTimexes.append([startIndex, endIndex])
            # print('Wait ---')
            # print(word)
            # print(wait)
    #print(indicesAll)
    #print(allWordsOIE)
    #print(allWordsHeidel)

    #print("IndicesTimexes : ")
    #print(indicesTimexes)
    return indicesTimexes
'''
'''
# Check of the dataloader
# train_set = NUS(mode='Little')
# dump_write(train_set, "little_dataset")
train_set = NUS(mode=sys.argv[1], path=sys.argv[2],pathToSave=sys.argv[3],domain=sys.argv[4],number=sys.argv[6])
#train_set.writeMetadata()
dump_write(train_set, sys.argv[5])
#train_set = dump_load( "trainLoader")
#test_set = dump_load("test/time/testDatasetIteratie2-" + "huca")
train_loader = DataLoader(train_set,
                          batch_size=32,
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
    print('Verb indices claim')
    print(c[7])
    print('Time expression indices claim')
    print(c[8])
    print('Position indices claim')
    print(c[9])
    print('Refs indices claim')
    print(c[10])
    print('Time Heidel claim')
    print(c[11])
    print('Verb indices snippet')
    print(c[12])
    print('Time expression indices snippet')
    print(c[13])
    print('Position indices snippet')
    print(c[14])
    print('Refs indices snippet')
    print(c[15])
    print('Time Heidel snippet')
    print(c[16])
    break
print("Done dataset")
'''






