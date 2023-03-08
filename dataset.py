import sys
from datetime import datetime

from allennlp.predictors.predictor import Predictor as pred
import pickle
import os

import spacy

import baseModel.Claim as Claim

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
import spacy


class NUS(Dataset):
    # change root to your root
    '''
        mode: the type of the given dataset (Dev, Train and Test)
        path: the path of the given dataset
        domain: the domain where the claims are come from
    '''
    def __init__(self, mode,path,domain):
        super().__init__()
        assert mode in ['Train', 'Dev', 'Test']
        self.domain = domain
        self.claimPreTextSize = []
        self.snippetPreTextSize = []
        print('Loading {} set...'.format(mode))
        self.mode = mode
        print('Get buckets of differences')
        self.getDifferences('differencePublicationDate.txt')
        print('Get buckets of difference of timexes in text')
        self.getDifferencesAbsolute('differenceTimexesInText.txt')
        print('Reading the claims')
        self.getClaims(mode,path)
        print('Done')

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
        underivable = -1
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

    def getClaims(self, mode,path):
        self.claimIds = []
        self.documents = []
        self.labels = []
        self.snippetDocs = []
        self.metadata = []
        self.metadataSet = set()
        self.labelsAll = set()
        self.bucketsSnippets = []
        self.claimDateAvailable = []
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
                                        "snippets", predictorOIE, predictorNER, nlp, coreference)
                    claim.readPublicationDate()
                    self.getNumbersOfTokensPreText(tokenizer, claim)
                    bucketsSnippetClaim = ''
                    verbsSnippetIndices =''
                    timeSnippetIndices = ''
                    positionSnippetIndices =''
                    refsSnippetIndices =''
                    timeSnippetElements =''
                    differenceSnippets = []

                    '''
                        publication time claim and snippet (division 1)
                    '''
                    if not type(claim.claimDate) is tuple:
                        if claim.claimDate != None:
                            self.claimDateAvailable.append(1)
                        else:
                            self.claimDateAvailable.append(0)

                        for snippet in claim.getSnippets():
                            snippet.readPublicationDate()
                            if (snippet.publishTime != None and claim.claimDate != None):
                                if type(snippet.publishTime) is tuple:
                                    if ((snippet.publishTime[0] - claim.claimDate).days <= 0 and (
                                            snippet.publishTime[1] - claim.claimDate).days >= 0):
                                        difference = 0
                                    else:
                                        if (snippet.publishTime[1] - claim.claimDate).days <= 0:
                                            difference = (snippet.publishTime[1] - claim.claimDate).days
                                        else:
                                            difference = (snippet.publishTime[0] - claim.claimDate).days
                                    bucketsSnippetClaim +=  '\t' + str(self.matchBucket(difference))
                                    differenceSnippets.append(difference)
                                else:
                                    difference = (snippet.publishTime-claim.claimDate).days
                                    bucketsSnippetClaim += '\t' + str(self.matchBucket(difference))
                                    differenceSnippets.append(difference)
                            else:
                                if snippet.publishTime != None:
                                    differenceSnippets.append("None")
                                    bucketsSnippetClaim += '\t' + str(20)
                                else:
                                    bucketsSnippetClaim += '\t' + str(21)
                                    differenceSnippets.append("None")
                    else:
                        if claim.claimDate != None:
                            self.claimDateAvailable.append(1)
                        else:
                            self.claimDateAvailable.append(0)
                        for snippet in claim.getSnippets():
                            snippet.readPublicationDate()
                            if (snippet.publishTime != None and claim.claimDate != None):
                                if type(snippet.publishTime) is tuple:
                                    if (snippet.publishTime[0] - claim.claimDate[1]).days >= 0:
                                        difference = (snippet.publishTime[0] - claim.claimDate[1]).days
                                        bucketsSnippetClaim += '\t' + str(self.matchBucket(difference))
                                        differenceSnippets.append(difference)
                                    else:
                                        if (snippet.publishTime[1] - claim.claimDate[0]).days <= 0:
                                            difference = (snippet.publishTime[1] - claim.claimDate[0]).days
                                            bucketsSnippetClaim +=  '\t' + str(self.matchBucket(difference))
                                            differenceSnippets.append(difference)
                                        else:
                                            difference = 0
                                            bucketsSnippetClaim +=  '\t' + str(self.matchBucket(difference))
                                            differenceSnippets.append(difference)
                                else:
                                    if ((snippet.publishTime - claim.claimDate[0]).days >= 0 and (
                                            snippet.publishTime - claim.claimDate[1]).days <= 0):
                                        difference = 0
                                        bucketsSnippetClaim += '\t' + str(self.matchBucket(difference))
                                        differenceSnippets.append(difference)
                                    else:
                                        if (snippet.publishTime - claim.claimDate[0]).days <= 0:
                                            difference = (snippet.publishTime - claim.claimDate[0]).days
                                        else:
                                            difference = (snippet.publishTime - claim.claimDate[1]).days
                                        bucketsSnippetClaim +=  '\t' + str(self.matchBucket(difference))
                                        differenceSnippets.append(difference)
                            else:
                                if snippet.publishTime != None:
                                    bucketsSnippetClaim += '\t' + str(20)
                                    differenceSnippets.append("None")
                                else:
                                    bucketsSnippetClaim += '\t' + str(21)
                                    differenceSnippets.append("None")

                    '''
                        timeInText for iteration 2
                    '''
                    input_ids = [i for i in tokenizer.encode(
                        text=getClaimTextLocal(claim))]
                    decoding = [tokenizer.decode([i]) for i in input_ids]
                    startIndex = claim.getIndex()
                    OIE = claim.readOpenInformationExtraction()
                    indicesAll, tagsAll, positionsAll, allWords, sentencesIndices = ProcessOpenInformation(
                        OIE, decoding, startIndex, nlp)
                    verbClaim = ""
                    for verb in tagsAll['Verbs']:
                        for part in verb:
                            verbClaim += str(part) +"\t"
                    verbClaim = verbClaim[:-1]
                    self.verbIndices.append(verbClaim)
                    positionsClaim = ""
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
                        timeClaim = ""
                        indicesRefsClaim = ""
                        for time in tagsAll['Time']:
                            if len(indicesRefs) <= index:
                                for position in positionsAll:
                                    if time[0] == position[0] + position[1]:
                                        positionsClaim += str(position[0]) + ',' + str(
                                            position[1]) + "\t"
                                for part in time:
                                    timeClaim += str(part) + "\t"
                            else:
                                if time[-1] < indicesRefs[index][0]:
                                    for position in positionsAll:
                                        if time[0] == position[0] + position[1]:
                                            positionsClaim += str(position[0]) + ',' + str(
                                                position[1]) + "\t"
                                    for part in time:
                                        timeClaim += str(part) + "\t"
                                else:
                                    if time[0] > indicesRefs[index][-1]:
                                        while time[0] > indicesRefs[index][-1]:
                                            indicesRefsClaim += str(indicesRefs[index][0]) + "\t"
                                            for timeI in range(indicesRefs[index][0],indicesRefs[index][-1]+1):
                                                timeClaim += str(timeI) + "\t"
                                            index += 1
                                            if len(indicesRefs) <= index:
                                                break
                                        if len(indicesRefs) <= index:
                                            for position in positionsAll:
                                                if time[0] == position[0] + position[1]:
                                                    positionsClaim += str(position[0]) + ',' + str(
                                                        position[1]) + "\t"
                                            for part in time:
                                                timeClaim += str(part) + "\t"
                                        else:
                                            if time[-1] < indicesRefs[index][0]:
                                                for position in positionsAll:
                                                    if time[0] == position[0] + position[1]:
                                                        positionsClaim += str(position[0]) + ',' + str(
                                                            position[1]) + "\t"
                                                for part in time:
                                                    timeClaim += str(part) + "\t"
                                            else:
                                                indicesRefsClaim += str(min(indicesRefs[index][0], time[0])) + "\t"
                                                for position in positionsAll:
                                                    if time[0]== position[0] + position[1]:
                                                        second =  min(indicesRefs[index][0], time[0])-position[0]
                                                        positionsClaim += str(position[0]) + ',' + str(second) + "\t"
                                                for timeI in range(min(indicesRefs[index][0], time[0]),
                                                                   max(indicesRefs[index][1]+1, time[-1]+1)):
                                                    timeClaim += str(timeI) + "\t"
                                                index += 1
                                    else:
                                        for position in positionsAll:
                                            if time[0] == position[0] + position[1]:
                                                second = min(indicesRefs[index][0], time[0]) - position[0]
                                                positionsClaim += str(position[0]) + ',' + str(second) + "\t"
                                        indicesRefsClaim += str(min(indicesRefs[index][0],time[0])) + "\t"
                                        for timeI in range(min(indicesRefs[index][0],time[0]),max(indicesRefs[index][1]+1,time[-1]+1)):
                                            timeClaim += str(timeI) + "\t"
                                        index += 1
                        while index<len(indicesRefs):
                            indicesRefsClaim += str(indicesRefs[index][0]) + "\t"
                            for timeI in range(indicesRefs[index][0], indicesRefs[index][-1]+1):
                                timeClaim += str(timeI) + "\t"
                            index += 1
                        positionsClaim = positionsClaim[:-1]
                        self.positions.append(positionsClaim)
                        timeClaim = timeClaim[:-1]
                        self.timeIndices.append(timeClaim)
                        indicesRefsClaim = indicesRefsClaim[:-1]
                        self.refsIndices.append(indicesRefsClaim)

                        for data in datasM:
                            if type(data) == list:
                                if not data[0] in durenM:
                                    if data[0] in refsM:
                                        timeClaim += "Refs-"+self.matchBucketRelativeTime(data[0],difference=0) + "\t"
                                    else:
                                        if (claim.claimDate != None):
                                            if type(data[0]) is tuple:
                                                if ((data[0][
                                                         0] - claim.claimDate).days <= 0 and (
                                                        data[0][
                                                            1] - claim.claimDate).days >= 0):
                                                    difference = 0
                                                else:
                                                    if (data[0][1] - claim.claimDate).days <= 0:
                                                        difference = (data[0][
                                                                        1] - claim.claimDate).days
                                                    else:
                                                        difference = (data[0][
                                                                        0] - claim.claimDate).days
                                                timeClaim += str(self.matchBucketAbsoluteTime(difference)) + "\t"
                                            else:
                                                if type(data[0]) == datetime:
                                                    difference = (data[0]-claim.claimDate).days
                                                    timeClaim += str(self.matchBucketAbsoluteTime(difference)) + "\t"
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
                        timeClaim = ""
                        indicesRefsClaim = ""
                        for time in tagsAll['Time']:
                            if len(indicesRefs) <= index:
                                for position in positionsAll:
                                    if time[0] == position[0] + position[1]:
                                        positionsClaim += str(position[0]) + ',' + str(
                                            position[1]) + "\t"
                                for part in time:
                                    timeClaim += str(part) + "\t"
                            else:
                                if time[-1] < indicesRefs[index][0]:
                                    for position in positionsAll:
                                        if time[0] == position[0] + position[1]:
                                            positionsClaim += str(position[0]) + ',' + str(
                                                position[1]) + "\t"
                                    for part in time:
                                        timeClaim += str(part) + "\t"
                                else:
                                    if time[0] > indicesRefs[index][-1]:
                                        while time[0] > indicesRefs[index][-1]:
                                            indicesRefsClaim += str(indicesRefs[index][0]) + "\t"
                                            for timeI in range(indicesRefs[index][0], indicesRefs[index][-1] + 1):
                                                timeClaim += str(timeI) + "\t"
                                            index += 1
                                            if len(indicesRefs) <= index:
                                                break
                                        if len(indicesRefs) <= index:
                                            for position in positionsAll:
                                                if time[0] == position[0] + position[1]:
                                                    positionsClaim += str(position[0]) + ',' + str(
                                                        position[1]) + "\t"
                                            for part in time:
                                                timeClaim += str(part) + "\t"
                                        else:
                                            if time[-1] < indicesRefs[index][0]:
                                                for position in positionsAll:
                                                    if time[0] == position[0] + position[1]:
                                                        positionsClaim += str(position[0]) + ',' + str(
                                                            position[1]) + "\t"
                                                for part in time:
                                                    timeClaim += str(part) + "\t"
                                            else:
                                                indicesRefsClaim += str(min(indicesRefs[index][0], time[0])) + "\t"
                                                for position in positionsAll:
                                                    if time[0] == position[0] + position[1]:
                                                        second = min(indicesRefs[index][0], time[0]) - position[0]
                                                        positionsClaim += str(position[0]) + ',' + str(
                                                            second) + "\t"
                                                for timeI in range(min(indicesRefs[index][0], time[0]),
                                                                   max(indicesRefs[index][1] + 1, time[-1] + 1)):
                                                    timeClaim += str(timeI) + "\t"
                                                index += 1
                                    else:
                                        for position in positionsAll:
                                            if time[0] == position[0] + position[1]:
                                                second = min(indicesRefs[index][0], time[0]) - position[0]
                                                positionsClaim += str(position[0]) + ',' + str(second) + "\t"
                                        indicesRefsClaim += str(min(indicesRefs[index][0], time[0])) + "\t"
                                        for timeI in range(min(indicesRefs[index][0], time[0]),
                                                           max(indicesRefs[index][1] + 1, time[-1] + 1)):
                                            timeClaim += str(timeI) + "\t"
                                        index += 1
                        while index < len(indicesRefs):
                            indicesRefsClaim += str(indicesRefs[index][0]) + "\t"
                            for timeI in range(indicesRefs[index][0], indicesRefs[index][-1] + 1):
                                timeClaim += str(timeI) + "\t"
                            index += 1
                        positionsClaim = positionsClaim[:-1]
                        self.positions.append(positionsClaim)
                        timeClaim = timeClaim[:-1]
                        self.timeIndices.append(timeClaim)
                        indicesRefsClaim = indicesRefsClaim[:-1]
                        self.refsIndices.append(indicesRefsClaim)
                        for data in datasM:
                            if type(data) == list:
                                if not data[0] in durenM:
                                    if data[0] in refsM:
                                        timeClaim += "Refs-"+self.matchBucketRelativeTime(data[0],difference=0) + "\t"
                                    else:
                                        if not data[0] in setsM:
                                            if type(data[0]) is tuple:
                                                if (data[0][0] - claim.claimDate[1]).days >= 0:
                                                    difference = (data[0][0] - claim.claimDate[
                                                        1]).days
                                                else:
                                                    if (data[0][1] - claim.claimDate[0]).days <= 0:
                                                        difference = (data[0][1] - claim.claimDate[
                                                            0]).days
                                                    else:
                                                        difference = 0
                                                timeClaim += str(self.matchBucketAbsoluteTime(difference)) + "\t"
                                            else:
                                                if ((data[0] - claim.claimDate[0]).days >= 0 and (
                                                        data[0] - claim.claimDate[1]).days <= 0):
                                                    difference = 0
                                                else:
                                                    if (data[0] - claim.claimDate[0]).days <= 0:
                                                        difference = (data[0] - claim.claimDate[
                                                            0]).days
                                                    else:
                                                        difference = (data[0] - claim.claimDate[
                                                            1]).days
                                                timeClaim += str(self.matchBucketAbsoluteTime(difference)) + "\t"
                        timeClaim = timeClaim[:-1]
                        self.time.append(timeClaim)
                    for snippet in claim.snippets:
                        input_ids = [i for i in tokenizer.encode(
                            text=getSnippetTextLocal(snippet))]
                        decoding = [tokenizer.decode([i]) for i in input_ids]
                        OIE = snippet.readOpenInformationExtraction()
                        startIndex = snippet.getIndex()
                        indicesAll, tagsAll, positionsAll, allWords, sentencesIndices = ProcessOpenInformation(
                            OIE, decoding, startIndex, nlp)
                        if len(tagsAll['Verbs']) > 0:
                            for verb in tagsAll['Verbs']:
                                for part in verb:
                                    verbsSnippetIndices += str(part) + "\t"
                            verbsSnippetIndices = verbsSnippetIndices[:-1]
                        verbsSnippetIndices += ' 0123456789 '
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
                            differenceSnippet = differenceSnippets[0]
                            differenceSnippets = differenceSnippets[1:]
                            for data in datasM:
                                if type(data) == list:
                                    if not data[0] in durenM:
                                        if data[0] in refsM:
                                            if differenceSnippet != "None":
                                                timeSnippetElements += "Refs-" + self.matchBucketRelativeTime(
                                                    data[0], difference=differenceSnippet) + "\t"
                                        else:
                                            if not data[0] in setsM :
                                                if (claim.claimDate != None):
                                                    if type(data[0]) is tuple:
                                                        if ((data[0][
                                                                 0] - claim.claimDate).days <= 0 and (
                                                                data[0][
                                                                    1] - claim.claimDate).days >= 0):
                                                            difference = 0
                                                        else:
                                                            if (data[0][
                                                                    1] - claim.claimDate).days <= 0:
                                                                difference = (data[0][
                                                                                1] - claim.claimDate).days
                                                            else:
                                                                difference = (data[0][
                                                                                0] - claim.claimDate).days
                                                        timeSnippetElements += str(
                                                            self.matchBucketAbsoluteTime(difference)) + "\t"
                                                    else:
                                                        difference = (data[0] - claim.claimDate).days
                                                        timeSnippetElements += str(
                                                            self.matchBucketAbsoluteTime(difference)) + "\t"
                            if len(timeSnippetElements) > 0 and timeSnippetElements[-1] == "\t":
                                timeSnippetElements = timeSnippetElements[:-1]
                            timeSnippetElements += ' 0123456789 '
                        else:
                            datasM, durenM, refsM, setsM = snippet.readTime()
                            tokenisation = []
                            refsIndices = []
                            startIndex = snippet.getIndexHeidel()
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
                            differenceSnippet = differenceSnippets[0]
                            differenceSnippets = differenceSnippets[1:]
                            for data in datasM:
                                if type(data) != int:
                                    if type(data[0]) != str:
                                        if not data[0] in durenM:
                                            if data[0] in refsM:
                                                if differenceSnippet != "None":
                                                    timeSnippetElements += "Refs-" + self.matchBucketRelativeTime(
                                                        data[0], difference=differenceSnippet) + "\t"
                                            else:
                                                if not data[0] in setsM:
                                                    if type(data[0]) is tuple:
                                                        if (data[0][0] - claim.claimDate[1]).days >= 0:
                                                            difference = (data[0][0] - claim.claimDate[
                                                                1]).days
                                                            timeSnippetElements += str(
                                                                self.matchBucketAbsoluteTime(difference)) + "\t"
                                                        else:
                                                            if (data[0][1] - claim.claimDate[
                                                                0]).days <= 0:
                                                                difference = (data[0][1] -
                                                                            claim.claimDate[0]).days
                                                                timeSnippetElements += str(
                                                                    self.matchBucketAbsoluteTime(difference)) + "\t"
                                                            else:
                                                                difference = 0
                                                                timeSnippetElements += str(
                                                                    self.matchBucketAbsoluteTime(difference)) + "\t"
                                                    else:
                                                        if ((data[0] - claim.claimDate[
                                                            0]).days >= 0 and (
                                                                data[0] - claim.claimDate[
                                                            1]).days <= 0):
                                                            difference = 0
                                                        else:
                                                            if (data[0] - claim.claimDate[0]).days <= 0:
                                                                difference = (data[0] - claim.claimDate[
                                                                    0]).days
                                                            else:
                                                                difference = (data[0] - claim.claimDate[
                                                                    1]).days
                                                        timeSnippetElements += str(
                                                            self.matchBucketAbsoluteTime(difference)) + "\t"
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
                                        "snippets", predictorOIE, predictorNER, nlp, coreference)
                    claim.readPublicationDate()
                    self.getNumbersOfTokensPreText(tokenizer, claim)
                    bucketsSnippetClaim = ''
                    verbsSnippetIndices = ''
                    timeSnippetIndices = ''
                    positionSnippetIndices = ''
                    refsSnippetIndices = ''
                    timeSnippetElements = ''
                    differenceSnippets = []

                    '''
                        publication time claim and snippet (iteration 1)
                    '''
                    if not type(claim.claimDate) is tuple:
                        if claim.claimDate != None:
                            self.claimDateAvailable.append(1)
                        else:
                            self.claimDateAvailable.append(0)

                        for snippet in claim.getSnippets():
                            snippet.readPublicationDate()
                            if (snippet.publishTime != None and claim.claimDate != None):
                                if type(snippet.publishTime) is tuple:
                                    if ((snippet.publishTime[0] - claim.claimDate).days <= 0 and (
                                            snippet.publishTime[1] - claim.claimDate).days >= 0):
                                        difference = 0
                                    else:
                                        if (snippet.publishTime[1] - claim.claimDate).days <= 0:
                                            difference = (snippet.publishTime[1] - claim.claimDate).days
                                        else:
                                            difference = (snippet.publishTime[0] - claim.claimDate).days
                                    bucketsSnippetClaim += '\t' + str(self.matchBucket(difference))
                                    differenceSnippets.append(difference)
                                else:
                                    difference = (snippet.publishTime - claim.claimDate).days
                                    bucketsSnippetClaim += '\t' + str(self.matchBucket(difference))
                                    differenceSnippets.append(difference)
                            else:
                                if snippet.publishTime != None:
                                    differenceSnippets.append("None")
                                    bucketsSnippetClaim += '\t' + str(20)
                                else:
                                    bucketsSnippetClaim += '\t' + str(21)
                                    differenceSnippets.append("None")
                    else:
                        if claim.claimDate != None:
                            self.claimDateAvailable.append(1)
                        else:
                            self.claimDateAvailable.append(0)
                        for snippet in claim.getSnippets():
                            snippet.readPublicationDate()
                            if (snippet.publishTime != None and claim.claimDate != None):
                                if type(snippet.publishTime) is tuple:
                                    if (snippet.publishTime[0] - claim.claimDate[1]).days >= 0:
                                        difference = (snippet.publishTime[0] - claim.claimDate[1]).days
                                        bucketsSnippetClaim += '\t' + str(self.matchBucket(difference))
                                        differenceSnippets.append(difference)
                                    else:
                                        if (snippet.publishTime[1] - claim.claimDate[0]).days <= 0:
                                            difference = (snippet.publishTime[1] - claim.claimDate[0]).days
                                            bucketsSnippetClaim += '\t' + str(self.matchBucket(difference))
                                            differenceSnippets.append(difference)
                                        else:
                                            difference = 0
                                            bucketsSnippetClaim += '\t' + str(self.matchBucket(difference))
                                            differenceSnippets.append(difference)
                                else:
                                    if ((snippet.publishTime - claim.claimDate[0]).days >= 0 and (
                                            snippet.publishTime - claim.claimDate[1]).days <= 0):
                                        difference = 0
                                        bucketsSnippetClaim += '\t' + str(self.matchBucket(difference))
                                        differenceSnippets.append(difference)
                                    else:
                                        if (snippet.publishTime - claim.claimDate[0]).days <= 0:
                                            difference = (snippet.publishTime - claim.claimDate[0]).days
                                        else:
                                            difference = (snippet.publishTime - claim.claimDate[1]).days
                                        bucketsSnippetClaim += '\t' + str(self.matchBucket(difference))
                                        differenceSnippets.append(difference)
                            else:
                                if snippet.publishTime != None:
                                    bucketsSnippetClaim += '\t' + str(20)
                                    differenceSnippets.append("None")
                                else:
                                    bucketsSnippetClaim += '\t' + str(21)
                                    differenceSnippets.append("None")

                    '''
                        timeInText for iteration 2
                    '''
                    input_ids = [i for i in tokenizer.encode(
                        text=getClaimTextLocal(claim))]
                    decoding = [tokenizer.decode([i]) for i in input_ids]
                    startIndex = claim.getIndex()
                    OIE = claim.readOpenInformationExtraction()
                    indicesAll, tagsAll, positionsAll, allWords, sentencesIndices = ProcessOpenInformation(
                        OIE, decoding, startIndex, nlp)
                    verbClaim = ""
                    for verb in tagsAll['Verbs']:
                        for part in verb:
                            verbClaim += str(part) + "\t"
                    verbClaim = verbClaim[:-1]
                    self.verbIndices.append(verbClaim)
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
                        timeClaim = ""
                        indicesRefsClaim = ""
                        for time in tagsAll['Time']:
                            if len(indicesRefs) <= index:
                                for position in positionsAll:
                                    if time[0] == position[0] + position[1]:
                                        positionsClaim += str(position[0]) + ',' + str(
                                            position[1]) + "\t"
                                for part in time:
                                    timeClaim += str(part) + "\t"
                            else:
                                if time[-1] < indicesRefs[index][0]:
                                    for position in positionsAll:
                                        if time[0] == position[0] + position[1]:
                                            positionsClaim += str(position[0]) + ',' + str(
                                                position[1]) + "\t"
                                    for part in time:
                                        timeClaim += str(part) + "\t"
                                else:
                                    if time[0] > indicesRefs[index][-1]:
                                        while time[0] > indicesRefs[index][-1]:
                                            indicesRefsClaim += str(indicesRefs[index][0]) + "\t"
                                            for timeI in range(indicesRefs[index][0], indicesRefs[index][-1] + 1):
                                                timeClaim += str(timeI) + "\t"
                                            index += 1
                                            if len(indicesRefs) <= index:
                                                break
                                        if len(indicesRefs) <= index:
                                            for position in positionsAll:
                                                if time[0] == position[0] + position[1]:
                                                    positionsClaim += str(position[0]) + ',' + str(
                                                        position[1]) + "\t"
                                            for part in time:
                                                timeClaim += str(part) + "\t"
                                        else:
                                            if time[-1] < indicesRefs[index][0]:
                                                for position in positionsAll:
                                                    if time[0] == position[0] + position[1]:
                                                        positionsClaim += str(position[0]) + ',' + str(
                                                            position[1]) + "\t"
                                                for part in time:
                                                    timeClaim += str(part) + "\t"
                                            else:
                                                indicesRefsClaim += str(min(indicesRefs[index][0], time[0])) + "\t"
                                                for position in positionsAll:
                                                    if time[0] == position[0] + position[1]:
                                                        second = min(indicesRefs[index][0], time[0]) - position[0]
                                                        positionsClaim += str(position[0]) + ',' + str(second) + "\t"
                                                for timeI in range(min(indicesRefs[index][0], time[0]),
                                                                   max(indicesRefs[index][1] + 1, time[-1] + 1)):
                                                    timeClaim += str(timeI) + "\t"
                                                index += 1
                                    else:
                                        for position in positionsAll:
                                            if time[0] == position[0] + position[1]:
                                                second = min(indicesRefs[index][0], time[0]) - position[0]
                                                positionsClaim += str(position[0]) + ',' + str(second) + "\t"
                                        indicesRefsClaim += str(min(indicesRefs[index][0], time[0])) + "\t"
                                        for timeI in range(min(indicesRefs[index][0], time[0]),
                                                           max(indicesRefs[index][1] + 1, time[-1] + 1)):
                                            timeClaim += str(timeI) + "\t"
                                        index += 1
                        while index < len(indicesRefs):
                            indicesRefsClaim += str(indicesRefs[index][0]) + "\t"
                            for timeI in range(indicesRefs[index][0], indicesRefs[index][-1] + 1):
                                timeClaim += str(timeI) + "\t"
                            index += 1
                        positionsClaim = positionsClaim[:-1]
                        self.positions.append(positionsClaim)
                        timeClaim = timeClaim[:-1]
                        self.timeIndices.append(timeClaim)
                        indicesRefsClaim = indicesRefsClaim[:-1]
                        self.refsIndices.append(indicesRefsClaim)

                        for data in datasM:
                            if type(data) == list:
                                if not data[0] in durenM:
                                    if data[0] in refsM:
                                        timeClaim += "Refs-" + self.matchBucketRelativeTime(data[0],
                                                                                            difference=0) + "\t"
                                    else:
                                        if not data[0] in setsM:
                                            if (claim.claimDate != None):
                                                if type(data[0]) is tuple:
                                                    if ((data[0][
                                                             0] - claim.claimDate).days <= 0 and (
                                                            data[0][
                                                                1] - claim.claimDate).days >= 0):
                                                        difference = 0
                                                    else:
                                                        if (data[0][1] - claim.claimDate).days <= 0:
                                                            difference = (data[0][
                                                                            1] - claim.claimDate).days
                                                        else:
                                                            difference = (data[0][
                                                                            0] - claim.claimDate).days
                                                    timeClaim += str(self.matchBucketAbsoluteTime(difference)) + "\t"
                                                else:
                                                    difference = (data[0] - claim.claimDate).days
                                                    timeClaim += str(self.matchBucketAbsoluteTime(difference)) + "\t"
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
                        timeClaim = ""
                        indicesRefsClaim = ""
                        for time in tagsAll['Time']:
                            if len(indicesRefs) <= index:
                                for position in positionsAll:
                                    if time[0] == position[0] + position[1]:
                                        positionsClaim += str(position[0]) + ',' + str(
                                            position[1]) + "\t"
                                for part in time:
                                    timeClaim += str(part) + "\t"
                            else:
                                if time[-1] < indicesRefs[index][0]:
                                    for position in positionsAll:
                                        if time[0] == position[0] + position[1]:
                                            positionsClaim += str(position[0]) + ',' + str(
                                                position[1]) + "\t"
                                    for part in time:
                                        timeClaim += str(part) + "\t"
                                else:
                                    if time[0] > indicesRefs[index][-1]:
                                        while time[0] > indicesRefs[index][-1]:
                                            indicesRefsClaim += str(indicesRefs[index][0]) + "\t"
                                            for timeI in range(indicesRefs[index][0], indicesRefs[index][-1] + 1):
                                                timeClaim += str(timeI) + "\t"
                                            index += 1
                                            if len(indicesRefs) <= index:
                                                break
                                        if len(indicesRefs) <= index:
                                            for position in positionsAll:
                                                if time[0] == position[0] + position[1]:
                                                    positionsClaim += str(position[0]) + ',' + str(
                                                        position[1]) + "\t"
                                            for part in time:
                                                timeClaim += str(part) + "\t"
                                        else:
                                            if time[-1] < indicesRefs[index][0]:
                                                for position in positionsAll:
                                                    if time[0] == position[0] + position[1]:
                                                        positionsClaim += str(position[0]) + ',' + str(
                                                            position[1]) + "\t"
                                                for part in time:
                                                    timeClaim += str(part) + "\t"
                                            else:
                                                indicesRefsClaim += str(min(indicesRefs[index][0], time[0])) + "\t"
                                                for position in positionsAll:
                                                    if time[0] == position[0] + position[1]:
                                                        second = min(indicesRefs[index][0], time[0]) - position[0]
                                                        positionsClaim += str(position[0]) + ',' + str(
                                                            second) + "\t"
                                                for timeI in range(min(indicesRefs[index][0], time[0]),
                                                                   max(indicesRefs[index][1] + 1, time[-1] + 1)):
                                                    timeClaim += str(timeI) + "\t"
                                                index += 1
                                    else:
                                        for position in positionsAll:
                                            if time[0] == position[0] + position[1]:
                                                second = min(indicesRefs[index][0], time[0]) - position[0]
                                                positionsClaim += str(position[0]) + ',' + str(second) + "\t"
                                        indicesRefsClaim += str(min(indicesRefs[index][0], time[0])) + "\t"
                                        for timeI in range(min(indicesRefs[index][0], time[0]),
                                                           max(indicesRefs[index][1] + 1, time[-1] + 1)):
                                            timeClaim += str(timeI) + "\t"
                                        index += 1
                        while index < len(indicesRefs):
                            indicesRefsClaim += str(indicesRefs[index][0]) + "\t"
                            for timeI in range(indicesRefs[index][0], indicesRefs[index][-1] + 1):
                                timeClaim += str(timeI) + "\t"
                            index += 1
                        positionsClaim = positionsClaim[:-1]
                        self.positions.append(positionsClaim)
                        timeClaim = timeClaim[:-1]
                        self.timeIndices.append(timeClaim)
                        indicesRefsClaim = indicesRefsClaim[:-1]
                        self.refsIndices.append(indicesRefsClaim)
                        for data in datasM:
                            if type(data) == list:
                                if not data[0] in durenM:
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
                                                    difference = (data[0][0] - claim.claimDate[
                                                        1]).days
                                                else:
                                                    if (data[0][1] - claim.claimDate[0]).days <= 0:
                                                        difference = (data[0][1] - claim.claimDate[
                                                            0]).days
                                                    else:
                                                        difference = 0
                                                timeClaim += str(self.matchBucketAbsoluteTime(difference)) + "\t"
                                            else:
                                                if ((data[0] - claim.claimDate[0]).days >= 0 and (
                                                        data[0] - claim.claimDate[1]).days <= 0):
                                                    difference = 0
                                                else:
                                                    if (data[0] - claim.claimDate[0]).days <= 0:
                                                        difference = (data[0] - claim.claimDate[
                                                            0]).days
                                                    else:
                                                        difference = (data[0] - claim.claimDate[
                                                            1]).days
                                                timeClaim += str(self.matchBucketAbsoluteTime(difference)) + "\t"
                        timeClaim = timeClaim[:-1]
                        self.time.append(timeClaim)
                    for snippet in claim.snippets:
                        input_ids = [i for i in tokenizer.encode(
                            text=getSnippetTextLocal(snippet))]
                        decoding = [tokenizer.decode([i]) for i in input_ids]
                        OIE = snippet.readOpenInformationExtraction()
                        startIndex = snippet.getIndex()
                        indicesAll, tagsAll, positionsAll, allWords, sentencesIndices = ProcessOpenInformation(
                            OIE, decoding, startIndex, nlp)
                        if len(tagsAll['Verbs']) > 0:
                            for verb in tagsAll['Verbs']:
                                for part in verb:
                                    verbsSnippetIndices += str(part) + "\t"
                            verbsSnippetIndices = verbsSnippetIndices[:-1]
                        verbsSnippetIndices += ' 0123456789 '
                        if not type(claim.claimDate) is tuple:
                            datasM, durenM, refsM, setsM = snippet.readTime()
                            tokenisation = []
                            refsIndices = []
                            startIndex = snippet.getIndexHeidel()
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
                            differenceSnippet = differenceSnippets[0]
                            differenceSnippets = differenceSnippets[1:]
                            for data in datasM:
                                if not data[0] in durenM:
                                    if type(data) == list:
                                        if data[0] in refsM:
                                            if differenceSnippet != "None":
                                                timeSnippetElements += "Refs-" + self.matchBucketRelativeTime(
                                                    data[0], difference=differenceSnippet) + "\t"
                                        else:
                                            if not data[0] in setsM:
                                                if (claim.claimDate != None):
                                                    if type(data[0]) is tuple:
                                                        if ((data[0][
                                                                 0] - claim.claimDate).days <= 0 and (
                                                                data[0][
                                                                    1] - claim.claimDate).days >= 0):
                                                            difference = 0
                                                        else:
                                                            if (data[0][
                                                                    1] - claim.claimDate).days <= 0:
                                                                difference = (data[0][
                                                                                1] - claim.claimDate).days
                                                            else:
                                                                difference = (data[0][
                                                                                0] - claim.claimDate).days
                                                        timeSnippetElements += str(
                                                            self.matchBucketAbsoluteTime(difference)) + "\t"
                                                    else:
                                                        difference = (data[0] - claim.claimDate).days
                                                        timeSnippetElements += str(
                                                            self.matchBucketAbsoluteTime(difference)) + "\t"
                            if len(timeSnippetElements) > 0 and timeSnippetElements[-1] == "\t":
                                timeSnippetElements = timeSnippetElements[:-1]
                            timeSnippetElements += ' 0123456789 '
                        else:
                            datasM, durenM, refsM, setsM = snippet.readTime()
                            tokenisation = []
                            refsIndices = []
                            startIndex = snippet.getIndexHeidel()
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
                            differenceSnippet = differenceSnippets[0]
                            differenceSnippets = differenceSnippets[1:]
                            for data in datasM:
                                if type(data) == list:
                                    if not data[0] in durenM:
                                        if data[0] in refsM:
                                            if differenceSnippet != "None":
                                                timeSnippetElements += "Refs-" + self.matchBucketRelativeTime(
                                                    data[0], difference=differenceSnippet) + "\t"
                                        else:
                                            if type(data[0]) is tuple:
                                                if (data[0][0] - claim.claimDate[1]).days >= 0:
                                                    difference = (data[0][0] - claim.claimDate[
                                                        1]).days
                                                    timeSnippetElements += str(
                                                        self.matchBucketAbsoluteTime(difference)) + "\t"
                                                else:
                                                    if (data[0][1] - claim.claimDate[
                                                        0]).days <= 0:
                                                        difference = (data[0][1] -
                                                                    claim.claimDate[0]).days
                                                        timeSnippetElements += str(
                                                            self.matchBucketAbsoluteTime(difference)) + "\t"
                                                    else:
                                                        difference = 0
                                                        timeSnippetElements += str(
                                                            self.matchBucketAbsoluteTime(difference)) + "\t"
                                            else:
                                                if ((data[0] - claim.claimDate[
                                                    0]).days >= 0 and (
                                                        data[0] - claim.claimDate[
                                                    1]).days <= 0):
                                                    difference = 0
                                                else:
                                                    if (data[0] - claim.claimDate[0]).days <= 0:
                                                        difference = (data[0] - claim.claimDate[
                                                            0]).days
                                                    else:
                                                        difference = (data[0] - claim.claimDate[
                                                            1]).days
                                                timeSnippetElements += str(
                                                    self.matchBucketAbsoluteTime(difference)) + "\t"
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
    def getClaimText(self, claim):
        file = open("text" + "/" + claim.claimID + "/" + "claim", encoding="utf-8")
        return file.read()
    def getSnippets(self, snippets):
        dataSnippets = ''
        for snippet in snippets:
            file = open("text" + "/" + snippet.claimID + "/" + snippet.number,encoding="utf-8")
            dataSnippets += file.read()
            dataSnippets += ' 0123456789 '
            file.close()
        return dataSnippets

    def getNumbersOfTokensPreText(self, tokenizer, claim):
        file = open("pretext" + "/" + claim.claimID + "/" + "claim", encoding="utf-8")
        preText = file.read()
        inputIds =[i for i in tokenizer.encode(text=preText)]
        self.claimPreTextSize.append(len(inputIds))
        snippetPreTextSize = ''
        for snippet in claim.snippets:
            file = open("pretext" + "/" + claim.claimID + "/" + snippet.number, encoding="utf-8")
            preText = file.read()
            input_ids = [i for i in tokenizer.encode(text=preText)]
            snippetPreTextSize += str(len(input_ids))
            snippetPreTextSize += ' 0123456789 '
            file.close()
        self.snippetPreTextSize.append(snippetPreTextSize)

    def getMetaDataSet(self):
        return self.metadataSet

    def __len__(self):
        return len(self.claimIds)

    def __getitem__(self, index):
        return self.claimIds[index], self.documents[index], self.snippetDocs[index],self.metadata[index], self.labels[index],\
               self.claimDateAvailable[index],self.bucketsSnippets[index],self.verbIndices[index], self.timeIndices[index],\
               self.positions[index],self.refsIndices[index],self.time[index],self.verbIndicesSnippets[index],\
               self.timeIndicesSnippets[index],self.positionsSnippets[index],self.refsIndicesSnippets[index],self.timeSnippets[index], \
               self.claimPreTextSize[index],self.snippetPreTextSize[index]

# Dump dataset into file
def dump_write(dataset, name):
    with open(name, 'wb') as f:
        pickle.dump(dataset, f)


# retrieve dataset from file
def dump_load(name):
    with open(name, 'rb') as f:
        dataset = pickle.load(f)
    return dataset

def getClaimTextLocal(claim):
    file = open("textLocal" + "/" + claim.claimID + "/" + "claim", encoding="utf-8")
    return file.read()

def getSnippetTextLocal(snippet):
    file = open("textLocal" + "/" + snippet.claimID + "/" + snippet.number, encoding="utf-8")
    return file.read()

def getClaimText(claim):
    return claim.claim

def getSentence(words):
    sentence = ""
    for word in words:
        sentence += word + " "
    return sentence

'''
    Process open information structure for getting the indices of the verbs and time-expressions in the text
'''
def ProcessOpenInformation(OIE,decoding,startIndex,nlp):
    tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-distilroberta-v1')
    indicesAll = []
    tagsAll = {"Verbs":[],"Time":[],"Relative":[]}
    positionsAll = []
    index = startIndex
    if startIndex == 7:
        firstSentence = True
    else:
        firstSentence = False
    allWords = []
    sentencesIndices = []
    nextWordD = None
    nextWordE = None
    for items in OIE:
        indices = []
        words = items['words']
        allWords += words
        lastIndex = index
        stillToDo = []
        start = 0
        window = 1
        if len(words)==0:
            continue
        word = str(words[start]).replace('!','').replace('İ','').replace('@','').replace(',','').replace('•','').replace('_','').replace('[','').replace(']','').replace('*','').replace('�','').replace('.','').replace(chr(8220),'').replace(' ', '').replace("'", "").replace('"', '').replace('<', '').replace('/',
                                                                                                                 '').replace(
            '>', '').replace("\\", '').replace('(','').replace(')','').lower().encode("ascii", "ignore").decode()

        if nextWordE != None:
            word = nextWordE + word
        while len(word) == 0:
            indices.append([lastIndex])
            start += 1
            if start < len(words):

                word = str(words[start]).replace('!','').replace('İ','').replace('@','').replace(',','').replace('•','').replace('_','').replace('[','').replace(']','').replace('*','').replace('�','').replace('.','').replace(chr(8220),'').replace(' ', '').replace("'", "").replace('"', '').replace('<', '').replace('/',
                                                                                                                     '').replace(
                    '>', '').replace("\\", '').replace('(','').replace(')','').lower().encode("ascii", "ignore").decode()
            else:
                start -=1
                break
        if len(word) == 0:
            nextWordD = None
            nextWordE = None
            indicesAll += indices
            sentencesIndices.append(len(indicesAll))
            continue
        if nextWordD == None:
            wordDecoding = str(decoding[lastIndex]).replace('!','').replace('İ','').replace('@','').replace(',','').replace('•','').replace('_','').replace('[','').replace(']','').replace('*','').replace(chr(8220),'').replace('.','').replace('�','').replace(' ', '').replace("'", "").replace('"', '').replace('<',
                                                                                                               '').replace(
                '/', '').replace(
                '>', '').replace("\\", '').replace('(','').replace(')','').lower().encode("ascii", "ignore").decode()

            while (len(wordDecoding) == 0):
                if lastIndex +1 < len(decoding):
                    lastIndex += 1

                    wordDecoding = str(decoding[lastIndex]).replace('!','').replace('İ','').replace('@','').replace(',','').replace('•','').replace('_','').replace('[','').replace(']','').replace('*','').replace('.','').replace(chr(8220),'').replace('�','').replace(' ', '').replace("'", "").replace('"', '').replace('<',
                                                                                                                       '').replace(
                        '/', '').replace(
                        '>', '').replace("\\", '').replace('(','').replace(')','').lower().encode("ascii", "ignore").decode()

                else:
                    break
        else:
            wordDecoding = nextWordD
        if lastIndex < len(decoding) and start < len(words):
            word2 = str(decoding[lastIndex]).replace(' ', '').lower()
            word2Decoding = str(words[start]).replace(' ', '').lower()

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

                    else:
                        break

        while (len(wordDecoding.split(word)[0]) != 0 and
               len(word.split(wordDecoding)[0]) != 0):
            if lastIndex +1 < len(decoding):
                lastIndex += 1

                wordDecoding = str(decoding[lastIndex]).replace('!','').replace('İ','').replace('@','').replace(',','').replace('•','').replace('_','').replace('[','').replace(']','').replace('*','').replace('.','').replace(chr(8220),'').replace('�','').replace(' ', '').replace("'", "").replace('"', '').replace(
                    '<',
                    '').replace(
                    '/', '').replace(
                    '>', '').replace("\\", '').replace('(','').replace(')','').lower().encode("ascii", "ignore").decode()

            else:
                break

            while (len(wordDecoding) == 0):
                if lastIndex +1 < len(decoding):
                    lastIndex += 1

                    wordDecoding = str(decoding[lastIndex]).replace('!','').replace('İ','').replace('@','').replace(',','').replace('•','').replace('_','').replace('[','').replace(']','').replace('*','').replace('.','').replace(chr(8220),'').replace('�','').replace(' ', '').replace("'", "").replace('"', '').replace(
                        '<',
                        '').replace(
                        '/', '').replace(
                        '>', '').replace("\\", '').replace('(','').replace(')','').lower().encode("ascii", "ignore").decode()

                else:
                    break
        lastIndexWord = start
        while(lastIndexWord<len(words)):
            indexWord = lastIndexWord
            word = str(words[indexWord]).replace('!','').replace('İ','').replace('@','').replace(',','').replace('•','').replace('_','').replace('[','').replace(']','').replace('*','').replace('.','').replace(chr(8220),'').replace('�','').replace(' ', '').replace("'","").replace('"','').replace('<','').replace('/','').replace(
                '>','').replace("\\",'').replace(".",'').replace('(','').replace(')','').lower().encode("ascii", "ignore").decode()
            if nextWordE != None:
                word = nextWordE + word

            if len(word)==0:
                for index in range(len(stillToDo)):
                    tuple = [lastIndex - len(stillToDo) + index]
                    indices.append(tuple)
                stillToDo = []
                window+=1
                indices.append([lastIndex])
                nextWordE = None
            else:
                for decod in range(lastIndex, len(decoding)):
                    if nextWordD == None:
                        wordDecoding = str(decoding[decod]).replace('!','').replace('İ','').replace('@','').replace(',','').replace('•','').replace('_','').replace('[','').replace(']','').replace('*','').replace('.','').replace(chr(8220),'').replace('�','').replace(' ', '').replace("'", "").replace('"', '').replace(
                            '<', '').replace('/', '').replace(
                            '>', '').replace("\\", '').replace('(','').replace('.','').replace(')','').lower().encode("ascii", "ignore").decode()

                        while (len(wordDecoding) == 0):
                            decod += 1
                            if decod < len(decoding):

                                wordDecoding = str(decoding[decod]).replace('!','').replace('İ','').replace('@','').replace(',','').replace('•','').replace('_','').replace('[','').replace(']','').replace('*','').replace('.','').replace(chr(8220),'').replace('�','').replace(' ', '').replace("'", "").replace('"',
                                                                                                              '').replace(
                                    '<', '').replace('/', '').replace(
                                    '>', '').replace("\\", '').replace('(','').replace(')','').lower().encode("ascii", "ignore").decode()

                            else:
                                break
                        if len(wordDecoding)==0:
                            break
                    else:
                        wordDecoding = nextWordD
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
                        parts = word.split(wordDecoding)
                        index = decod
                        if len(parts) >= 2 and index +1 <len(decoding) and len(parts[0])==0:
                            nextWordD = None
                            nextWordE = None
                            length = len(wordDecoding)
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
                                else:
                                    break
                            if len(wordDecoding)!= 0:
                                nextWord = word[length:]
                                parts = nextWord.lower().split(wordDecoding)
                                if len(parts) >= 2 and len(parts[0]) == 0 and not(nextWord==wordDecoding):
                                    for index2 in range(len(stillToDo)):
                                        tuple = [index - len(stillToDo) + index2]
                                        indices.append(tuple)
                                    stillToDo = []
                                    window = 1
                                    newIndices.append(index)
                                    index += 1
                                    done = True
                                    while len(parts) >= 2 and len(parts[0]) == 0 and not(nextWord==wordDecoding):
                                        length= len(wordDecoding)
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
                                            else:
                                                index -= 1
                                                break
                                        if len(wordDecoding) != 0:
                                            nextWord = nextWord[length:]
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
                            parts = wordDecoding.split(word)
                            index = indexWord
                            if len(parts) >= 2 and index + 1 < len(words) and len(parts[0]) == 0:
                                nextWordD = None
                                nextWordE = None
                                indices.append([decod])
                                length = len(word)
                                index += 1
                                word = str(words[index]).replace('!','').replace('İ','').replace('@','').replace(',','').replace('•','').replace('_','').replace('[','').replace(']','').replace('*','').replace('.','').replace(chr(8220),'').replace('�','').replace(' ', '').replace("'", "").replace('"',
                                                                                                       '').replace(
                                    '<', '').replace('/', '').replace(
                                    '>', '').replace("\\", '').replace('(','').replace(')','').lower().encode("ascii", "ignore").decode()
                                while len(word) == 0:
                                    indices.append([decod])
                                    index += 1
                                    if index < len(words):

                                        word = str(words[index]).replace('!','').replace('İ','').replace('@','').replace(',','').replace('•','').replace('_','').replace('[','').replace(']','').replace('*','').replace('.','').replace('�', '').replace(chr(8220), '').replace(' ',
                                                                                                                 '').replace(
                                            "'", "").replace('"', '').replace('<', '').replace('/',
                                                                                               '').replace(
                                            '>', '').replace("\\", '').replace('(', '').replace(')', '').lower().encode(
                                            "ascii", "ignore").decode()
                                    else:
                                        index -= 1
                                        break
                                if len(word)>0:
                                    nextWord = wordDecoding[length:]
                                    parts = nextWord.lower().split(word.lower())
                                    if len(parts) >= 2 and len(parts[0]) == 0 and not (nextWord == word):
                                        nextWordE = None
                                        for index2 in range(len(stillToDo)):
                                            tuple = [decod - len(stillToDo) + index2]
                                            indices.append(tuple)
                                        stillToDo = []
                                        indices.append([decod])
                                        index += 1
                                        while len(parts) >= 2 and len(parts[0])==0 and not(nextWord==word):
                                            length = len(word)
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

                                                else:
                                                    index -= 1
                                                    break

                                            if len(word)>0 :
                                                nextWord = nextWord[length:]
                                                parts = nextWord.lower().split(word.lower())
                                                if len(parts) >= 2 and len(parts[0]) == 0 and not(nextWord==word):
                                                    indices.append([decod])
                                                    nextWordE = None
                                                    index += 1
                                                    if index >= len(words):
                                                        parts = []
                                            else:
                                                parts = []
                                        if nextWord == word:
                                            nextWordD = None
                                            nextWordE = None
                                            for index2 in range(len(stillToDo)):
                                                tuple = [decod - len(stillToDo) + index2]
                                                indices.append(tuple)
                                            stillToDo = []
                                            window = 1
                                            indices.append([decod])
                                            lastIndexWord = index
                                            lastIndex = decod + 1
                                        else:
                                            nextWordD = nextWord
                                            stillToDo = []
                                            window = 1
                                            lastIndexWord = index - 1
                                            lastIndex = decod

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
                                            lastIndexWord = index
                                            lastIndex = decod + 1
                                            break
                                        else:
                                            nextWordD = nextWord
                                            stillToDo = []
                                            window = 1
                                            lastIndexWord = index - 1
                                            lastIndex = decod
                                            break

            lastIndexWord += 1
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
            for index in range(len(tags)):
                if tags[index][2:]=="V":
                    tagsSentence["Verbs"].append(indices[index])
                    if verbIndex == -1:
                        verbIndex = indices[index][0]
                        done = False
                        for indexNLP in range(indexNLPCurrent,len(sentenceNLP)):
                            if(str.lower(words[index]).find(str.lower(str(sentenceNLP[indexNLP])))!=-1):
                                '''
                                    0 = past
                                    1 = current
                                    2 = future
                                    -1 = underivable
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
        sentencesIndices.append(len(indicesAll))
        index = lastIndex
        tagsAll["Verbs"] += tagsSentence["Verbs"]
        tagsAll["Time"] += tagsSentence["Time"]
        tagsAll["Relative"] += tagsSentence["Relative"]
        for pos in positionsSentence:
            if len(pos)>0:
                positionsAll.append(pos)
    return indicesAll,tagsAll,positionsAll,allWords,sentencesIndices


'''
    Match the indices of OIE with that of the detraction of the timeML coming from HeidelTime
'''
def match(indicesAll, allWordsOIE, allWordsHeidel, hasTitle, indicesRefs):
    indicesAllHeidel = []
    indexO = 0
    indicesTimexes = []
    if hasTitle:
        firstSentence = True
    else:
        firstSentence = False
    lastIndexHeidel = 0
    indicesRefs2 = indicesRefs.copy()
    nextWordD = None
    nextWordE = None
    while lastIndexHeidel<len(allWordsHeidel):
        goToNext = False
        indexHeidel = lastIndexHeidel
        lastIndex = indexO
        if firstSentence:
            if indexHeidel + 1 < len(allWordsHeidel) and lastIndex + 1 < len(allWordsOIE):
                word2 = str(allWordsHeidel[indexHeidel]).replace(' ', '').lower()
                nextWord = str(allWordsHeidel[indexHeidel + 1]).replace(' ', '').lower()
                word2D = str(allWordsOIE[lastIndex]).replace(' ', '').lower()
                nextWordDecoding = str(allWordsOIE[lastIndex + 1]).replace(' ', '').lower()
                if not (word2D == word2 and nextWord == nextWordDecoding):
                    if (len(word2)>0 and word2[-1] == "'" and nextWord == 'says'):
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
            nextWordD = None
            indicesAllHeidel.append(indicesAll[min(lastIndex,len(indicesAll)-1)])
            if indexHeidel +1 < len(allWordsHeidel):
                if firstSentence and lastIndex + 1 < len(allWordsOIE):
                    word2 = str(allWordsHeidel[indexHeidel]).replace(' ', '').lower()
                    nextWord = str(allWordsHeidel[indexHeidel + 1]).replace(' ', '').lower()
                    word2D = str(allWordsOIE[indexDecoding]).replace(' ', '').lower()
                    nextWordDecoding = str(allWordsOIE[indexDecoding + 1]).replace(' ', '').lower()

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
        lastIndexHeidel = indexHeidel + 1
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

            if word == wordDecoding:
                nextWordD = None
                nextWordE = None
                indicesAllHeidel.append(indicesAll[indexWord])
                indexO = indexWord + 1
                break
            else:
                index = indexWord
                parts = word.lower().split(wordDecoding)
                if len(parts) >=2 and index +1 <len(allWordsOIE) and len(parts[0])==0:
                    nextWordD = None
                    nextWordE = None
                    length = len(wordDecoding)
                    newIndices = []
                    newIndices += indicesAll[index]
                    index += 1

                    wordDecoding = str(allWordsOIE[index]).replace('!','').replace('İ','').replace('@','').replace(',','').replace('•','').replace('_','').replace('[','').replace(']','').replace('*','').replace('.','').replace(chr(8220),'').replace('�','').replace(' ', '').replace("'", "").replace('"',
                                                                                                  '').replace(
                        '<', '').replace('/', '').replace(
                        '>', '').replace("\\", '').replace('(','').replace(')','').lower().encode("ascii", "ignore").decode()
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
                            break
                    if len(wordDecoding) != 0:
                        nextWord = word[length:]
                        parts = nextWord.split(wordDecoding)
                        if len(parts) >= 2 and len(parts[0]) == 0 and not(nextWord==wordDecoding):
                            newIndices += indicesAll[index]

                            index += 1
                            while len(parts) >= 2 and len(parts[0]) == 0 and not(nextWord==wordDecoding):
                                length = len(wordDecoding)
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
                                        break
                                if len(wordDecoding) != 0:
                                    nextWord = nextWord[length:]
                                    parts = nextWord.split(wordDecoding)
                                    if len(parts)>=2 and len(parts[0]) == 0 and not(nextWord==wordDecoding):
                                        newIndices += indicesAll[index]
                                        index += 1
                                        if index >= len(allWordsOIE):
                                            parts = []
                                else:
                                    parts = []
                            if nextWord == wordDecoding:
                                nextWordD = None
                                nextWordE = None
                                newIndices += indicesAll[index]
                                index += 1
                                indexO = index
                                indicesAllHeidel.append(newIndices)
                                break
                            else:
                                nextWordD = None
                                nextWordE = nextWord
                                indicesAllHeidel.append(newIndices)
                                indexO = index
                            break
                        else:
                            if nextWord == wordDecoding:
                                nextWordD = None
                                nextWordE = None
                                newIndices += indicesAll[index]
                                index += 1
                                indexO = index
                                indicesAllHeidel.append(newIndices)
                                break
                            else:
                                nextWordD = None
                                nextWordE = nextWord
                                indicesAllHeidel.append(newIndices)
                                indexO = index
                            break
                else:
                    parts = wordDecoding.split(word)
                    index = indexHeidel
                    if len(parts) >= 2 and index +1 <len(allWordsHeidel) and len(parts[0])==0:
                        nextWordD = None
                        nextWordE = None
                        indicesAllHeidel.append(indicesAll[indexWord])
                        length = len(word)
                        index += 1
                        word = str(allWordsHeidel[index]).replace('!','').replace('İ','').replace('@','').replace(',','').replace('•','').replace('_','').replace('[','').replace(']','').replace('*','').replace('.','').replace(chr(8220),'').replace('�','').replace(' ', '').replace("'", "").replace('"',
                                                                                                       '').replace(
                                    '<', '').replace('/', '').replace(
                                    '>', '').replace("\\", '').replace('(','').replace(')','').lower().encode("ascii", "ignore").decode()
                        while len(word) == 0:
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
                            nextWord = wordDecoding[length:]
                            parts = nextWord.lower().split(word.lower())
                            if len(parts) >= 2 and len(parts[0]) == 0 and not(nextWord==word):
                                indicesAllHeidel.append(indicesAll[indexWord])
                                index += 1
                                while len(parts) >= 2 and len(parts[0]) == 0 and not(nextWord==word):
                                    length = len(word)
                                    word = str(allWordsHeidel[index]).replace('!','').replace('İ','').replace('@','').replace(',','').replace('•','').replace('_','').replace('[','').replace(']','').replace('*','').replace('.','').replace(chr(8220),'').replace('�','').replace(' ', '').replace("'", "").replace('"',
                                                                                                               '').replace(
                                                '<', '').replace('/', '').replace(
                                                '>', '').replace("\\", '').replace('(','').replace(')','').lower().encode("ascii", "ignore").decode()
                                    while len(word) == 0:
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
                                        nextWord = nextWord[length:]
                                        parts = nextWord.lower().split(word.lower())
                                        if len(parts) >= 2 and len(parts[0]) == 0 and not(nextWord==word):
                                            indicesAllHeidel.append(indicesAll[indexWord])
                                            index += 1
                                            if index>= len(allWordsHeidel):
                                                parts = []
                                    else:
                                        parts = []
                                if nextWord == word:
                                    nextWordD = None
                                    nextWordE = None
                                    indicesAllHeidel.append(indicesAll[indexWord])
                                    indexO = indexWord + 1
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
                                else:
                                    nextWordD = nextWord
                                    nextWordE = None
                                    indexO = indexWord
                                    lastIndexHeidel = index
                                break
    for indices in indicesRefs2:
        startIndex = indicesAllHeidel[indices[0]][0]
        endIndex = indicesAllHeidel[indices[-1]][-1]
        indicesTimexes.append([startIndex, endIndex])

    return indicesTimexes

'''
test for loading the dataset for the huca domain
domain = "huca"
train_set = NUS(mode='Test', path=os.pardir + '/test/test-' + domain + '.tsv', domain=domain,pathToSave=sys.argv[1],number=1)
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
    print('Number of tokens pretext claim')
    print(c[17])
    print('Number of tokens pretext snippet')
    print(c[18])
    break
print("Done dataset")
'''






