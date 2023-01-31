import json
import os
import pickle
from datetime import datetime
import datetime

import numpy as np
import pandas as pd
import spacy
from transformers import AutoTokenizer
from allennlp.predictors.predictor import Predictor as pred

import Claim

'''
Make a file with all differences in days between the evidence dates and the claimdatum
'''
def analyseExpansion1(mode,path):
    notCorrectClaim = 0
    notCorrectSnippet = 0
    totalClaims = 0
    totalSnippets = 0
    nlp = "None"
    predictorOIE = "None"
    predictorNER = "None"
    coreference = "None"
    days = dict()
    with open('differenceDaysPublicationDate.txt','w',encoding='utf-8') as F:
        if mode != "Test":
            with open(path, 'r', encoding='utf-8') as file:
                for claim in file:
                    elements = claim.split('\t')
                    claim = Claim.claim(elements[0], elements[1], elements[2], elements[3], elements[4], elements[5],
                                        elements[6],
                                        elements[7], elements[8], elements[9], elements[10], elements[11], elements[12],
                                        "snippets", predictorOIE, predictorNER, nlp, coreference)
                    totalClaims += 1
                    claim.readPublicationDate()
                    if claim.claimDate == 'None':
                        notCorrectClaim += 1
                    if not type(claim.claimDate) is tuple:
                        for snippet in claim.getSnippets():
                            totalSnippets += 1
                            snippet.readPublicationDate()
                            if (snippet.publishTime!=None and claim.claimDate!=None):
                                if type(snippet.publishTime) is tuple:
                                    if((snippet.publishTime[0] - claim.claimDate).days<=0 and (snippet.publishTime[1]-claim.claimDate).days>=0):
                                        difference = 0
                                    else:
                                        if (snippet.publishTime[1]-claim.claimDate).days<=0:
                                            difference = (snippet.publishTime[1]-claim.claimDate).days
                                        else:
                                            difference = (snippet.publishTime[0] - claim.claimDate).days
                                    F.write(str(difference))
                                    F.write('\n')
                                else:
                                    F.write(str((snippet.publishTime-claim.claimDate).days))
                                    F.write('\n')
                                    if abs((claim.claimDate - snippet.publishTime).days) in days.keys():
                                        days[abs((claim.claimDate - snippet.publishTime).days)] +=1
                                    else:
                                        days[abs((claim.claimDate - snippet.publishTime).days)] = 1
                            else:
                                notCorrectSnippet += 1
                    else:
                        for snippet in claim.getSnippets():
                            totalSnippets += 1
                            snippet.readPublicationDate()
                            if (snippet.publishTime!=None and claim.claimDate!=None):
                                if type(snippet.publishTime) is tuple:
                                    if(snippet.publishTime[0]-claim.claimDate[1]).days>=0:
                                        difference = (snippet.publishTime[0]-claim.claimDate[1]).days
                                        F.write(str(difference))
                                        F.write('\n')
                                    else:
                                        if(snippet.publishTime[1]-claim.claimDate[0]).days<=0:
                                            difference = (snippet.publishTime[1]-claim.claimDate[0]).days
                                            F.write(str(difference))
                                            F.write('\n')
                                        else:
                                            difference = 0
                                            F.write(str(difference))
                                            F.write('\n')
                                else:
                                    if ((snippet.publishTime-claim.claimDate[0]).days >= 0 and (
                                            snippet.publishTime-claim.claimDate[1]).days <= 0):
                                        difference = 0
                                    else:
                                        if (snippet.publishTime-claim.claimDate[0]).days <= 0:
                                            difference = (snippet.publishTime-claim.claimDate[0]).days
                                        else:
                                            difference = (snippet.publishTime - claim.claimDate[1]).days
                                    F.write(str(difference))
                                    F.write('\n')
                            else:
                                notCorrectSnippet += 1
            print('not good claim ' + str(notCorrectClaim))
            print('not good snippet ' + str(notCorrectSnippet))
            print('total claims ' + str(totalClaims))
            print('total snippets ' + str(totalSnippets))
        else:
            with open(path, 'r', encoding='utf-8') as file:
                for claim in file:
                    elements = claim.split('\t')
                    claim = Claim.claim(elements[0], elements[1], 'None', elements[2], elements[3], elements[4],
                                        elements[5],
                                        elements[6],
                                        elements[7], elements[8], elements[9], elements[10], elements[11],
                                        "snippets", predictorOIE, predictorNER, nlp, coreference)
                    totalClaims += 1
                    claim.readPublicationDate()
                    if claim.claimDate == 'None':
                        notCorrectClaim += 1
                    if not type(claim.claimDate) is tuple:
                        for snippet in claim.getSnippets():
                            totalSnippets += 1
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
                                    F.write(str(difference))
                                    F.write('\n')
                                else:
                                    F.write(str((snippet.publishTime - claim.claimDate).days))
                                    F.write('\n')
                                    if abs((claim.claimDate - snippet.publishTime).days) in days.keys():
                                        days[abs((claim.claimDate - snippet.publishTime).days)] += 1
                                    else:
                                        days[abs((claim.claimDate - snippet.publishTime).days)] = 1
                            else:
                                notCorrectSnippet += 1

                    else:
                        for snippet in claim.getSnippets():
                            totalSnippets += 1
                            snippet.processDate()
                            if (snippet.publishTime != None and claim.claimDate != None):
                                if type(snippet.publishTime) is tuple:
                                    if (snippet.publishTime[0] - claim.claimDate[1]).days >= 0:
                                        difference = (snippet.publishTime[0] - claim.claimDate[1]).days
                                        F.write(str(difference))
                                        F.write('\n')
                                    else:
                                        if (snippet.publishTime[1] - claim.claimDate[0]).days <= 0:
                                            difference = (snippet.publishTime[1] - claim.claimDate[0]).days
                                            F.write(str(difference))
                                            F.write('\n')
                                        else:
                                            difference = 0
                                            F.write(str(difference))
                                            F.write('\n')
                                else:
                                    if ((snippet.publishTime - claim.claimDate[0]).days >= 0 and (
                                            snippet.publishTime - claim.claimDate[1]).days <= 0):
                                        difference = 0
                                    else:
                                        if (snippet.publishTime - claim.claimDate[0]).days <= 0:
                                            difference = (snippet.publishTime - claim.claimDate[0]).days
                                        else:
                                            difference = (snippet.publishTime - claim.claimDate[1]).days
                                    F.write(str(difference))
                                    F.write('\n')
                            else:
                                notCorrectSnippet += 1
                print('not good claim ' + str(notCorrectClaim))
                print('not good snippet ' + str(notCorrectSnippet))
                print('total claims ' + str(totalClaims))
                print('total snippets ' + str(totalSnippets))
    F.close()

'''
Construct the bins with the file of all differences in days with the claimdatum
'''
def dataBinning(file,numberOfBins):
    diferences = []
    indices = []
    index = 0
    with open(file, 'r', encoding='utf-8') as file:
        lines = file.readlines()
        for difference in lines:
            diferences.append(int(difference.replace('\n','')))
            indices.append(index)
            index +=1
    file.close()
    df = pd.DataFrame({'days': differences,'indices' : indices })
    df['days_bin'] = pd.qcut(df['days'], q=numberOfBins)
    print(df['days_bin'].value_counts())

'''
Make a file with all differences in days between the time entities in the text and the claimdatum
'''
def analyseU2(mode,path):
    notCorrectClaim = 0
    notCorrectSnippet = 0
    totalClaims = 0
    totalSnippets = 0
    nlp = "None"
    predictorOIE = "None"
    predictorNER = "None"
    coreference = "None"
    days = dict()
    datas = []
    durations = set()
    refs = set()
    sets = set()
    numberOfDurations = dict()
    numberOfRefs = dict()
    numberOfSets = dict()
    with open('differenceDaysTimexesInText.txt', 'w', encoding='utf-8') as F:
        if mode != "Test":
            with open(path, 'r', encoding='utf-8') as file:
                for lineF in file:
                    elements = lineF.split('\t')
                    claim = Claim.claim(elements[0], elements[1], elements[2], elements[3], elements[4], elements[5],
                                        elements[6],
                                        elements[7], elements[8], elements[9], elements[10], elements[11], elements[12],
                                        "snippets/", predictorOIE, predictorNER, nlp, coreference)

                    claim.readPublicationDate()
                    print(claim.claimID + '/' + 'claim')
                    path = "ProcessedTimes" + "/" + claim.claimID + "/" + "claim" + ".xml"
                    if os.path.exists(path):
                        if not type(claim.claimDate) is tuple:
                            totalClaims += 1
                            datasM,durationsM,refsM,setsM = claim.readTime()
                            for data in datasM:
                                if type(data) != str:
                                    if data[0] in durationsM:
                                        durations.add(data[0])
                                        if data[0] in numberOfDurations.keys():
                                            numberOfDurations[data[0]] += 1
                                        else:
                                            numberOfDurations[data[0]] = 1
                                    else:
                                        if data[0] in refsM:
                                            refs.add(data[0])
                                            if data[0] in numberOfRefs.keys():
                                                numberOfRefs[data[0]] +=1
                                            else:
                                                numberOfRefs[data[0]] = 1
                                        else:
                                            if data[0] in setsM:
                                                sets.add(data[0])
                                                if data[0] in numberOfSets.keys():
                                                    numberOfSets[data[0]] += 1
                                                else:
                                                    numberOfSets[data[0]] = 1
                                            else:
                                                datas.append(data[0])
                                                if (claim.claimDate != None):
                                                    if type(data[0]) is tuple:
                                                        if ((data[0][0]-claim.claimDate).days <= 0 and (
                                                            data[0][1]-claim.claimDate).days >= 0):
                                                            difference = 0
                                                        else:
                                                            if (data[0][1]-claim.claimDate).days <= 0:
                                                                difference = (data[0][1]-claim.claimDate).days
                                                            else:
                                                                difference = (data[0][0]-claim.claimDate).days
                                                        F.write(str(difference))
                                                        F.write('\n')
                                                    else:
                                                        F.write(str((data[0]-claim.claimDate).days))
                                                        F.write('\n')
                                                datas.append(data)
                            for snippet in claim.getSnippets():
                                pathS = claim.claimID +'/'+snippet.number
                                print(pathS)
                                datasM,durationsM,refsM,setsM = snippet.readTime()
                                for data in datasM:
                                    if type(data) != str:
                                        if data[0] in durationsM:
                                            durations.add(data[0])
                                            if data[0] in numberOfDurations.keys():
                                                numberOfDurations[data[0]] += 1
                                            else:
                                                numberOfDurations[data[0]] = 1
                                        else:
                                            if data[0] in refsM:
                                                refs.add(data[0])
                                                if data[0] in numberOfRefs.keys():
                                                    numberOfRefs[data[0]] += 1
                                                else:
                                                    numberOfRefs[data[0]] = 1
                                            else:
                                                if data[0] in setsM:
                                                    sets.add(data[0])
                                                    if data[0] in numberOfSets.keys():
                                                        numberOfSets[data[0]] += 1
                                                    else:
                                                        numberOfSets[data[0]] = 1
                                                else:
                                                    if (claim.claimDate != None):
                                                        if type(data[0]) is tuple:
                                                            if ((data[0][0]-claim.claimDate).days <= 0 and (
                                                                    data[0][1] - claim.claimDate).days >= 0):
                                                                difference = 0
                                                            else:
                                                                if (data[0][1] - claim.claimDate).days <= 0:
                                                                    difference = (data[0][1]-claim.claimDate).days
                                                                else:
                                                                    difference = (data[0][0]-claim.claimDate).days
                                                            F.write(str(difference))
                                                            F.write('\n')
                                                        else:
                                                            F.write(str((data[0] - claim.claimDate).days))
                                                            F.write('\n')
                                                    else:
                                                        notCorrectSnippet += 1
                                                    datas.append(data[0])

                        else:
                            datasM, durationsM, refsM, setsM = claim.readTime()
                            for data in datasM:
                                if type(data)!= str:
                                    if data[0] in durationsM:
                                        durations.add(data[0])
                                        if data[0] in numberOfDurations.keys():
                                            numberOfDurations[data[0]] += 1
                                        else:
                                            numberOfDurations[data[0]] = 1
                                    else:
                                        if data[0] in refsM:
                                            refs.add(data[0])
                                            if data[0] in numberOfRefs.keys():
                                                numberOfRefs[data[0]] += 1
                                            else:
                                                numberOfRefs[data[0]] = 1
                                        else:
                                            if data[0] in setsM:
                                                sets.add(data[0])
                                                if data[0] in numberOfSets.keys():
                                                    numberOfSets[data[0]] += 1
                                                else:
                                                    numberOfSets[data[0]] = 1
                                            else:
                                                if type(data[0]) is tuple:
                                                    if (data[0][0]-claim.claimDate[1]).days >= 0:
                                                        difference = (data[0][0]-claim.claimDate[1]).days
                                                        F.write(str(difference))
                                                        F.write('\n')
                                                    else:
                                                        if (data[0][1]-claim.claimDate[0]).days <= 0:
                                                            difference = (data[0][1]-claim.claimDate[0]).days
                                                            F.write(str(difference))
                                                            F.write('\n')
                                                        else:
                                                            difference = 0
                                                            F.write(str(difference))
                                                            F.write('\n')
                                                    datas.append(data[0])
                                                else:
                                                    if ((data[0] - claim.claimDate[0]).days >= 0 and (
                                                            data[0] - claim.claimDate[1] ).days <= 0):
                                                        difference = 0
                                                    else:
                                                        if (data[0]-claim.claimDate[0]).days <= 0:
                                                            difference = (data[0]-claim.claimDate[0]).days
                                                        else:
                                                            difference = (data[0]-claim.claimDate[1]).days
                                                    F.write(str(difference))
                                                    F.write('\n')
                            for snippet in claim.getSnippets():
                                pathS = claim.claimID + '/' + snippet.number
                                print(claim.claimID + '/' + snippet.number)
                                datasM, durationsM, refsM, setsM = snippet.readTime()
                                for data in datasM:
                                    if type(data[0]) != str:
                                        if data[0] in durationsM:
                                            durations.add(data[0])
                                            if data[0] in numberOfDurations.keys():
                                                numberOfDurations[data[0]] += 1
                                            else:
                                                numberOfDurations[data[0]] = 1
                                        else:
                                            if data[0] in refsM:
                                                refs.add(data[0])
                                                if data[0] in numberOfRefs.keys():
                                                    numberOfRefs[data[0]] += 1
                                                else:
                                                    numberOfRefs[data[0]] = 1
                                            else:
                                                if data[0] in setsM:
                                                    sets.add(data[0])
                                                    if data[0] in numberOfSets.keys():
                                                        numberOfSets[data[0]] += 1
                                                    else:
                                                        numberOfSets[data[0]] = 1
                                                else:
                                                    if type(data[0]) is tuple:
                                                        if (data[0][0] - claim.claimDate[1]).days >= 0:
                                                            difference = (data[0][0] - claim.claimDate[1]).days
                                                            F.write(str(difference))
                                                            F.write('\n')
                                                        else:
                                                            if (data[0][1] - claim.claimDate[0]).days <= 0:
                                                                difference = (data[0][1] - claim.claimDate[0]).days
                                                                F.write(str(difference))
                                                                F.write('\n')
                                                            else:
                                                                difference = 0
                                                                F.write(str(difference))
                                                                F.write('\n')
                                                        datas.append(data[0])
                                                    else:
                                                        if ((data[0] - claim.claimDate[0]).days >= 0 and (
                                                                data[0] - claim.claimDate[1]).days <= 0):
                                                            difference = 0
                                                        else:
                                                            if (data[0] - claim.claimDate[0]).days <= 0:
                                                                difference = (data[0] - claim.claimDate[0]).days
                                                            else:
                                                                difference = (data[0] - claim.claimDate[1]).days
                                                        F.write(str(difference))
                                                        F.write('\n')
                                                    datas.append(data)
                    else:
                        with open('NoTimexes.tsv','a',encoding='utf-8')as fp:
                            fp.write(lineF)
        else:
            with open(path, 'r', encoding='utf-8') as file:
                for lineF in file:
                    elements = lineF.split('\t')
                    claim = Claim.claim(elements[0], elements[1], "None", elements[2], elements[3],
                                        elements[4], elements[5],
                                        elements[6],
                                        elements[7], elements[8], elements[9], elements[10], elements[11],
                                        "snippets", predictorOIE, predictorNER, nlp, coreference)

                    claim.readPublicationDate()
                    print(claim.claimID + '/' + 'claim')
                    path = "ProcessedTimes" + "/" + claim.claimID + "/" + "claim" + ".xml"
                    if os.path.exists(path):
                        if not type(claim.claimDate) is tuple:
                            totalClaims += 1
                            datasM,durationsM,refsM,setsM = claim.readTime()
                            for data in datasM:
                                if type(data) != str:
                                    if data[0] in durationsM:
                                        durations.add(data[0])
                                        if data[0] in numberOfDurations.keys():
                                            numberOfDurations[data[0]] += 1
                                        else:
                                            numberOfDurations[data[0]] = 1
                                    else:
                                        if data[0] in refsM:
                                            refs.add(data[0])
                                            if data[0] in numberOfRefs.keys():
                                                numberOfRefs[data[0]] +=1
                                            else:
                                                numberOfRefs[data[0]] = 1
                                        else:
                                            if data[0] in setsM:
                                                sets.add(data[0])
                                                if data[0] in numberOfSets.keys():
                                                    numberOfSets[data[0]] += 1
                                                else:
                                                    numberOfSets[data[0]] = 1
                                            else:
                                                datas.append(data[0])
                                                if (claim.claimDate != None):
                                                    if type(data[0]) is tuple:
                                                        if ((data[0][0]-claim.claimDate).days <= 0 and (
                                                            data[0][1]-claim.claimDate).days >= 0):
                                                            difference = 0
                                                        else:
                                                            if (data[0][1]-claim.claimDate).days <= 0:
                                                                difference = (data[0][1]-claim.claimDate).days
                                                            else:
                                                                difference = (data[0][0]-claim.claimDate).days
                                                        F.write(str(difference))
                                                        F.write('\n')
                                                    else:
                                                        F.write(str((data[0]-claim.claimDate).days))
                                                        F.write('\n')
                                                datas.append(data)
                            for snippet in claim.getSnippets():
                                pathS = claim.claimID +'/'+snippet.number
                                datasM,durationsM,refsM,setsM = snippet.readTime()
                                for data in datasM:
                                    if type(data) != str:
                                        if data[0] in durationsM:
                                            durations.add(data[0])
                                            if data[0] in numberOfDurations.keys():
                                                numberOfDurations[data[0]] += 1
                                            else:
                                                numberOfDurations[data[0]] = 1
                                        else:
                                            if data[0] in refsM:
                                                refs.add(data[0])
                                                if data[0] in numberOfRefs.keys():
                                                    numberOfRefs[data[0]] += 1
                                                else:
                                                    numberOfRefs[data[0]] = 1
                                            else:
                                                if data[0] in setsM:
                                                    sets.add(data[0])
                                                    if data[0] in numberOfSets.keys():
                                                        numberOfSets[data[0]] += 1
                                                    else:
                                                        numberOfSets[data[0]] = 1
                                                else:
                                                    if (claim.claimDate != None):
                                                        if type(data[0]) is tuple:
                                                            if ((data[0][0]-claim.claimDate).days <= 0 and (
                                                                    data[0][1] - claim.claimDate).days >= 0):
                                                                difference = 0
                                                            else:
                                                                if (data[0][1] - claim.claimDate).days <= 0:
                                                                    difference = (data[0][1]-claim.claimDate).days
                                                                else:
                                                                    difference = (data[0][0]-claim.claimDate).days
                                                            F.write(str(difference))
                                                            F.write('\n')
                                                        else:
                                                            F.write(str((data[0] - claim.claimDate).days))
                                                            F.write('\n')
                                                    else:
                                                        notCorrectSnippet += 1
                                                    datas.append(data[0])
                        else:
                            datasM, durationsM, refsM, setsM = claim.readTime()
                            for data in datasM:
                                if type(data)!= str:
                                    if data[0] in durationsM:
                                        durations.add(data[0])
                                        if data[0] in numberOfDurations.keys():
                                            numberOfDurations[data[0]] += 1
                                        else:
                                            numberOfDurations[data[0]] = 1
                                    else:
                                        if data[0] in refsM:
                                            refs.add(data[0])
                                            if data[0] in numberOfRefs.keys():
                                                numberOfRefs[data[0]] += 1
                                            else:
                                                numberOfRefs[data[0]] = 1
                                        else:
                                            if data[0] in setsM:
                                                sets.add(data[0])
                                                if data[0] in numberOfSets.keys():
                                                    numberOfSets[data[0]] += 1
                                                else:
                                                    numberOfSets[data[0]] = 1
                                            else:
                                                if type(data[0]) is tuple:
                                                    if (data[0][0]-claim.claimDate[1]).days >= 0:
                                                        difference = (data[0][0]-claim.claimDate[1]).days
                                                        F.write(str(difference))
                                                        F.write('\n')
                                                    else:
                                                        if (data[0][1]-claim.claimDate[0]).days <= 0:
                                                            difference = (data[0][1]-claim.claimDate[0]).days
                                                            F.write(str(difference))
                                                            F.write('\n')
                                                        else:
                                                            difference = 0
                                                            F.write(str(difference))
                                                            F.write('\n')
                                                    datas.append(data[0])
                                                else:
                                                    if ((data[0] - claim.claimDate[0]).days >= 0 and (
                                                            data[0] - claim.claimDate[1] ).days <= 0):
                                                        difference = 0
                                                    else:
                                                        if (data[0]-claim.claimDate[0]).days <= 0:
                                                            difference = (data[0]-claim.claimDate[0]).days
                                                        else:
                                                            difference = (data[0]-claim.claimDate[1]).days
                                                    F.write(str(difference))
                                                    F.write('\n')
                            for snippet in claim.getSnippets():
                                pathS = claim.claimID + '/' + snippet.number
                                print(claim.claimID + '/' + snippet.number)
                                datasM, durationsM, refsM, setsM = snippet.readTime()
                                for data in datasM:
                                    if type(data) != str:
                                        if data[0] in durationsM:
                                            durations.add(data[0])
                                            if data[0] in numberOfDurations.keys():
                                                numberOfDurations[data[0]] += 1
                                            else:
                                                numberOfDurations[data[0]] = 1
                                        else:
                                            if data[0] in refsM:
                                                refs.add(data[0])
                                                if data[0] in numberOfRefs.keys():
                                                    numberOfRefs[data[0]] += 1
                                                else:
                                                    numberOfRefs[data[0]] = 1
                                            else:
                                                if data[0] in setsM:
                                                    sets.add(data[0])
                                                    if data[0] in numberOfSets.keys():
                                                        numberOfSets[data[0]] += 1
                                                    else:
                                                        numberOfSets[data[0]] = 1
                                                else:
                                                    if type(data[0]) is tuple:
                                                        if (data[0][0] - claim.claimDate[1]).days >= 0:
                                                            difference = (data[0][0] - claim.claimDate[1]).days
                                                            F.write(str(difference))
                                                            F.write('\n')
                                                        else:
                                                            if (data[0][1] - claim.claimDate[0]).days <= 0:
                                                                difference = (data[0][1] - claim.claimDate[0]).days
                                                                F.write(str(difference))
                                                                F.write('\n')
                                                            else:
                                                                difference = 0
                                                                F.write(str(difference))
                                                                F.write('\n')
                                                        datas.append(data[0])
                                                    else:
                                                        if ((data[0] - claim.claimDate[0]).days >= 0 and (
                                                                data[0] - claim.claimDate[1]).days <= 0):
                                                            difference = 0
                                                        else:
                                                            if (data[0] - claim.claimDate[0]).days <= 0:
                                                                difference = (data[0] - claim.claimDate[0]).days
                                                            else:
                                                                difference = (data[0] - claim.claimDate[1]).days
                                                        F.write(str(difference))
                                                        F.write('\n')
                                                    datas.append(data)
                    else:
                        with open('NoTimexes.tsv', 'a', encoding='utf-8') as fp:
                            fp.write(lineF)
    return numberOfDurations,numberOfRefs,numberOfSets

if sys.argv[1] == "divisionByPublicationDate":
    analyseExpansion1(sys.argv[2], sys.argv[3])
    dataBinning('differenceDaysPublicationDate.txt', sys.argv[4])
else:
    analyseExpansion2(sys.argv[2], sys.argv[3])
    dataBinning('differenceDaysTimexesInText.txt', sys.argv[4])

