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
def analyseUitbreiding1(mode,path):
    notCorrectClaim = 0
    notCorrectSnippet = 0
    totalClaims = 0
    totalSnippets = 0
    nlp = "None"
    predictorOIE = "None"
    predictorNER = "None"
    coreference = "None"
    days = dict()
    with open('verschilDays.txt','w',encoding='utf-8') as F:
        if mode != "Test":
            with open(path, 'r', encoding='utf-8') as file:
                for claim in file:
                    elements = claim.split('\t')
                    claim = Claim.claim(elements[0], elements[1], elements[2], elements[3], elements[4], elements[5],
                                        elements[6],
                                        elements[7], elements[8], elements[9], elements[10], elements[11], elements[12],
                                        os.pardir+"/snippets/", predictorOIE, predictorNER, nlp, coreference)
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
                                        verschil = 0
                                    else:
                                        if (snippet.publishTime[1]-claim.claimDate).days<=0:
                                            verschil = (snippet.publishTime[1]-claim.claimDate).days
                                        else:
                                            verschil = (snippet.publishTime[0] - claim.claimDate).days
                                    F.write(str(verschil))
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
                                        verschil = (snippet.publishTime[0]-claim.claimDate[1]).days
                                        F.write(str(verschil))
                                        F.write('\n')
                                    else:
                                        if(snippet.publishTime[1]-claim.claimDate[0]).days<=0:
                                            verschil = (snippet.publishTime[1]-claim.claimDate[0]).days
                                            F.write(str(verschil))
                                            F.write('\n')
                                        else:
                                            verschil = 0
                                            F.write(str(verschil))
                                            F.write('\n')
                                else:
                                    if ((snippet.publishTime-claim.claimDate[0]).days >= 0 and (
                                            snippet.publishTime-claim.claimDate[1]).days <= 0):
                                        verschil = 0
                                    else:
                                        if (snippet.publishTime-claim.claimDate[0]).days <= 0:
                                            verschil = (snippet.publishTime-claim.claimDate[0]).days
                                        else:
                                            verschil = (snippet.publishTime - claim.claimDate[1]).days
                                    F.write(str(verschil))
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
                                        os.pardir+os.pardir+"/snippets/", predictorOIE, predictorNER, nlp, coreference)
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
                                        verschil = 0
                                    else:
                                        if (snippet.publishTime[1] - claim.claimDate).days <= 0:
                                            verschil = (snippet.publishTime[1] - claim.claimDate).days
                                        else:
                                            verschil = (snippet.publishTime[0] - claim.claimDate).days
                                    F.write(str(verschil))
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
                                        verschil = (snippet.publishTime[0] - claim.claimDate[1]).days
                                        F.write(str(verschil))
                                        F.write('\n')
                                    else:
                                        if (snippet.publishTime[1] - claim.claimDate[0]).days <= 0:
                                            verschil = (snippet.publishTime[1] - claim.claimDate[0]).days
                                            F.write(str(verschil))
                                            F.write('\n')
                                        else:
                                            verschil = 0
                                            F.write(str(verschil))
                                            F.write('\n')
                                else:
                                    if ((snippet.publishTime - claim.claimDate[0]).days >= 0 and (
                                            snippet.publishTime - claim.claimDate[1]).days <= 0):
                                        verschil = 0
                                    else:
                                        if (snippet.publishTime - claim.claimDate[0]).days <= 0:
                                            verschil = (snippet.publishTime - claim.claimDate[0]).days
                                        else:
                                            verschil = (snippet.publishTime - claim.claimDate[1]).days
                                    F.write(str(verschil))
                                    F.write('\n')
                            else:
                                notCorrectSnippet += 1
                print('not good claim ' + str(notCorrectClaim))
                print('not good snippet ' + str(notCorrectSnippet))
                print('total claims ' + str(totalClaims))
                print('total snippets ' + str(totalSnippets))
    F.close()

'''
Calulate the numbers of appearences for each day between -100 and +100 days from claimdatum
'''
def dataBinningExtra(file):
    verschillen = np.zeros(201)
    with open(file, 'r', encoding='utf-8') as file:
        lines = file.readlines()
        for verschil in lines:
            if int(verschil.replace('\n',''))>=-100 and int(verschil.replace('\n',''))<=100 :
                verschillen[(int(verschil.replace('\n',''))+100)] += 1
    file.close()

'''
Construct the bins with the file of all differences in days with the claimdatum
'''
def dataBinning(file,numberOfBins):
    verschillen = []
    indices = []
    index = 0
    with open(file, 'r', encoding='utf-8') as file:
        lines = file.readlines()
        for verschil in lines:
            verschillen.append(int(verschil.replace('\n','')))
            indices.append(index)
            index +=1
    file.close()
    df = pd.DataFrame({'days': verschillen,'indices' : indices })
    df['days_bin'] = pd.qcut(df['days'], q=numberOfBins)
    print(df['days_bin'].value_counts())

'''
Make a file with all differences in days between the time entities in the text and the claimdatum
'''
def analyseUitbreiding2(mode,path):
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
    duren = set()
    refs = set()
    sets = set()
    aantalDuren = dict()
    aantalRefs = dict()
    aantalSets = dict()
    with open('verschilDaysUitbreiding2.txt', 'w', encoding='utf-8') as F:
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
                    path = os.pardir+"ProcessedTimes" + "/" + claim.claimID + "/" + "claim" + ".xml"
                    if os.path.exists(path):
                        if not type(claim.claimDate) is tuple:
                            totalClaims += 1
                            datasM,durenM,refsM,setsM = claim.readTime()
                            for data in datasM:
                                if type(data) == int:
                                    with open("bad.txt", 'a', encoding='utf-8') as bad:
                                        bad.write(path+"\n")
                                else:
                                    if type(data) != str:
                                        if data[0] in durenM:
                                            duren.add(data[0])
                                            if data[0] in aantalDuren.keys():
                                                aantalDuren[data[0]] += 1
                                            else:
                                                aantalDuren[data[0]] = 1
                                        else:
                                            if data[0] in refsM:
                                                refs.add(data[0])
                                                if data[0] in aantalRefs.keys():
                                                    aantalRefs[data[0]] +=1
                                                else:
                                                    aantalRefs[data[0]] = 1
                                            else:
                                                if data[0] in setsM:
                                                    sets.add(data[0])
                                                    if data[0] in aantalSets.keys():
                                                        aantalSets[data[0]] += 1
                                                    else:
                                                        aantalSets[data[0]] = 1
                                                else:
                                                    datas.append(data[0])
                                                    if (claim.claimDate != None):
                                                        if type(data[0]) is tuple:
                                                            if ((data[0][0]-claim.claimDate).days <= 0 and (
                                                                data[0][1]-claim.claimDate).days >= 0):
                                                                verschil = 0
                                                            else:
                                                                if (data[0][1]-claim.claimDate).days <= 0:
                                                                    verschil = (data[0][1]-claim.claimDate).days
                                                                else:
                                                                    verschil = (data[0][0]-claim.claimDate).days
                                                            F.write(str(verschil))
                                                            F.write('\n')
                                                        else:
                                                            F.write(str((data[0]-claim.claimDate).days))
                                                            F.write('\n')
                                                    datas.append(data)
                            for snippet in claim.getSnippets():
                                pathS = claim.claimID +'/'+snippet.number
                                print(pathS)
                                datasM,durenM,refsM,setsM = snippet.readTime()
                                for data in datasM:
                                    if type(data) == int:
                                        with open("bad.txt", 'a', encoding='utf-8') as bad:
                                            bad.write(pathS+"\n")
                                    else:
                                        if type(data) != str:
                                            if data[0] in durenM:
                                                duren.add(data[0])
                                                if data[0] in aantalDuren.keys():
                                                    aantalDuren[data[0]] += 1
                                                else:
                                                    aantalDuren[data[0]] = 1
                                            else:
                                                if data[0] in refsM:
                                                    refs.add(data[0])
                                                    if data[0] in aantalRefs.keys():
                                                        aantalRefs[data[0]] += 1
                                                    else:
                                                        aantalRefs[data[0]] = 1
                                                else:
                                                    if data[0] in setsM:
                                                        sets.add(data[0])
                                                        if data[0] in aantalSets.keys():
                                                            aantalSets[data[0]] += 1
                                                        else:
                                                            aantalSets[data[0]] = 1
                                                    else:
                                                        if (claim.claimDate != None):
                                                            if type(data[0]) is tuple:
                                                                if ((data[0][0]-claim.claimDate).days <= 0 and (
                                                                        data[0][1] - claim.claimDate).days >= 0):
                                                                    verschil = 0
                                                                else:
                                                                    if (data[0][1] - claim.claimDate).days <= 0:
                                                                        verschil = (data[0][1]-claim.claimDate).days
                                                                    else:
                                                                        verschil = (data[0][0]-claim.claimDate).days
                                                                F.write(str(verschil))
                                                                F.write('\n')
                                                            else:
                                                                F.write(str((data[0] - claim.claimDate).days))
                                                                F.write('\n')
                                                        else:
                                                            notCorrectSnippet += 1
                                                        datas.append(data[0])

                        else:
                            datasM, durenM, refsM, setsM = claim.readTime()
                            for data in datasM:
                                if type(data) == int:
                                    with open("bad.txt", 'a', encoding='utf-8') as bad:
                                        bad.write(path+"\n")
                                else:
                                    if type(data)!= str:
                                        if data[0] in durenM:
                                            duren.add(data[0])
                                            if data[0] in aantalDuren.keys():
                                                aantalDuren[data[0]] += 1
                                            else:
                                                aantalDuren[data[0]] = 1
                                        else:
                                            if data[0] in refsM:
                                                refs.add(data[0])
                                                if data[0] in aantalRefs.keys():
                                                    aantalRefs[data[0]] += 1
                                                else:
                                                    aantalRefs[data[0]] = 1
                                            else:
                                                if data[0] in setsM:
                                                    sets.add(data[0])
                                                    if data[0] in aantalSets.keys():
                                                        aantalSets[data[0]] += 1
                                                    else:
                                                        aantalSets[data[0]] = 1
                                                else:
                                                    if type(data[0]) is tuple:
                                                        if (data[0][0]-claim.claimDate[1]).days >= 0:
                                                            verschil = (data[0][0]-claim.claimDate[1]).days
                                                            F.write(str(verschil))
                                                            F.write('\n')
                                                        else:
                                                            if (data[0][1]-claim.claimDate[0]).days <= 0:
                                                                verschil = (data[0][1]-claim.claimDate[0]).days
                                                                F.write(str(verschil))
                                                                F.write('\n')
                                                            else:
                                                                verschil = 0
                                                                F.write(str(verschil))
                                                                F.write('\n')
                                                        datas.append(data[0])
                                                    else:
                                                        if ((data[0] - claim.claimDate[0]).days >= 0 and (
                                                                data[0] - claim.claimDate[1] ).days <= 0):
                                                            verschil = 0
                                                        else:
                                                            if (data[0]-claim.claimDate[0]).days <= 0:
                                                                verschil = (data[0]-claim.claimDate[0]).days
                                                            else:
                                                                verschil = (data[0]-claim.claimDate[1]).days
                                                        F.write(str(verschil))
                                                        F.write('\n')
                            for snippet in claim.getSnippets():
                                pathS = claim.claimID + '/' + snippet.number
                                print(claim.claimID + '/' + snippet.number)
                                datasM, durenM, refsM, setsM = snippet.readTime()
                                for data in datasM:
                                    if type(data) == int:
                                        with open("bad.txt", 'a', encoding='utf-8') as bad:
                                            bad.write(pathS+"\n")
                                    else:
                                        if type(data[0]) != str:
                                            if data[0] in durenM:
                                                duren.add(data[0])
                                                if data[0] in aantalDuren.keys():
                                                    aantalDuren[data[0]] += 1
                                                else:
                                                    aantalDuren[data[0]] = 1
                                            else:
                                                if data[0] in refsM:
                                                    refs.add(data[0])
                                                    if data[0] in aantalRefs.keys():
                                                        aantalRefs[data[0]] += 1
                                                    else:
                                                        aantalRefs[data[0]] = 1
                                                else:
                                                    if data[0] in setsM:
                                                        sets.add(data[0])
                                                        if data[0] in aantalSets.keys():
                                                            aantalSets[data[0]] += 1
                                                        else:
                                                            aantalSets[data[0]] = 1
                                                    else:
                                                        if type(data[0]) is tuple:
                                                            if (data[0][0] - claim.claimDate[1]).days >= 0:
                                                                verschil = (data[0][0] - claim.claimDate[1]).days
                                                                F.write(str(verschil))
                                                                F.write('\n')
                                                            else:
                                                                if (data[0][1] - claim.claimDate[0]).days <= 0:
                                                                    verschil = (data[0][1] - claim.claimDate[0]).days
                                                                    F.write(str(verschil))
                                                                    F.write('\n')
                                                                else:
                                                                    verschil = 0
                                                                    F.write(str(verschil))
                                                                    F.write('\n')
                                                            datas.append(data[0])
                                                        else:
                                                            if ((data[0] - claim.claimDate[0]).days >= 0 and (
                                                                    data[0] - claim.claimDate[1]).days <= 0):
                                                                verschil = 0
                                                            else:
                                                                if (data[0] - claim.claimDate[0]).days <= 0:
                                                                    verschil = (data[0] - claim.claimDate[0]).days
                                                                else:
                                                                    verschil = (data[0] - claim.claimDate[1]).days
                                                            F.write(str(verschil))
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
                                        os.pardir+"/snippets/", predictorOIE, predictorNER, nlp, coreference)

                    claim.readPublicationDate()
                    print(claim.claimID + '/' + 'claim')
                    path = os.pardir+"/ProcessedTimes" + "/" + claim.claimID + "/" + "claim" + ".xml"
                    if os.path.exists(path):
                        if not type(claim.claimDate) is tuple:
                            totalClaims += 1
                            datasM,durenM,refsM,setsM = claim.readTime()
                            for data in datasM:
                                if type(data) == int:
                                    with open("bad.txt", 'a', encoding='utf-8') as bad:
                                        bad.write(path+"\n")
                                else:
                                    if type(data) != str:
                                        if data[0] in durenM:
                                            duren.add(data[0])
                                            if data[0] in aantalDuren.keys():
                                                aantalDuren[data[0]] += 1
                                            else:
                                                aantalDuren[data[0]] = 1
                                        else:
                                            if data[0] in refsM:
                                                refs.add(data[0])
                                                if data[0] in aantalRefs.keys():
                                                    aantalRefs[data[0]] +=1
                                                else:
                                                    aantalRefs[data[0]] = 1
                                            else:
                                                if data[0] in setsM:
                                                    sets.add(data[0])
                                                    if data[0] in aantalSets.keys():
                                                        aantalSets[data[0]] += 1
                                                    else:
                                                        aantalSets[data[0]] = 1
                                                else:
                                                    datas.append(data[0])
                                                    if (claim.claimDate != None):
                                                        if type(data[0]) is tuple:
                                                            if ((data[0][0]-claim.claimDate).days <= 0 and (
                                                                data[0][1]-claim.claimDate).days >= 0):
                                                                verschil = 0
                                                            else:
                                                                if (data[0][1]-claim.claimDate).days <= 0:
                                                                    verschil = (data[0][1]-claim.claimDate).days
                                                                else:
                                                                    verschil = (data[0][0]-claim.claimDate).days
                                                            F.write(str(verschil))
                                                            F.write('\n')
                                                        else:
                                                            F.write(str((data[0]-claim.claimDate).days))
                                                            F.write('\n')
                                                    datas.append(data)
                            for snippet in claim.getSnippets():
                                pathS = claim.claimID +'/'+snippet.number
                                datasM,durenM,refsM,setsM = snippet.readTime()
                                for data in datasM:
                                    if type(data) == int:
                                        with open("bad.txt", 'a', encoding='utf-8') as bad:
                                            bad.write(pathS+"\n")
                                    else:
                                        if type(data) != str:
                                            if data[0] in durenM:
                                                duren.add(data[0])
                                                if data[0] in aantalDuren.keys():
                                                    aantalDuren[data[0]] += 1
                                                else:
                                                    aantalDuren[data[0]] = 1
                                            else:
                                                if data[0] in refsM:
                                                    refs.add(data[0])
                                                    if data[0] in aantalRefs.keys():
                                                        aantalRefs[data[0]] += 1
                                                    else:
                                                        aantalRefs[data[0]] = 1
                                                else:
                                                    if data[0] in setsM:
                                                        sets.add(data[0])
                                                        if data[0] in aantalSets.keys():
                                                            aantalSets[data[0]] += 1
                                                        else:
                                                            aantalSets[data[0]] = 1
                                                    else:
                                                        if (claim.claimDate != None):
                                                            if type(data[0]) is tuple:
                                                                if ((data[0][0]-claim.claimDate).days <= 0 and (
                                                                        data[0][1] - claim.claimDate).days >= 0):
                                                                    verschil = 0
                                                                else:
                                                                    if (data[0][1] - claim.claimDate).days <= 0:
                                                                        verschil = (data[0][1]-claim.claimDate).days
                                                                    else:
                                                                        verschil = (data[0][0]-claim.claimDate).days
                                                                F.write(str(verschil))
                                                                F.write('\n')
                                                            else:
                                                                F.write(str((data[0] - claim.claimDate).days))
                                                                F.write('\n')
                                                        else:
                                                            notCorrectSnippet += 1
                                                        datas.append(data[0])
                        else:
                            datasM, durenM, refsM, setsM = claim.readTime()
                            for data in datasM:
                                if type(data) == int:
                                    with open("bad.txt", 'a', encoding='utf-8') as bad:
                                        bad.write(path+"\n")
                                else:
                                    if type(data)!= str:
                                        if data[0] in durenM:
                                            duren.add(data[0])
                                            if data[0] in aantalDuren.keys():
                                                aantalDuren[data[0]] += 1
                                            else:
                                                aantalDuren[data[0]] = 1
                                        else:
                                            if data[0] in refsM:
                                                refs.add(data[0])
                                                if data[0] in aantalRefs.keys():
                                                    aantalRefs[data[0]] += 1
                                                else:
                                                    aantalRefs[data[0]] = 1
                                            else:
                                                if data[0] in setsM:
                                                    sets.add(data[0])
                                                    if data[0] in aantalSets.keys():
                                                        aantalSets[data[0]] += 1
                                                    else:
                                                        aantalSets[data[0]] = 1
                                                else:
                                                    if type(data[0]) is tuple:
                                                        if (data[0][0]-claim.claimDate[1]).days >= 0:
                                                            verschil = (data[0][0]-claim.claimDate[1]).days
                                                            F.write(str(verschil))
                                                            F.write('\n')
                                                        else:
                                                            if (data[0][1]-claim.claimDate[0]).days <= 0:
                                                                verschil = (data[0][1]-claim.claimDate[0]).days
                                                                F.write(str(verschil))
                                                                F.write('\n')
                                                            else:
                                                                verschil = 0
                                                                F.write(str(verschil))
                                                                F.write('\n')
                                                        datas.append(data[0])
                                                    else:
                                                        if ((data[0] - claim.claimDate[0]).days >= 0 and (
                                                                data[0] - claim.claimDate[1] ).days <= 0):
                                                            verschil = 0
                                                        else:
                                                            if (data[0]-claim.claimDate[0]).days <= 0:
                                                                verschil = (data[0]-claim.claimDate[0]).days
                                                            else:
                                                                verschil = (data[0]-claim.claimDate[1]).days
                                                        F.write(str(verschil))
                                                        F.write('\n')
                            for snippet in claim.getSnippets():
                                pathS = claim.claimID + '/' + snippet.number
                                print(claim.claimID + '/' + snippet.number)
                                datasM, durenM, refsM, setsM = snippet.readTime()
                                for data in datasM:
                                    if type(data) == int:
                                        with open("bad.txt", 'a', encoding='utf-8') as bad:
                                            bad.write(pathS+"\n")
                                    else:
                                        if type(data) != str:
                                            if data[0] in durenM:
                                                duren.add(data[0])
                                                if data[0] in aantalDuren.keys():
                                                    aantalDuren[data[0]] += 1
                                                else:
                                                    aantalDuren[data[0]] = 1
                                            else:
                                                if data[0] in refsM:
                                                    refs.add(data[0])
                                                    if data[0] in aantalRefs.keys():
                                                        aantalRefs[data[0]] += 1
                                                    else:
                                                        aantalRefs[data[0]] = 1
                                                else:
                                                    if data[0] in setsM:
                                                        sets.add(data[0])
                                                        if data[0] in aantalSets.keys():
                                                            aantalSets[data[0]] += 1
                                                        else:
                                                            aantalSets[data[0]] = 1
                                                    else:
                                                        if type(data[0]) is tuple:
                                                            if (data[0][0] - claim.claimDate[1]).days >= 0:
                                                                verschil = (data[0][0] - claim.claimDate[1]).days
                                                                F.write(str(verschil))
                                                                F.write('\n')
                                                            else:
                                                                if (data[0][1] - claim.claimDate[0]).days <= 0:
                                                                    verschil = (data[0][1] - claim.claimDate[0]).days
                                                                    F.write(str(verschil))
                                                                    F.write('\n')
                                                                else:
                                                                    verschil = 0
                                                                    F.write(str(verschil))
                                                                    F.write('\n')
                                                            datas.append(data[0])
                                                        else:
                                                            if ((data[0] - claim.claimDate[0]).days >= 0 and (
                                                                    data[0] - claim.claimDate[1]).days <= 0):
                                                                verschil = 0
                                                            else:
                                                                if (data[0] - claim.claimDate[0]).days <= 0:
                                                                    verschil = (data[0] - claim.claimDate[0]).days
                                                                else:
                                                                    verschil = (data[0] - claim.claimDate[1]).days
                                                            F.write(str(verschil))
                                                            F.write('\n')
                                                        datas.append(data)
                    else:
                        with open('NoTimexes.tsv', 'a', encoding='utf-8') as fp:
                            fp.write(lineF)
    return aantalDuren,aantalRefs,aantalSets

def getClaimText(claim):
    basepath = os.path.normpath(os.getcwd() + os.sep + os.pardir)
    file = open(basepath + "/text" + "/" + claim.claimID + "/" + "claim",encoding="utf-8")
    return file.read()

def getSnippetText(snippet):
    basepath = os.path.normpath(os.getcwd() + os.sep + os.pardir)
    file = open(basepath + "/text" + "/" + snippet.claimID + "/" + snippet.number,encoding="utf-8")
    return file.read()

def getSentence(words):
    sentence = ""
    for word in words:
        sentence+= word + " "
    return sentence

def ProcessOpenInformation(OIE,decoding,startIndex,nlp):
    tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-distilroberta-v1')
    indicesAll = []
    tagsAll = {"Verbs":[],"Time":[],"Relative":[]}
    positionsAll = []
    index = startIndex
    hasTitle = False
    if startIndex == 7:
        hasTitle = True
    numberS = 0
    allWords = []
    zinnenIndices = []
    for items in OIE:
        indices = []
        words = items['words']
        allWords += words
        lastIndex = index
        lastPart = None
        stillToDo = []
        for indexWord in range(len(words)):
            done = False
            for decod in range(lastIndex,min(lastIndex+10,len(decoding))):
                if '<' in decoding[decod].lower() or '/' in decoding[decod].lower() \
                        or '>' in decoding[decod].lower() or "\ ".replace(' ', '') in decoding[decod].lower() \
                        or '."' in decoding[decod].lower():
                    if lastPart != None:
                        parts = lastPart.replace(' ', '').split(words[indexWord].lower().replace(' ', ''))
                        if len(parts[0]) == 0:
                            indexW = len(words[indexWord].lower().replace(' ', ''))
                            lastPart = lastPart[indexW:]
                            for index in range(len(stillToDo)):
                                tuple = [decod-len(stillToDo)+index]
                                indices.append(tuple)
                            stillToDo= []
                            tuple = [decod]
                            indices.append(tuple)
                            done = True
                            if len(lastPart) == 0:
                                lastIndex = decod + 1
                                lastPart = None
                            break
                        else:
                            lastPart = None
                    else:
                        lastPart = decoding[decod].lower()
                        parts = lastPart.replace(' ', '').split(words[indexWord].lower().replace(' ', ''))
                        if len(parts[0]) == 0:
                            indexW = len(words[indexWord].lower().replace(' ', ''))
                            lastPart = lastPart[indexW:]
                            for index in range(len(stillToDo)):
                                tuple = [decod-len(stillToDo)+index]
                                indices.append(tuple)
                            stillToDo = []
                            tuple = [decod]
                            indices.append(tuple)
                            done = True
                            if len(lastPart) == 0:
                                lastIndex = decod + 1
                                lastPart = None
                            break
                        else:
                            lastPart = None
                if decoding[decod].lower() != ' ':
                    if words[indexWord].lower().replace(' ','').find(decoding[decod].lower().replace(' ',''))!=-1:
                        if decoding[decod][0] == ' ':
                            encode = [i for i in tokenizer.encode(text=" "+ words[indexWord]) if
                                      i not in {0, 2, 3, 1, 50264}]
                            for index in range(len(stillToDo)):
                                tuple = [decod-len(stillToDo)+index]
                                indices.append(tuple)
                            tuple = []
                            for j in range(len(encode)):
                                tuple.append(decod+j)
                            stillToDo = []
                            indices.append(tuple)
                            startEncoding = True
                            lastIndex = decod + len(encode)
                            done = True
                            break
                        else:
                            encode = [i for i in tokenizer.encode(text=words[indexWord]) if
                                      i not in {0, 2, 3, 1, 50264}]
                            for index in range(len(stillToDo)):
                                tuple = [decod-len(stillToDo)+index]
                                indices.append(tuple)
                            tuple = []
                            for j in range(len(encode)):
                                tuple.append(decod + j)
                            stillToDo = []
                            indices.append(tuple)
                            lastIndex = decod + len(encode)
                            done = True
                            break

            if not done:
                stillToDo.append(indexWord)
        for index in range(len(stillToDo)):
            tuple = [lastIndex]
            indices.append(tuple)
            lastIndex += 1

        tagsSentence= {"Verbs":[],"Time":[],"Relative":[]}
        positionsSentence = []
        sentence = getSentence(words)
        print(sentence)
        sentenceNLP = nlp(sentence)
        indexNLPCurrent = 0
        for item in items['verbs']:
            positionsItem = False
            tags = item['tags']
            verbIndex = -1
            lastVerbs = []
            timeIndex = -1
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
                                    0 = verleden
                                    1 = heden
                                    2 = toekomst
                                    -1 = vaag
                                '''
                                if sentenceNLP[indexNLP].pos_ == "VERB" or sentenceNLP[indexNLP].pos_ =="AUX":
                                    indexNLPCurrent = indexNLP
                                    if sentenceNLP[indexNLP].tag_ == "VBD":
                                        tagsSentence["Relative"].append(0)
                                        done = True
                                    else:
                                        if sentenceNLP[indexNLP].tag_ == "VBP" or sentenceNLP[indexNLP].tag_ == "VBZ":
                                            tagsSentence["Relative"].append(1)
                                            done = True
                                        else:
                                            if sentenceNLP[indexNLP].pos_ == "AUX" and (sentenceNLP[indexNLP].lemma_ == "shall" or sentenceNLP[indexNLP].lemma_ == "will"):
                                                tagsSentence["Relative"].append(2)
                                                done = True
                                            else:
                                                for child in sentenceNLP[indexNLP].children:
                                                    if child.dep_ == "aux" and (child.lemma_ == "shall" or child.lemma_ == "will"):
                                                        tagsSentence["Relative"].append(2)
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
                        tagsSentence["Time"].append(indices[index])
                        if timeIndex == -1:
                            timeIndex = indices[index][0]
                            positionsItem = True
                        args.add(tags[index][2:])
                    else:
                        if tags[index] != "O":
                            args.add(tags[index][2:])
            if len(args)>1:
                if positionsItem:
                    positionsItem = ([verbIndex, timeIndex - verbIndex])
                else:
                    positionsItem = []
                positionsSentence.append(positionsItem)
            else:
                for index in lastVerbs:
                    tagsSentence["Verbs"].remove(index)
                tagsSentence["Relative"].pop()

        if hasTitle:
            lastIndex += 2
            hasTitle = False
        indicesAll += indices
        zinnenIndices.append(len(indicesAll))
        index = lastIndex
        tagsAll["Verbs"] += tagsSentence["Verbs"]
        tagsAll["Time"] += tagsSentence["Time"]
        tagsAll["Relative"] += tagsSentence["Relative"]
        positionsAll += positionsSentence

    print(indicesAll)
    print(tagsAll)
    print(positionsAll)
    return indicesAll,tagsAll,positionsAll,allWords,zinnenIndices

def linkToVerb(OIE,sentenceIndices,indicesTimexes,positionsAll,tagsTime):
    nlp = spacy.load("en_core_web_trf")
    tagsIndices = []
    newPostions = []
    for index in indicesTimexes:
        done = False
        for position in positionsAll:
            if len(position)==2:
                startIndex = position[0] + position[1]
                times = tagsTime['Time']
                endIndex = startIndex
                for time in times:
                    if endIndex == time[0]:
                        endIndex = time[-1]
                if bool(set(range(startIndex,endIndex)) & set(range(index[0],index[1]))):
                    done = True
                    if index[0] < startIndex:
                        tagsIndices.append(index[0])
                        newPostions.append([position[0],index[0]])
                    else:
                        tagsIndices.append(startIndex)
                        newPostions.append([position[0], startIndex])




def match(indicesAll,allWordsOIE,allWordsHeidel,hasTitle,indicesRefs):
    indicesAllHeidel = []
    index = 0
    wait = 0
    indicesTimexes = []
    firstSentence = True
    timexes = dict()
    for indexHeidel in range(len(allWordsHeidel)):
        lastIndex = index
        word = str(allWordsHeidel[indexHeidel]).replace(' ','')
        if len(word) != 0:
            if indexHeidel + 1 < len(allWordsHeidel):
                nextWord = str(allWordsHeidel[indexHeidel + 1]).replace(' ', '')
                if hasTitle and word == "'" and nextWord == 'says' and firstSentence:
                    firstSentence = False
                    wait = 2
            if wait == 0:
                for indexWord in range(lastIndex,len(allWordsOIE)):
                    if(allWordsOIE[indexWord] != '...') :
                        if word.lower() == allWordsOIE[indexWord].lower():
                            indicesAllHeidel.append(indicesAll[indexWord])
                            for indices in indicesRefs:
                                if indices[0] == indexHeidel:
                                    timexes[indices[0]] = indicesAll[indexWord][0]
                                if indices[1] == indexHeidel:
                                    startIndex = timexes.get(indices[0])
                                    endIndex = indicesAll[indexWord][0]
                                    indicesTimexes.append([startIndex,endIndex])
                            index = indexWord + 1
                            break
                        else:
                            parts = word.lower().split(allWordsOIE[indexWord].lower())
                            index = indexWord
                            if len(parts)>1:
                                for indices in indicesRefs:
                                    if indices[0] == indexHeidel:
                                        timexes[indices[0]] = indicesAll[indexWord][0]
                                    if indices[1] == indexHeidel:
                                        startIndex = timexes.get(indices[0])
                                        endIndex = indicesAll[indexWord][0]
                                        indicesTimexes.append([startIndex, endIndex])
                            while len(parts)>1:
                                if len(parts[0]) == 0:
                                    indicesAllHeidel.append(indicesAll[index])
                                    index += 1
                                    parts = parts[1].lower().split(allWordsOIE[index].lower())
                                else:
                                    for indices in indicesRefs:
                                        if indices[1] == indexHeidel:
                                            startIndex = timexes.get(indices[0])
                                            endIndex = indicesAll[index][0]
                                            indicesTimexes.append([startIndex, endIndex])
                                    parts = parts[0]
                            break

            wait = max(0,wait-1)
    return indicesTimexes




def analyse3(mode,path):
    tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-distilroberta-v1')
    predictorOIE = "None"

    coreference = "None"
    days = dict()
    datas = []
    duren = set()
    refs = set()
    sets = set()
    aantalDuren = dict()
    aantalRefs = dict()
    aantalSets = dict()

    predictorNER = pred.from_path(
        "https://storage.googleapis.com/allennlp-public-models/ner-elmo.2021-02-12.tar.gz", cuda_device=-1)
    nlp = spacy.load("en_core_web_sm")
    nlpOIE = spacy.load("en_core_web_trf")
    with open('verschilDaysAbsolute2.txt', 'w', encoding='utf-8') as F:
        if mode != "Test":
            with open(path, 'r', encoding='utf-8') as file:
                for claim in file:
                    elements = claim.split('\t')
                    claim = Claim.claim(elements[0], elements[1], elements[2], elements[3], elements[4], elements[5],
                                        elements[6],
                                        elements[7], elements[8], elements[9], elements[10], elements[11], elements[12],
                                        os.pardir+"/snippets/", predictorOIE, predictorNER, nlp, coreference)

                    claim.processDate()
                    input_ids = [i for i in tokenizer.encode(
                        text=getClaimText(claim))]
                    decoding = [tokenizer.decode([i]) for i in input_ids]
                    startIndex = claim.getIndex()
                    OIE = claim.readOpenInformationExtraction()
                    indicesAll,tagsAll,positionsAll,allWords,zinnenIndices = ProcessOpenInformation(OIE,decoding,startIndex,nlpOIE)
--
                    path = os.pardir+"/ProcessedTimes" + "/" + claim.claimID + "/" + "claim" + ".xml"
                    print(path)
                    if os.path.exists(path):
                        if not type(claim.claimDate) is tuple:
                            datasM, durenM, refsM, setsM = claim.readTime()
                            refsIndices = []
                            tokenisation = []
                            startIndex = claim.getIndexHeidel()
                            for data in datasM:
                                if type(data) == str:
                                    tokenisation += [token for token in nlp.tokenizer(str(data))]
                                else:
                                    startItem = len(tokenisation)-startIndex
                                    tokenisation += [token for token in nlp.tokenizer(str(data[1]))]
                                    endItem = len(tokenisation) - startIndex-1
                                    refsIndices.append([startItem,endItem])
                            tokenisation = tokenisation[startIndex:]
                            indicesRefs = match(indicesAll,allWords,tokenisation,startIndex==6,refsIndices)
                            for data in datasM:
                                if type(data) == list:
                                    if data[0] in durenM:
                                        duren.add(data[0])
                                        if data in aantalDuren.keys():
                                            aantalDuren[data[0]] += 1
                                        else:
                                            aantalDuren[data[0]] = 1
                                    else:
                                        if data[0] in refsM:
                                            refs.add(data[0])
                                            if data in aantalRefs.keys():
                                                aantalRefs[data[0]] += 1
                                            else:
                                                aantalRefs[data[0]] = 1
                                        else:
                                            if data[0] in setsM:
                                                sets.add(data[0])
                                                if data[0] in aantalSets.keys():
                                                    aantalSets[data[0]] += 1
                                                else:
                                                    aantalSets[data[0]] = 1
                                            else:
                                                datas.append(data)
                                                if (claim.claimDate != None):
                                                    if type(data[0]) is tuple:
                                                        if ((data[0][0] - claim.claimDate).days <= 0 and (
                                                                data[0][1] - claim.claimDate).days >= 0):
                                                            verschil = 0
                                                        else:
                                                            if (data[0][1] - claim.claimDate).days <= 0:
                                                                verschil = (data[0][1] - claim.claimDate).days
                                                            else:
                                                                verschil = (data[0][0] - claim.claimDate).days
                                                        F.write(str(verschil))
                                                        F.write('\n')
                                                    else:
                                                        F.write(str((data[0] - claim.claimDate).days))
                                                        F.write('\n')
                                                datas.append(data)
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
                                    endItem = len(tokenisation) - startIndex-1
                                    refsIndices.append([startItem, endItem])
                            tokenisation = tokenisation[startIndex:]
                            indicesRefs = match(indicesAll, allWords, tokenisation, startIndex == 6, refsIndices)
                            for data in datasM:
                                if type(data[0]) != str:
                                    if data[0] in durenM:
                                        duren.add(data[0])
                                        if data[0] in aantalDuren.keys():
                                            aantalDuren[data[0]] += 1
                                        else:
                                            aantalDuren[data[0]] = 1
                                    else:
                                        if data[0] in refsM:
                                            refs.add(data[0])
                                            if data[0] in aantalRefs.keys():
                                                aantalRefs[data[0]] += 1
                                            else:
                                                aantalRefs[data[0]] = 1
                                        else:
                                            if data[0] in setsM:
                                                sets.add(data[0])
                                                if data[0] in aantalSets.keys():
                                                    aantalSets[data[0]] += 1
                                                else:
                                                    aantalSets[data[0]] = 1
                                            else:
                                                if type(data[0]) is tuple:
                                                    if (data[0][0] - claim.claimDate[1]).days >= 0:
                                                        verschil = (data[0][0] - claim.claimDate[1]).days
                                                        F.write(str(verschil))
                                                        F.write('\n')
                                                    else:
                                                        if (data[0][1] - claim.claimDate[0]).days >= 0:
                                                            verschil = (data[0][1] - claim.claimDate[0]).days
                                                            F.write(str(verschil))
                                                            F.write('\n')
                                                        else:
                                                            verschil = 0
                                                            F.write(str(verschil))
                                                            F.write('\n')
                                                    datas.append(data[0])
                                                else:
                                                    if ((data[0] - claim.claimDate[0]).days >= 0 and (
                                                            data[0] - claim.claimDate[1]).days <= 0):
                                                        verschil = 0
                                                    else:
                                                        if (data[0] - claim.claimDate[0]).days <= 0:
                                                            verschil = (data[0] - claim.claimDate[0]).days
                                                        else:
                                                            verschil = (data[0] - claim.claimDate[1]).days
                                                    F.write(str(verschil))
                                                    F.write('\n')

                    for snippet in claim.snippets:
                        print(claim.claimID + '/' + 'claim'+'/'+snippet.number)
                        input_ids = [i for i in tokenizer.encode(
                            text=getSnippetText(snippet))]
                        decoding = [tokenizer.decode([i]) for i in input_ids]
                        OIE = snippet.readOpenInformationExtraction()
                        startIndex = snippet.getIndex()
                        indicesAll,tagsAll,positionsAll,allWords,zinnenIndices = ProcessOpenInformation(OIE, decoding,startIndex,nlpOIE)
                        print('Indices')
                        print(indicesAll)
                        print('Tags')
                        print(tagsAll)
                        print('Positions')
                        print(positionsAll)
                        print('Words')
                        print(allWords)
                        print("Zinnen indices")
                        print(zinnenIndices)
                        path = os.pardir+"/ProcessedTimes" + "/" + claim.claimID + "/" + "claim" + ".xml"
                        print(path)
                        if os.path.exists(path):
                            if not type(claim.claimDate) is tuple:
                                datasM, durenM, refsM, setsM = snippet.readTime()
                                tokenisation = []
                                refsIndices = []
                                startIndex = snippet.getIndexHeidel()
                                for data in datasM:
                                    if type(data) == str:
                                        tokenisation += [token for token in nlp.tokenizer(str(data))]
                                    else:
                                        startItem = len(tokenisation) - startIndex
                                        tokenisation += [token for token in nlp.tokenizer(str(data[1]))]
                                        endItem = len(tokenisation) - startIndex-1
                                        refsIndices.append([startItem, endItem])
                                tokenisation = tokenisation[startIndex:]
                                indicesRefs = match(indicesAll, allWords, tokenisation,startIndex==6,refsIndices)
                                for data in datasM:
                                    if type(data) != str:
                                        if data[0] in durenM:
                                            duren.add(data[0])
                                            if data[0] in aantalDuren.keys():
                                                aantalDuren[data[0]] += 1
                                            else:
                                                aantalDuren[data[0]] = 1
                                        else:
                                            if data[0] in refsM:
                                                refs.add(data[0])
                                                if data[0] in aantalRefs.keys():
                                                    aantalRefs[data[0]] += 1
                                                else:
                                                    aantalRefs[data[0]] = 1
                                            else:
                                                if data[0] in setsM:
                                                    sets.add(data[0])
                                                    if data[0] in aantalSets.keys():
                                                        aantalSets[data[0]] += 1
                                                    else:
                                                        aantalSets[data[0]] = 1
                                                else:
                                                    if (claim.claimDate != None):
                                                        if type(data[0]) is tuple:
                                                            if ((data[0][0] - claim.claimDate).days <= 0 and (
                                                                    data[0][1] - claim.claimDate).days >= 0):
                                                                verschil = 0
                                                            else:
                                                                if (data[0][1] - claim.claimDate).days <= 0:
                                                                    verschil = (data[0][1] - claim.claimDate).days
                                                                else:
                                                                    verschil = (data[0][0] - claim.claimDate).days
                                                            F.write(str(verschil))
                                                            F.write('\n')
                                                        else:
                                                            F.write(str((claim.claimDate - data[0]).days))
                                                            F.write('\n')
                                                    datas.append(data[0])
                            else:
                                print(claim.claimID + '/' + snippet.number)
                                datasM, durenM, refsM, setsM = snippet.readTime()
                                tokenisation = []
                                refsIndices = []
                                startIndex = snippet.getIndexHeidel()
                                for data in datasM:
                                    if type(data) == str:
                                        tokenisation += [token for token in nlp.tokenizer(str(data))]
                                    else:
                                        startItem = len(tokenisation) - startIndex
                                        tokenisation += [token for token in nlp.tokenizer(str(data[1]))]
                                        endItem = len(tokenisation) - startIndex-1
                                        refsIndices.append([startItem, endItem])
                                tokenisation = tokenisation[startIndex:]
                                indicesRefs = match(indicesAll, allWords, tokenisation, startIndex == 6, refsIndices)
                                for data in datasM:
                                    if type(data[0]) != str:
                                        if data[0] in durenM:
                                            duren.add(data[0])
                                            if data[0] in aantalDuren.keys():
                                                aantalDuren[data[0]] += 1
                                            else:
                                                aantalDuren[data[0]] = 1
                                        else:
                                            if data[0] in refsM:
                                                refs.add(data[0])
                                                if data[0] in aantalRefs.keys():
                                                    aantalRefs[data[0]] += 1
                                                else:
                                                    aantalRefs[data[0]] = 1
                                            else:
                                                if data[0] in setsM:
                                                    sets.add(data[0])
                                                    if data[0] in aantalSets.keys():
                                                        aantalSets[data[0]] += 1
                                                    else:
                                                        aantalSets[data[0]] = 1
                                                else:
                                                    if type(data[0]) is tuple:
                                                        if (data[0][0] - claim.claimDate[1]).days >= 0:
                                                            verschil = (data[0][0] - claim.claimDate[1]).days
                                                            F.write(str(verschil))
                                                            F.write('\n')
                                                        else:
                                                            if (data[0][1] - claim.claimDate[0]).days >= 0:
                                                                verschil = (data[0][1] - claim.claimDate[0]).days
                                                                F.write(str(verschil))
                                                                F.write('\n')
                                                            else:
                                                                verschil = 0
                                                                F.write(str(verschil))
                                                                F.write('\n')
                                                        datas.append(data[0])
                                                    else:
                                                        if ((data[0] - claim.claimDate[0]).days >= 0 and (
                                                                data[0] - claim.claimDate[1]).days <= 0):
                                                            verschil = 0
                                                        else:
                                                            if (data[0] - claim.claimDate[0]).days <= 0:
                                                                verschil = (data[0] - claim.claimDate[0]).days
                                                            else:
                                                                verschil = (data[0] - claim.claimDate[1]).days
                                                        F.write(str(verschil))
                                                        F.write('\n')
                                                    datas.append(data)




def days_between(d1, d2):
    return abs((d2 - d1).days)

def durationToSeconds(duren):
    with open('duraties.txt', 'w', encoding='utf-8') as F:
        calculation = {"Y" : 31556926,
                       "M" : 2629746,
                       "W" : 604800,
                       "D" : 86400,
                       "H" : 3600,
                       "Min" : 60,
                        "WE": 172800,
                        "Q" : 15778476,
                       "DE" : 315569260,
                       "S" : 1,
                       "CE" : 3155692600}
        for item in duren.keys():
            if item[0:2] == "PT":
                if item[-1] == "M":
                    symbol = "Min"
                else:
                    symbol = item[-1]
                if item[2:-1] =="X":
                    value = 1
                else:
                    value = int(item[2:-1])
                result =  value *calculation[symbol]
                for j in range(duren[item]):
                    F.write(str(result))
                    F.write("\n")
            else:
                if item[0] == "P":
                    if item[-2:] == "WE" or item[-2:] == "DE" or item[-2:] =="CE":
                        symbol = item[-2:]
                        if item[1:-2] == "X":
                            value = 1
                        else:
                            value = int(item[1:-2])
                        result = value * calculation[symbol]
                        for j in range(duren[item]):
                            F.write(str(result))
                            F.write("\n")
                    else:
                        symbol = item[-1]
                        if item[1:-1] == "X":
                            value = 1
                        else:
                            value = int(item[1:-1])
                        result = value * calculation[symbol]
                        for j in range(duren[item]):
                            F.write(str(result))
                            F.write("\n")
                else:
                    print(item)

'''
uitbreiding 1
'''
analyseUitbreiding1('Train',os.pardir+'/train/train.tsv')
dataBinning('verschilDays.txt',20)
'''
uitbreiding 2
'''
analyseUitbreiding2('Train',os.pardir+'/train/train.tsv')
dataBinning('verschilDaysUitbreiding2.txt',25)
