import math
import pickle
import os
import sys
import random
import scipy.stats as ss
import numpy
import numpy as np
import torch.nn.functional as F
from scipy import stats
from division1DifferencePublication.encoderGlobal import encoder as encoderPublicatie
from division1DifferencePublication.verificationModelGlobal import verifactionModel as verificationPublicatie
from division1DifferencePublication import OneHotEncoder, labelEmbeddingLayer, encoderMetadata, \
    instanceEncoder, evidence_ranker, labelMaskDomain
from datasetIteratie2Combiner import NUS
import torch
from torch.utils.data import DataLoader

'''
    Calculate intra and inter SpearmanRankingCoefficient for division1DifferencePublication according to experiment 3
'''
def spearmanRanking(loaders,models):
    labelsBins = [{},{},{}]
    labelsBinsDomain = [{},{},{}]
    for loader in loaders:
        for data in loader[0]:
            for i in range(len(data[0])):
                for j in range(0,len(models)):
                    model = models[j]
                    metaDataClaim = model.metaDataEncoder.oneHotEncoder.encode(data[3][i], device)
                    metadata_encoding = model.metaDataEncoder(metaDataClaim.unsqueeze(0)).to(device)

                    labelsAll,labelsDomain = model.getRankingLabelsPerBin(data[1][i], data[2][i],
                                                                                               metadata_encoding,
                                                                                               domain,data[5][i],data[6][i])
                    for k, v in labelsAll.items():
                        if k in labelsBins[j].keys():
                            labelsBins[j][k] += v
                        else:
                            labelsBins[j][k] = v
                    for k, v in labelsDomain.items():
                        if k in labelsBinsDomain[j].keys():
                            labelsBinsDomain[j][k] += v
                        else:
                            labelsBinsDomain[j][k] = v
    return labelsBins,labelsBinsDomain

def getIntraRankingLabelsAll(labelsAll):
    spearmanLabelsAll = [{},{},{}]
    for model in range(0,len(spearmanLabelsAll)):
        for time in labelsAll[model]:
            print("Time - " + str(time))
            if(len(labelsAll[model][time]))>1:
                correlation,_ = stats.spearmanr(labelsAll[model][time],axis=1)
                if len(labelsAll[model][time]) == 2:
                    spearmanLabelsAll[model][time] = [correlation]
                else:
                    for i in range(0,len(correlation)):
                        for j in range(i+1,len(correlation[i])):
                            if time in spearmanLabelsAll[model]:
                                spearmanLabelsAll[model][time].append(correlation[i][j])
                            else:
                                spearmanLabelsAll[model][time] = [correlation[i][j]]
    return spearmanLabelsAll

def getInterRankingLabelsDomain(labelsDomain,domains):
    spearmanLabelsDomainAll = [{},{},{}]
    spearmanLabelsDomain = [{},{},{}]
    for domain in domains:
        times = list(labelsDomain[domain][0])
        for model in range(0, len(labelsDomain[domain])):
            spearmanLabelsDomain[model][domain] = {}
            for x in range(0, len(times)):
                time1 = times[x]
                for y in range(x + 1, len(times)):
                    time2 = times[y]
                    print("Evaluation : " + str(time1) + "-" + str(time2))
                    correlation,_ = stats.spearmanr(labelsDomain[domain][model][time1], labelsDomain[domain][model][time2],axis=1)
                    if(len(labelsDomain[domain][model][time1])+len(labelsDomain[domain][model][time2]))==2:
                        spearmanLabelsDomainAll[model][(time1, time2)] = [
                            correlation]
                    else:
                        for i in range(0,len(labelsDomain[domain][model][time1])):
                            for j in range(len(labelsDomain[domain][model][time1]),len(correlation[0])):
                                if (time1,time2) in spearmanLabelsDomain[model][domain]:
                                    spearmanLabelsDomain[model][domain][(time1,time2)].append(correlation[i][j])
                                else:
                                    spearmanLabelsDomain[model][domain][(time1,time2)] = [correlation[i][j]]
                                if (time1,time2) in spearmanLabelsDomainAll[model]:
                                    spearmanLabelsDomainAll[model][(time1,time2)].append(
                                        correlation[i][j])
                                else:
                                    spearmanLabelsDomainAll[model][(time1,time2)] = [
                                        correlation[i][j]]
    return spearmanLabelsDomainAll,spearmanLabelsDomain

def getIntraRankingLabelsDomain(labelsDomain,domains):
    spearmanLabelsDomainAll = [{},{},{}]
    spearmanLabelsDomain = [{},{},{}]
    for domain in domains:
        for model in range(0, len(labelsDomain[domain])):
            spearmanLabelsDomain[model][domain] = {}
            for time in labelsDomain[domain][model]:
                print("Time - " + str(time))
                if (len(labelsDomain[domain][model][time])) > 1:
                    correlation, _ = stats.spearmanr(labelsDomain[domain][model][time], axis=1)
                    if (len(labelsDomain[domain][model][time])) == 2:
                        spearmanLabelsDomain[model][domain][time] = [correlation]
                        spearmanLabelsDomainAll[model][time] = [
                            correlation]
                    else:
                        for i in range(0,len(correlation)):
                            for j in range(i+1,len(correlation)):
                                if time in spearmanLabelsDomain[model][domain]:
                                    spearmanLabelsDomain[model][domain][time].append(correlation[i][j])
                                else:
                                    spearmanLabelsDomain[model][domain][time] = [correlation[i][j]]
                                if time in spearmanLabelsDomainAll[model]:
                                    spearmanLabelsDomainAll[model][time].append(
                                        correlation[i][j])
                                else:
                                    spearmanLabelsDomainAll[model][time] = [
                                        correlation[i][j]]
    return spearmanLabelsDomainAll,spearmanLabelsDomain

def getInterRankingLabelsAll(labelsAll):
    spearmanLabelsAll = [{},{},{}]
    times = list(labelsAll[0])
    for model in range(0,len(spearmanLabelsAll)):
        for x in range(0,len(times)):
            time1  = times[x]
            for y in range(x+1,len(times)):
                time2 = times[y]
                print("Evaluation : " + str(time1) + "-" + str(time2))
                correlation, _ = stats.spearmanr(labelsAll[model][time1], labelsAll[model][time2],
                                                 axis=1)
                if (len(labelsAll[model][time1]) + len(labelsAll[model][time2])) == 2:
                    spearmanLabelsAll[model][(time1, time2)] = [
                        correlation]
                else:
                    for i in range(0, len(labelsAll[model][time1])):
                        for j in range(len(labelsAll[model][time1]), len(correlation[0])):
                            if (time1, time2) in spearmanLabelsAll[model]:
                                spearmanLabelsAll[model][(time1, time2)].append(correlation[i][j])
                            else:
                                spearmanLabelsAll[model][(time1, time2)] = [correlation[i][j]]
    return spearmanLabelsAll

def calculateMeanAndStdAll(spearmanLabelsAll):
    means = [{},{},{}]
    stds = [{},{},{}]
    for model in range(0,len(means)):
        for time in spearmanLabelsAll[model]:
            means[model][time] = np.mean(spearmanLabelsAll[model][time])
            stds[model][time] = np.std(spearmanLabelsAll[model][time])
    meansTogether = {}
    stdsTogether = {}
    for time in spearmanLabelsAll[0]:
        meansTogether[time] = (np.mean([means[0][time],means[1][time],means[2][time]]),np.std([means[0][time],means[1][time],means[2][time]]))
        stdsTogether[time] = (np.mean([stds[0][time],stds[1][time],stds[2][time]]),np.std([stds[0][time],stds[1][time],stds[2][time]]))
    return meansTogether,stdsTogether

def calculateMeanAndStdDomain(spearmanLabelsAll,domains):
    means = [{},{},{}]
    stds = [{},{},{}]
    for domain in domains:
        for model in range(0,len(means)):
            means[model][domain] = {}
            stds[model][domain] = {}
            for time in spearmanLabelsAll[model][domain]:
                means[model][domain][time] = np.mean(spearmanLabelsAll[model][domain][time])
                stds[model][domain][time] = np.std(spearmanLabelsAll[model][domain][time])
    meansTogether = {}
    stdsTogether = {}
    for domain in domains:
        meansTogether[domain] = {}
        stdsTogether[domain] = {}
        for time in spearmanLabelsAll[0][domain]:
            meansTogether[domain][time] = (np.mean([means[0][domain][time],means[1][domain][time],means[2][domain][time]]),np.std([means[0][domain][time],means[1][domain][time],means[2][domain][time]]))
            stdsTogether[domain][time] = (np.mean([stds[0][domain][time],stds[1][domain][time],stds[2][domain][time]]),np.std([stds[0][domain][time],stds[1][domain][time],stds[2][domain][time]]))
    return meansTogether,stdsTogether

def getLabelIndicesDomain(domainPath,labelPath,weightsPath):
    domainsIndices = dict()
    domainsLabels = dict()
    domainLabelIndices = dict()
    domainWeights = dict()
    labelSequence = []
    file = open(labelPath,'r')
    lines = file.readlines()
    for line in lines:
        labelSequence.append(line.replace('\n',''))
    file = open(domainPath,'r')
    lines = file.readlines()
    for line in lines:
        parts = line.split("\t")
        labelsDomain = parts[1].split(",")
        labelsDomain[-1] = labelsDomain[-1].replace('\n','')
        labelIndices = []
        for label in labelsDomain:
            labelIndices.append(labelSequence.index(label.replace('\n','')))
        labelIndicesDomainM = sorted(labelIndices)
        labelIndicesDomain = []
        for index in labelIndices:
            labelIndicesDomain.append(labelIndicesDomainM.index(index))
        domainsIndices[parts[0]] = labelIndices
        domainsLabels[parts[0]] = labelsDomain
        domainLabelIndices[parts[0]] = labelIndicesDomain
    file = open(weightsPath, 'r')
    lines = file.readlines()
    for line in lines:

        parts = line.split("\t")
        weightsDomainNormal = parts[1:]
        weightsDomainNormal[-1] = weightsDomainNormal[-1].replace('\n','')
        domainWeights[parts[0]] = torch.zeros(len(weightsDomainNormal))
        for i in range(len(weightsDomainNormal)):
            domainWeights[parts[0]][domainLabelIndices[parts[0]][i]] = float(weightsDomainNormal[i])
    #print(domainWeights)
    return domainsIndices,domainsLabels,domainLabelIndices,domainWeights

'''
argument 1 path of model
argument 2 name of domain to take examples from to calculate attribution
alpha=0.9
'''
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
domainIndices, domainLabels, domainLabelIndices, domainWeights = getLabelIndicesDomain(
    'labels/labels.tsv', 'labels/labelSequence', 'labels/weights.tsv')
domains = domainIndices.keys()
models = []
for domain in domains:
    test_set = NUS(mode='Test', path='test/test-' + domain + '.tsv',
                   pathToSave="test/time/dataset2/",
                   domain=domain)
    test_loader = DataLoader(test_set,
                            batch_size=1,
                            shuffle=False)
    models.append([test_loader,domain])
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
labelsDomainAll = {}
labelsAllAll = [{},{},{}]

with torch.no_grad():
    for model in models:
        oneHotEncoder = OneHotEncoder.oneHotEncoder('Metadata_sequence/metadata')
        labelEmbeddingLayerM = labelEmbeddingLayer.labelEmbeddingLayer(772, domainIndices)
        encoderM = encoderPublicatie(300, 128, 0.9).to(device)
        encoderMetadataM = encoderMetadata.encoderMetadata(3, 3, oneHotEncoder).to(device)
        instanceEncoderM = instanceEncoder.instanceEncoder().to(device)
        evidenceRankerM = evidence_ranker.evidenceRanker(772, 100).to(device)
        labelMaskDomainM = labelMaskDomain.labelMaskDomain(772, domainIndices, model[1],
                                                           len(domainIndices[model[1]])).to(device)
        verificationModelTime90A = verificationPublicatie(encoderM, encoderMetadataM, instanceEncoderM,
                                                          evidenceRankerM,
                                                          labelEmbeddingLayerM, labelMaskDomainM,
                                                          domainIndices, domainWeights,
                                                          model[1]).to(device)
        verificationModelTime90A.loading_NeuralNetwork(sys.argv[1])
        oneHotEncoder = OneHotEncoder.oneHotEncoder('Metadata_sequence/metadata')
        labelEmbeddingLayerM = labelEmbeddingLayer.labelEmbeddingLayer(772, domainIndices)
        encoderM = encoderPublicatie(300, 128, 0.9).to(device)
        encoderMetadataM = encoderMetadata.encoderMetadata(3, 3, oneHotEncoder).to(device)
        instanceEncoderM = instanceEncoder.instanceEncoder().to(device)
        evidenceRankerM = evidence_ranker.evidenceRanker(772, 100).to(device)
        labelMaskDomainM = labelMaskDomain.labelMaskDomain(772, domainIndices, model[1],
                                                           len(domainIndices[model[1]])).to(device)
        verificationModelTime90B = verificationPublicatie(encoderM, encoderMetadataM, instanceEncoderM,
                                                          evidenceRankerM,
                                                          labelEmbeddingLayerM, labelMaskDomainM,
                                                          domainIndices, domainWeights,
                                                          model[1]).to(device)
        verificationModelTime90B.loading_NeuralNetwork(sys.argv[1])
        oneHotEncoder = OneHotEncoder.oneHotEncoder('Metadata_sequence/metadata')
        labelEmbeddingLayerM = labelEmbeddingLayer.labelEmbeddingLayer(772, domainIndices)
        encoderM = encoderPublicatie(300, 128, 0.9).to(device)
        encoderMetadataM = encoderMetadata.encoderMetadata(3, 3, oneHotEncoder).to(device)
        instanceEncoderM = instanceEncoder.instanceEncoder().to(device)
        evidenceRankerM = evidence_ranker.evidenceRanker(772, 100).to(device)
        labelMaskDomainM = labelMaskDomain.labelMaskDomain(772, domainIndices, model[1],
                                                           len(domainIndices[model[1]])).to(device)
        verificationModelTime90C = verificationPublicatie(encoderM, encoderMetadataM, instanceEncoderM,
                                                          evidenceRankerM,
                                                          labelEmbeddingLayerM, labelMaskDomainM,
                                                          domainIndices, domainWeights,
                                                          model[1]).to(device)
        verificationModelTime90C.loading_NeuralNetwork(sys.argv[1])

        timeModels = [verificationModelTime90A,verificationModelTime90B,verificationModelTime90A]
        labelsDomain,labelsAll = spearmanRanking([model],timeModels)
        labelsDomainAll[model[1]] = labelsDomain
        for j in range(0,3):
            for k, v in labelsAll[j].items():
                if k in labelsAllAll[j].keys():
                    labelsAllAll[j][k] += v
                else:
                    labelsAllAll[j][k] = v
        '''
        file = open('resultPearson/biLSTM/versusBase/'+model[1],"w")
        
        if len(pearsonDomain) > 0:
            print("Ranking Time")
            print("Gemiddelde : " + model[1] + " : " + str(np.mean(pearsonDomain)))
            print("Deviatie: " + model[1] + " : " + str(np.std(pearsonDomain)))
            print("Deviation under the models:: " + model[1] + " : " + str(np.mean(standaardDomain)))
        if len(labelsDomain) >0:
            print("Labels domain Time")
            print("Gemiddelde : " + model[1] + " : " + str(np.mean(labelsDomain)))
            print("Deviatie: " + model[1] + " : " + str(np.std(labelsDomain)))
            print("Deviation under the models:: " + model[1] + " : " + str(np.mean(standaardLabelDomain)))
            print("Labels all Time")
        if  len(labelsAll) > 0 :
            print("Gemiddelde : " + model[1] + " : " + str(np.mean(labelsAll)))
            print("Deviatie: " + model[1] + " : " + str(np.std(labelsAll)))
            print("Deviation under the models:: " + model[1] + " : " + str(np.mean(standaardAll)))
        if len(pearsonAllAbsolute) > 0:
            print("Ranking Absolute")
            print("Gemiddelde : " + model[1] + " : " + str(np.mean(pearsonAllAbsolute)))
            print("Deviatie: " + model[1] + " : " + str(np.std(pearsonAllAbsolute)))
            print("Deviation under the models:: " + model[1] + " : " + str(np.mean(standaardDomainAbsolute)))
        if len(labelsDomainAbsolute) >0:
            print("Labels domain")
            print("Gemiddelde : " + model[1] + " : " + str(np.mean(labelsDomainAbsolute)))
            print("Deviatie: " + model[1] + " : " + str(np.std(labelsDomainAbsolute)))
            print("Deviation under the models:: " + model[1] + " : " + str(np.mean(standaardLabelDomainAbsolute)))
            print("Labels all")
        if  len(labelsAllAbsolute) > 0 :
            print("Gemiddelde : " + model[1] + " : " + str(np.mean(labelsAllAbsolute)))
            print("Deviatie: " + model[1] + " : " + str(np.std(labelsAllAbsolute)))
            print("Deviation under the models:: " + model[1] + " : " + str(np.mean(standaardLabelAllDomainAbsolute)))
        if len(pearsonAllEverything) > 0:
            print("Ranking Everything")
            print("Gemiddelde : " + model[1] + " : " + str(np.mean(pearsonAllEverything)))
            print("Deviatie: " + model[1] + " : " + str(np.std(pearsonAllEverything)))
            print("Deviation under the models:: " + model[1] + " : " + str(np.mean(standaardDomainEverything)))
        if len(labelsDomainEverything) >0:
            print("Labels domain")
            print("Gemiddelde : " + model[1] + " : " + str(np.mean(labelsDomainEverything)))
            print("Deviatie: " + model[1] + " : " + str(np.std(labelsDomainEverything)))
            print("Deviation under the models:: " + model[1] + " : " + str(np.mean(standaardLabelDomainEverything)))
            print("Labels all")
        if  len(labelsAllEverything) > 0 :
            print("Gemiddelde : " + model[1] + " : " + str(np.mean(labelsAllEverything)))
            print("Deviatie: " + model[1] + " : " + str(np.std(labelsAllEverything)))
            print("Deviation under the models:: " + model[1] + " : " + str(np.mean(standaardLabelAllDomainEverything)))
    
        if len(pearsonDomain) > 0:
            file.write("Ranking Time"+"\n")
            file.write("Gemiddelde : " + model[1] + " : " + str(np.mean(pearsonDomain)) +"\n")
            file.write("Deviatie: " + model[1] + " : " + str(np.std(pearsonDomain))+"\n")
            file.write("Deviation under the models:: " + model[1] + " : " + str(np.mean(standaardDomain))+"\n")
        if len(labelsDomain) >0:
            file.write("Labels domain Time"+"\n")
            file.write("Gemiddelde : " + model[1] + " : " + str(np.mean(labelsDomain))+"\n")
            file.write("Deviatie: " + model[1] + " : " + str(np.std(labelsDomain))+"\n")
            file.write("Deviation under the models:: " + model[1] + " : " + str(np.mean(standaardLabelDomain))+"\n")
            file.write("Labels all Time"+"\n")
        if  len(labelsAll) > 0 :
            file.write("Gemiddelde : " + model[1] + " : " + str(np.mean(labelsAll))+"\n")
            file.write("Deviatie: " + model[1] + " : " + str(np.std(labelsAll))+"\n")
            file.write("Deviation under the models:: " + model[1] + " : " + str(np.mean(standaardAll))+"\n")
        if len(pearsonAllAbsolute) > 0:
            file.write("Ranking Absolute"+"\n")
            file.write("Gemiddelde : " + model[1] + " : " + str(np.mean(pearsonAllAbsolute))+"\n")
            file.write("Deviatie: " + model[1] + " : " + str(np.std(pearsonAllAbsolute))+"\n")
            file.write("Deviation under the models:: " + model[1] + " : " + str(np.mean(standaardDomainAbsolute))+"\n")
        if len(labelsDomainAbsolute) >0:
            file.write("Labels domain"+"\n")
            file.write("Gemiddelde : " + model[1] + " : " + str(np.mean(labelsDomainAbsolute))+"\n")
            file.write("Deviatie: " + model[1] + " : " + str(np.std(labelsDomainAbsolute))+"\n")
            file.write("Deviation under the models:: " + model[1] + " : " + str(np.mean(standaardLabelDomainAbsolute))+"\n")
            file.write("Labels all"+"\n")
        if  len(labelsAllAbsolute) > 0 :
            file.write("Gemiddelde : " + model[1] + " : " + str(np.mean(labelsAllAbsolute))+"\n")
            file.write("Deviatie: " + model[1] + " : " + str(np.std(labelsAllAbsolute))+"\n")
            file.write("Deviation under the models:: " + model[1] + " : " + str(np.mean(standaardLabelAllDomainAbsolute))+"\n")
        if len(pearsonAllEverything) > 0:
            file.write("Ranking Everything"+"\n")
            file.write("Gemiddelde : " + model[1] + " : " + str(np.mean(pearsonAllEverything))+"\n")
            file.write("Deviatie: " + model[1] + " : " + str(np.std(pearsonAllEverything))+"\n")
            file.write("Deviation under the models:: " + model[1] + " : " + str(np.mean(standaardDomainEverything))+"\n")
        if len(labelsDomainEverything) >0:
            file.write("Labels domain"+"\n")
            file.write("Gemiddelde : " + model[1] + " : " + str(np.mean(labelsDomainEverything))+"\n")
            file.write("Deviatie: " + model[1] + " : " + str(np.std(labelsDomainEverything))+"\n")
            file.write("Deviation under the models:: " + model[1] + " : " + str(np.mean(standaardLabelDomainEverything))+"\n")
            file.write("Labels all"+"\n")
        if  len(labelsAllEverything) > 0 :
            file.write("Gemiddelde : " + model[1] + " : " + str(np.mean(labelsAllEverything))+"\n")
            file.write("Deviatie: " + model[1] + " : " + str(np.std(labelsAllEverything))+"\n")
            file.write("Deviation under the models:: " + model[1] + " : " + str(np.mean(standaardLabelAllDomainEverything))+"\n")
        file.write(str(numberEqual))
        file.close()
        pearsonAll.extend(pearsonDomain)
        standaard.extend(standaardDomain)
        labelsAllE.extend(labelsAll)
        standaardLabelsAll.extend(standaardAll)
        labelsDomainE.extend(labelsDomain)
        standaardDomainLabelsAll.extend(standaardLabelDomain)
        pearsonAllAbsoluteD.extend(pearsonAllAbsolute)
        standaardAbsolute.extend(standaardDomainAbsolute)
        labelsDomainEAbsolute.extend(labelsDomainAbsolute)
        standaardDomainLabelsAllAbsolute.extend(standaardLabelDomainAbsolute)
        labelsAllEAbsolute.extend(labelsAllAbsolute)
        standaardLabelsAllAbsolute.extend(standaardLabelAllDomainAbsolute)
        pearsonAllEverythingD.extend(pearsonAllEverything)
        standaardEverything.extend(standaardDomainEverything)
        labelsDomainEEverything.extend(labelsDomainEverything)
        standaardDomainLabelsAllEverything.extend(standaardLabelDomainEverything)
        labelsAllEEverything.extend(labelsAllEverything)
        standaardLabelsAllEverything.extend(standaardLabelAllDomainEverything)
        numberEqualAbsolute, pearsonAllAbsolute, standaardDomainAbsolute, labelsDomainAbsolute, standaardLabelDomainAbsolute, \
        labelsAllAbsolute, standaardLabelAllDomainAbsolute, pearsonAllEverything, standaardDomainEverything, \
        labelsDomainEverything, standaardLabelDomainEverything, labelsAllEverything, standaardLabelAllDomainEverything \
            = pearsonRankingTime([model],referenceModels, modelsAbsolute, modelsEverything)
        numberEqual2.update(numberEqualAbsolute)
        #print("Time with absolute and everything")
        file = open('resultPearson/biLSTM/versusTime/' + model[1], "w")
        
        if len(pearsonAllAbsolute) > 0:
            print("Ranking Absolute")
            print("Gemiddelde : " + model[1] + " : " + str(np.mean(pearsonAllAbsolute)))
            print("Deviatie: " + model[1] + " : " + str(np.std(pearsonAllAbsolute)))
            print("Deviation under the models:: " + model[1] + " : " + str(np.mean(standaardDomainAbsolute)))
        if len(labelsDomainAbsolute) >0:
            print("Labels domain")
            print("Gemiddelde : " + model[1] + " : " + str(np.mean(labelsDomainAbsolute)))
            print("Deviatie: " + model[1] + " : " + str(np.std(labelsDomainAbsolute)))
            print("Deviation under the models:: " + model[1] + " : " + str(np.mean(standaardLabelDomainAbsolute)))
            print("Labels all")
        if  len(labelsAllAbsolute) > 0 :
            print("Gemiddelde : " + model[1] + " : " + str(np.mean(labelsAllAbsolute)))
            print("Deviatie: " + model[1] + " : " + str(np.std(labelsAllAbsolute)))
            print("Deviation under the models:: " + model[1] + " : " + str(np.mean(standaardLabelAllDomainAbsolute)))
        if len(pearsonAllEverything) > 0:
            print("Ranking Everything")
            print("Gemiddelde : " + model[1] + " : " + str(np.mean(pearsonAllEverything)))
            print("Deviatie: " + model[1] + " : " + str(np.std(pearsonAllEverything)))
            print("Deviation under the models:: " + model[1] + " : " + str(np.mean(standaardDomainEverything)))
        if len(labelsDomainEverything) >0:
            print("Labels domain")
            print("Gemiddelde : " + model[1] + " : " + str(np.mean(labelsDomainEverything)))
            print("Deviatie: " + model[1] + " : " + str(np.std(labelsDomainEverything)))
            print("Deviation under the models:: " + model[1] + " : " + str(np.mean(standaardLabelDomainEverything)))
            print("Labels all")
        if  len(labelsAllEverything) > 0 :
            print("Gemiddelde : " + model[1] + " : " + str(np.mean(labelsAllEverything)))
            print("Deviatie: " + model[1] + " : " + str(np.std(labelsAllEverything)))
            print("Deviation under the models:: " + model[1] + " : " + str(np.mean(standaardLabelAllDomainEverything)))
        
        if len(pearsonAllAbsolute) > 0:
            file.write("Ranking Absolute"+"\n")
            file.write("Gemiddelde : " + model[1] + " : " + str(np.mean(pearsonAllAbsolute))+"\n")
            file.write("Deviatie: " + model[1] + " : " + str(np.std(pearsonAllAbsolute))+"\n")
            file.write("Deviation under the models:: " + model[1] + " : " + str(np.mean(standaardDomainAbsolute))+"\n")
        if len(labelsDomainAbsolute) >0:
            file.write("Labels domain"+"\n")
            file.write("Gemiddelde : " + model[1] + " : " + str(np.mean(labelsDomainAbsolute))+"\n")
            file.write("Deviatie: " + model[1] + " : " + str(np.std(labelsDomainAbsolute))+"\n")
            file.write("Deviation under the models:: " + model[1] + " : " + str(np.mean(standaardLabelDomainAbsolute))+"\n")
            file.write("Labels all"+"\n")
        if  len(labelsAllAbsolute) > 0 :
            file.write("Gemiddelde : " + model[1] + " : " + str(np.mean(labelsAllAbsolute))+"\n")
            file.write("Deviatie: " + model[1] + " : " + str(np.std(labelsAllAbsolute))+"\n")
            file.write("Deviation under the models:: " + model[1] + " : " + str(np.mean(standaardLabelAllDomainAbsolute))+"\n")
        if len(pearsonAllEverything) > 0:
            file.write("Ranking Everything"+"\n")
            file.write("Gemiddelde : " + model[1] + " : " + str(np.mean(pearsonAllEverything))+"\n")
            file.write("Deviatie: " + model[1] + " : " + str(np.std(pearsonAllEverything))+"\n")
            file.write("Deviation under the models:: " + model[1] + " : " + str(np.mean(standaardDomainEverything))+"\n")
        if len(labelsDomainEverything) >0:
            file.write("Labels domain"+"\n")
            file.write("Gemiddelde : " + model[1] + " : " + str(np.mean(labelsDomainEverything))+"\n")
            file.write("Deviatie: " + model[1] + " : " + str(np.std(labelsDomainEverything))+"\n")
            file.write("Deviation under the models:: " + model[1] + " : " + str(np.mean(standaardLabelDomainEverything))+"\n")
            file.write("Labels all"+"\n")
        if  len(labelsAllEverything) > 0 :
            file.write("Gemiddelde : " + model[1] + " : " + str(np.mean(labelsAllEverything))+"\n")
            file.write("Deviatie: " + model[1] + " : " + str(np.std(labelsAllEverything))+"\n")
            file.write("Deviation under the models:: " + model[1] + " : " + str(np.mean(standaardLabelAllDomainEverything))+"\n")
        file.write(str(numberEqualAbsolute))
        file.close()
        pearsonAllAbsolute2.extend(pearsonAllAbsolute)
        standaardAbsolute2.extend(standaardDomainAbsolute)
        labelsDomainEAbsolute2.extend(labelsDomainAbsolute)
        standaardDomainLabelsAllAbsolute2.extend(standaardLabelDomainAbsolute)
        labelsAllEAbsolute2.extend(labelsAllAbsolute)
        standaardLabelsAllAbsolute2.extend(standaardLabelAllDomainAbsolute)
        pearsonAllEverything2.extend(pearsonAllEverything)
        standaardEverything2.extend(standaardDomainEverything)
        labelsDomainEEverything2.extend(labelsDomainEverything)
        standaardDomainLabelsAllEverything2.extend(standaardLabelDomainEverything)
        labelsAllEEverything2.extend(labelsAllEverything)
        standaardLabelsAllEverything2.extend(standaardLabelAllDomainEverything)
        numberEqualEverything,pearsonAllEverything, standaardDomainEverything, \
        labelsDomainEverything, standaardLabelDomainEverything, labelsAllEverything, standaardLabelAllDomainEverything \
            = pearsonRankingAbsolute([model],modelsAbsolute, modelsEverything)
        numberEqual3.update(numberEqualEverything)
        file = open('resultPearson/biLSTM/versusAbsolute/' + model[1], "w")
        
        print("Absolute with  everything")
        if len(pearsonAllEverything) > 0:
            print("Ranking Everything")
            print("Gemiddelde : " + model[1] + " : " + str(np.mean(pearsonAllEverything)))
            print("Deviatie: " + model[1] + " : " + str(np.std(pearsonAllEverything)))
            print("Deviation under the models:: " + model[1] + " : " + str(np.mean(standaardDomainEverything)))
        if len(labelsDomainEverything) > 0:
            print("Labels domain")
            print("Gemiddelde : " + model[1] + " : " + str(np.mean(labelsDomainEverything)))
            print("Deviatie: " + model[1] + " : " + str(np.std(labelsDomainEverything)))
            print("Deviation under the models:: " + model[1] + " : " + str(np.mean(standaardLabelDomainEverything)))
            print("Labels all")
        if len(labelsAllEverything) > 0:
            print("Gemiddelde : " + model[1] + " : " + str(np.mean(labelsAllEverything)))
            print("Deviatie: " + model[1] + " : " + str(np.std(labelsAllEverything)))
            print("Deviation under the models:: " + model[1] + " : " + str(np.mean(standaardLabelAllDomainEverything)))
        
        if len(pearsonAllEverything) > 0:
            file.write("Ranking Everything"+"\n")
            file.write("Gemiddelde : " + model[1] + " : " + str(np.mean(pearsonAllEverything))+"\n")
            file.write("Deviatie: " + model[1] + " : " + str(np.std(pearsonAllEverything))+"\n")
            file.write("Deviation under the models:: " + model[1] + " : " + str(np.mean(standaardDomainEverything))+"\n")
        if len(labelsDomainEverything) > 0:
            file.write("Labels domain"+"\n")
            file.write("Gemiddelde : " + model[1] + " : " + str(np.mean(labelsDomainEverything))+"\n")
            file.write("Deviatie: " + model[1] + " : " + str(np.std(labelsDomainEverything))+"\n")
            file.write("Deviation under the models:: " + model[1] + " : " + str(np.mean(standaardLabelDomainEverything))+"\n")
            file.write("Labels all"+"\n")
        if len(labelsAllEverything) > 0:
            file.write("Gemiddelde : " + model[1] + " : " + str(np.mean(labelsAllEverything))+"\n")
            file.write("Deviatie: " + model[1] + " : " + str(np.std(labelsAllEverything))+"\n")
            file.write("Deviation under the models:: " + model[1] + " : " + str(np.mean(standaardLabelAllDomainEverything))+"\n")
        file.write(str(numberEqualEverything))
        file.close()
        pearsonAllEverything3.extend(pearsonAllEverything)
        standaardEverything3.extend(standaardDomainEverything)
        labelsDomainEEverything3.extend(labelsDomainEverything)
        standaardDomainLabelsAllEverything3.extend(standaardLabelDomainEverything)
        labelsAllEEverything3.extend(labelsAllEverything)
        standaardLabelsAllEverything3.extend(standaardLabelAllDomainEverything)
    file = open("resultPearson/biLSTM/versusBase/AllDomains","w")
    
    print("All domains for baseModel versus time,absolute and everything")
    print("Ranking")
    print("Gemiddelde all : " + str(np.mean(pearsonAll)))
    print("Deviatie all : "  + str(np.std(pearsonAll)))
    print("Deviation all : " + str(np.mean(standaard)))
    print("stand mean deviation all : " + str(np.std(standaard)) )
    print("Labels all")
    print("Gemiddelde all : " + str(np.mean(labelsAllE)))
    print("Deviatie all : " + str(np.std(labelsAllE)))
    print("Deviation all : " + str(np.mean(standaardAll)))
    print("stand mean deviation all : " + str(np.std(standaardAll)))
    print("Labels domain")
    print("Gemiddelde all : " + str(np.mean(labelsDomainE)))
    print("Deviatie all : " + str(np.std(labelsDomainE)))
    print("Deviation all : " + str(np.mean(standaardDomainLabelsAll)))
    print("stand mean deviation all : " + str(np.std(standaardDomainLabelsAll)))
    print("Ranking absolute")
    print("Gemiddelde all : " + str(np.mean(pearsonAllAbsolute)))
    print("Deviatie all : " + str(np.std(pearsonAllAbsolute)))
    print("Deviation all : " + str(np.mean(standaardAbsolute)))
    print("stand mean deviation all : " + str(np.std(standaardAbsolute)))
    print("Labels all absolute")
    print("Gemiddelde all : " + str(np.mean(labelsAllEAbsolute)))
    print("Deviatie all : " + str(np.std(labelsAllEAbsolute)))
    print("Deviation all : " + str(np.mean(standaardLabelsAllAbsolute)))
    print("stand mean deviation all : " + str(np.std(standaardLabelsAllAbsolute)))
    print("Labels domain absolute")
    print("Gemiddelde all : " + str(np.mean(labelsDomainEAbsolute)))
    print("Deviatie all : " + str(np.std(labelsDomainEAbsolute)))
    print("Deviation all : " + str(np.mean(standaardDomainLabelsAllAbsolute)))
    print("stand mean deviation all : " + str(np.std(standaardDomainLabelsAllAbsolute)))
    print("Ranking everything")
    print("Gemiddelde all : " + str(np.mean(pearsonAllEverything)))
    print("Deviatie all : " + str(np.std(pearsonAllEverything)))
    print("Deviation all : " + str(np.mean(standaardEverything)))
    print("stand mean deviation all : " + str(np.std(standaardEverything)))
    print("Labels all everything")
    print("Gemiddelde all : " + str(np.mean(labelsAllEEverything)))
    print("Deviatie all : " + str(np.std(labelsAllEEverything)))
    print("Deviation all : " + str(np.mean(standaardLabelsAllEverything)))
    print("stand mean deviation all : " + str(np.std(standaardLabelsAllEverything)))
    print("Labels domain everything")
    print("Gemiddelde all : " + str(np.mean(labelsDomainEEverything)))
    print("Deviatie all : " + str(np.std(labelsDomainEEverything)))
    print("Deviation all : " + str(np.mean(standaardDomainLabelsAllEverything)))
    print("stand mean deviation all : " + str(np.std(standaardDomainLabelsAllEverything)))
    print(numberEqual1)
    
    file.write("All domains for baseModel versus time,absolute and everything"+"\n")
    file.write("Ranking"+"\n")
    file.write("Gemiddelde all : " + str(np.mean(pearsonAll))+"\n")
    file.write("Deviatie all : " + str(np.std(pearsonAll))+"\n")
    file.write("Deviation all : " + str(np.mean(standaard))+"\n")
    file.write("stand mean deviation all : " + str(np.std(standaard))+"\n")
    file.write("Labels all"+"\n")
    file.write("Gemiddelde all : " + str(np.mean(labelsAllE))+"\n")
    file.write("Deviatie all : " + str(np.std(labelsAllE))+"\n")
    file.write("Deviation all : " + str(np.mean(standaardAll))+"\n")
    file.write("stand mean deviation all : " + str(np.std(standaardAll))+"\n")
    file.write("Labels domain"+"\n")
    file.write("Gemiddelde all : " + str(np.mean(labelsDomainE))+"\n")
    file.write("Deviatie all : " + str(np.std(labelsDomainE))+"\n")
    file.write("Deviation all : " + str(np.mean(standaardDomainLabelsAll))+"\n")
    file.write("stand mean deviation all : " + str(np.std(standaardDomainLabelsAll))+"\n")
    file.write("Ranking absolute"+"\n")
    file.write("Gemiddelde all : " + str(np.mean(pearsonAllAbsoluteD))+"\n")
    file.write("Deviatie all : " + str(np.std(pearsonAllAbsoluteD))+"\n")
    file.write("Deviation all : " + str(np.mean(standaardAbsolute))+"\n")
    file.write("stand mean deviation all : " + str(np.std(standaardAbsolute))+"\n")
    file.write("Labels all absolute"+"\n")
    file.write("Gemiddelde all : " + str(np.mean(labelsAllEAbsolute))+"\n")
    file.write("Deviatie all : " + str(np.std(labelsAllEAbsolute))+"\n")
    file.write("Deviation all : " + str(np.mean(standaardLabelsAllAbsolute))+"\n")
    file.write("stand mean deviation all : " + str(np.std(standaardLabelsAllAbsolute))+"\n")
    file.write("Labels domain absolute"+"\n")
    file.write("Gemiddelde all : " + str(np.mean(labelsDomainEAbsolute))+"\n")
    file.write("Deviatie all : " + str(np.std(labelsDomainEAbsolute))+"\n")
    file.write("Deviation all : " + str(np.mean(standaardDomainLabelsAllAbsolute))+"\n")
    file.write("stand mean deviation all : " + str(np.std(standaardDomainLabelsAllAbsolute))+"\n")
    file.write("Ranking everything"+"\n")
    file.write("Gemiddelde all : " + str(np.mean(pearsonAllEverythingD))+"\n")
    file.write("Deviatie all : " + str(np.std(pearsonAllEverythingD))+"\n")
    file.write("Deviation all : " + str(np.mean(standaardEverything))+"\n")
    file.write("stand mean deviation all : " + str(np.std(standaardEverything))+"\n")
    file.write("Labels all everything"+"\n")
    file.write("Gemiddelde all : " + str(np.mean(labelsAllEEverything))+"\n")
    file.write("Deviatie all : " + str(np.std(labelsAllEEverything))+"\n")
    file.write("Deviation all : " + str(np.mean(standaardLabelsAllEverything))+"\n")
    file.write("stand mean deviation all : " + str(np.std(standaardLabelsAllEverything))+"\n")
    file.write("Labels domain everything"+"\n")
    file.write("Gemiddelde all : " + str(np.mean(labelsDomainEEverything))+"\n")
    file.write("Deviatie all : " + str(np.std(labelsDomainEEverything))+"\n")
    file.write("Deviation all : " + str(np.mean(standaardDomainLabelsAllEverything))+"\n")
    file.write("stand mean deviation all : " + str(np.std(standaardDomainLabelsAllEverything))+"\n")
    file.write(str(numberEqual1))
    file.close()

    file = open("resultPearson/biLSTM/versusTime/AllDomains", "w")
    
    print("All domains for time versus absolute and everything")
    print("Ranking absolute")
    print("Gemiddelde all : " + str(np.mean(pearsonAllAbsolute2)))
    print("Deviatie all : " + str(np.std(pearsonAllAbsolute2)))
    print("Deviation all : " + str(np.mean(standaardAbsolute2)))
    print("stand mean deviation all : " + str(np.std(standaardAbsolute2)))
    print("Labels all absolute")
    print("Gemiddelde all : " + str(np.mean(labelsAllEAbsolute2)))
    print("Deviatie all : " + str(np.std(labelsAllEAbsolute2)))
    print("Deviation all : " + str(np.mean(standaardLabelsAllAbsolute2)))
    print("stand mean deviation all : " + str(np.std(standaardLabelsAllAbsolute2)))
    print("Labels domain absolute")
    print("Gemiddelde all : " + str(np.mean(labelsDomainEAbsolute2)))
    print("Deviatie all : " + str(np.std(labelsDomainEAbsolute2)))
    print("Deviation all : " + str(np.mean(standaardDomainLabelsAllAbsolute2)))
    print("stand mean deviation all : " + str(np.std(standaardDomainLabelsAllAbsolute2)))
    print("Ranking everything")
    print("Gemiddelde all : " + str(np.mean(pearsonAllEverything2)))
    print("Deviatie all : " + str(np.std(pearsonAllEverything2)))
    print("Deviation all : " + str(np.mean(standaardEverything2)))
    print("stand mean deviation all : " + str(np.std(standaardEverything2)))
    print("Labels all everything")
    print("Gemiddelde all : " + str(np.mean(labelsAllEEverything2)))
    print("Deviatie all : " + str(np.std(labelsAllEEverything2)))
    print("Deviation all : " + str(np.mean(standaardLabelsAllEverything2)))
    print("stand mean deviation all : " + str(np.std(standaardLabelsAllEverything2)))
    print("Labels domain everything")
    print("Gemiddelde all : " + str(np.mean(labelsDomainEEverything2)))
    print("Deviatie all : " + str(np.std(labelsDomainEEverything2)))
    print("Deviation all : " + str(np.mean(standaardDomainLabelsAllEverything2)))
    print("stand mean deviation all : " + str(np.std(standaardDomainLabelsAllEverything2)))
    print(numberEqual2)
    
    file.write("All domains for time versus absolute and everything"+"\n")
    file.write("Ranking absolute"+"\n")
    file.write("Gemiddelde all : " + str(np.mean(pearsonAllAbsolute2))+"\n")
    file.write("Deviatie all : " + str(np.std(pearsonAllAbsolute2))+"\n")
    file.write("Deviation all : " + str(np.mean(standaardAbsolute2))+"\n")
    file.write("stand mean deviation all : " + str(np.std(standaardAbsolute2))+"\n")
    file.write("Labels all absolute"+"\n")
    file.write("Gemiddelde all : " + str(np.mean(labelsAllEAbsolute2))+"\n")
    file.write("Deviatie all : " + str(np.std(labelsAllEAbsolute2))+"\n")
    file.write("Deviation all : " + str(np.mean(standaardLabelsAllAbsolute2))+"\n")
    file.write("stand mean deviation all : " + str(np.std(standaardLabelsAllAbsolute2))+"\n")
    file.write("Labels domain absolute"+"\n")
    file.write("Gemiddelde all : " + str(np.mean(labelsDomainEAbsolute2))+"\n")
    file.write("Deviatie all : " + str(np.std(labelsDomainEAbsolute2))+"\n")
    file.write("Deviation all : " + str(np.mean(standaardDomainLabelsAllAbsolute2))+"\n")
    file.write("stand mean deviation all : " + str(np.std(standaardDomainLabelsAllAbsolute2))+"\n")
    file.write("Ranking everything"+"\n")
    file.write("Gemiddelde all : " + str(np.mean(pearsonAllEverything2))+"\n")
    file.write("Deviatie all : " + str(np.std(pearsonAllEverything2))+"\n")
    file.write("Deviation all : " + str(np.mean(standaardEverything2))+"\n")
    file.write("stand mean deviation all : " + str(np.std(standaardEverything2))+"\n")
    file.write("Labels all everything"+"\n")
    file.write("Gemiddelde all : " + str(np.mean(labelsAllEEverything2))+"\n")
    file.write("Deviatie all : " + str(np.std(labelsAllEEverything2))+"\n")
    file.write("Deviation all : " + str(np.mean(standaardLabelsAllEverything2))+"\n")
    file.write("stand mean deviation all : " + str(np.std(standaardLabelsAllEverything2))+"\n")
    file.write("Labels domain everything"+"\n")
    file.write("Gemiddelde all : " + str(np.mean(labelsDomainEEverything2))+"\n")
    file.write("Deviatie all : " + str(np.std(labelsDomainEEverything2))+"\n")
    file.write("Deviation all : " + str(np.mean(standaardDomainLabelsAllEverything2))+"\n")
    file.write("stand mean deviation all : " + str(np.std(standaardDomainLabelsAllEverything2))+"\n")
    file.write(str(numberEqual2))
    file.close()

    file = open("resultPearson/biLSTM/versusAbsolute/AllDomains", "w")
    file.write("All domains for absolute versus everything"+"\n")
    file.write("Ranking everything"+"\n")
    file.write("Gemiddelde all : " + str(np.mean(pearsonAllEverything3))+"\n")
    file.write("Deviatie all : " + str(np.std(pearsonAllEverything3))+"\n")
    file.write("Deviation all : " + str(np.mean(standaardEverything3))+"\n")
    file.write("stand mean deviation all : " + str(np.std(standaardEverything3))+"\n")
    file.write("Labels all everything"+"\n")
    file.write("Gemiddelde all : " + str(np.mean(labelsAllEEverything3))+"\n")
    file.write("Deviatie all : " + str(np.std(labelsAllEEverything3))+"\n")
    file.write("Deviation all : " + str(np.mean(standaardLabelsAllEverything3))+"\n")
    file.write("stand mean deviation all : " + str(np.std(standaardLabelsAllEverything3))+"\n")
    file.write("Labels domain everything"+"\n")
    file.write("Gemiddelde all : " + str(np.mean(labelsDomainEEverything3))+"\n")
    file.write("Deviatie all : " + str(np.std(labelsDomainEEverything3))+"\n")
    file.write("Deviation all : " + str(np.mean(standaardDomainLabelsAllEverything3))+"\n")
    file.write("stand mean deviation all : " + str(np.std(standaardDomainLabelsAllEverything3))+"\n")
    file.write(str(numberEqual3))
    '''
    file = open("labelsDomainAll", "wb")
    pickle.dump(labelsDomainAll, file)
    file.close()
    file = open("labelsAllAll", "wb")
    pickle.dump(labelsAllAll, file)
    file.close()
    file = open("domains", "wb")
    pickle.dump(list(domains), file)
    file.close()
    print("Time")
    SpearmanLabelsAllIntra = getIntraRankingLabelsAll(labelsAllAll)
    meansAllIntra,stdAllIntra = calculateMeanAndStdAll(SpearmanLabelsAllIntra)
    print("Intra")
    print(meansAllIntra)
    print(stdAllIntra)
    file = open("Time-Intra-means", "wb")
    pickle.dump(meansAllIntra, file)
    file.close()
    file = open("Time-Intra-std", "wb")
    pickle.dump(stdAllIntra, file)
    file.close()
    print("Inter")
    SpearmanLabelsAllInter = getInterRankingLabelsAll(labelsAllAll)
    meansAllInter, stdAllInter = calculateMeanAndStdAll(SpearmanLabelsAllInter)
    print(meansAllInter)
    print(stdAllInter)
    file = open("Time-Inter-means", "wb")
    pickle.dump(meansAllInter, file)
    file.close()
    file = open("Time-Inter-std", "wb")
    pickle.dump(stdAllInter, file)
    file.close()
    print("Domain intra all")
    SpearmanLabelsDomainAllIntra,SpearmanLabelsDomainIntra = getIntraRankingLabelsDomain(labelsDomainAll,domains)
    meansDomainAll, stdDomainAll = calculateMeanAndStdAll(SpearmanLabelsDomainAllIntra)
    print(meansDomainAll)
    print(stdDomainAll)
    file = open("Time-Intra-DomainAll-means", "wb")
    pickle.dump(meansDomainAll, file)
    file.close()
    file = open("Time-Intra-DomainAll-std", "wb")
    pickle.dump(stdDomainAll, file)
    file.close()
    print("Domain intra domain")
    meansDomain, stdDomain = calculateMeanAndStdDomain(SpearmanLabelsDomainIntra,domains)
    print(meansDomain)
    print(stdDomain)
    file = open("Time-Intra-Domain-means", "wb")
    pickle.dump(meansDomain, file)
    file.close()
    file = open("Time-Intra-Domain-std", "wb")
    pickle.dump(stdDomain, file)
    file.close()
    print("Domain inter all")
    SpearmanLabelsDomainAllInter, SpearmanLabelsDomainInter = getInterRankingLabelsDomain(labelsDomainAll, domains)
    meansDomainAllInter, stdDomainAllInter = calculateMeanAndStdAll(SpearmanLabelsDomainAllInter)
    print(meansDomainAllInter)
    print(stdDomainAllInter)
    file = open("Time-Inter-DomainAll-means", "wb")
    pickle.dump(meansDomainAllInter, file)
    file.close()
    file = open("Time-Inter-DomainAll-std", "wb")
    pickle.dump(stdDomainAllInter, file)
    file.close()
    print("Domain inter domain")
    meansDomain, stdDomain = calculateMeanAndStdDomain(SpearmanLabelsDomainInter, domains)
    print(meansDomain)
    print(stdDomain)
    file = open("Time-Inter-Domain-means", "wb")
    pickle.dump(meansDomain, file)
    file.close()
    file = open("Time-Inter-Domain-std", "wb")
    pickle.dump(stdDomain, file)
    file.close()