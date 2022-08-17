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
from transformers import AutoModel

import encoderClaim90
import encoderClaimAbsoluteTimeAdding25
import encoderClaimAbsoluteTimeEverything2040
import encoderMetadata
import evidence_ranker
import instanceEncoder
import labelMaskDomain
import verificationModel90A
import verificationModel90B
import verificationModel90C
import verificationModelBERT75A
import verificationModelBERT75B
import verificationModelFineTuningAbsoluteTimeConstantLRAdamAdding25
import verificationModelFineTuningAbsoluteTimeConstantLRAdamAdding25B
import verificationModelFineTuningAbsoluteTimeConstantLRAdamAdding25C
import verificationModelFineTuningAbsoluteTimeConstantLRAdamEverything2040A
import verificationModelFineTuningAbsoluteTimeConstantLRAdamEverything2040B
import verificationModelFineTuningAbsoluteTimeConstantLRAdamEverything2040C
from basisModel import OneHotEncoderBasis, labelEmbeddingLayerBasis, verificationModelBasis, encoderBasis, encoderMetadataB, \
    instanceEncoderBasis, evidence_rankerBasis, labelMaskDomainBasis, verificationModelC, verificationModelD
import torch
from torch.utils.data import DataLoader
from datasetIteratie2CombinerOld import dump_load, dump_write, NUS
from labelEmbeddingLayer import labelEmbeddingLayer


def spearmanRanking(loaders,models):
    labelsBins = [{},{}]
    labelsBinsDomain = [{},{}]
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
    spearmanLabelsAll = [{},{}]
    for model in range(0,len(spearmanLabelsAll)):
        for time in labelsAll[model]:
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
    spearmanLabelsDomainAll = [{},{}]
    spearmanLabelsDomain = [{},{}]
    for domain in domains:
        times = list(labelsDomain[domain][0])
        for model in range(0, len(labelsDomain[domain])):
            spearmanLabelsDomain[model][domain] = {}
            for x in range(0, len(times)):
                time1 = times[x]
                for y in range(x + 1, len(times)):
                    time2 = times[y]
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
    spearmanLabelsDomainAll = [{},{}]
    spearmanLabelsDomain = [{},{}]
    for domain in domains:
        for model in range(0, len(labelsDomain[domain])):
            spearmanLabelsDomain[model][domain] = {}
            for time in labelsDomain[domain][model]:
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
    spearmanLabelsAll = [{},{}]
    times = list(labelsAll[0])
    for model in range(0,len(spearmanLabelsAll)):
        for x in range(0,len(times)):
            time1  = times[x]
            for y in range(x+1,len(times)):
                time2 = times[y]
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
    means = [{},{}]
    stds = [{},{}]
    for model in range(0,len(means)):
        for time in spearmanLabelsAll[model]:
            means[model][time] = np.mean(spearmanLabelsAll[model][time])
            stds[model][time] = np.std(spearmanLabelsAll[model][time])
    meansTogether = {}
    stdsTogether = {}
    for time in spearmanLabelsAll[0]:
        meansTogether[time] = (np.mean([means[0][time],means[1][time]]),np.std([means[0][time],means[1][time]]))
        stdsTogether[time] = (np.mean([stds[0][time],stds[1][time]]),np.std([stds[0][time],stds[1][time]]))
    return meansTogether,stdsTogether

def calculateMeanAndStdDomain(spearmanLabelsAll,domains):
    means = [{},{}]
    stds = [{},{}]
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
            meansTogether[domain][time] = (np.mean([means[0][domain][time],means[1][domain][time]]),np.std([means[0][domain][time],means[1][domain][time]]))
            stdsTogether[domain][time] = (np.mean([stds[0][domain][time],stds[1][domain][time]]),np.std([stds[0][domain][time],stds[1][domain][time]]))
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

numpy.seterr(divide='ignore', invalid='ignore')
domainIndices,domainLabels,domainLabelIndices,domainWeights = getLabelIndicesDomain('timeModels/labels/labels.tsv','timeModels/labels/labelSequence','timeModels/labels/weights.tsv')
domains = domainIndices.keys()
models = []
for domain in domains:
        #dev_set = NUS(mode="Dev", path='timeModels/dev/dev-' + domain + '.tsv', pathToSave="timeModels/dev/time/dataset2/", domain=domain)
        #train_set = NUS(mode='Train', path='timeModels/train/train-' + domain + '.tsv', pathToSave="timeModels/train/time/dataset2/",
        #                domain=domain)
        test_set = NUS(mode='Test', path='timeModels/test/test-' + domain + '.tsv', pathToSave="timeModels/test/time/dataset2/",
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
        domain = model[1]
        print("Start loading model " + domain)
        oneHotEncoderBasis = OneHotEncoderBasis.oneHotEncoder('timeModels/Metadata_sequence/metadata')
        oneHotEncoderM = OneHotEncoderBasis.oneHotEncoder('timeModels/Metadata_sequence/metadata')
        labelEmbeddingLayerM = labelEmbeddingLayer(2308, domainIndices)
        encoderMetadataM = encoderMetadata.encoderMetadata(3, 3, oneHotEncoderBasis).to(device)
        instanceEncoderM = instanceEncoder.instanceEncoder().to(device)
        evidenceRankerM = evidence_ranker.evidenceRanker(2308, 100).to(device)
        labelMaskDomainM = labelMaskDomain.labelMaskDomain(2308, domainIndices, model[1],
                                                           len(domainIndices[model[1]])).to(device)
        transformer = AutoModel.from_pretrained('sentence-transformers/all-distilroberta-v1').to(device)
        verificationModelTime75A = verificationModelBERT75A.verifactionModel(transformer, encoderMetadataM,
                                                                             instanceEncoderM,
                                                                             evidenceRankerM,
                                                                             labelEmbeddingLayerM, labelMaskDomainM,
                                                                             domainIndices, domainWeights, domain).to(
            device)
        verificationModelTime75A.loading_NeuralNetwork()
        oneHotEncoderM = OneHotEncoderBasis.oneHotEncoder('timeModels/Metadata_sequence/metadata')
        labelEmbeddingLayerM = labelEmbeddingLayer(2308, domainIndices)
        encoderMetadataM = encoderMetadata.encoderMetadata(3, 3, oneHotEncoderBasis).to(device)
        instanceEncoderM = instanceEncoder.instanceEncoder().to(device)
        evidenceRankerM = evidence_ranker.evidenceRanker(2308, 100).to(device)
        labelMaskDomainM = labelMaskDomain.labelMaskDomain(2308, domainIndices, model[1],
                                                           len(domainIndices[model[1]])).to(device)
        transformer = AutoModel.from_pretrained('sentence-transformers/all-distilroberta-v1').to(device)
        verificationModelTime75B = verificationModelBERT75B.verifactionModel(transformer, encoderMetadataM,
                                                                             instanceEncoderM,
                                                                             evidenceRankerM,
                                                                             labelEmbeddingLayerM, labelMaskDomainM,
                                                                             domainIndices, domainWeights, domain).to(
            device)
        verificationModelTime75B.loading_NeuralNetwork()
        print("Done loading " + domain)
        timeModels = [verificationModelTime75A,verificationModelTime75B]
        labelsDomain,labelsAll = spearmanRanking([model],timeModels)
        labelsDomainAll[model[1]] = labelsDomain
        for j in range(0,2):
            for k, v in labelsAll[j].items():
                if k in labelsAllAll[j].keys():
                    labelsAllAll[j][k] += v
                else:
                    labelsAllAll[j][k] = v
    file = open("labelsDomainAllBERT", "wb")
    pickle.dump(labelsDomainAll, file)
    file.close()
    file = open("labelsAllAllBERT", "wb")
    pickle.dump(labelsAllAll, file)
    file.close()
    file = open("domainsBERT", "wb")
    pickle.dump(list(domains), file)
    file.close()
    print("Time")
    SpearmanLabelsAllIntra = getIntraRankingLabelsAll(labelsAllAll)
    meansAllIntra,stdAllIntra = calculateMeanAndStdAll(SpearmanLabelsAllIntra)
    print("Intra")
    print(meansAllIntra)
    print(stdAllIntra)
    file = open("Time-Intra-meansBERT", "wb")
    pickle.dump(meansAllIntra, file)
    file.close()
    file = open("Time-Intra-stdBERT", "wb")
    pickle.dump(stdAllIntra, file)
    file.close()
    print("Inter")
    SpearmanLabelsAllInter = getInterRankingLabelsAll(labelsAllAll)
    meansAllInter, stdAllInter = calculateMeanAndStdAll(SpearmanLabelsAllInter)
    print(meansAllInter)
    print(stdAllInter)
    file = open("Time-Inter-meansBERT", "wb")
    pickle.dump(meansAllInter, file)
    file.close()
    file = open("Time-Inter-stdBERT", "wb")
    pickle.dump(stdAllInter, file)
    file.close()
    print("Domain intra all")
    SpearmanLabelsDomainAllIntra,SpearmanLabelsDomainIntra = getIntraRankingLabelsDomain(labelsDomainAll,domains)
    meansDomainAll, stdDomainAll = calculateMeanAndStdAll(SpearmanLabelsDomainAllIntra)
    print(meansDomainAll)
    print(stdDomainAll)
    file = open("Time-Intra-DomainAll-meansBERT", "wb")
    pickle.dump(meansDomainAll, file)
    file.close()
    file = open("Time-Intra-DomainAll-stdBERT", "wb")
    pickle.dump(stdDomainAll, file)
    file.close()
    print("Domain intra domain")
    meansDomain, stdDomain = calculateMeanAndStdDomain(SpearmanLabelsDomainIntra,domains)
    print(meansDomain)
    print(stdDomain)
    file = open("Time-Intra-Domain-meansBERT", "wb")
    pickle.dump(meansDomain, file)
    file.close()
    file = open("Time-Intra-Domain-stdBERT", "wb")
    pickle.dump(stdDomain, file)
    file.close()
    print("Domain inter all")
    SpearmanLabelsDomainAllInter, SpearmanLabelsDomainInter = getInterRankingLabelsDomain(labelsDomainAll, domains)
    meansDomainAllInter, stdDomainAllInter = calculateMeanAndStdAll(SpearmanLabelsDomainAllInter)
    print(meansDomainAllInter)
    print(stdDomainAllInter)
    file = open("Time-Inter-DomainAll-meansBERT", "wb")
    pickle.dump(meansDomainAllInter, file)
    file.close()
    file = open("Time-Inter-DomainAll-stdBERT", "wb")
    pickle.dump(stdDomainAllInter, file)
    file.close()
    print("Domain inter domain")
    meansDomain, stdDomain = calculateMeanAndStdDomain(SpearmanLabelsDomainInter, domains)
    print(meansDomain)
    print(stdDomain)
    file = open("Time-Inter-Domain-meansBERT", "wb")
    pickle.dump(meansDomain, file)
    file.close()
    file = open("Time-Inter-Domain-stdBERT", "wb")
    pickle.dump(stdDomain, file)
    file.close()