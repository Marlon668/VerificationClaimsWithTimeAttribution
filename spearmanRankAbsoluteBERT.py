import math
import os
import sys
import random
import pickle
import scipy.stats as ss
import numpy
import numpy as np
import torch.nn.functional as F
from scipy import stats
from transformers import AutoModel

import encoderClaimAbsoluteTimeAdding25
import encoderMetadata
import evidence_ranker
import instanceEncoder
import labelMaskDomain
import verificationModel90A
import verificationModel90B
import verificationModel90C
import verificationModelAbsoluteTimeBERTAdamWAdding10A
import verificationModelAbsoluteTimeBERTAdamWAdding10B
import verificationModelFineTuningAbsoluteTimeConstantLRAdamAdding25
import verificationModelFineTuningAbsoluteTimeConstantLRAdamAdding25B
import verificationModelFineTuningAbsoluteTimeConstantLRAdamAdding25C
import verificationModelFineTuningAbsoluteTimeConstantLRAdamEverything2040A
import verificationModelFineTuningAbsoluteTimeConstantLRAdamEverything2040B
import verificationModelFineTuningAbsoluteTimeConstantLRAdamEverything2040C
from base import OneHotEncoderB, labelEmbeddingLayerB, verificationModelB, encoderClaimB, encoderMetadataB, \
    instanceEncoderB, evidence_rankerB, labelMaskDomainB, verificationModelC, verificationModelD
import torch
from torch.utils.data import DataLoader
from datasetIteratie2CombinerOld import dump_load, dump_write, NUS
from labelEmbeddingLayer import labelEmbeddingLayer


def spearmanRanking(loaders,models):
    labelsAbsoluteBins = [{},{}]
    labelsAbsoluteBinsDomain = [{},{}]
    labelsAbsoluteBinsIndices = [{}, {}]
    labelsAbsoluteBinsDomainIndices = [{}, {}]
    for loader in loaders:
        for data in loader[0]:
            for i in range(len(data[0])):
                for j in range(0,len(models)):
                    model = models[j]
                    metaDataClaim = model.metaDataEncoder.oneHotEncoder.encode(data[3][i], device)
                    metadata_encoding = model.metaDataEncoder(metaDataClaim.unsqueeze(0)).to(device)
                    labelsDomainAbsolute,labelsAllAbsolute,labelsDomainIndices,labelsAllIndices = model.getRankingLabelsPerBin(data[0][i],data[1][i], data[2][i],
                                                                                                     metadata_encoding, domain,
                                                                                                     data[5][i], data[6][i],
                                                                                                     data[7][i],
                          data[8][i], data[9][i], data[10][i], data[11][i], data[12][i], data[13][i],
                          data[14][i], data[15][i], data[16][i])
                    for k, v in labelsAllAbsolute.items():
                        if k in labelsAbsoluteBins[j].keys():
                            labelsAbsoluteBins[j][k] += v
                        else:
                            labelsAbsoluteBins[j][k] = v
                    for k, v in labelsAllIndices.items():
                        if k in labelsAbsoluteBinsIndices[j].keys():
                            labelsAbsoluteBinsIndices[j][k] += v
                        else:
                            labelsAbsoluteBinsIndices[j][k] = v
                    for k, v in labelsDomainAbsolute.items():
                        if k in labelsAbsoluteBinsDomain[j].keys():
                            labelsAbsoluteBinsDomain[j][k] += v
                        else:
                            labelsAbsoluteBinsDomain[j][k] = v
                    for k, v in labelsDomainIndices.items():
                        if k in labelsAbsoluteBinsDomainIndices[j].keys():
                            labelsAbsoluteBinsDomainIndices[j][k] += v
                        else:
                            labelsAbsoluteBinsDomainIndices[j][k] = v
    return labelsAbsoluteBins,labelsAbsoluteBinsDomain,labelsAbsoluteBinsIndices,labelsAbsoluteBinsDomainIndices

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
        meansTogether[time] = (np.mean([means[0][time],means[1][time],means[2][time]]),np.std([means[0][time],means[1][time],means[2][time]]))
        stdsTogether[time] = (np.mean([stds[0][time],stds[1][time],stds[2][time]]),np.std([stds[0][time],stds[1][time],stds[2][time]]))
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
            meansTogether[domain][time] = (np.mean([means[0][domain][time],means[1][domain][time],means[2][domain][time]]),np.std([means[0][domain][time],means[1][domain][time],means[2][domain][time]]))
            stdsTogether[domain][time] = (np.mean([stds[0][domain][time],stds[1][domain][time],stds[2][domain][time]]),np.std([stds[0][domain][time],stds[1][domain][time],stds[2][domain][time]]))
    return meansTogether,stdsTogether

def getIntraRankingLabelsAllAbsolute(labelsAll):
    spearmanLabelsAll = [{},{}]
    for model in range(0,len(spearmanLabelsAll)):
        for time in labelsAll[model]:
            if (len(labelsAll[model][time])) > 1:
                correlation, _ = stats.spearmanr(labelsAll[model][time], axis=1)
                if len(labelsAll[model][time]) == 2:
                    if time in spearmanLabelsAll[model]:
                        spearmanLabelsAll[model][time].append(correlation)
                    else:
                        spearmanLabelsAll[model][time] = [correlation]
                else:
                    for i in range(0, len(correlation)):
                        for j in range(i + 1, len(correlation[i])):
                            if time in spearmanLabelsAll[model]:
                                spearmanLabelsAll[model][time].append(correlation[i][j])
                            else:
                                spearmanLabelsAll[model][time] = [correlation[i][j]]
    return spearmanLabelsAll

def getInterRankingLabelsDomainAbsolute(labelsDomain,domains,indicesLabelsDomain):
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
                    correlation, _ = stats.spearmanr(labelsDomain[domain][model][time1],
                                                     labelsDomain[domain][model][time2], axis=1)
                    if (len(labelsDomain[domain][model][time1]) + len(labelsDomain[domain][model][time2])) == 2:
                        if indicesLabelsDomain[domain][model][time1][0] !=indicesLabelsDomain[domain][model][time2][0]:
                            if (time1, time2) in spearmanLabelsDomain[model][domain]:
                                spearmanLabelsDomain[model][domain][(time1, time2)].append(correlation)
                            else:
                                spearmanLabelsDomain[model][domain][(time1, time2)] = [correlation]
                            if (time1, time2) in spearmanLabelsDomainAll[model]:
                                spearmanLabelsDomainAll[model][(time1, time2)].append(
                                    correlation)
                            else:
                                spearmanLabelsDomainAll[model][(time1, time2)] = [
                                    correlation]
                    else:
                        for i in range(0, len(labelsDomain[domain][model][time1])):
                            for j in range(len(labelsDomain[domain][model][time1]), len(correlation[0])):
                                if indicesLabelsDomain[domain][model][time1][i] !=indicesLabelsDomain[domain][model][time2][j-len(labelsDomain[domain][model][time1])]:
                                    if (time1, time2) in spearmanLabelsDomain[model][domain]:
                                        spearmanLabelsDomain[model][domain][(time1, time2)].append(correlation[i][j])
                                    else:
                                        spearmanLabelsDomain[model][domain][(time1, time2)] = [correlation[i][j]]
                                    if (time1, time2) in spearmanLabelsDomainAll[model]:
                                        spearmanLabelsDomainAll[model][(time1, time2)].append(
                                            correlation[i][j])
                                    else:
                                        spearmanLabelsDomainAll[model][(time1, time2)] = [
                                            correlation[i][j]]
    return spearmanLabelsDomainAll,spearmanLabelsDomain

def getIntraRankingLabelsDomainAbsolute(labelsDomain,domains):
    spearmanLabelsDomainAll = [{},{}]
    spearmanLabelsDomain = [{},{}]
    for domain in domains:
        for model in range(0, len(labelsDomain[domain])):
            spearmanLabelsDomain[model][domain] = {}
            for time in labelsDomain[domain][model]:
                if (len(labelsDomain[domain][model][time])) > 1:
                    correlation, _ = stats.spearmanr(labelsDomain[domain][model][time], axis=1)
                    if (len(labelsDomain[domain][model][time])) == 2:
                        if time in spearmanLabelsDomain[model][domain]:
                            spearmanLabelsDomain[model][domain][time].append(correlation)
                        else:
                            spearmanLabelsDomain[model][domain][time] = [correlation]
                        if time in spearmanLabelsDomainAll[model]:
                            spearmanLabelsDomainAll[model][time].append(
                                correlation)
                        else:
                            spearmanLabelsDomainAll[model][time] = [
                                correlation]
                    else:
                        for i in range(0, len(correlation)):
                            for j in range(i + 1, len(correlation)):
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

def getInterRankingLabelsAllAbsolute(labelsAll,indicesLabelsAll):
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
                    if (time1, time2) in spearmanLabelsAll[model]:
                        spearmanLabelsAll[model][(time1, time2)].append(correlation)
                    else:
                        spearmanLabelsAll[model][(time1, time2)] = [correlation]
                else:
                    for i in range(0, len(labelsAll[model][time1])):
                        for j in range(len(labelsAll[model][time1]), len(correlation[0])):
                            if indicesLabelsAll[model][time1][i] != indicesLabelsAll[model][time2][j-len(labelsAll[model][time1])]:
                                if (time1, time2) in spearmanLabelsAll[model]:
                                    spearmanLabelsAll[model][(time1, time2)].append(correlation[i][j])
                                else:
                                    spearmanLabelsAll[model][(time1, time2)] = [correlation[i][j]]
    return spearmanLabelsAll

def calculateMeanAndStdAllAbsolute(spearmanLabelsAll):
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

def calculateMeanAndStdDomainAbsolute(spearmanLabelsAll,domains):
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
labelsDomainAllAbsolute = {}
labelsAllAllAbsolute = [{},{},{}]
labelsDomainAllIndicesAbsolute = {}
labelsAllAllIndicesAbsolute = [{},{},{}]

with torch.no_grad():
    for model in models:
        domain = model[1]
        oneHotEncoderM = OneHotEncoderB.oneHotEncoder('timeModels/Metadata_sequence/metadata')
        labelEmbeddingLayerM = labelEmbeddingLayer(2308, domainIndices)
        encoderMetadataM = encoderMetadata.encoderMetadata(3, 3, oneHotEncoderM).to(device)
        instanceEncoderM = instanceEncoder.instanceEncoder().to(device)
        evidenceRankerM = evidence_ranker.evidenceRanker(2308, 100).to(device)
        labelMaskDomainM = labelMaskDomain.labelMaskDomain(2308, domainIndices, model[1],
                                                           len(domainIndices[model[1]])).to(device)
        transformer = AutoModel.from_pretrained('sentence-transformers/all-distilroberta-v1').to(device)
        verificationModelTimeAbsolute10A = verificationModelAbsoluteTimeBERTAdamWAdding10A.verifactionModel(transformer,
                                                                                                            encoderMetadataM,
                                                                                                            instanceEncoderM,
                                                                                                            evidenceRankerM,
                                                                                                            labelEmbeddingLayerM,
                                                                                                            labelMaskDomainM,
                                                                                                            domainIndices,
                                                                                                            domainWeights,
                                                                                                            domain).to(
            device)
        verificationModelTimeAbsolute10A.loading_NeuralNetwork()
        oneHotEncoderM = OneHotEncoderB.oneHotEncoder('timeModels/Metadata_sequence/metadata')
        labelEmbeddingLayerM = labelEmbeddingLayer(2308, domainIndices)
        encoderMetadataM = encoderMetadata.encoderMetadata(3, 3, oneHotEncoderM).to(device)
        instanceEncoderM = instanceEncoder.instanceEncoder().to(device)
        evidenceRankerM = evidence_ranker.evidenceRanker(2308, 100).to(device)
        labelMaskDomainM = labelMaskDomain.labelMaskDomain(2308, domainIndices, model[1],
                                                           len(domainIndices[model[1]])).to(device)
        transformer = AutoModel.from_pretrained('sentence-transformers/all-distilroberta-v1').to(device)
        verificationModelTimeAbsolute10B = verificationModelAbsoluteTimeBERTAdamWAdding10B.verifactionModel(transformer,
                                                                                                            encoderMetadataM,
                                                                                                            instanceEncoderM,
                                                                                                            evidenceRankerM,
                                                                                                            labelEmbeddingLayerM,
                                                                                                            labelMaskDomainM,
                                                                                                            domainIndices,
                                                                                                            domainWeights,
                                                                                                            domain).to(
            device)
        verificationModelTimeAbsolute10B.loading_NeuralNetwork()

        timeModels = [verificationModelTimeAbsolute10A,verificationModelTimeAbsolute10B]
        labelsAbsoluteBins,labelsAbsoluteBinsDomain,labelsAbsoluteBinsIndices,labelsAbsoluteBinsDomainIndices = spearmanRanking([model],timeModels)
        labelsDomainAllAbsolute[model[1]] = labelsAbsoluteBinsDomain
        labelsDomainAllIndicesAbsolute[model[1]] = labelsAbsoluteBinsDomainIndices
        for j in range(0,2):
            for k, v in labelsAbsoluteBins[j].items():
                if k in labelsAllAllAbsolute[j].keys():
                    labelsAllAllAbsolute[j][k] += v
                else:
                    labelsAllAllAbsolute[j][k] = v
        for j in range(0,2):
            for k, v in labelsAbsoluteBinsIndices[j].items():
                if k in labelsAllAllIndicesAbsolute[j].keys():
                    labelsAllAllIndicesAbsolute[j][k] += v
                else:
                    labelsAllAllIndicesAbsolute[j][k] = v

    file = open("labelsDomainAllAbsoluteBERT", "wb")
    pickle.dump(labelsDomainAllAbsolute, file)
    file.close()
    file=open("labelsAllAllAbsoluteBERT", "wb")
    pickle.dump(labelsAllAllAbsolute, file)
    file.close()
    file = open("labelsDomainAllIndicesAbsoluteBERT", "wb")
    pickle.dump(labelsDomainAllIndicesAbsolute, file)
    file.close()
    file=open("labelsAllAllIndicesAbsoluteBERT", "wb")
    pickle.dump(labelsAllAllIndicesAbsolute, file)
    file.close()
    file = open("domainsBERT", "wb")
    pickle.dump(list(domains), file)
    file.close()
    print("Absolute version")
    SpearmanLabelsAllIntra = getIntraRankingLabelsAllAbsolute(labelsAllAllAbsolute)

    meansAllIntra,stdAllIntra = calculateMeanAndStdAllAbsolute(SpearmanLabelsAllIntra)
    print("Intra")
    print(meansAllIntra)
    print(stdAllIntra)
    file = open("Time-Intra-meansABERT", "wb")
    pickle.dump(meansAllIntra, file)
    file.close()
    file = open("Time-Intra-stdABERT", "wb")
    pickle.dump(stdAllIntra, file)
    file.close()
    print("Inter")
    SpearmanLabelsAllInter = getInterRankingLabelsAllAbsolute(labelsAllAllAbsolute,labelsAllAllIndicesAbsolute)
    meansAllInter, stdAllInter = calculateMeanAndStdAllAbsolute(SpearmanLabelsAllInter)
    print(meansAllInter)
    print(stdAllInter)
    file = open("Time-Inter-meansABERT", "wb")
    pickle.dump(meansAllInter, file)
    file.close()
    file = open("Time-Inter-stdABERT", "wb")
    pickle.dump(stdAllInter, file)
    file.close()
    print("Domain intra all")
    SpearmanLabelsDomainAllIntra,SpearmanLabelsDomainIntra = getIntraRankingLabelsDomainAbsolute(labelsDomainAllAbsolute,domains)
    meansDomainAll, stdDomainAll = calculateMeanAndStdAllAbsolute(SpearmanLabelsDomainAllIntra)
    print(meansDomainAll)
    print(stdDomainAll)
    file = open("Time-Intra-DomainAll-meansABERT", "wb")
    pickle.dump(meansDomainAll, file)
    file.close()
    file = open("Time-Intra-DomainAll-stdABERT", "wb")
    pickle.dump(stdDomainAll, file)
    file.close()
    print("Domain intra domain")
    meansDomain, stdDomain = calculateMeanAndStdDomainAbsolute(SpearmanLabelsDomainIntra,domains)
    print(meansDomain)
    print(stdDomain)
    file = open("Time-Intra-Domain-meansABERT", "wb")
    pickle.dump(meansDomain, file)
    file.close()
    file = open("Time-Intra-Domain-stdABERT", "wb")
    pickle.dump(stdDomain, file)
    file.close()
    print("Domain inter all")
    SpearmanLabelsDomainAllInter, SpearmanLabelsDomainInter = getInterRankingLabelsDomainAbsolute(labelsDomainAllAbsolute, domains,labelsDomainAllIndicesAbsolute)
    meansDomainAllInter, stdDomainAllInter = calculateMeanAndStdAllAbsolute(SpearmanLabelsDomainAllInter)
    print(meansDomainAllInter)
    print(stdDomainAllInter)
    file = open("Time-Inter-DomainAll-meansABERT", "wb")
    pickle.dump(meansDomainAllInter, file)
    file.close()
    file = open("Time-Inter-DomainAll-stdABERT", "wb")
    pickle.dump(stdDomainAllInter, file)
    file.close()
    print("Domain inter domain")
    meansDomain, stdDomain = calculateMeanAndStdDomainAbsolute(SpearmanLabelsDomainInter, domains)
    print(meansDomain)
    print(stdDomain)
    file = open("Time-Inter-Domain-meansABERT", "wb")
    pickle.dump(meansDomain, file)
    file.close()
    file = open("Time-Inter-Domain-stdABERT", "wb")
    pickle.dump(stdDomain, file)
    file.close()