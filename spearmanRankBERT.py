import math
import os
import sys
import random
import scipy.stats as ss
import numpy
import numpy as np
import torch.nn.functional as F
from scipy import stats
import encoderClaim90
import encoderClaimAbsoluteTimeAdding25
import encoderClaimAbsoluteTimeEverything2040
import encoderMetadata
import evidence_ranker
import instanceEncoder
import labelMaskDomain
import verificationModelBERT75A
import verificationModelBERT75B
import verificationModelAbsoluteTimeBERTAdamWAdding10A
import verificationModelAbsoluteTimeBERTAdamWAdding10B
import verificationModelAbsoluteTimeBERTAdamWAddingEverything2020A
import verificationModelAbsoluteTimeBERTAdamWAddingEverything2020B
from base import OneHotEncoderBasis, labelEmbeddingLayerBasis, verificationModelBasis, encoderBasis, encoderMetadataB, \
    instanceEncoderBasis, evidence_rankerBasis, labelMaskDomainBasis, verificationModelBERT, verificationModelBERT2
import torch
from torch.utils.data import DataLoader
from datasetIteratie2CombinerOld import dump_load, dump_write, NUS
from labelEmbeddingLayer import labelEmbeddingLayer
from transformers import BertTokenizer, AutoModel, AutoTokenizer


def spearmanRanking(loaders,basisModels,models,absoluteModels,modelsEverything):
    pearonsEvidenceRanking = {}
    numberEqual = {}
    spearmanAll = []
    labelsDomain = []
    standaardDomain = []
    standaardLabelDomain = []
    labelsAll = []
    standaardLabelAllDomain = []
    spearmanAllAbsolute = []
    labelsDomainAbsolute = []
    standaardDomainAbsolute = []
    standaardLabelDomainAbsolute = []
    labelsAllAbsolute = []
    standaardLabelAllDomainAbsolute = []
    spearmanAllEverything = []
    labelsDomainEverything = []
    standaardDomainEverything = []
    standaardLabelDomainEverything = []
    labelsAllEverything = []
    standaardLabelAllDomainEverything = []
    for loader in loaders:
        pearonsEvidenceRanking[loader[1]] = []
        numberEqual[loader[1]] = {}
        for data in loader[0]:
            for i in range(len(data[0])):
                spearmanRankEntry = []
                labelsEntry = []
                labelsAllEntry = []
                metaDataClaim = oneHotEncoderBasis.encode(data[3][i], device)
                metadata_encoding = basisModels[0].metaDataEncoder(metaDataClaim.unsqueeze(0)).to(device)
                domain = data[0][0].split('-')[0]
                rankingB1, labelsAllB1, labelsDomainB1,allEqualB1 = basisModels[0].getRankingEvidencesLabels(data[1][i], data[2][i],
                                                                                        metadata_encoding, domain)
                if allEqualB1:
                    if "Base" in numberEqual[domain]:
                        numberEqual[domain]["Base"] += 1
                    else:
                        numberEqual[domain]["Base"] = 1
                rankB1 = [sorted(rankingB1,reverse=True).index(x)+1 for x in rankingB1]
                labelB1 = [sorted(labelsDomainB1,reverse=True).index(x)+1 for x in labelsDomainB1]
                labelsAllB1 = [sorted(labelsAllB1,reverse=True).index(x)+1 for x in labelsAllB1]
                metaDataClaim = oneHotEncoderBasis.encode(data[3][i], device)
                metadata_encoding = basisModels[1].metaDataEncoder(metaDataClaim.unsqueeze(0)).to(device)
                rankingB2, labelsAllB2, labelsDomainB2, allEqualB2 = basisModels[1].getRankingEvidencesLabels(
                    data[1][i], data[2][i],
                    metadata_encoding, domain)
                if allEqualB2:
                    if "Base2" in numberEqual[domain]:
                        numberEqual[domain]["Base2"] += 1
                    else:
                        numberEqual[domain]["Base2"] = 1
                rankB2 = [sorted(rankingB2, reverse=True).index(x) + 1 for x in rankingB2]
                labelB2 = [sorted(labelsDomainB2, reverse=True).index(x) + 1 for x in labelsDomainB2]
                labelsAllB2 = [sorted(labelsAllB2, reverse=True).index(x) + 1 for x in labelsAllB2]
                index = 1
                for model in models:
                    metaDataClaim = model.metaDataEncoder.oneHotEncoder.encode(data[3][i], device)
                    metadata_encoding = model.metaDataEncoder(metaDataClaim.unsqueeze(0)).to(device)
                    rankingM, labelsAllM, labelsDomainM,allEqualM = model.getRankingEvidencesLabels(data[1][i], data[2][i],
                                                                                               metadata_encoding,
                                                                                               domain,data[5][i],data[6][i])
                    if allEqualM:
                        if "time"+str(index) in numberEqual[domain]:
                            numberEqual[domain]["time"+str(index)] += 1
                        else:
                            numberEqual[domain]["time"+str(index)] = 1

                    rankM = [sorted(rankingM,reverse=True).index(x)+1 for x in rankingM]
                    corref = stats.spearmanr(rankB1,rankM)[0]
                    if not math.isnan(corref):
                        spearmanRankEntry.append(corref)
                    corref = stats.spearmanr(rankB2, rankM)[0]
                    if not math.isnan(corref):
                        spearmanRankEntry.append(corref)
                    labelM = [sorted(labelsDomainM, reverse=True).index(x) + 1 for x in labelsDomainM]
                    corref = stats.spearmanr(labelB1, labelM)[0]
                    if not math.isnan(corref):
                        labelsEntry.append(corref)
                    corref = stats.spearmanr(labelB2, labelM)[0]
                    if not math.isnan(corref):
                        labelsEntry.append(corref)
                    labelsAllM = [sorted(labelsAllM, reverse=True).index(x) + 1 for x in labelsAllM]
                    corref = stats.spearmanr(labelsAllB1, labelsAllM)[0]
                    if not math.isnan(corref):
                        labelsAllEntry.append(corref)
                    corref = stats.spearmanr(labelsAllB2, labelsAllM)[0]
                    if not math.isnan(corref):
                        labelsAllEntry.append(corref)
                    index += 1

                if len(spearmanRankEntry)>0:
                    standaardDomain.append(numpy.std(spearmanRankEntry))
                if len(labelsEntry) > 0:
                    standaardLabelDomain.append(numpy.std(labelsEntry))
                if len(labelsAllEntry) > 0:
                    standaardLabelAllDomain.append(numpy.std(labelsAllEntry))
                pearonsEvidenceRanking[loader[1]].extend(spearmanRankEntry)
                spearmanAll.extend(spearmanRankEntry)
                labelsDomain.extend(labelsEntry)
                labelsAll.extend(labelsAllEntry)
                index = 1
                spearmanRankEntry = []
                labelsEntry = []
                labelsAllEntry = []
                for model in absoluteModels:
                    metaDataClaim = model.metaDataEncoder.oneHotEncoder.encode(data[3][i], device)
                    metadata_encoding = model.metaDataEncoder(metaDataClaim.unsqueeze(0)).to(device)
                    rankingM, labelsAllM, labelsDomainM,allEqualM = model.getRankingEvidencesLabels(data[1][i], data[2][i],
                                                                                                     metadata_encoding, domain,
                                                                                                     data[5][i], data[6][i],
                                                                                                     data[7][i],
                          data[8][i], data[9][i], data[10][i], data[11][i], data[12][i], data[13][i],
                          data[14][i], data[15][i], data[16][i])
                    if allEqualM:
                        if "absolute"+str(index) in numberEqual[domain]:
                            numberEqual[domain]["absolute"+str(index)] += 1
                        else:
                            numberEqual[domain]["absolute"+str(index)] = 1

                    rankM = [sorted(rankingM,reverse=True).index(x)+1 for x in rankingM]
                    corref = stats.spearmanr(rankB1,rankM)[0]
                    if not math.isnan(corref):
                        spearmanRankEntry.append(corref)
                    corref = stats.spearmanr(rankB2, rankM)[0]
                    if not math.isnan(corref):
                        spearmanRankEntry.append(corref)
                    labelM = [sorted(labelsDomainM, reverse=True).index(x) + 1 for x in labelsDomainM]
                    corref = stats.spearmanr(labelB1, labelM)[0]
                    if not math.isnan(corref):
                        labelsEntry.append(corref)
                    corref = stats.spearmanr(labelB2, labelM)[0]
                    if not math.isnan(corref):
                        labelsEntry.append(corref)
                    labelsAllM = [sorted(labelsAllM, reverse=True).index(x) + 1 for x in labelsAllM]
                    corref = stats.spearmanr(labelsAllB1, labelsAllM)[0]
                    if not math.isnan(corref):
                        labelsAllEntry.append(corref)
                    corref = stats.spearmanr(labelsAllB2, labelsAllM)[0]
                    if not math.isnan(corref):
                        labelsAllEntry.append(corref)
                    index += 1

                if len(spearmanRankEntry)>0:
                    standaardDomainAbsolute.append(numpy.std(spearmanRankEntry))
                if len(labelsEntry) > 0:
                    standaardLabelDomainAbsolute.append(numpy.std(labelsEntry))
                if len(labelsAllEntry) > 0:
                    standaardLabelAllDomainAbsolute.append(numpy.std(labelsAllEntry))
                spearmanAllAbsolute.extend(spearmanRankEntry)
                labelsDomainAbsolute.extend(labelsEntry)
                labelsAllAbsolute.extend(labelsAllEntry)
                spearmanRankEntry = []
                labelsEntry = []
                labelsAllEntry = []
                for model in modelsEverything:
                    metaDataClaim = model.metaDataEncoder.oneHotEncoder.encode(data[3][i], device)
                    metadata_encoding = model.metaDataEncoder(metaDataClaim.unsqueeze(0)).to(device)
                    rankingM, labelsAllM, labelsDomainM, allEqualM = model.getRankingEvidencesLabels(data[1][i], data[2][i],
                                                                                                     metadata_encoding, domain,
                                                                                                     data[5][i], data[6][i],
                                                                                                     data[7][i],
                          data[8][i], data[9][i], data[10][i], data[11][i], data[12][i], data[13][i],
                          data[14][i], data[15][i], data[16][i])
                    if allEqualM:
                        if "everything" + str(index) in numberEqual[domain]:
                            numberEqual[domain]["everything" + str(index)] += 1
                        else:
                            numberEqual[domain]["everything" + str(index)] = 1

                    rankM = [sorted(rankingM, reverse=True).index(x) + 1 for x in rankingM]
                    corref = stats.spearmanr(rankB1, rankM)[0]
                    if not math.isnan(corref):
                        spearmanRankEntry.append(corref)
                    corref = stats.spearmanr(rankB2, rankM)[0]
                    if not math.isnan(corref):
                        spearmanRankEntry.append(corref)
                    labelM = [sorted(labelsDomainM, reverse=True).index(x) + 1 for x in labelsDomainM]
                    corref = stats.spearmanr(labelB1, labelM)[0]
                    if not math.isnan(corref):
                        labelsEntry.append(corref)
                    corref = stats.spearmanr(labelB2, labelM)[0]
                    if not math.isnan(corref):
                        labelsEntry.append(corref)
                    labelsAllM = [sorted(labelsAllM, reverse=True).index(x) + 1 for x in labelsAllM]
                    corref = stats.spearmanr(labelsAllB1, labelsAllM)[0]
                    if not math.isnan(corref):
                        labelsAllEntry.append(corref)
                    corref = stats.spearmanr(labelsAllB2, labelsAllM)[0]
                    if not math.isnan(corref):
                        labelsAllEntry.append(corref)
                    index += 1

                if len(spearmanRankEntry) > 0:
                    standaardDomainEverything.append(numpy.std(spearmanRankEntry))
                if len(labelsEntry) > 0:
                    standaardLabelDomainEverything.append(numpy.std(labelsEntry))
                if len(labelsAllEntry) > 0:
                    standaardLabelAllDomainEverything.append(numpy.std(labelsAllEntry))
                spearmanAllEverything.extend(spearmanRankEntry)
                labelsDomainEverything.extend(labelsEntry)
                labelsAllEverything.extend(labelsAllEntry)
    return spearmanAll,standaardDomain,labelsAll,standaardLabelAllDomain,labelsDomain,standaardLabelDomain,numberEqual,\
           spearmanAllAbsolute, standaardDomainAbsolute,labelsDomainAbsolute,standaardLabelDomainAbsolute,\
           labelsAllAbsolute,standaardLabelAllDomainAbsolute,spearmanAllEverything,standaardDomainEverything,\
           labelsDomainEverything,standaardLabelDomainEverything,labelsAllEverything,standaardLabelAllDomainEverything

def spearmanRankingTime(loaders,basisModels,absoluteModels,modelsEverything):
    pearonsEvidenceRanking = {}
    numberEqual = {}
    spearmanAllAbsolute = []
    labelsDomainAbsolute = []
    standaardDomainAbsolute = []
    standaardLabelDomainAbsolute = []
    labelsAllAbsolute = []
    standaardLabelAllDomainAbsolute = []
    spearmanAllEverything = []
    labelsDomainEverything = []
    standaardDomainEverything = []
    standaardLabelDomainEverything = []
    labelsAllEverything = []
    standaardLabelAllDomainEverything = []
    for loader in loaders:
        pearonsEvidenceRanking[loader[1]] = []
        numberEqual[loader[1]] = {}
        for data in loader[0]:
            for i in range(len(data[0])):
                spearmanRankEntry = []
                labelsEntry = []
                labelsAllEntry = []
                metaDataClaim = oneHotEncoderBasis.encode(data[3][i], device)
                metadata_encoding = basisModels[0].metaDataEncoder(metaDataClaim.unsqueeze(0)).to(device)
                domain = data[0][0].split('-')[0]
                rankingB1, labelsAllB1, labelsDomainB1,allEqualB1 = basisModels[0].getRankingEvidencesLabels(data[1][i], data[2][i],
                                                                                               metadata_encoding,
                                                                                               domain,data[5][i],data[6][i])
                if allEqualB1:
                    if "Time" in numberEqual[domain]:
                        numberEqual[domain]["Time"] += 1
                    else:
                        numberEqual[domain]["Time"] = 1
                rankB1 = [sorted(rankingB1,reverse=True).index(x)+1 for x in rankingB1]
                labelB1 = [sorted(labelsDomainB1,reverse=True).index(x)+1 for x in labelsDomainB1]
                labelsAllB1 = [sorted(labelsAllB1,reverse=True).index(x)+1 for x in labelsAllB1]
                metaDataClaim = basisModels[1].metaDataEncoder.oneHotEncoder.encode(data[3][i], device)
                metadata_encoding = basisModels[1].metaDataEncoder(metaDataClaim.unsqueeze(0)).to(device)
                rankingB2, labelsAllB2, labelsDomainB2, allEqualB2 = basisModels[1].getRankingEvidencesLabels(
                    data[1][i], data[2][i],
                    metadata_encoding,
                    domain, data[5][i], data[6][i])
                if allEqualB2:
                    if "Time2" in numberEqual[domain]:
                        numberEqual[domain]["Time2"] += 1
                    else:
                        numberEqual[domain]["Time2"] = 1
                rankB2 = [sorted(rankingB2, reverse=True).index(x) + 1 for x in rankingB2]
                labelB2 = [sorted(labelsDomainB2, reverse=True).index(x) + 1 for x in labelsDomainB2]
                labelsAllB2 = [sorted(labelsAllB2, reverse=True).index(x) + 1 for x in labelsAllB2]
                index = 1

                spearmanRankEntry = []
                labelsEntry = []
                labelsAllEntry = []
                for model in absoluteModels:
                    metaDataClaim = model.metaDataEncoder.oneHotEncoder.encode(data[3][i], device)
                    metadata_encoding = model.metaDataEncoder(metaDataClaim.unsqueeze(0)).to(device)
                    rankingM, labelsAllM, labelsDomainM,allEqualM = model.getRankingEvidencesLabels(data[1][i], data[2][i],
                                                                                                     metadata_encoding, domain,
                                                                                                     data[5][i], data[6][i],
                                                                                                     data[7][i],
                          data[8][i], data[9][i], data[10][i], data[11][i], data[12][i], data[13][i],
                          data[14][i], data[15][i], data[16][i])
                    if allEqualM:
                        if "absolute"+str(index) in numberEqual[domain]:
                            numberEqual[domain]["absolute"+str(index)] += 1
                        else:
                            numberEqual[domain]["absolute"+str(index)] = 1

                    rankM = [sorted(rankingM,reverse=True).index(x)+1 for x in rankingM]
                    corref = stats.spearmanr(rankB1,rankM)[0]
                    if not math.isnan(corref):
                        spearmanRankEntry.append(corref)
                    corref = stats.spearmanr(rankB2, rankM)[0]
                    if not math.isnan(corref):
                        spearmanRankEntry.append(corref)
                    labelM = [sorted(labelsDomainM, reverse=True).index(x) + 1 for x in labelsDomainM]
                    corref = stats.spearmanr(labelB1, labelM)[0]
                    if not math.isnan(corref):
                        labelsEntry.append(corref)
                    corref = stats.spearmanr(labelB2, labelM)[0]
                    if not math.isnan(corref):
                        labelsEntry.append(corref)
                    labelsAllM = [sorted(labelsAllM, reverse=True).index(x) + 1 for x in labelsAllM]
                    corref = stats.spearmanr(labelsAllB1, labelsAllM)[0]
                    if not math.isnan(corref):
                        labelsAllEntry.append(corref)
                    corref = stats.spearmanr(labelsAllB2, labelsAllM)[0]
                    if not math.isnan(corref):
                        labelsAllEntry.append(corref)
                    index += 1

                if len(spearmanRankEntry)>0:
                    standaardDomainAbsolute.append(numpy.std(spearmanRankEntry))
                if len(labelsEntry) > 0:
                    standaardLabelDomainAbsolute.append(numpy.std(labelsEntry))
                if len(labelsAllEntry) > 0:
                    standaardLabelAllDomainAbsolute.append(numpy.std(labelsAllEntry))
                spearmanAllAbsolute.extend(spearmanRankEntry)
                labelsDomainAbsolute.extend(labelsEntry)
                labelsAllAbsolute.extend(labelsAllEntry)
                spearmanRankEntry = []
                labelsEntry = []
                labelsAllEntry = []
                for model in modelsEverything:
                    metaDataClaim = model.metaDataEncoder.oneHotEncoder.encode(data[3][i], device)
                    metadata_encoding = model.metaDataEncoder(metaDataClaim.unsqueeze(0)).to(device)
                    metaDataClaim = model.metaDataEncoder.oneHotEncoder.encode(data[3][i], device)
                    metadata_encoding = model.metaDataEncoder(metaDataClaim.unsqueeze(0)).to(device)
                    rankingM, labelsAllM, labelsDomainM, allEqualM = model.getRankingEvidencesLabels(data[1][i], data[2][i],
                                                                                                     metadata_encoding, domain,
                                                                                                     data[5][i], data[6][i],
                                                                                                     data[7][i],
                          data[8][i], data[9][i], data[10][i], data[11][i], data[12][i], data[13][i],
                          data[14][i], data[15][i], data[16][i])
                    if allEqualM:
                        if "everything" + str(index) in numberEqual[domain]:
                            numberEqual[domain]["everything" + str(index)] += 1
                        else:
                            numberEqual[domain]["everything" + str(index)] = 1

                    rankM = [sorted(rankingM, reverse=True).index(x) + 1 for x in rankingM]
                    corref = stats.spearmanr(rankB1, rankM)[0]
                    if not math.isnan(corref):
                        spearmanRankEntry.append(corref)
                    corref = stats.spearmanr(rankB2, rankM)[0]
                    if not math.isnan(corref):
                        spearmanRankEntry.append(corref)
                    labelM = [sorted(labelsDomainM, reverse=True).index(x) + 1 for x in labelsDomainM]
                    corref = stats.spearmanr(labelB1, labelM)[0]
                    if not math.isnan(corref):
                        labelsEntry.append(corref)
                    corref = stats.spearmanr(labelB2, labelM)[0]
                    if not math.isnan(corref):
                        labelsEntry.append(corref)
                    labelsAllM = [sorted(labelsAllM, reverse=True).index(x) + 1 for x in labelsAllM]
                    corref = stats.spearmanr(labelsAllB1, labelsAllM)[0]
                    if not math.isnan(corref):
                        labelsAllEntry.append(corref)
                    corref = stats.spearmanr(labelsAllB2, labelsAllM)[0]
                    if not math.isnan(corref):
                        labelsAllEntry.append(corref)
                    index += 1

                if len(spearmanRankEntry) > 0:
                    standaardDomainEverything.append(numpy.std(spearmanRankEntry))
                if len(labelsEntry) > 0:
                    standaardLabelDomainEverything.append(numpy.std(labelsEntry))
                if len(labelsAllEntry) > 0:
                    standaardLabelAllDomainEverything.append(numpy.std(labelsAllEntry))
                spearmanAllEverything.extend(spearmanRankEntry)
                labelsDomainEverything.extend(labelsEntry)
                labelsAllEverything.extend(labelsAllEntry)
    return numberEqual,\
           spearmanAllAbsolute, standaardDomainAbsolute,labelsDomainAbsolute,standaardLabelDomainAbsolute,\
           labelsAllAbsolute,standaardLabelAllDomainAbsolute,spearmanAllEverything,standaardDomainEverything,\
           labelsDomainEverything,standaardLabelDomainEverything,labelsAllEverything,standaardLabelAllDomainEverything


def spearmanRankingAbsolute(loaders, basisModels, modelsEverything):
    pearonsEvidenceRanking = {}
    numberEqual = {}
    spearmanAllEverything = []
    labelsDomainEverything = []
    standaardDomainEverything = []
    standaardLabelDomainEverything = []
    labelsAllEverything = []
    standaardLabelAllDomainEverything = []
    for loader in loaders:
        pearonsEvidenceRanking[loader[1]] = []
        numberEqual[loader[1]] = {}
        for data in loader[0]:
            for i in range(len(data[0])):

                metaDataClaim = oneHotEncoderBasis.encode(data[3][i], device)
                metadata_encoding = basisModels[0].metaDataEncoder(metaDataClaim.unsqueeze(0)).to(device)
                domain = data[0][0].split('-')[0]
                rankingB1, labelsAllB1, labelsDomainB1, allEqualB1 = basisModels[0].getRankingEvidencesLabels(
                    data[1][i], data[2][i],
                    metadata_encoding, domain,
                    data[5][i], data[6][i],
                    data[7][i],
                    data[8][i], data[9][i], data[10][i], data[11][i], data[12][i], data[13][i],
                    data[14][i], data[15][i], data[16][i])
                if allEqualB1:
                    if "Absolute" in numberEqual[domain]:
                        numberEqual[domain]["Absolute"] += 1
                    else:
                        numberEqual[domain]["Absolute"] = 1
                rankB1 = [sorted(rankingB1, reverse=True).index(x) + 1 for x in rankingB1]
                labelB1 = [sorted(labelsDomainB1, reverse=True).index(x) + 1 for x in labelsDomainB1]
                labelsAllB1 = [sorted(labelsAllB1, reverse=True).index(x) + 1 for x in labelsAllB1]
                metaDataClaim = oneHotEncoderBasis.encode(data[3][i], device)
                metadata_encoding = basisModels[1].metaDataEncoder(metaDataClaim.unsqueeze(0)).to(device)
                rankingB2, labelsAllB2, labelsDomainB2, allEqualB2 = basisModels[1].getRankingEvidencesLabels(
                    data[1][i], data[2][i],
                    metadata_encoding, domain,
                    data[5][i], data[6][i],
                    data[7][i],
                    data[8][i], data[9][i], data[10][i], data[11][i], data[12][i], data[13][i],
                    data[14][i], data[15][i], data[16][i])
                if allEqualB2:
                    if "Absolute2" in numberEqual[domain]:
                        numberEqual[domain]["Absolute2"] += 1
                    else:
                        numberEqual[domain]["Absolute2"] = 1
                rankB2 = [sorted(rankingB2, reverse=True).index(x) + 1 for x in rankingB2]
                labelB2 = [sorted(labelsDomainB2, reverse=True).index(x) + 1 for x in labelsDomainB2]
                labelsAllB2 = [sorted(labelsAllB2, reverse=True).index(x) + 1 for x in labelsAllB2]
                index = 1

                spearmanRankEntry = []
                labelsEntry = []
                labelsAllEntry = []
                for model in modelsEverything:
                    metaDataClaim = model.metaDataEncoder.oneHotEncoder.encode(data[3][i], device)
                    metadata_encoding = model.metaDataEncoder(metaDataClaim.unsqueeze(0)).to(device)
                    rankingM, labelsAllM, labelsDomainM, allEqualM = model.getRankingEvidencesLabels(data[1][i],
                                                                                                     data[2][i],
                                                                                                     metadata_encoding,
                                                                                                     domain,
                                                                                                     data[5][i],
                                                                                                     data[6][i],
                                                                                                     data[7][i],
                                                                                                     data[8][i],
                                                                                                     data[9][i],
                                                                                                     data[10][i],
                                                                                                     data[11][i],
                                                                                                     data[12][i],
                                                                                                     data[13][i],
                                                                                                     data[14][i],
                                                                                                     data[15][i],
                                                                                                     data[16][i])
                    if allEqualM:
                        if "everything" + str(index) in numberEqual[domain]:
                            numberEqual[domain]["everything" + str(index)] += 1
                        else:
                            numberEqual[domain]["everything" + str(index)] = 1

                    rankM = [sorted(rankingM, reverse=True).index(x) + 1 for x in rankingM]
                    corref = stats.spearmanr(rankB1, rankM)[0]
                    if not math.isnan(corref):
                        spearmanRankEntry.append(corref)
                    corref = stats.spearmanr(rankB2, rankM)[0]
                    if not math.isnan(corref):
                        spearmanRankEntry.append(corref)
                    labelM = [sorted(labelsDomainM, reverse=True).index(x) + 1 for x in labelsDomainM]
                    corref = stats.spearmanr(labelB1, labelM)[0]
                    if not math.isnan(corref):
                        labelsEntry.append(corref)
                    corref = stats.spearmanr(labelB2, labelM)[0]
                    if not math.isnan(corref):
                        labelsEntry.append(corref)
                    labelsAllM = [sorted(labelsAllM, reverse=True).index(x) + 1 for x in labelsAllM]
                    corref = stats.spearmanr(labelsAllB1, labelsAllM)[0]
                    if not math.isnan(corref):
                        labelsAllEntry.append(corref)
                    corref = stats.spearmanr(labelsAllB2, labelsAllM)[0]
                    if not math.isnan(corref):
                        labelsAllEntry.append(corref)
                    index += 1

                if len(spearmanRankEntry) > 0:
                    standaardDomainEverything.append(numpy.std(spearmanRankEntry))
                if len(labelsEntry) > 0:
                    standaardLabelDomainEverything.append(numpy.std(labelsEntry))
                if len(labelsAllEntry) > 0:
                    standaardLabelAllDomainEverything.append(numpy.std(labelsAllEntry))
                spearmanAllEverything.extend(spearmanRankEntry)
                labelsDomainEverything.extend(labelsEntry)
                labelsAllEverything.extend(labelsAllEntry)
    return numberEqual, \
           spearmanAllEverything, standaardDomainEverything, \
           labelsDomainEverything, standaardLabelDomainEverything, labelsAllEverything, standaardLabelAllDomainEverything
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
#domains = ["chct","faan","faly","fani","farg","goop","hoer","mpws","obry","para","peck","pomt","pose","ranz","snes","thal","thet","tron","vees","vogo","wast"]
models = []
for domain in domains:
        #dev_set = NUS(mode="Dev", path='timeModels/dev/dev-' + domain + '.tsv', pathToSave="timeModels/dev/time/dataset2/", domain=domain)
        #train_set = NUS(mode='Train', path='timeModels/train/train-' + domain + '.tsv', pathToSave="timeModels/train/time/dataset2/",
        #                domain=domain)
        test_set = NUS(mode='Test', path='timeModels/test/test-' + domain + '.tsv', pathToSave="timeModels/test/time/dataset2/",
                       domain=domain)
        test_loader = DataLoader(test_set,
                                 batch_size=1,
                                 shuffle=True)
        models.append([test_loader,domain])
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
with torch.no_grad():
    spearmanAll = []
    standaard = []
    labelsAllE = []
    standaardLabelsAll = []
    labelsDomainE = []
    standaardDomainLabelsAll = []
    spearmanAllAbsoluteD = []
    standaardAbsolute = []
    labelsDomainEAbsolute = []
    standaardDomainLabelsAllAbsolute = []
    labelsAllEAbsolute = []
    standaardLabelsAllAbsolute = []
    spearmanAllEverythingD = []
    standaardEverything = []
    labelsDomainEEverything = []
    standaardDomainLabelsAllEverything = []
    labelsAllEEverything = []
    standaardLabelsAllEverything = []
    numberEqual1 = {}
    numberEqual2 = {}
    numberEqual3 = {}
    spearmanAllAbsolute2 = []
    standaardAbsolute2 = []
    labelsDomainEAbsolute2 = []
    standaardDomainLabelsAllAbsolute2 = []
    labelsAllEAbsolute2 = []
    standaardLabelsAllAbsolute2 = []
    spearmanAllEverything2 = []
    standaardEverything2 = []
    labelsDomainEEverything2 = []
    standaardDomainLabelsAllEverything2 = []
    labelsAllEEverything2 = []
    standaardLabelsAllEverything2 = []
    spearmanAllAbsolute3 = []
    standaardAbsolute3 = []
    labelsDomainEAbsolute3 = []
    standaardDomainLabelsAllAbsolute3 = []
    labelsAllEAbsolute3 = []
    standaardLabelsAllAbsolute3 = []
    spearmanAllEverything3 = []
    standaardEverything3 = []
    labelsDomainEEverything3 = []
    standaardDomainLabelsAllEverything3 = []
    labelsAllEEverything3 = []
    standaardLabelsAllEverything3 = []
    for model in models:
        domain = model[1]
        print("Start loading model " + domain)
        oneHotEncoderBasis = OneHotEncoderBasis.oneHotEncoder('timeModels/Metadata_sequence/metadata')
        labelEmbeddingLayerBasis = labelEmbeddingLayerBasis.labelEmbeddingLayer(2308, domainIndices)
        encoderMetadataBasis = encoderMetadataB.encoderMetadata(3, 3, oneHotEncoderBasis).to(device)
        instanceEncoderBasis = instanceEncoderBasis.instanceEncoder().to(device)
        evidenceRankerBasis = evidence_rankerBasis.evidenceRanker(2308, 100).to(device)
        labelMaskDomainBasis = labelMaskDomainBasis.labelMaskDomain(2308, domainIndices, model[1], len(domainIndices[model[1]])).to(device)
        transformer = AutoModel.from_pretrained('sentence-transformers/all-distilroberta-v1').to(device)
        basisModel = verificationModelBERT.verifactionModel(transformer, encoderMetadataBasis, instanceEncoderBasis, evidenceRankerBasis, labelEmbeddingLayerBasis, labelMaskDomainBasis, domainIndices, domainWeights, domain)
        basisModel.loading_NeuralNetwork()
        oneHotEncoderBasis2 = OneHotEncoderBasis.oneHotEncoder('timeModels/Metadata_sequence/metadata')
        labelEmbeddingLayerBasis2 = labelEmbeddingLayerBasis.labelEmbeddingLayer(2308, domainIndices)
        encoderMetadataBasis2 = encoderMetadataB.encoderMetadata(3, 3, oneHotEncoderBasis).to(device)
        instanceEncoderBasis2 = instanceEncoderBasis.instanceEncoder().to(device)
        evidenceRankerBasis2 = evidence_rankerBasis.evidenceRanker(2308, 100).to(device)
        labelMaskDomainBasis2 = labelMaskDomainBasis.labelMaskDomain(2308, domainIndices, model[1], len(domainIndices[model[1]])).to(device)
        transformer = AutoModel.from_pretrained('sentence-transformers/all-distilroberta-v1').to(device)
        basisModel2 = verificationModelBERT2.verifactionModel(transformer,encoderMetadataBasis, instanceEncoderBasis,
                                            evidenceRankerBasis,
                                            labelEmbeddingLayerBasis,labelMaskDomainBasis, domainIndices,domainWeights,domain)
        basisModel2.loading_NeuralNetwork()
        oneHotEncoderM = OneHotEncoderBasis.oneHotEncoder('timeModels/Metadata_sequence/metadata')
        labelEmbeddingLayerM = labelEmbeddingLayer(2308, domainIndices)
        encoderMetadataM = encoderMetadata.encoderMetadata(3, 3, oneHotEncoderBasis).to(device)
        instanceEncoderM = instanceEncoder.instanceEncoder().to(device)
        evidenceRankerM = evidence_ranker.evidenceRanker(2308, 100).to(device)
        labelMaskDomainM = labelMaskDomain.labelMaskDomain(2308, domainIndices, model[1], len(domainIndices[model[1]])).to(device)
        transformer = AutoModel.from_pretrained('sentence-transformers/all-distilroberta-v1').to(device)
        verificationModelTime75A = verificationModelBERT75A.verifactionModel(transformer,encoderMetadataM, instanceEncoderM,
                                            evidenceRankerM,
                                            labelEmbeddingLayerM,labelMaskDomainM, domainIndices,domainWeights,domain).to(device)
        verificationModelTime75A.loading_NeuralNetwork()
        oneHotEncoderM = OneHotEncoderBasis.oneHotEncoder('timeModels/Metadata_sequence/metadata')
        labelEmbeddingLayerM = labelEmbeddingLayer(2308, domainIndices)
        encoderMetadataM = encoderMetadata.encoderMetadata(3, 3, oneHotEncoderBasis).to(device)
        instanceEncoderM = instanceEncoder.instanceEncoder().to(device)
        evidenceRankerM = evidence_ranker.evidenceRanker(2308, 100).to(device)
        labelMaskDomainM = labelMaskDomain.labelMaskDomain(2308, domainIndices, model[1], len(domainIndices[model[1]])).to(device)
        transformer = AutoModel.from_pretrained('sentence-transformers/all-distilroberta-v1').to(device)
        verificationModelTime75B = verificationModelBERT75B.verifactionModel(transformer,encoderMetadataM, instanceEncoderM,
                                            evidenceRankerM,
                                            labelEmbeddingLayerM,labelMaskDomainM, domainIndices,domainWeights,domain).to(device)
        verificationModelTime75B.loading_NeuralNetwork()
        oneHotEncoderM = OneHotEncoderBasis.oneHotEncoder('timeModels/Metadata_sequence/metadata')
        labelEmbeddingLayerM = labelEmbeddingLayer(2308, domainIndices)
        encoderMetadataM = encoderMetadata.encoderMetadata(3, 3, oneHotEncoderBasis).to(device)
        instanceEncoderM = instanceEncoder.instanceEncoder().to(device)
        evidenceRankerM = evidence_ranker.evidenceRanker(2308, 100).to(device)
        labelMaskDomainM = labelMaskDomain.labelMaskDomain(2308, domainIndices, model[1], len(domainIndices[model[1]])).to(device)
        transformer = AutoModel.from_pretrained('sentence-transformers/all-distilroberta-v1').to(device)
        verificationModelTimeAbsolute10A = verificationModelAbsoluteTimeBERTAdamWAdding10A.verifactionModel(transformer,encoderMetadataM, instanceEncoderM,
                                            evidenceRankerM,
                                            labelEmbeddingLayerM,labelMaskDomainM, domainIndices,domainWeights,domain).to(device)
        verificationModelTimeAbsolute10A .loading_NeuralNetwork()
        oneHotEncoderM = OneHotEncoderBasis.oneHotEncoder('timeModels/Metadata_sequence/metadata')
        labelEmbeddingLayerM = labelEmbeddingLayer(2308, domainIndices)
        encoderMetadataM = encoderMetadata.encoderMetadata(3, 3, oneHotEncoderBasis).to(device)
        instanceEncoderM = instanceEncoder.instanceEncoder().to(device)
        evidenceRankerM = evidence_ranker.evidenceRanker(2308, 100).to(device)
        labelMaskDomainM = labelMaskDomain.labelMaskDomain(2308, domainIndices, model[1], len(domainIndices[model[1]])).to(device)
        transformer = AutoModel.from_pretrained('sentence-transformers/all-distilroberta-v1').to(device)
        verificationModelTimeAbsolute10B = verificationModelAbsoluteTimeBERTAdamWAdding10B.verifactionModel(transformer,encoderMetadataM, instanceEncoderM,
                                            evidenceRankerM,
                                            labelEmbeddingLayerM,labelMaskDomainM, domainIndices,domainWeights,domain).to(device)
        verificationModelTimeAbsolute10B .loading_NeuralNetwork()
        oneHotEncoderM = OneHotEncoderBasis.oneHotEncoder('timeModels/Metadata_sequence/metadata')
        labelEmbeddingLayerM = labelEmbeddingLayer(2308, domainIndices)
        encoderMetadataM = encoderMetadata.encoderMetadata(3, 3, oneHotEncoderBasis).to(device)
        instanceEncoderM = instanceEncoder.instanceEncoder().to(device)
        evidenceRankerM = evidence_ranker.evidenceRanker(2308, 100).to(device)
        labelMaskDomainM = labelMaskDomain.labelMaskDomain(2308, domainIndices, model[1], len(domainIndices[model[1]])).to(device)
        transformer = AutoModel.from_pretrained('sentence-transformers/all-distilroberta-v1').to(device)
        verificationModelTimeEverything2020A = verificationModelAbsoluteTimeBERTAdamWAddingEverything2020A.verifactionModel(
            transformer,encoderMetadataM, instanceEncoderM,
                                            evidenceRankerM,
                                            labelEmbeddingLayerM,labelMaskDomainM, domainIndices,domainWeights,domain).to(device)
        verificationModelTimeEverything2020A.loading_NeuralNetwork()
        oneHotEncoderM = OneHotEncoderBasis.oneHotEncoder('timeModels/Metadata_sequence/metadata')
        labelEmbeddingLayerM = labelEmbeddingLayer(2308, domainIndices)
        encoderMetadataM = encoderMetadata.encoderMetadata(3, 3, oneHotEncoderBasis).to(device)
        instanceEncoderM = instanceEncoder.instanceEncoder().to(device)
        evidenceRankerM = evidence_ranker.evidenceRanker(2308, 100).to(device)
        labelMaskDomainM = labelMaskDomain.labelMaskDomain(2308, domainIndices, model[1], len(domainIndices[model[1]])).to(device)
        transformer = AutoModel.from_pretrained('sentence-transformers/all-distilroberta-v1').to(device)
        verificationModelTimeEverything2020B = verificationModelAbsoluteTimeBERTAdamWAddingEverything2020B.verifactionModel(
            transformer,encoderMetadataM, instanceEncoderM,
                                            evidenceRankerM,
                                            labelEmbeddingLayerM,labelMaskDomainM, domainIndices,domainWeights,domain).to(device)
        verificationModelTimeEverything2020B.loading_NeuralNetwork()
        basisModels = [basisModel,basisModel2]
        referenceModels = [verificationModelTime75A,verificationModelTime75B]
        modelsAbsolute = [verificationModelTimeAbsolute10A,verificationModelTimeAbsolute10B]
        modelsEverything = [verificationModelTimeEverything2020A, verificationModelTimeEverything2020B]
        print("Done loading "+domain)
        spearmanDomain, standaardDomain, labelsAll, standaardAll, labelsDomain, standaardLabelDomain, numberEqual, \
        spearmanAllAbsolute, standaardDomainAbsolute, labelsDomainAbsolute, standaardLabelDomainAbsolute, \
        labelsAllAbsolute, standaardLabelAllDomainAbsolute, spearmanAllEverything, standaardDomainEverything, \
        labelsDomainEverything, standaardLabelDomainEverything, labelsAllEverything, standaardLabelAllDomainEverything \
            = spearmanRanking([model], basisModels, referenceModels, modelsAbsolute, modelsEverything)
        numberEqual1.update(numberEqual)
        file = open('resultspearman/BERT/versusBase/' + model[1], "w")
        '''
        if len(spearmanDomain) > 0:
            print("Ranking Time")
            print("Gemiddelde : " + model[1] + " : " + str(np.mean(spearmanDomain)))
            print("Deviatie: " + model[1] + " : " + str(np.std(spearmanDomain)))
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
        if len(spearmanAllAbsolute) > 0:
            print("Ranking Absolute")
            print("Gemiddelde : " + model[1] + " : " + str(np.mean(spearmanAllAbsolute)))
            print("Deviatie: " + model[1] + " : " + str(np.std(spearmanAllAbsolute)))
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
        if len(spearmanAllEverything) > 0:
            print("Ranking Everything")
            print("Gemiddelde : " + model[1] + " : " + str(np.mean(spearmanAllEverything)))
            print("Deviatie: " + model[1] + " : " + str(np.std(spearmanAllEverything)))
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
        '''
        if len(spearmanDomain) > 0:
            file.write("Ranking Time" + "\n")
            file.write("Gemiddelde : " + model[1] + " : " + str(np.mean(spearmanDomain)) + "\n")
            file.write("Deviatie: " + model[1] + " : " + str(np.std(spearmanDomain)) + "\n")
            file.write("Deviation under the models:: " + model[1] + " : " + str(np.mean(standaardDomain)) + "\n")
        if len(labelsDomain) > 0:
            file.write("Labels domain Time" + "\n")
            file.write("Gemiddelde : " + model[1] + " : " + str(np.mean(labelsDomain)) + "\n")
            file.write("Deviatie: " + model[1] + " : " + str(np.std(labelsDomain)) + "\n")
            file.write("Deviation under the models:: " + model[1] + " : " + str(np.mean(standaardLabelDomain)) + "\n")
            file.write("Labels all Time" + "\n")
        if len(labelsAll) > 0:
            file.write("Gemiddelde : " + model[1] + " : " + str(np.mean(labelsAll)) + "\n")
            file.write("Deviatie: " + model[1] + " : " + str(np.std(labelsAll)) + "\n")
            file.write("Deviation under the models:: " + model[1] + " : " + str(np.mean(standaardAll)) + "\n")
        if len(spearmanAllAbsolute) > 0:
            file.write("Ranking Absolute" + "\n")
            file.write("Gemiddelde : " + model[1] + " : " + str(np.mean(spearmanAllAbsolute)) + "\n")
            file.write("Deviatie: " + model[1] + " : " + str(np.std(spearmanAllAbsolute)) + "\n")
            file.write(
                "Deviation under the models:: " + model[1] + " : " + str(np.mean(standaardDomainAbsolute)) + "\n")
        if len(labelsDomainAbsolute) > 0:
            file.write("Labels domain" + "\n")
            file.write("Gemiddelde : " + model[1] + " : " + str(np.mean(labelsDomainAbsolute)) + "\n")
            file.write("Deviatie: " + model[1] + " : " + str(np.std(labelsDomainAbsolute)) + "\n")
            file.write(
                "Deviation under the models:: " + model[1] + " : " + str(np.mean(standaardLabelDomainAbsolute)) + "\n")
            file.write("Labels all" + "\n")
        if len(labelsAllAbsolute) > 0:
            file.write("Gemiddelde : " + model[1] + " : " + str(np.mean(labelsAllAbsolute)) + "\n")
            file.write("Deviatie: " + model[1] + " : " + str(np.std(labelsAllAbsolute)) + "\n")
            file.write("Deviation under the models:: " + model[1] + " : " + str(
                np.mean(standaardLabelAllDomainAbsolute)) + "\n")
        if len(spearmanAllEverything) > 0:
            file.write("Ranking Everything" + "\n")
            file.write("Gemiddelde : " + model[1] + " : " + str(np.mean(spearmanAllEverything)) + "\n")
            file.write("Deviatie: " + model[1] + " : " + str(np.std(spearmanAllEverything)) + "\n")
            file.write(
                "Deviation under the models:: " + model[1] + " : " + str(np.mean(standaardDomainEverything)) + "\n")
        if len(labelsDomainEverything) > 0:
            file.write("Labels domain" + "\n")
            file.write("Gemiddelde : " + model[1] + " : " + str(np.mean(labelsDomainEverything)) + "\n")
            file.write("Deviatie: " + model[1] + " : " + str(np.std(labelsDomainEverything)) + "\n")
            file.write("Deviation under the models:: " + model[1] + " : " + str(
                np.mean(standaardLabelDomainEverything)) + "\n")
            file.write("Labels all" + "\n")
        if len(labelsAllEverything) > 0:
            file.write("Gemiddelde : " + model[1] + " : " + str(np.mean(labelsAllEverything)) + "\n")
            file.write("Deviatie: " + model[1] + " : " + str(np.std(labelsAllEverything)) + "\n")
            file.write("Deviation under the models:: " + model[1] + " : " + str(
                np.mean(standaardLabelAllDomainEverything)) + "\n")
        file.write(str(numberEqual))
        file.close()
        spearmanAll.extend(spearmanDomain)
        standaard.extend(standaardDomain)
        labelsAllE.extend(labelsAll)
        standaardLabelsAll.extend(standaardAll)
        labelsDomainE.extend(labelsDomain)
        standaardDomainLabelsAll.extend(standaardLabelDomain)
        spearmanAllAbsoluteD.extend(spearmanAllAbsolute)
        standaardAbsolute.extend(standaardDomainAbsolute)
        labelsDomainEAbsolute.extend(labelsDomainAbsolute)
        standaardDomainLabelsAllAbsolute.extend(standaardLabelDomainAbsolute)
        labelsAllEAbsolute.extend(labelsAllAbsolute)
        standaardLabelsAllAbsolute.extend(standaardLabelAllDomainAbsolute)
        spearmanAllEverythingD.extend(spearmanAllEverything)
        standaardEverything.extend(standaardDomainEverything)
        labelsDomainEEverything.extend(labelsDomainEverything)
        standaardDomainLabelsAllEverything.extend(standaardLabelDomainEverything)
        labelsAllEEverything.extend(labelsAllEverything)
        standaardLabelsAllEverything.extend(standaardLabelAllDomainEverything)
        numberEqualAbsolute, spearmanAllAbsolute, standaardDomainAbsolute, labelsDomainAbsolute, standaardLabelDomainAbsolute, \
        labelsAllAbsolute, standaardLabelAllDomainAbsolute, spearmanAllEverything, standaardDomainEverything, \
        labelsDomainEverything, standaardLabelDomainEverything, labelsAllEverything, standaardLabelAllDomainEverything \
            = spearmanRankingTime([model], referenceModels, modelsAbsolute, modelsEverything)
        numberEqual2.update(numberEqualAbsolute)
        # print("Time with absolute and everything")
        file = open('resultspearman/BERT/versusTime/' + model[1], "w")
        ''''
        if len(spearmanAllAbsolute) > 0:
            print("Ranking Absolute")
            print("Gemiddelde : " + model[1] + " : " + str(np.mean(spearmanAllAbsolute)))
            print("Deviatie: " + model[1] + " : " + str(np.std(spearmanAllAbsolute)))
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
        if len(spearmanAllEverything) > 0:
            print("Ranking Everything")
            print("Gemiddelde : " + model[1] + " : " + str(np.mean(spearmanAllEverything)))
            print("Deviatie: " + model[1] + " : " + str(np.std(spearmanAllEverything)))
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
        '''
        if len(spearmanAllAbsolute) > 0:
            file.write("Ranking Absolute" + "\n")
            file.write("Gemiddelde : " + model[1] + " : " + str(np.mean(spearmanAllAbsolute)) + "\n")
            file.write("Deviatie: " + model[1] + " : " + str(np.std(spearmanAllAbsolute)) + "\n")
            file.write(
                "Deviation under the models:: " + model[1] + " : " + str(np.mean(standaardDomainAbsolute)) + "\n")
        if len(labelsDomainAbsolute) > 0:
            file.write("Labels domain" + "\n")
            file.write("Gemiddelde : " + model[1] + " : " + str(np.mean(labelsDomainAbsolute)) + "\n")
            file.write("Deviatie: " + model[1] + " : " + str(np.std(labelsDomainAbsolute)) + "\n")
            file.write(
                "Deviation under the models:: " + model[1] + " : " + str(np.mean(standaardLabelDomainAbsolute)) + "\n")
            file.write("Labels all" + "\n")
        if len(labelsAllAbsolute) > 0:
            file.write("Gemiddelde : " + model[1] + " : " + str(np.mean(labelsAllAbsolute)) + "\n")
            file.write("Deviatie: " + model[1] + " : " + str(np.std(labelsAllAbsolute)) + "\n")
            file.write("Deviation under the models:: " + model[1] + " : " + str(
                np.mean(standaardLabelAllDomainAbsolute)) + "\n")
        if len(spearmanAllEverything) > 0:
            file.write("Ranking Everything" + "\n")
            file.write("Gemiddelde : " + model[1] + " : " + str(np.mean(spearmanAllEverything)) + "\n")
            file.write("Deviatie: " + model[1] + " : " + str(np.std(spearmanAllEverything)) + "\n")
            file.write(
                "Deviation under the models:: " + model[1] + " : " + str(np.mean(standaardDomainEverything)) + "\n")
        if len(labelsDomainEverything) > 0:
            file.write("Labels domain" + "\n")
            file.write("Gemiddelde : " + model[1] + " : " + str(np.mean(labelsDomainEverything)) + "\n")
            file.write("Deviatie: " + model[1] + " : " + str(np.std(labelsDomainEverything)) + "\n")
            file.write("Deviation under the models:: " + model[1] + " : " + str(
                np.mean(standaardLabelDomainEverything)) + "\n")
            file.write("Labels all" + "\n")
        if len(labelsAllEverything) > 0:
            file.write("Gemiddelde : " + model[1] + " : " + str(np.mean(labelsAllEverything)) + "\n")
            file.write("Deviatie: " + model[1] + " : " + str(np.std(labelsAllEverything)) + "\n")
            file.write("Deviation under the models:: " + model[1] + " : " + str(
                np.mean(standaardLabelAllDomainEverything)) + "\n")
        file.write(str(numberEqualAbsolute))
        file.close()
        spearmanAllAbsolute2.extend(spearmanAllAbsolute)
        standaardAbsolute2.extend(standaardDomainAbsolute)
        labelsDomainEAbsolute2.extend(labelsDomainAbsolute)
        standaardDomainLabelsAllAbsolute2.extend(standaardLabelDomainAbsolute)
        labelsAllEAbsolute2.extend(labelsAllAbsolute)
        standaardLabelsAllAbsolute2.extend(standaardLabelAllDomainAbsolute)
        spearmanAllEverything2.extend(spearmanAllEverything)
        standaardEverything2.extend(standaardDomainEverything)
        labelsDomainEEverything2.extend(labelsDomainEverything)
        standaardDomainLabelsAllEverything2.extend(standaardLabelDomainEverything)
        labelsAllEEverything2.extend(labelsAllEverything)
        standaardLabelsAllEverything2.extend(standaardLabelAllDomainEverything)
        numberEqualEverything, spearmanAllEverything, standaardDomainEverything, \
        labelsDomainEverything, standaardLabelDomainEverything, labelsAllEverything, standaardLabelAllDomainEverything \
            = spearmanRankingAbsolute([model], modelsAbsolute, modelsEverything)
        numberEqual3.update(numberEqualEverything)
        file = open('resultspearman/BERT/versusAbsolute/' + model[1], "w")
        '''
        print("Absolute with  everything")
        if len(spearmanAllEverything) > 0:
            print("Ranking Everything")
            print("Gemiddelde : " + model[1] + " : " + str(np.mean(spearmanAllEverything)))
            print("Deviatie: " + model[1] + " : " + str(np.std(spearmanAllEverything)))
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
        '''
        if len(spearmanAllEverything) > 0:
            file.write("Ranking Everything" + "\n")
            file.write("Gemiddelde : " + model[1] + " : " + str(np.mean(spearmanAllEverything)) + "\n")
            file.write("Deviatie: " + model[1] + " : " + str(np.std(spearmanAllEverything)) + "\n")
            file.write(
                "Deviation under the models:: " + model[1] + " : " + str(np.mean(standaardDomainEverything)) + "\n")
        if len(labelsDomainEverything) > 0:
            file.write("Labels domain" + "\n")
            file.write("Gemiddelde : " + model[1] + " : " + str(np.mean(labelsDomainEverything)) + "\n")
            file.write("Deviatie: " + model[1] + " : " + str(np.std(labelsDomainEverything)) + "\n")
            file.write("Deviation under the models:: " + model[1] + " : " + str(
                np.mean(standaardLabelDomainEverything)) + "\n")
            file.write("Labels all" + "\n")
        if len(labelsAllEverything) > 0:
            file.write("Gemiddelde : " + model[1] + " : " + str(np.mean(labelsAllEverything)) + "\n")
            file.write("Deviatie: " + model[1] + " : " + str(np.std(labelsAllEverything)) + "\n")
            file.write("Deviation under the models:: " + model[1] + " : " + str(
                np.mean(standaardLabelAllDomainEverything)) + "\n")
        file.write(str(numberEqualEverything))
        file.close()
        spearmanAllEverything3.extend(spearmanAllEverything)
        standaardEverything3.extend(standaardDomainEverything)
        labelsDomainEEverything3.extend(labelsDomainEverything)
        standaardDomainLabelsAllEverything3.extend(standaardLabelDomainEverything)
        labelsAllEEverything3.extend(labelsAllEverything)
        standaardLabelsAllEverything3.extend(standaardLabelAllDomainEverything)
    file = open("resultspearman/BERT/versusBase/AllDomains", "w")
    '''
    print("All domains for base versus time,absolute and everything")
    print("Ranking")
    print("Gemiddelde all : " + str(np.mean(spearmanAll)))
    print("Deviatie all : "  + str(np.std(spearmanAll)))
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
    print("Gemiddelde all : " + str(np.mean(spearmanAllAbsolute)))
    print("Deviatie all : " + str(np.std(spearmanAllAbsolute)))
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
    print("Gemiddelde all : " + str(np.mean(spearmanAllEverything)))
    print("Deviatie all : " + str(np.std(spearmanAllEverything)))
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
    '''
    file.write("All domains for base versus time,absolute and everything" + "\n")
    file.write("Ranking" + "\n")
    file.write("Gemiddelde all : " + str(np.mean(spearmanAll)) + "\n")
    file.write("Deviatie all : " + str(np.std(spearmanAll)) + "\n")
    file.write("Deviation all : " + str(np.mean(standaard)) + "\n")
    file.write("stand mean deviation all : " + str(np.std(standaard)) + "\n")
    file.write("Labels all" + "\n")
    file.write("Gemiddelde all : " + str(np.mean(labelsAllE)) + "\n")
    file.write("Deviatie all : " + str(np.std(labelsAllE)) + "\n")
    file.write("Deviation all : " + str(np.mean(standaardAll)) + "\n")
    file.write("stand mean deviation all : " + str(np.std(standaardAll)) + "\n")
    file.write("Labels domain" + "\n")
    file.write("Gemiddelde all : " + str(np.mean(labelsDomainE)) + "\n")
    file.write("Deviatie all : " + str(np.std(labelsDomainE)) + "\n")
    file.write("Deviation all : " + str(np.mean(standaardDomainLabelsAll)) + "\n")
    file.write("stand mean deviation all : " + str(np.std(standaardDomainLabelsAll)) + "\n")
    file.write("Ranking absolute" + "\n")
    file.write("Gemiddelde all : " + str(np.mean(spearmanAllAbsoluteD)) + "\n")
    file.write("Deviatie all : " + str(np.std(spearmanAllAbsoluteD)) + "\n")
    file.write("Deviation all : " + str(np.mean(standaardAbsolute)) + "\n")
    file.write("stand mean deviation all : " + str(np.std(standaardAbsolute)) + "\n")
    file.write("Labels all absolute" + "\n")
    file.write("Gemiddelde all : " + str(np.mean(labelsAllEAbsolute)) + "\n")
    file.write("Deviatie all : " + str(np.std(labelsAllEAbsolute)) + "\n")
    file.write("Deviation all : " + str(np.mean(standaardLabelsAllAbsolute)) + "\n")
    file.write("stand mean deviation all : " + str(np.std(standaardLabelsAllAbsolute)) + "\n")
    file.write("Labels domain absolute" + "\n")
    file.write("Gemiddelde all : " + str(np.mean(labelsDomainEAbsolute)) + "\n")
    file.write("Deviatie all : " + str(np.std(labelsDomainEAbsolute)) + "\n")
    file.write("Deviation all : " + str(np.mean(standaardDomainLabelsAllAbsolute)) + "\n")
    file.write("stand mean deviation all : " + str(np.std(standaardDomainLabelsAllAbsolute)) + "\n")
    file.write("Ranking everything" + "\n")
    file.write("Gemiddelde all : " + str(np.mean(spearmanAllEverythingD)) + "\n")
    file.write("Deviatie all : " + str(np.std(spearmanAllEverythingD)) + "\n")
    file.write("Deviation all : " + str(np.mean(standaardEverything)) + "\n")
    file.write("stand mean deviation all : " + str(np.std(standaardEverything)) + "\n")
    file.write("Labels all everything" + "\n")
    file.write("Gemiddelde all : " + str(np.mean(labelsAllEEverything)) + "\n")
    file.write("Deviatie all : " + str(np.std(labelsAllEEverything)) + "\n")
    file.write("Deviation all : " + str(np.mean(standaardLabelsAllEverything)) + "\n")
    file.write("stand mean deviation all : " + str(np.std(standaardLabelsAllEverything)) + "\n")
    file.write("Labels domain everything" + "\n")
    file.write("Gemiddelde all : " + str(np.mean(labelsDomainEEverything)) + "\n")
    file.write("Deviatie all : " + str(np.std(labelsDomainEEverything)) + "\n")
    file.write("Deviation all : " + str(np.mean(standaardDomainLabelsAllEverything)) + "\n")
    file.write("stand mean deviation all : " + str(np.std(standaardDomainLabelsAllEverything)) + "\n")
    file.write(str(numberEqual1))
    file.close()

    file = open("resultspearman/BERT/versusTime/AllDomains", "w")
    '''
    print("All domains for time versus absolute and everything")
    print("Ranking absolute")
    print("Gemiddelde all : " + str(np.mean(spearmanAllAbsolute2)))
    print("Deviatie all : " + str(np.std(spearmanAllAbsolute2)))
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
    print("Gemiddelde all : " + str(np.mean(spearmanAllEverything2)))
    print("Deviatie all : " + str(np.std(spearmanAllEverything2)))
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
    '''
    file.write("All domains for time versus absolute and everything" + "\n")
    file.write("Ranking absolute" + "\n")
    file.write("Gemiddelde all : " + str(np.mean(spearmanAllAbsolute2)) + "\n")
    file.write("Deviatie all : " + str(np.std(spearmanAllAbsolute2)) + "\n")
    file.write("Deviation all : " + str(np.mean(standaardAbsolute2)) + "\n")
    file.write("stand mean deviation all : " + str(np.std(standaardAbsolute2)) + "\n")
    file.write("Labels all absolute" + "\n")
    file.write("Gemiddelde all : " + str(np.mean(labelsAllEAbsolute2)) + "\n")
    file.write("Deviatie all : " + str(np.std(labelsAllEAbsolute2)) + "\n")
    file.write("Deviation all : " + str(np.mean(standaardLabelsAllAbsolute2)) + "\n")
    file.write("stand mean deviation all : " + str(np.std(standaardLabelsAllAbsolute2)) + "\n")
    file.write("Labels domain absolute" + "\n")
    file.write("Gemiddelde all : " + str(np.mean(labelsDomainEAbsolute2)) + "\n")
    file.write("Deviatie all : " + str(np.std(labelsDomainEAbsolute2)) + "\n")
    file.write("Deviation all : " + str(np.mean(standaardDomainLabelsAllAbsolute2)) + "\n")
    file.write("stand mean deviation all : " + str(np.std(standaardDomainLabelsAllAbsolute2)) + "\n")
    file.write("Ranking everything" + "\n")
    file.write("Gemiddelde all : " + str(np.mean(spearmanAllEverything2)) + "\n")
    file.write("Deviatie all : " + str(np.std(spearmanAllEverything2)) + "\n")
    file.write("Deviation all : " + str(np.mean(standaardEverything2)) + "\n")
    file.write("stand mean deviation all : " + str(np.std(standaardEverything2)) + "\n")
    file.write("Labels all everything" + "\n")
    file.write("Gemiddelde all : " + str(np.mean(labelsAllEEverything2)) + "\n")
    file.write("Deviatie all : " + str(np.std(labelsAllEEverything2)) + "\n")
    file.write("Deviation all : " + str(np.mean(standaardLabelsAllEverything2)) + "\n")
    file.write("stand mean deviation all : " + str(np.std(standaardLabelsAllEverything2)) + "\n")
    file.write("Labels domain everything" + "\n")
    file.write("Gemiddelde all : " + str(np.mean(labelsDomainEEverything2)) + "\n")
    file.write("Deviatie all : " + str(np.std(labelsDomainEEverything2)) + "\n")
    file.write("Deviation all : " + str(np.mean(standaardDomainLabelsAllEverything2)) + "\n")
    file.write("stand mean deviation all : " + str(np.std(standaardDomainLabelsAllEverything2)) + "\n")
    file.write(str(numberEqual2))
    file.close()

    file = open("resultspearman/BERT/versusAbsolute/AllDomains", "w")
    file.write("All domains for absolute versus everything" + "\n")
    file.write("Ranking everything" + "\n")
    file.write("Gemiddelde all : " + str(np.mean(spearmanAllEverything3)) + "\n")
    file.write("Deviatie all : " + str(np.std(spearmanAllEverything3)) + "\n")
    file.write("Deviation all : " + str(np.mean(standaardEverything3)) + "\n")
    file.write("stand mean deviation all : " + str(np.std(standaardEverything3)) + "\n")
    file.write("Labels all everything" + "\n")
    file.write("Gemiddelde all : " + str(np.mean(labelsAllEEverything3)) + "\n")
    file.write("Deviatie all : " + str(np.std(labelsAllEEverything3)) + "\n")
    file.write("Deviation all : " + str(np.mean(standaardLabelsAllEverything3)) + "\n")
    file.write("stand mean deviation all : " + str(np.std(standaardLabelsAllEverything3)) + "\n")
    file.write("Labels domain everything" + "\n")
    file.write("Gemiddelde all : " + str(np.mean(labelsDomainEEverything3)) + "\n")
    file.write("Deviatie all : " + str(np.std(labelsDomainEEverything3)) + "\n")
    file.write("Deviation all : " + str(np.mean(standaardDomainLabelsAllEverything3)) + "\n")
    file.write("stand mean deviation all : " + str(np.std(standaardDomainLabelsAllEverything3)) + "\n")
    file.write(str(numberEqual3))
