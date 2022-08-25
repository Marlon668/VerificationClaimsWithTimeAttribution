import io
import math
import os
import statistics
import sys
import random

import matplotlib
import scipy.stats as ss
import numpy
import numpy as np
import torch.nn.functional as F
from fontTools.merge import cmap
from matplotlib import pyplot as plt, colors
from datasetIteratie2Combiner import NUS
from basisModel import OneHotEncoderBasis, labelEmbeddingLayerBasis, verificationModelBasis, encoderBasis, encoderMetadataBasis, \
    instanceEncoderBasis, evidence_rankerBasis, labelMaskDomainBasis
import torch
from torch.utils.data import DataLoader

'''
Calculate attribution for text for basismodel
code based on https://github.com/ankurtaly/Integrated-Gradients
'''
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
    return domainsIndices,domainsLabels,domainLabelIndices,domainWeights

def calculate_outputs_and_gradients(data, model):
    # do the pre-processing
    predict_idx = None
    gradientsEncoding = []
    for i in range(len(data[0])):
        print(data[1][i])
        print(data[2][i])
        metaDataClaim = oneHotEncoderBasis.encode(data[3][i], device)
        metadata_encoding = model.metaDataEncoder(metaDataClaim.unsqueeze(0)).to(device)
        domain = data[0][0].split('-')[0]
        output,claim_encoding,evidenceEncodings = model.forwardAttribution(data[1][i], data[2][i],metadata_encoding, domain,data[5][i],data[6][i])
        output = F.softmax(output, dim=0)
        target_label_idx = torch.argmax(output).item()
        index = np.ones((1)) * target_label_idx
        index = torch.tensor(index, dtype=torch.int64)
        output = output.gather(0, index)
        # clear grad
        model.zero_grad()
        claim_encoding.retain_grad()
        for evidenceEncoding in evidenceEncodings:
            evidenceEncoding.retain_grad()
        output.backward()
        gradient = claim_encoding.grad.detach().cpu().numpy()[0]
        gradientsEncoding.append(gradient)
        for evidenceEncoding in evidenceEncodings:
            gradient = evidenceEncoding.grad.detach().cpu().numpy()[0]
            gradientsEncoding.append(gradient)
        invoer = [claim_encoding,evidenceEncodings]
    gradientsEncoding = np.array(gradientsEncoding)
    return gradientsEncoding, target_label_idx,invoer,metadata_encoding

def getInputs(data, model):
    for i in range(len(data[0])):
        metaDataClaim = oneHotEncoderBasis.encode(data[3][i], device)
        metadata_encoding = model.metaDataEncoder(metaDataClaim.unsqueeze(0)).to(device)
        domain = data[0][0].split('-')[0]
        output,claim_encoding,evidenceEncodings = model.forwardAttribution(data[1][i], data[2][i],metadata_encoding, domain,data[5][i],data[6][i])
        invoer = [claim_encoding,evidenceEncodings]
    return invoer,metadata_encoding

def calculate_outputs_and_gradientsIntegrated(inputs,metadata_encoding, model,target_label_idx):
    # do the pre-processing
    predict_idx = None
    gradientsEncoding = []
    for input in inputs:
        gradientsEncodingEntry = []

        input[0].requires_grad = True
        input[0].retain_grad()
        for i in range(len(input[1])):
            input[1][i].requires_grad = True
            input[1][i].retain_grad()

        output = model.forwardIntegrated(input,metadata_encoding)
        output = F.softmax(output, dim=0)
        index = np.ones((1)) * target_label_idx
        index = torch.tensor(index, dtype=torch.int64)
        output = output.gather(0, index)
        # clear grad
        model.zero_grad()
        output.backward()
        gradient = input[0].grad.detach().cpu().numpy()[0]
        gradientsEncodingEntry.append(gradient)
        for i in range(len(input[1])):
            gradient = input[1][i].grad.detach().cpu().numpy()[0]
            gradientsEncodingEntry.append(gradient)
        gradientsEncoding.append(gradientsEncodingEntry)
    gradientsEncoding = np.array(gradientsEncoding)
    return gradientsEncoding

# integrated gradients
def integrated_gradients(inputs,metadata_encoding, model, target_label_idx, predict_and_gradients, baseline, steps=50, cuda=False):
    baselineClaim = torch.zeros([256])
    baselineEvidence = torch.zeros([256])
    baselineClaim, baselineEvidence = model.getBaseLine(baselineClaim, baselineEvidence, metadata_encoding.squeeze(0))
    scaled_inputs = []
    print(inputs)
    for i in range(0, steps + 1):
        entry = []
        entry.append(baselineClaim + (float(i)/ steps) * (inputs[0]-baselineClaim))
        evidences = []
        for j in range(len(inputs[1])):
            evidences.append(baselineEvidence + (float(i) / steps) * (inputs[1][j]-baselineEvidence))
        entry.append(evidences)
        scaled_inputs.append(entry)

    gradientsEncoding = predict_and_gradients(scaled_inputs,metadata_encoding, model, target_label_idx)

    sumGradsEncoding = [np.zeros(len(inputs[1])+1)]
    for i in range(0,steps +1):
        sumGradsEncoding += gradientsEncoding[i]
    avg_gradientsEncoding = [element/(steps+1) for element in sumGradsEncoding]
    integrated_gradEncoding = []
    integrated_gradEncoding.append((inputs[0].numpy()-baselineClaim.numpy())*avg_gradientsEncoding[0][0])
    print(inputs[1])
    for i in range(len(inputs[1])):
        integrated_gradEncoding.append((inputs[1][i].numpy()-baselineEvidence.numpy())*avg_gradientsEncoding[0][i+1])
    return integrated_gradEncoding

def random_baseline_integrated_gradients(inputs,metadata_encoding, model, target_label_idx, predict_and_gradients, steps, num_random_trials):
    all_intgradsEncoding = []
    integrated_gradEncoding= integrated_gradients(inputs,metadata_encoding, model, target_label_idx, predict_and_gradients, \
                                            baseline=torch.zeros(256), steps=steps)
    all_intgradsEncoding.append(integrated_gradEncoding)
    return integrated_gradEncoding

if __name__ == '__main__':
    '''
    argument 1 path of model
    argument 2 name of domain to take examples from to calculate attributin
    '''
    oneHotEncoderBasis = OneHotEncoderBasis.oneHotEncoder('Metadata_sequence/metadata')
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    domainIndices, domainLabels, domainLabelIndices, domainWeights = getLabelIndicesDomain(
        'labels/labels.tsv', 'labels/labelSequence', 'labels/weights.tsv')
    domains = {sys.argv[2]}
    datas= []
    for domain in domains:
        test_set = NUS(mode='Test', path='test/test-' + domain + '.tsv',
                       pathToSave="test/time/dataset2/",
                       domain=domain)
        dev_loader = DataLoader(test_set,
                                 batch_size=1,
                                 shuffle=False)
        datas.append([dev_loader, domain])


    for data in datas:
        labelEmbeddingLayerBasis = labelEmbeddingLayerBasis.labelEmbeddingLayer(772, domainIndices)
        encoderBasis = encoderBasis.encoder(300, 128).to(device)
        encoderMetadataBasis = encoderMetadataBasis.encoderMetadata(3, 3, oneHotEncoderBasis).to(device)
        instanceEncoderBasis = instanceEncoderBasis.instanceEncoder().to(device)
        evidenceRankerBasis = evidence_rankerBasis.evidenceRanker(772, 100).to(device)
        labelMaskDomainBasis = labelMaskDomainBasis.labelMaskDomain(772, domainIndices, data[1],
                                                                    len(domainIndices[data[1]])).to(device)
        basisModel = verificationModelBasis.verifactionModel(encoderBasis, encoderMetadataBasis, instanceEncoderBasis,
                                                             evidenceRankerBasis,
                                                             labelEmbeddingLayerBasis, labelMaskDomainBasis, domainIndices,
                                                             domainWeights, data[1]).to(device)
        basisModel.loading_NeuralNetwork(sys.argv[1])
        for entry in data[0]:
            print(entry)
            gradientsEncoding, predictedLabel,_,_ = calculate_outputs_and_gradients(entry, basisModel)
            with torch.no_grad():
                inputs, metadata_encoding = getInputs(entry, basisModel)
                label_index = domainLabelIndices[data[1]][domainLabels[data[1]].index(entry[4][0])]
            if predictedLabel != label_index:
                attributionsEncoding = random_baseline_integrated_gradients(inputs,metadata_encoding, basisModel, label_index,
                                                                    calculate_outputs_and_gradientsIntegrated, \
                                                                    steps=1000, num_random_trials=1)

                print("Summation of the attributions encoding")
                print(numpy.sum(attributionsEncoding,axis=1))
                print("Summation of the absolute attributions encoding")
                print(numpy.sum(np.abs(attributionsEncoding), axis=1))
