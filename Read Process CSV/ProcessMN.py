import copy
from tqdm import tqdm
import numpy as np
import sympy as sy
import json
from scipy import stats

#relative imports of Isotomics
import sys
import os
from pathlib import Path
cwd = Path().resolve()

sys.path.insert(1, os.path.join(cwd, 'Isotomics Scripts'))

import initializeMethionine as metTest
import readInput as ri
import fragmentAndSimulate as fas
import solveSystem as ss
import basicDeltaOperations as op
import calcIsotopologues as ci

fragSubset = ['133','104','102','61','56']

processFragKeys = {'133':'133',
                '104':'104',
                '102':'102',
                '61':'61',
                '56':'56'}
deltasStd = [-53.9,-23.7,-24.0,-24.3,0,2.215,0,0,0,0,0,0,0]
deltasSmp = [41.05,-23.7,-24.0,-24.3,0,2.215,0,0,0,0,0,0,0]

def standardizeLR(standardValues):
    #Perform a linear regression
    n = len(standardValues)
    xs = np.arange(0,n,1)
    slope, intercept, r_value, p_value, slope_serr = stats.linregress(xs,standardValues)

    #calculate useful intermediates
    predictions = slope * xs + intercept 
    yerr = standardValues - predictions 
    s_err = np.sum(yerr**2)
    mean_x = np.mean(xs)       
    # appropriate t value (where n=4, two tailed 68%)            
    t = stats.t.ppf(1-0.16, n-2)              

    #Define a function to calculate the error in this regression at a certain x value
    def errorAtX(thisX):
        pointErr = np.sqrt(s_err / (n-2)) * np.sqrt(1.0/n + (thisX - mean_x)**2 / np.sum((xs-mean_x)**2))
        return pointErr

    #Standardize sample values
    smpxs = np.arange(0.5,3.5,1)
    smpPreds = slope * smpxs + intercept
    smpErr = errorAtX(smpxs)

    return smpPreds, smpErr
'''
This code accomplishes the following:

1) Precalculate the isotopologues of methionine and select only the ones we need
'''
def toJsonEncodeable(MNDict):
    '''
    Allows us to encode the MNDicts as jsons so we don't need to compute each time 
    '''
    for MNKey, MNData in MNDict.items():
        for isoKey, isoData in MNData.items(): 
            isoData['Number'] = int(isoData['Number'])
            isoData['Conc'] = float(isoData['Conc'])
            isoData['Mass'] = int(isoData['Mass'])
            isoData['Stochastic U'] = float(isoData['Stochastic U'])

MNKeyData = {}
toRun = ['Generate Predictions','M1','M2','M3','M4']
'''
(1) Precalculate the isotopologues of methionine and select only the ones we need
'''
deltas = [0] * 13
precalc = False
if precalc:
    #precalculate isotopologues of interest
    preCalcMet = metTest.initializeMethionine(deltas, fragSubset, printHeavy = False)
    byAtom = ci.inputToAtomDict(preCalcMet['molecularDataFrame'], overrideIsoDict = {}, disable = False, M1Only = False)
    MN = ci.massSelections(byAtom, massThreshold = 4)

    overrideIsoDict = ci.pullOutIsotopologues(MN)
        
    with open(str(cwd) + '/Reference Files/Stored Isotopologues5Frags.json', 'w', encoding='utf-8') as f:
        json.dump(overrideIsoDict, f, ensure_ascii=False, indent=4)

with open(str(cwd) + '/Reference Files/Stored Isotopologues5Frags.json', 'r') as j:
    overrideIsoDict = json.loads(j.read())

'''
2) Generate a 'perfect' dataset, including all possible peaks; then read in the observed peaks, and determine which went unobserved. We can use this list of forbidden peaks to help generate our forward models. 
'''
#Need molecularDf and fragmentation information in all cases.
#Approximation for standard
stdMethionine = metTest.initializeMethionine(deltasStd, fragSubset)


withoutCutoff, MNDict, fractionationFactors = metTest.simulateMeasurement(stdMethionine,overrideIsoDict = overrideIsoDict, abundanceThreshold = 0.00,
                                                massThreshold = 4,
                                                calcFF = False,
                                                omitMeasurements = {})

#Compute forbidden peaks dictionary
forbiddenPeaks = {}
for MNKey, MNData in withoutCutoff.items():
    forbiddenPeaks[MNKey] = {}
    if MNKey in ['M1','M2','M3','M4']:
        with open('Met_' + MNKey + 'Results.json') as f:
            sampleOutputDict = json.load(f)
        firstFileKey = list(sampleOutputDict.keys())[0]
        obsPeaks = sampleOutputDict[firstFileKey]
        for fragKey, fragData in MNData.items():
            forbiddenPeaks[MNKey][fragKey] = []
            for subKey, subData in fragData.items():
                if subKey not in obsPeaks[fragKey]:
                    forbiddenPeaks[MNKey][fragKey].append(subKey)

'''
3) Generate a forward model of the sample and standard, using the forbidden peaks
'''
deltasStd = [-53.9,-23.7,-24.0,-24.3,0,2.215,0,0,0,0,0,0,0]
smpMethionineFM = metTest.initializeMethionine(deltasStd, fragSubset)
    
predictedMeasurementSmpExp, MNDictSmp, fractionationFactors = metTest.simulateMeasurement(smpMethionineFM, overrideIsoDict=overrideIsoDict,
                                                abundanceThreshold = 0.00,
                                                massThreshold = 4,
                                                calcFF = False,
                                                omitMeasurements = forbiddenPeaks)

'''
4) Solve the (analytically solved) M+1 problem
'''
if 'M1' in toRun:
    with open(str(cwd) + '/Read Process CSV/M1Results.json') as f:
        sampleOutputDict = json.load(f)

    replicateData = ri.readObservedData(sampleOutputDict, theory = withoutCutoff,
                                    standard = [True, False, True, False,
                                                True, False, True],
                                    processFragKeys = processFragKeys)

    replicateDataKeys = list(replicateData.keys())

    R13C = op.deltaToConcentration('13C',-11)
    U13CAppx = 5*R13C[1] / R13C[0]

    #Run through the M+1 algorithm
    fullResults = {}
    linearStandardization = {}
    for smpFileIdx in range(1,7,2):
        #compute predicted standard values
        linear = False
        firstBracket = replicateData[replicateDataKeys[smpFileIdx-1]]
        secondBracket = replicateData[replicateDataKeys[smpFileIdx+1]]

        processStandard = {'M1':{}}
        
        for fragKey, fragInfo in firstBracket['M1'].items():
            if linear:
                if fragKey not in linearStandardization:
                    linearStandardization[fragKey] = {}
                #calculate linear prediction of standard composition once
                if smpFileIdx == 1:
                    thisFragLinearStandardization = {'Values':[],'PredSmp':[],'PredSmpErr':[]}
                    for stdFileIdx in [0,2,4,6]:
                        thisFragLinearStandardization['Values'].append(np.array(replicateData[replicateDataKeys[stdFileIdx]]['M1'][fragKey]['Observed Abundance']))

                    thisFragLinearStandardization['Values'] = np.array(thisFragLinearStandardization['Values'])
                    
                    #Each substitution is a column of the resulting array
                    for subCol in thisFragLinearStandardization['Values'].T:

                        smpPreds, smpErr = standardizeLR(subCol)

                        #fill in dictionary
                        thisFragLinearStandardization['PredSmp'].append(smpPreds)
                        thisFragLinearStandardization['PredSmpErr'].append(smpErr)

                    linearStandardization[fragKey]['PredSmp'] = np.array(thisFragLinearStandardization['PredSmp'])
                    linearStandardization[fragKey]['PredSmpErr'] = np.array(thisFragLinearStandardization['PredSmpErr'])

                #file Indices (1,3,5) should go to number indices (0,1,2)
                smpNmIdx = smpFileIdx // 2
                predStdForThisSmp = linearStandardization[fragKey]['PredSmp'].T[smpNmIdx]
                predStdErrForThisSmp = linearStandardization[fragKey]['PredSmpErr'].T[smpNmIdx]

                processStandard['M1'][fragKey] = {'Subs':fragInfo['Subs'],
                                                'Predicted Abundance':fragInfo['Predicted Abundance'],
                                                'Observed Abundance':predStdForThisSmp,
                                                'Error':predStdErrForThisSmp}
            
            else:
                avgAbund = (np.array(fragInfo['Observed Abundance']) + np.array(secondBracket['M1'][fragKey]['Observed Abundance'])) / 2
                combinedErr = (np.array(fragInfo['Error']) + np.array(secondBracket['M1'][fragKey]['Error'])) / 2
                processStandard['M1'][fragKey] = {'Subs':fragInfo['Subs'],
                                                'Predicted Abundance':fragInfo['Predicted Abundance'],
                                                'Observed Abundance':avgAbund,
                                                'Error':combinedErr}

        processSample = replicateData[replicateDataKeys[smpFileIdx]]
        UValuesSmp = {'13C':{'Observed': U13CAppx, 'Error': U13CAppx * 0.0001},
                    '33S':{'Observed':0.007901812549, 'Error':0.007901812549 * 0.0004 * 0.515}}

        isotopologuesDict = fas.isotopologueDataFrame(MNDict, stdMethionine['molecularDataFrame'])
        OValueCorrection = ss.OValueCorrectTheoretical(withoutCutoff, 
                                                    processSample,
                                                    massThreshold = 1)

        M1Results = ss.M1MonteCarlo(processStandard, processSample, OValueCorrection, isotopologuesDict, stdMethionine['fragmentationDictionary'], experimentalOCorrectList = [],  N = 1000, GJ = False, debugMatrix = False,abundanceCorrect = True,perturbTheoryOAmt = 0.0005)

        processedResults = ss.processM1MCResults(M1Results, UValuesSmp, isotopologuesDict, stdMethionine['molecularDataFrame'], GJ = False,UMNSub = ['13C'])

        ss.updateSiteSpecificDfM1MC(processedResults, stdMethionine['molecularDataFrame'])

        fullResults['Replicate ' + str(smpFileIdx // 2)] = processedResults

    fullResultsMeansStds = {'Mean':[],'Std':[]}
    for replicate, replicateInfo in fullResults.items():
        mean = np.array(replicateInfo['Relative Deltas']).T.mean(axis = 1)
        std = np.array(replicateInfo['Relative Deltas']).T.std(axis = 1)
        
        fullResultsMeansStds['Mean'].append(mean)
        fullResultsMeansStds['Std'].append(std)
        
    fullResultsMeansStds['Mean'] = np.array(fullResultsMeansStds['Mean']).T.tolist()
    fullResultsMeansStds['Std'] = np.array(fullResultsMeansStds['Std']).T.tolist()

    with open(str(cwd) + '/Read Process CSV/M1ProcessedResults.json', 'w', encoding='utf-8') as f:
        json.dump(fullResultsMeansStds, f, ensure_ascii=False, indent=4)

Conc34 = op.deltaToConcentration('34S',4.3)
U34SAppx = Conc34[2] / Conc34[0]
U34SAppxError = U34SAppx * 0.0004

MNKeyParams = {'M2':{'fileName':'M2Results.json',
                     'U Value to Use':'34S',
                     'U Value':U34SAppx,
                     'U Value Error':U34SAppxError},

                'M3':{'fileName':'M3Results.json',
                     'U Value to Use':'34S15N',
                     'U Value':0.00016303994459899808,
                     'U Value Error':0.00016303994459899808 * 0.000412},

                'M4':{'fileName':'M4Results.json',
                     'U Value to Use':'36S',
                     'U Value':0.00010613428277475725,
                     'U Value Error':0.00010613428277475725 * 0.0004*1.9}}

for currentMNKey, currentMNParams in MNKeyParams.items():
    if currentMNKey in toRun:
        with open(str(cwd) + '/Read Process CSV/' + currentMNParams['fileName']) as f:
            sampleOutputDict = json.load(f)

        processFragKeys = {'133':'133',
                    '104':'104',
                    '102':'102',
                    '61':'61',
                    '56':'56'}
            
        replicateDataRead = ri.readObservedData(sampleOutputDict, MNKey = currentMNKey, theory = withoutCutoff,
                                        standard = [True, False, True, False, True, False, True],
                                        processFragKeys = processFragKeys)

        replicateData = {**replicateDataRead}

        replicateDataKeys = list(replicateData.keys())

        fullResults = {}
        linearStandardization = {}
        linear = True
        for smpFileIdx in range(1,7,2):
            firstBracket = replicateData[replicateDataKeys[smpFileIdx-1]]
            secondBracket = replicateData[replicateDataKeys[smpFileIdx+1]]

            processStandard = {currentMNKey:{}}
            
            for fragKey, fragInfo in firstBracket[currentMNKey].items():
                if linear:
                    if fragKey not in linearStandardization:
                        linearStandardization[fragKey] = {}

                    if smpFileIdx == 1:
                        thisFragLinearStandardization = {'Values':[],'PredSmp':[],'PredSmpErr':[]}
                        for stdFileIdx in [0,2,4,6]:
                            thisFragLinearStandardization['Values'].append(np.array(replicateData[replicateDataKeys[stdFileIdx]][currentMNKey][fragKey]['Observed Abundance']))

                        thisFragLinearStandardization['Values'] = np.array(thisFragLinearStandardization['Values'])
                        
                        #Each substitution is a column of the resulting array
                        for subCol in thisFragLinearStandardization['Values'].T:
                            smpPreds, smpErr = standardizeLR(subCol)

                            #fill in dictionary
                            thisFragLinearStandardization['PredSmp'].append(smpPreds)
                            thisFragLinearStandardization['PredSmpErr'].append(smpErr)

                        linearStandardization[fragKey]['PredSmp'] = np.array(thisFragLinearStandardization['PredSmp'])
                        linearStandardization[fragKey]['PredSmpErr'] = np.array(thisFragLinearStandardization['PredSmpErr'])

                    #file Indices (1,3,5) should go to number indices (0,1,2)
                    smpNmIdx = smpFileIdx // 2
                    predStdForThisSmp = linearStandardization[fragKey]['PredSmp'].T[smpNmIdx]
                    predStdErrForThisSmp = linearStandardization[fragKey]['PredSmpErr'].T[smpNmIdx]

                    processStandard[currentMNKey][fragKey] = {'Subs':fragInfo['Subs'],
                                                    'Predicted Abundance':fragInfo['Predicted Abundance'],
                                                    'Observed Abundance':predStdForThisSmp,
                                                    'Error':predStdErrForThisSmp}

                else:
                    avgAbund = (np.array(fragInfo['Observed Abundance']) + np.array(secondBracket[currentMNKey][fragKey]['Observed Abundance'])) / 2
                    combinedErr = (np.array(fragInfo['Error']) + np.array(secondBracket[currentMNKey][fragKey]['Error'])) / 2
                    processStandard[currentMNKey][fragKey] = {'Subs':fragInfo['Subs'],
                                                    'Predicted Abundance':fragInfo['Predicted Abundance'],
                                                    'Observed Abundance':avgAbund,
                                                    'Error':combinedErr}

            processSample = replicateData[replicateDataKeys[smpFileIdx]]
            UValuesSmp = {currentMNParams['U Value to Use']:{'Observed': currentMNParams['U Value'], 'Error':currentMNParams['U Value Error']}}

            isotopologuesDict = fas.isotopologueDataFrame(MNDict, stdMethionine['molecularDataFrame'])
            OCorrection = ss.OValueCorrectTheoretical(predictedMeasurementSmpExp, 
                                                        processSample,
                                                        massThreshold = 4)
            
            MNSol = {currentMNKey:0}
            UMNSubs = {currentMNKey:[currentMNParams['U Value to Use']]}

            for MNKey in MNSol.keys():
                print("Solving " + MNKey)

                Isotopologues = isotopologuesDict[MNKey]
                results, comp, GJSol, meas = ss.MonteCarloMN(MNKey, Isotopologues, processStandard, processSample, 
                                                    OCorrection, stdMethionine['fragmentationDictionary'], N = 100,
                                                    perturbTheoryOAmt = 0.001,
                                                    abundanceCorrect = True)

                dfOutput = ss.checkSolutionIsotopologues(GJSol, Isotopologues, MNKey, numerical = False)
                nullSpaceCycles = ss.findNullSpaceCycles(comp, Isotopologues)
                actuallyConstrained = ss.findFullyConstrained(nullSpaceCycles)
                processedResults = ss.processMNMonteCarloResults(MNKey, results, UValuesSmp, dfOutput,  stdMethionine['molecularDataFrame'], MNDict,
                                                                UMNSub = UMNSubs[MNKey])
                dfOutput = ss.updateMNMonteCarloResults(dfOutput, processedResults)
                MNSol[MNKey] = dfOutput.loc[dfOutput.index.isin(actuallyConstrained)].copy()
                
            fullResults['Replicate ' + str(smpFileIdx)] = MNSol[MNKey]

        fullResultsMeansStds = {}

        for replicateKey, replicateData in fullResults.items():
            constrainedSubs = list(replicateData.index)

            for subIdx, sub in enumerate(constrainedSubs):
                if sub not in fullResultsMeansStds:
                    fullResultsMeansStds[sub] = {'Means':[],'Stds':[]}

                fullResultsMeansStds[sub]['Means'].append(list(replicateData['Clumped Deltas Relative'])[subIdx])
                fullResultsMeansStds[sub]['Stds'].append(list(replicateData['Clumped Deltas Relative Error'])[subIdx])

        with open(str(cwd) + '/Read Process CSV/' + currentMNKey + 'ProcessedResults.json', 'w', encoding='utf-8') as f:
            json.dump(fullResultsMeansStds, f, ensure_ascii=False, indent=4)