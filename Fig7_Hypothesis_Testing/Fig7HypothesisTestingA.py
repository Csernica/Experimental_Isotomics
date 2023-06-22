import json
import copy
import numpy as np
import itertools
from operator import itemgetter
import pandas as pd 
import matplotlib.pyplot as plt

#relative imports of Isotomics
import sys
import os
from pathlib import Path
cwd = Path().resolve()

sys.path.insert(1, os.path.join(cwd, 'Isotomics Scripts'))

import readInput as ri
import calcIsotopologues as ci
import initializeMethionine as metInit

def prettyLabel(string):
    '''
    Add nicer labels to the plot
    '''
    if string == 'Unsub':
        return "Unsub"
    split = string.split('-')
    combined = []
    for sub in split:
        if sub == 'D':
            combined.append('$^{2}H$')
        else:
            num = sub[:-1]
            pretty = '$^{' + num + '}' + sub[-1] + "$"
            combined.append(pretty)
    return ''.join(combined)

'''
To proceed we need 4 datasets: observed sample/standard and predicted sample/standard. 

The problem is complicated by the fact that the observed and predicted must contain the same observations, i.e. for the observation we will not observe many peaks (due to unresolved or low abundance issues) and we must account for these. To deal with this, we proceed in the following manner:

1) Read in the observed datasets; perform the sample standard comparison. 

1a) Calculate all isotopologues of methionine and select the subset we are interested in ***this is not necessary for the computation but speeds later steps. 

2) Generate a 'perfect' dataset where all beams are observed.

3) Check which beams are observed in the perfect dataset which are not in the observed dataset, and track these as forbidden peaks. 

4) Generate the predicted datasets using (3) as input 'forbidden peaks'; perform a sample/standard comparison of these datasets. 

5) Construct and output plots. 
'''
'''
(1a) Precalculate the isotopologues of methionine and select only the ones we need
'''
precalc = True
if precalc:
    #precalculate isotopologues of interest
    deltasPreCalc = [0] * 16
    fragSubset = ['full','133','104','102','87','74High','74Low','61','56']
    thisMethionine = metInit.initializeMethionine(deltasPreCalc, fragSubset, smallFrags = True, printHeavy = False)
    byAtom = ci.inputToAtomDict(thisMethionine['molecularDataFrame'], overrideIsoDict = {}, disable = False, M1Only = False)
    MN = ci.massSelections(byAtom, massThreshold = 4)

    overrideIsoDict = ci.pullOutIsotopologues(MN)
        
    with open(str(cwd) + '/Reference Files/Stored Isotopologues 8 Frags.json', 'w', encoding='utf-8') as f:
        json.dump(overrideIsoDict, f, ensure_ascii=False, indent=4)

with open(str(cwd) + '/Reference Files/Stored Isotopologues 8 Frags.json', 'r') as j:
    overrideIsoDict = json.loads(j.read())
'''
1) Read in the observed dataset; perform sample/standard comparison.
'''
#output
OBSSmpStdComparison = {}
#Need this to get sizes for points later
avgMNAbund = {}

processFragKeys = {'133':'133',
                   '104':'104',
                   '102':'102',
                   '74High':'74High',
                   '74Low':'74Low',
                   '87':'87',
                   '61':'61',
                   '56':'56'}

for MNKey in ['M1','M2','M3','M4']:
    with open(str(cwd) + '/Read Process CSV/' + MNKey + 'Results.json') as f:
        sampleOutputDict = json.load(f)
        
    #'standard' is used in this function for forward model standardization in 'isotopologue reconstruction mode'; set it false here. 
    replicateData = ri.readObservedData(sampleOutputDict, MNKey = MNKey, theory = {},
                                    standard = [False] * 7,
                                    processFragKeys = processFragKeys)

    replicateDataKeys = list(replicateData.keys())

    observedAvg = {'Std':{},'Smp':{}}

    #Assign keys
    for repKey, repData in replicateData.items():
        idx = replicateDataKeys.index(repKey)
        if idx % 2 == 0:
            smpStdStr = 'Std'
        else:
            smpStdStr = 'Smp'

        #Fill in the 'standard' and 'sample' with all 4 (3) replicates
        for fragKey, fragData in repData[MNKey].items():
            if fragKey not in observedAvg[smpStdStr]:
                observedAvg[smpStdStr][fragKey] = {}
                observedAvg[smpStdStr][fragKey]['Observed Abundance'] = []
                observedAvg[smpStdStr][fragKey]['Observed RSE'] = []
                
            observedAvg[smpStdStr][fragKey]['Subs'] = fragData['Subs']
            observedAvg[smpStdStr][fragKey]['Observed Abundance'].append(fragData['Observed Abundance'].copy())

            thisRse = np.array(fragData['Error']) / np.array(fragData['Observed Abundance'])
            observedAvg[smpStdStr][fragKey]['Observed RSE'].append(thisRse.copy())

    #take the average value of these for both sample and standard
    for smpStdStr, value in observedAvg.items():
        for fragKey, fragData in value.items():
            fragData['Observed Abundance'] = np.array(fragData['Observed Abundance']).mean(axis = 0)
            fragData['Observed RSE'] = np.array(fragData['Observed RSE']).mean(axis = 0)
    
    #Compare average sample and standard values
    thisSmpStdComparison = {}
    thisAvgMNAbund = {}
    for smpStdStr, value in observedAvg.items():
        for fragKey, fragData in value.items():
            if fragKey not in thisSmpStdComparison:
                thisSmpStdComparison[fragKey] = {}
            if fragKey not in thisAvgMNAbund:
                thisAvgMNAbund[fragKey] = {}
                for subIdx, subKey in enumerate(fragData['Subs']):
                    if subKey not in thisSmpStdComparison[fragKey]:
                        thisSmpStdComparison[fragKey][subKey] = {}
                    if subKey not in thisAvgMNAbund[fragKey]:
                        thisAvgMNAbund[fragKey][subKey] = {}

                    subComparison = observedAvg['Smp'][fragKey]['Observed Abundance'][subIdx] / observedAvg['Std'][fragKey]['Observed Abundance'][subIdx]
                    errorComparison = np.sqrt(observedAvg['Smp'][fragKey]['Observed RSE'][subIdx]**2 + observedAvg['Smp'][fragKey]['Observed RSE'][subIdx]**2)

                    thisSmpStdComparison[fragKey][subKey]['Mean'] = subComparison
                    thisSmpStdComparison[fragKey][subKey]['RSE'] = errorComparison

                    thisAvgMNAbund[fragKey][subKey] = {'Smp':observedAvg['Smp'][fragKey]['Observed Abundance'][subIdx], 
                    'Std': observedAvg['Std'][fragKey]['Observed Abundance'][subIdx],
                    'Smp Error':observedAvg['Smp'][fragKey]['Observed RSE'][subIdx],
                    'Std Error':observedAvg['Std'][fragKey]['Observed RSE'][subIdx]}

    #copy this to an output dictionary
    OBSSmpStdComparison[MNKey] = copy.deepcopy(thisSmpStdComparison)
    avgMNAbund[MNKey] = copy.deepcopy(thisAvgMNAbund)

'''
(2) Generate a 'perfect' dataset where all beams are observed
'''
deltasPerfect = [0] * 16
fragSubset = ['133','104','102','87','74High','74Low','61','56']

methioninePerfect = metInit.initializeMethionine(deltasPerfect, fragSubset, smallFrags = True, printHeavy = False)

predictedMeasurementPerfect, MNDictSmp, FFSmp = metInit.simulateMeasurement(methioninePerfect, overrideIsoDict = overrideIsoDict,abundanceThreshold = 0, massThreshold = 4,disableProgress = False)
'''
3) Check which beams are observed in the perfect dataset which are not in the observed dataset, and track these as forbidden peaks. 
'''
forbiddenPeaks = {}
for MNKey, MNData in predictedMeasurementPerfect.items():
    if MNKey in OBSSmpStdComparison.keys():
        forbiddenPeaks[MNKey] = {}
        for fragKey, fragData in MNData.items():
            forbiddenPeaks[MNKey][fragKey] = []
            for subKey, subData in fragData.items():
                if subKey not in OBSSmpStdComparison[MNKey][fragKey]:
                    forbiddenPeaks[MNKey][fragKey].append(subKey)
'''
4) Generate the predicted datasets using (3) as input 'forbidden peaks'. This is a limitation in the sense that unresolved peaks may contribute to the abundances of some observed peaks, and we do not model this. Note that we reuse the precalculated M+1 through M+4 isotopologues for both sample and standard.
'''
def smpStdCompare(predictedMeasurementSmp, predictedMeasurementStd, OBSSmpStdComparison):
    '''
    Helper function for 4; performs a sample/standard comparison of predicted datasets. Checks OBSSmpStdComparison to make sure we are including the same sets of peaks.
    '''
    SIMSmpStdComparison = {}

    #Iterate through the sample observation
    for MNKey, MNData in predictedMeasurementSmp.items():
        #Check that we really have this MNKey
        if MNKey in OBSSmpStdComparison.keys():
            if MNKey not in SIMSmpStdComparison:
                SIMSmpStdComparison[MNKey] = {}
            #Iterate by fragment
            for fragKey, fragData in MNData.items():
                if fragKey not in SIMSmpStdComparison[MNKey]:
                    SIMSmpStdComparison[MNKey][fragKey] = {}
                #Iterate by sub
                for subKey, subData in fragData.items():
                    #Perform comparison
                    smpAbund = subData['Adj. Rel. Abundance']
                    stdAbund = predictedMeasurementStd[MNKey][fragKey][subKey]['Adj. Rel. Abundance']
                    predSmpStd = smpAbund / stdAbund

                    SIMSmpStdComparison[MNKey][fragKey][subKey] = predSmpStd

    #brief check the datasets contain the same information
    for MNKey, MNData in SIMSmpStdComparison.items():
        for fragKey, fragData in MNData.items():
            for subKey, subData in fragData.items():
                if subKey not in OBSSmpStdComparison[MNKey][fragKey]:
                    print(MNKey + ' ' + fragKey + ' ' + subKey + ' in SIM but not OBS')

    for MNKey, MNData in OBSSmpStdComparison.items():
        for fragKey, fragData in MNData.items():
            for subKey, subData in fragData.items():
                if subKey not in SIMSmpStdComparison[MNKey][fragKey]:
                    print(MNKey + ' ' + fragKey + ' ' + subKey + ' in OBS but not SIM')

    return SIMSmpStdComparison

def ObsVsSim(OBSSmpStdComparison, SIMSmpStdComparison):
    '''
    A helper function for 4; compares the observed sample standard comparison to the simulated sample standard comparison.
    '''
    OBSvsSIM = {}

    #Iterate through by MNKey
    for MNKey, MNData in OBSSmpStdComparison.items():
        if MNKey not in OBSvsSIM:
            OBSvsSIM[MNKey] = {}
        #By fragment
        for fragKey, fragData in MNData.items():
            if fragKey not in OBSvsSIM[MNKey]:
                OBSvsSIM[MNKey][fragKey] = {'Subs':[],'Compare':[],'RSE OBS':[]}
            #and by substitution
            for subKey, subData in fragData.items():
                #Perform comparison
                thisSIM = SIMSmpStdComparison[MNKey][fragKey][subKey]
                thisOBS = subData['Mean']
                thisComp = 1000*(thisOBS / thisSIM -1)
                thisErr = 1000* subData['RSE'] 

                #Add to output
                OBSvsSIM[MNKey][fragKey]['Subs'].append(subKey)
                OBSvsSIM[MNKey][fragKey]['Compare'].append(thisComp)
                OBSvsSIM[MNKey][fragKey]['RSE OBS'].append(thisErr)

    return OBSvsSIM

#Start by generating datasets; fix delta values.
#Predicted sample composition
deltasSmp = [41.05,-23.7,-24.0,-24.0,-24.3,0,0,2.215,0,0,0,0,0,0,0,0]
#Predicted standard composition
deltasStd = [-53.9,-23.7,-24.0,-24.0,-24.3,0,0,2.215,0,0,0,0,0,0,0,0]

#Check all permutations of possible 'clean' fragment geometries of 74 fragment (clean in that they either sample a site or do not sample a site, no in between).
perms = list(set(itertools.permutations([0,0,1,1,1])))
sortedPerms = sorted(perms, key=itemgetter(0,1,2,3,4))
filtered = []
for perm in sortedPerms:
    thisFilter = ['x' if x == 0 else 1 for x in perm]
    filtered.append(thisFilter)

byHypothesis = {}
#Test every permutation
for thisCarbonComp in filtered:
    print(thisCarbonComp)
    #Generate the fragmentation vector
    this74Hypothesis = {'01':{'subgeometry':thisCarbonComp + [1,'x','x',1,'x',1,1,1,1,'x',1],'relCont':1}}

    #Generate sample data
    methionineSmp = metInit.initializeMethionine(deltasSmp, fragSubset, smallFrags = True, hypothFrag74High = this74Hypothesis, printHeavy = False)

    predictedMeasurementSmp, MNDictSmp, FFSmp = metInit.simulateMeasurement(methionineSmp, overrideIsoDict = overrideIsoDict,massThreshold = 4, disableProgress = True, omitMeasurements = forbiddenPeaks)

    #Generate standard data
    methionineStd = metInit.initializeMethionine(deltasStd, fragSubset, smallFrags = True, hypothFrag74High = this74Hypothesis, printHeavy = False)
    predictedMeasurementStd, MNDictStd, FFStd = metInit.simulateMeasurement(methionineStd, overrideIsoDict = overrideIsoDict,massThreshold = 4, disableProgress = True, omitMeasurements = forbiddenPeaks)

    #Compare simulated sample and standard
    SIMSmpStd = smpStdCompare(predictedMeasurementSmp, predictedMeasurementStd, OBSSmpStdComparison)

    #Compare simulated sample/std versus observed sample/standard
    OBSSIMCompare = ObsVsSim(OBSSmpStdComparison, SIMSmpStd)

    #Output to plot
    byHypothesis[''.join([str(x) for x in thisCarbonComp])] = copy.deepcopy(OBSSIMCompare)
'''
5) Create and output plots
'''
allLists = []
expKeys = []
allRSELists = []

#Pull out relevant data from the simulation (i.e., the 74High fragment comparisons)
for expKey, expData in byHypothesis.items():
    expKeys.append(expKey)
    fullKeys = []
    thisData = []
    thisRSE = []
    for MNKey, MNData in expData.items():
        thisFragData = MNData['74High']
        subMNKeys = [MNKey + x for x in thisFragData['Subs']]
        fullKeys += subMNKeys
        thisData += copy.deepcopy(thisFragData['Compare'])
        thisRSE += copy.deepcopy(thisFragData['RSE OBS'])
        
    allLists.append(copy.deepcopy(thisData))
    allRSELists.append(thisRSE)

#Put this information into a dataframe for easier access
thisDf = pd.DataFrame(allLists, columns = fullKeys, index = expKeys)
thisDf = thisDf.T
#Put errors into a separate dataframe
thisRSEDf = pd.DataFrame(allRSELists, columns = fullKeys, index = expKeys)
thisRSEDf = thisRSEDf.T

#Calculate WRMSE for two fragments, one with and one without. (We did all 10 geometries, but the major differences occur only for having vs not having the methyl carbon, so we pick an arbitrary two which have/do not have that carbon.)
WRMSE = np.sqrt(np.mean(1 / thisRSEDf['1x11x'].values**2 * thisDf['1x11x'].values**2))
WRMSEMethyl = np.sqrt(np.mean(1 / thisRSEDf['xx111'].values**2 * thisDf['xx111'].values**2))

#Plot
fig, ax = plt.subplots(figsize = (6*0.88,4*0.88), dpi = 120)
ax.errorbar(range(len(thisDf)), thisDf['1x11x'],thisRSEDf['1x11x'], fmt = 'o', mfc = 'None', mec = 'k', ecolor = 'k',label = 'Not Including Methyl; WRMSE: ' + f'{WRMSE:.2f}')
ax.errorbar(range(len(thisDf)), thisDf['x111x'],thisRSEDf['x111x'], fmt = 's', mfc = 'None', mec = 'tab:blue', ecolor = 'tab:blue',label = 'Including Methyl; WRMSE: ' + f'{WRMSEMethyl:.2f}')
ax.plot([-5,len(thisDf)+5], [0,0], c = 'k')
ax.set_xlim(-1, len(thisDf))

ax.set_xticks(range(len(thisDf)))
xtickLabels = [x[:2] + ' ' + prettyLabel(x[2:]) for x in thisDf.index]
ax.set_xticklabels(xtickLabels, rotation = 45)

ax.legend()

ax.set_ylabel("$\delta^{OBS}_{SIM}$")

#Save the output figure
fig.savefig(str(cwd) + '/Fig7_Hypothesis_Testing/Fig7_PanelA.jpeg', bbox_inches='tight',dpi = 1000)