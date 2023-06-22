import json
import copy
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import cm 

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

def prettyYLabel(string):
    if string == '74High':
        return '$74_{High}$'
    elif string == '74Low':
        return "$74_{Low}$"
    else:
        return string
'''
To proceed we need 4 datasets: observed sample/standard and predicted sample/standard. 

The problem is complicated by the fact that the observed and predicted must contain the same observations, i.e. for the observation we will not observe many peaks (due to unresolved or low abundance issues) and we must account for these. To deal with this, we proceed in the following manner:

1a) Calculate all isotopologues of methionine and select the subset we are interested in ***this is not necessary for the computation but speeds later steps. 

1) Read in the observed datasets; perform the sample standard comparison. 

2) Generate a 'perfect' dataset where all beams are observed.

3) Check which beams are observed in the perfect dataset which are not in the observed dataset, and track these as forbidden peaks. 

4) Generate the predicted datasets using (3) as input 'forbidden peaks', and perform a sample/standard comparison.  

5) Construct and output plots
'''
'''
(1a) Precalculate the isotopologues of methionine and select only the ones we need
'''
precalc = False
if precalc:
    #precalculate isotopologues of interest
    deltasPreCalc = [0] * 16
    fragSubset = ['full','133','104','102','61','56']
    thisMethionine = metInit.initializeMethionine(deltasPreCalc, fragSubset, smallFrags = True, printHeavy = False)
    byAtom = ci.inputToAtomDict(thisMethionine['molecularDataFrame'], overrideIsoDict = {}, disable = False, M1Only = False)
    MN = ci.massSelections(byAtom, massThreshold = 4)

    overrideIsoDict = ci.pullOutIsotopologues(MN)
        
    with open(str(cwd) + '/Reference Files/Stored Isotopologues  8 Frags.json', 'w', encoding='utf-8') as f:
        json.dump(overrideIsoDict, f, ensure_ascii=False, indent=4)

with open(str(cwd) + '/Reference Files/Stored Isotopologues 8 Frags.json', 'r') as j:
    overrideIsoDict = json.loads(j.read())
'''
1) Read in the observed dataset; perform sample/standard comparison.
'''
OBSSmpStdComparison = {}
#Need this to get sizes for points later
avgMNAbund = {}

processFragKeys = {'133':'133',
                   '104':'104',
                   '102':'102',
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
fragSubset = ['133','104','102','61','56']

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

#Predicted sample composition
clumpD =  {'01':{'Sites':['Cmethyl','Cgamma'],'Amount':1e-6}}
np.random.seed(0)

met1 = [41.05,-23.7,-24.0,-24.0,-24.3,0,0,2.215,0,0,0,0,0,0,0,0]
met2 = [np.random.normal(x,5) for x in met1[:9]] + [np.random.normal(x,20) for x in met1[9:]]
met2 = [49.87026172983832, -21.69921395816388, -19.106310079471303, -12.79553400399271, -14.962210049250164, 0.1, 0.1, 1.4582139585115104, -2.064377035871157, 8.211970038767447, 2.8808714232175596, 29.085470139259503, 15.220754502939869, 2.4335003298565683, 8.877264654908513, 6.673486547485337]

print(met2)

metSmpDict = {'Met_1':{'Deltas':met1,'Clumps':{}},'Met_2':{'Deltas':met2,'Clumps':{}}}

#Predicted standard composition
deltasStd = [-53.9,-23.7,-24.0,-24.0,-24.3,0,0,2.215,0,0,0,0,0,0,0,0]

byHypothesis = {}
fragSubset = ['133','104','102','61','56']
#Try both hypotheses
for thisSmpHypothesisKey, thisSmpHypothesisData in metSmpDict.items():
    thisDeltas = thisSmpHypothesisData['Deltas']
    thisClumps = thisSmpHypothesisData['Clumps']

    #Generate sample data
    methionineSmp = metInit.initializeMethionine(thisDeltas, fragSubset, smallFrags = True,  printHeavy = True)

    predictedMeasurementSmp, MNDictSmp, FFSmp = metInit.simulateMeasurement(methionineSmp, overrideIsoDict = overrideIsoDict, clumpD = thisClumps, massThreshold = 4, disableProgress = True, omitMeasurements = forbiddenPeaks)

    #Generate standard data
    methionineStd = metInit.initializeMethionine(deltasStd, fragSubset, smallFrags = True, printHeavy = False)
    predictedMeasurementStd, MNDictStd, FFStd = metInit.simulateMeasurement(methionineStd, overrideIsoDict = overrideIsoDict,massThreshold = 4, disableProgress = True, omitMeasurements = forbiddenPeaks)

    #Compare
    SIMSmpStd = smpStdCompare(predictedMeasurementSmp, predictedMeasurementStd, OBSSmpStdComparison)

    OBSSIMCompare = ObsVsSim(OBSSmpStdComparison, SIMSmpStd)

    byHypothesis[thisSmpHypothesisKey] = copy.deepcopy(OBSSIMCompare)

'''
5) Construct and output plots
'''
allLists = []
expKeys = []
allRSELists = []

#Pull out relevant data into lists
for expKey, expData in byHypothesis.items():
    expKeys.append(expKey)
    fullKeys = []
    thisData = []
    thisRSE = []
    for MNKey, MNData in expData.items():
        for fragKey, fragData in MNData.items():
            subMNKeys = [MNKey + '_' + fragKey + '_' + x for x in fragData['Subs']]
            fullKeys += subMNKeys
            thisData += copy.deepcopy(fragData['Compare'])
            thisRSE += copy.deepcopy(fragData['RSE OBS'])
        
    allLists.append(copy.deepcopy(thisData))
    allRSELists.append(thisRSE)

#Use these to generate dataframes
thisDf = pd.DataFrame(allLists, columns = fullKeys, index = expKeys)
thisDf = thisDf.T

thisRSEDf = pd.DataFrame(allRSELists, columns = fullKeys, index = expKeys)
thisRSEDf = thisRSEDf.T

#Calculated WRMSE
WRMSE = np.sqrt(np.mean(1 / thisRSEDf['Met_1'].values**2 * thisDf['Met_1'].values**2))
WRMSEPerturb = np.sqrt(np.mean(1 / thisRSEDf['Met_2'].values**2 * thisDf['Met_2'].values**2))

#Generate panel B
fig, ax = plt.subplots(figsize = (6,4))

ax.plot(1 / thisRSEDf['Met_1'].values**2 * thisDf['Met_1'].values**2, c = 'k',linestyle = '-', label = "Computed Met; WRMSE = " + f'{WRMSE:.1f}')
ax.plot(1 / thisRSEDf['Met_2'].values**2 * thisDf['Met_2'].values**2, c = 'tab:orange',linestyle = '-', label = "Perturbed Met; WRMSE = " + f'{WRMSEPerturb:.1f}')

ax.set_ylabel("$(\dfrac{\delta^{OBS}_{SIM}}{\sigma_{PAE}})^2$", fontsize = 12)
ax.set_xlabel("Variable Number")
ax.legend(loc = 'upper left')

ax.axvspan(xmin=19, xmax = 40, alpha = 0.3, fc = 'gray')
ax.axvspan(xmin=67, xmax = 120, alpha = 0.3, fc = 'gray')
ax.set_xlim(-2,102)

fig.savefig(str(cwd) + '/Fig7_Hypothesis_Testing/Fig7_PanelB.jpeg', bbox_inches = 'tight', dpi = 1000)

#Begin generating panel C
#Set explicit locations in x, y space for each substitution and fragment
placementDict = {'M1': {'Unsub':0,'15N':1,'33S':2,'13C':3,'D':4},
                    'M2': {'Unsub':0,'15N':1,'33S':2,'13C':3,'18O':4,'34S':5,'13C-13C':6,'13C-15N':7,'13C-D':8,'13C-33S':9},
                    'M3': {'Unsub':0,'15N':1,'33S':2,'13C':3,'D':4,'18O':5,'34S':6,'13C-13C':7,'13C-34S':8,'34S-15N':9,'18O-33S':10,'34S-D':11,'13C-18O':12,'13C-13C-13C':13},
                    'M4': {'Unsub':0,'15N':1,'13C':2,'18O':3,'34S':4,'36S':5,'13C-13C':6,'13C-15N':7,'13C-18O':8,'13C-D':9,'18O-34S':10,'13C-34S':11,'13C-13C-34S':12,'13C-34S-D':13,'34S-15N':14,'13C-13C-18O':15,'18O-18O':16}}
fragPlacementDict = {'133':0,'104':1,'102':2,'61':3,'56':4}

for expKey, expData in byHypothesis.items():
    #Generate a plot for each MNKey
    for MNKey, MNData in expData.items():
        #Get placement dict
        thisPlacementDict = placementDict[MNKey]

        cBarLims = (-7,7)
        sns.set(style='whitegrid', rc = {'legend.labelspacing': 2.5})
        RdBu = cm.get_cmap('RdBu_r', 256)
        figSizes = {'M1':(3,3.7),'M2':(5,3.7),'M3':(6,3.7),'M4':(7,3.7)}

        #Generate a plot for each MNKey
        fig, ax = plt.subplots(nrows = 1, ncols = 1, figsize = figSizes[MNKey], dpi = 300)
        ax.grid(True, zorder = 0)

        #Generate proper labels with subscripts/superscripts for each fragment & substitution
        yticks = ['133','104','102','61','56']
        xticksUgly = list(thisPlacementDict.keys())
        xticks = [prettyLabel(x) for x in xticksUgly]

        #no yticks for M2 and M3
        if (MNKey == 'M2') or (MNKey == 'M3') or (MNKey == 'M4'):
            yticks = [''] * 8

        #each observation is plotted individually via this loop
        for fragKey, fragData in MNData.items():    
            for subIdx, subKey in enumerate(fragData['Subs']):
                thisDelta = fragData['Compare'][subIdx]
                thisErr = fragData['RSE OBS']
                #need to come back and add this information in somehow
                MNRelAbund = avgMNAbund[MNKey][fragKey][subKey]['Smp']
                #Compute the size of the circle based on the MNRelAbund
                thisSize = 10 * min(10 + (MNRelAbund / 0.5) * 20, 30)
                #Size based on error bars
                #thisSize = 10 * max(  10 + 20 * ((3-thisErr) / 3)  , 10)
                
                #Compute the color of the circle based on the delta value
                scaledDelta = np.abs(cBarLims[0]) + thisDelta
                scaleHeight = np.abs(cBarLims[0]) + np.abs(cBarLims[1])
                fraction = scaledDelta / scaleHeight
                thisColor = RdBu(fraction)
                
                #Plot the point
                ax.scatter(thisPlacementDict[subKey], fragPlacementDict[fragKey], s = thisSize, color = thisColor, edgecolors = 'k',zorder = 3)

        #Optionally add a legend
        legend = False
        if legend:
            #Show these error setpoints in the legend
            errorSetpoints = [0,0.17,0.33,0.5]
            for errorSet in errorSetpoints: 
                thisSize = 10 * min(10 + (errorSet / 0.5) * 20, 30)
                #Plot points of this size way off the scale, so we can use them in the legend
                ax.scatter(100,100,s = thisSize, color = 'w', edgecolors = 'k', label = str(errorSet))

            #Generate the legend for this dummy plot
            leg = ax.legend(loc=(1.04,0.1), fontsize = 20, labelspacing = 0.8)
            leg.set_title('M+N Relative\n  Abundance',prop={'size':'20'})

        #Set x & y ticks, title
        ax.set_xticks(range(len(xticks)))
        ax.set_xticklabels(xticks, fontsize = 16, rotation = 90)

        ax.set_yticks(range(len(yticks)))
        ax.set_yticklabels(yticks, fontsize = 16)

        ax.set_title("M+" + MNKey[1], fontsize = 18)

        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.spines['bottom'].set_color('k')
        ax.spines['left'].set_color('k')

        #Set x & y limits
        ax.set_xlim(-0.5,len(xticks)-0.5)
        ax.set_ylim(-0.5,4.5)

        ax.invert_yaxis()

        #Output panel C
        fig.savefig(str(cwd) + '/Fig7_Hypothesis_Testing/' + "Fig7" + expKey + MNKey + ".jpeg", bbox_inches = 'tight')