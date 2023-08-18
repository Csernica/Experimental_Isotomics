import os
import json
import numpy as np
import copy

#relative imports of Isotomics
import sys
import os
from pathlib import Path
cwd = Path().resolve()

sys.path.insert(1, os.path.join(cwd, 'Isotomics Scripts'))

import initializeMethionine as metInit
import fragmentAndSimulate as fas
import solveSystem as ss
import readInput as ri
import calcIsotopologues as ci

'''
This code is divided into the following sections:

1a) Precalculate isotopologues and save them for later. Here, do both the '5 frags' version with 13 sites and '8 frags' version with 16 sites. 

2) Prepare to make the 'observed isotopologues' portion. Calclulate how many isotopologues are constrained assuming a perfect measurement relative to how many isotopologues exist. 

3) Construct the figure for the 'isotopologues observed' plot

4) Generate a perfect dataset for 8 fragments, and compute how many beams we observe relative to how many we could observe
'''
'''
1a) Precalculate isotopologues and save them for later; this saves time. 
'''
precalc = False
if precalc:
    #precalculate isotopologues of interest: 5 fragments
    deltasPreCalc = [0] * 13
    fragSubset = ['full','133','104','102','61','56']
    thisMethionine = metInit.initializeMethionine(deltasPreCalc, fragSubset, printHeavy = False)
    byAtom = ci.inputToAtomDict(thisMethionine['molecularDataFrame'], overrideIsoDict = {}, disable = False, M1Only = False)
    MN = ci.massSelections(byAtom, massThreshold = 4)
    overrideIsoDict = ci.pullOutIsotopologues(MN)
        
    with open(str(cwd) + '/Reference Files/Stored Isotopologues 5 Frags.json', 'w', encoding='utf-8') as f:
        json.dump(overrideIsoDict, f, ensure_ascii=False, indent=4)


    #precalculate isotopologues of interest: 8 fragments
    deltasPreCalc = [0] * 16
    fragSubset = ['full','133','104','102','87','74High','74Low','61','56']
    thisMethionine = metInit.initializeMethionine(deltasPreCalc, fragSubset, smallFrags = True, printHeavy = False)
    byAtom = ci.inputToAtomDict(thisMethionine['molecularDataFrame'], overrideIsoDict = {}, disable = False, M1Only = False)
    MN = ci.massSelections(byAtom, massThreshold = 4)
    overrideIsoDict = ci.pullOutIsotopologues(MN)
        
    with open(str(cwd) + '/Reference Files/Stored Isotopologues 8 Frags.json', 'w', encoding='utf-8') as f:
        json.dump(overrideIsoDict, f, ensure_ascii=False, indent=4)

with open(str(cwd) + '/Reference Files/Stored Isotopologues 5 Frags.json', 'r') as j:
    overrideIsoDict5 = json.loads(j.read())

with open(str(cwd) + '/Reference Files/Stored Isotopologues 8 Frags.json', 'r') as j:
    overrideIsoDict8 = json.loads(j.read())
'''
2) Prepare to make the 'observed isotopologues' portion. Calclulate how many isotopologues are constrained assuming a perfect measurement relative to how many isotopologues exist. Uses 5 fragments. 
'''

#Initialize dictionaries to hold output
resultsAbbr = {}
deltasPerfect = [0] * 13
fragSubset = ['full','133','104','102','61','56']
thisMethionine = metInit.initializeMethionine(deltasPerfect, fragSubset,printHeavy = False)


#Simulate sample and standard datasets, then standard forward model. 
predictedMeasurementSmp, MNDictSmp, FFSmp = metInit.simulateMeasurement(thisMethionine,
                                                        overrideIsoDict = overrideIsoDict5,
                                                       abundanceThreshold = 0,
                                                       massThreshold = 4, outputPath = 'Fig 5 Sample',
                                                       disableProgress = True)

predictedMeasurementStd, MNDictStd, FFStd = metInit.simulateMeasurement(thisMethionine,
                                                        overrideIsoDict = overrideIsoDict5,
                                                       abundanceThreshold = 0,
                                                       massThreshold = 4, outputPath = 'Fig 5 Standard',
                                                       disableProgress = True)

predictedMeasurementFMStd, MNDictFMStd, FFFMStd = metInit.simulateMeasurement(thisMethionine,
                                                        overrideIsoDict = overrideIsoDict5,
                                                       abundanceThreshold = 0,
                                                       massThreshold = 4,
                                                       disableProgress = True)
    
#Read in simulated sample and standard data, generated above.  
standardJSON = ri.readJSON('Fig 5 Standard.json')
processStandard = ri.readComputedData(standardJSON, error = 0, theory = predictedMeasurementFMStd)

sampleJSON = ri.readJSON('Fig 5 Sample.json')
processSample = ri.readComputedData(sampleJSON, error = 0)
UValuesSmp = ri.readComputedUValues(sampleJSON, error = 0)

ri.checkSampleStandard(processSample, processStandard)
#solve M+1
isotopologuesDict = fas.isotopologueDataFrame(MNDictFMStd, thisMethionine['molecularDataFrame'])
OCorrection = ss.OValueCorrectTheoretical(predictedMeasurementFMStd, processSample, massThreshold = 4)
M1Results = ss.M1MonteCarlo(processStandard, processSample, OCorrection, isotopologuesDict,
                            thisMethionine['fragmentationDictionary'], perturbTheoryOAmt = 0,
                            N = 1, GJ = False, debugMatrix = False, disableProgress = True)


#Process results from M+N relative abundance space to U value space, using the UM+1 value calculated via the sub(s) in UMNSub. 
processedResults = ss.processM1MCResults(M1Results, UValuesSmp, isotopologuesDict, thisMethionine['molecularDataFrame'], disableProgress = True,
                                        UMNSub = ['13C'])
    
#Update the dataframe with results. 
ss.updateSiteSpecificDfM1MC(processedResults, thisMethionine['molecularDataFrame'])
    
M1Df = thisMethionine['molecularDataFrame'].copy()
M1Df['deltas'] = M1Df['VPDB etc. Deltas']
    
#Initialize dictionary for M+N solutions
MNSol = {'M2':0,'M3':0,'M4':0}
#Determine which substitutions are used for U Value scaling
UMNSubs = {'M2':['34S'],'M3':['34S15N'],'M4':['36S']}

for MNKey in MNSol.keys():
    Isotopologues = isotopologuesDict[MNKey]

    #Solve for specific M+N isotopologues
    results, comp, GJSol, meas = ss.MonteCarloMN(MNKey, Isotopologues, processStandard, processSample, OCorrection, thisMethionine['fragmentationDictionary'], N = 1, disableProgress = True)

    dfOutput = ss.checkSolutionIsotopologues(GJSol, Isotopologues, MNKey, numerical = False)
    #Many will be codependent, not individually constarined. Determine which isotopologues have actually been solved for.
    nullSpaceCycles = ss.findNullSpaceCycles(comp, Isotopologues)
    actuallyConstrained = ss.findFullyConstrained(nullSpaceCycles)
    
    #Scale results to U values, update the dataframe.
    processedResults = ss.processMNMonteCarloResults(MNKey, results, UValuesSmp, dfOutput, M1Df, MNDictFMStd, UMNSub = UMNSubs[MNKey], disableProgress = True)
    dfOutput = ss.updateMNMonteCarloResults(dfOutput, processedResults)
    MNSol[MNKey] = dfOutput.loc[dfOutput.index.isin(actuallyConstrained)].copy()
        
    if os.path.exists('Fig 5 Sample.json'):
        os.remove('Fig 5 Sample.json')
    if os.path.exists('Fig 5 Standard.json'):
        os.remove('Fig 5 Standard.json')
        
#Generate an abbreviated results file for easy use.
resultsAbbr = {'M1':{},'M2':{},'M3':{},'M4':{}}
resultsAbbr['M1']["Values"] = list(M1Df['Relative Deltas'])
resultsAbbr['M1']["Errors"] = list(M1Df['Relative Deltas Error'])

resultsAbbr['M2']['Identity'] = list(MNSol['M2'].index)
resultsAbbr['M2']['Values'] = list(MNSol['M2']['Clumped Deltas Relative'])
resultsAbbr['M2']['Errors'] = list(MNSol['M2']['Clumped Deltas Relative Error'])

resultsAbbr['M3']['Identity'] = list(MNSol['M3'].index)
resultsAbbr['M3']['Values'] = list(MNSol['M3']['Clumped Deltas Relative'])
resultsAbbr['M3']['Errors'] = list(MNSol['M3']['Clumped Deltas Relative Error'])

resultsAbbr['M4']['Identity'] = list(MNSol['M4'].index)
resultsAbbr['M4']['Values'] = list(MNSol['M4']['Clumped Deltas Relative'])
resultsAbbr['M4']['Errors'] = list(MNSol['M4']['Clumped Deltas Relative Error'])

'''
3) Construct the figure for the 'isotopologues observed' plot
'''
siteObs = {}
for MNKey in ['M2','M3','M4']:
    siteObs[MNKey] = {'Obs':0,'Possibly Observed':0}
    possiblyObservedIsotopologues = resultsAbbr[MNKey]['Identity']
    with open(str(cwd) + '/Read Process CSV/' + MNKey + 'ProcessedResults.json', 'r') as f:
        thisMNObs = json.load(f)

        for isotopologueID in possiblyObservedIsotopologues:
            siteObs[MNKey]['Possibly Observed'] += 1
            if isotopologueID in thisMNObs:
                siteObs[MNKey]['Obs'] += 1

    siteObs[MNKey]['Total Isotopologues'] = len(MNDictSmp[MNKey])

with open(str(cwd) + '/Read Process CSV/M1ProcessedResults.json', 'r') as f:
    thisMNObs = json.load(f)

M1Amt = len(thisMNObs['Mean'])
siteObs['M1'] = {'Obs':len(thisMNObs['Mean']),'Possibly Observed':len(resultsAbbr['M1']['Values']),'Total Isotopologues':len(thisMethionine['molecularDataFrame'])}

import matplotlib.pyplot as plt
import numpy as np
fig, axes = plt.subplots(nrows = 1, ncols = 4, figsize = (9,4), dpi = 150)
fig.set_facecolor('white')
blue3 = [0.7161860822760477,0.8332026143790849,0.916155324875048]
blue6 = [0.2909803921568628,0.5945098039215686,0.7890196078431373]
blue8 = [0.09019607843137256,0.39294117647058824,0.6705882352941177]
colors = [blue8, blue6, blue3]
labels = ['# Isotopologues Observed', '# Observable Isotopologues', '# Isotopologues Total']
for MNIdx, MNKey in enumerate(['M1','M2','M3','M4']):
    cAx = axes[MNIdx]
    nObs = siteObs[MNKey]['Obs']
    nPosObs =  siteObs[MNKey]['Possibly Observed']
    nTot = siteObs[MNKey]['Total Isotopologues']
    sizes = [nObs, nPosObs - nObs, nTot - nPosObs]

    print(MNKey)
    print(sizes)

    def actualValues(val):
        a  = np.round(val/100.*nTot, 0)
        return int(a)

    patches, texts, other = cAx.pie(sizes, colors = colors, autopct=actualValues, startangle = 90)
    #patches, texts = cAx.pie(sizes, colors = colors, startangle = 90)
    #cAx.set_title('M+' + MNKey[-1])


plt.legend(patches, labels, loc="best", bbox_to_anchor= [1.00,0.5])

plt.tight_layout()

#Output figure
fig.savefig(str(cwd) + '/Fig5_Percentage_Complete/Fig5Percent_Completion_Panel1.eps')

'''
4) Generate a perfect dataset for 8 fragments, and compute how many beams we observe relative to how many we could observe
'''
#Get how many beams we do observe here
with open(str(cwd) + '/Reference Files/MasterMNDict.json') as f:
    masterMNDict = json.load(f)

#Generate perfect dataset
deltasPerfect = [0] * 16
fragSubset = ['full','133','104','102','87','74High','74Low','61','56']

methioninePerfect = metInit.initializeMethionine(deltasPerfect, fragSubset, smallFrags = True, printHeavy = False)

predictedMeasurementPerfect, MNDictSmp, FFSmp = metInit.simulateMeasurement(methioninePerfect, overrideIsoDict = overrideIsoDict8,abundanceThreshold = 0, massThreshold = 4,disableProgress = False)

#Compare the number of beams observed
beamObs = {}
for MNKey in ['M1','M2','M3','M4']:
    beamObs[MNKey] = {'Obs':0,'Total':0}
    for fragKey, fragData in predictedMeasurementPerfect[MNKey].items():
        for subKey, subData in fragData.items():
            beamObs[MNKey]['Total'] += 1
            if fragKey in masterMNDict[MNKey]:
                if subKey in masterMNDict[MNKey][fragKey]:
                    beamObs[MNKey]['Obs'] += 1

#Construct figure
fig, axes = plt.subplots(nrows = 1, ncols = 4, figsize = (9,4), dpi = 150)
fig.set_facecolor('white')
blue3 = [0.7161860822760477,0.8332026143790849,0.916155324875048]
blue6 = [0.2909803921568628,0.5945098039215686,0.7890196078431373]
blue8 = [0.09019607843137256,0.39294117647058824,0.6705882352941177]
colors = [blue8, blue6]
for MNIdx, MNKey in enumerate(['M1','M2','M3','M4']):
    cAx = axes[MNIdx]
    nObs = beamObs[MNKey]['Obs']
    nTot =  beamObs[MNKey]['Total']
    sizes = [nObs, nTot - nObs]
    labels = ['# Ion Beams Observed', '# Ion Beams Total']

    def actualValues(val):
        a  = np.round(val/100.*nTot, 0)
        return int(a)

    patches, texts, other = cAx.pie(sizes, colors = colors, autopct=actualValues, startangle = 90)
    #patches, texts = cAx.pie(sizes, colors = colors, startangle = 90)
    #cAx.set_title('M+' + MNKey[-1])

plt.legend(patches, labels, loc="best", bbox_to_anchor= [1.00,0.5])

plt.tight_layout()

#Output figure
fig.savefig(str(cwd) + '/Fig5_Percentage_Complete/Fig5Percent_Completion_Panel2.eps')