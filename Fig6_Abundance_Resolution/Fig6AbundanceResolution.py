import json
import copy
import os
import sys
from pathlib import Path
cwd = Path().resolve()

sys.path.insert(1, os.path.join(cwd, 'Isotomics Scripts'))
sys.path.insert(1, os.path.join(cwd, 'Read Process CSV'))

import initializeMethionine as initMet
import calcIsotopologues as ci
import fragmentAndSimulate as fas
import DataAnalyzerWithPeakInteg

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from tqdm import tqdm
from scipy.stats import norm

'''
This .py file will generate a seaborn heatmap showing the number of obserable ion beams at a range of resolutions and abundance sensitivities. It is split into 5 parts. Parts 1 and 3 require longer computations which can be bypassed by setting a boolean. 

1) Precalculate the isotopologues of methionine and select only the ones we need. We will recalculate the isotome of methionine several times; it has over a million isotopologues and is time-consuming (on the order of minutes). However, only 3402 of these isotopologues are in the M+1 through M+4 populations. So we precalculate all isotopologues only once, save those 3402, and in later calls, only calculate information for those 3402. 

2) Calculate which peaks are unresolved at which resolutions. For each resolution of interest (specified via resSetpoints), we compute all of the peaks, abundances, and locations. We then check if any peaks are sufficiently close to be unresolved at the target resolution. If they are, we assign the abundance of the less abundance peak to the more abundant one. We specify this all in an 'unresolved dict' for each resolution. 

3) We next which to calculate which peaks are observed at each relative abundance threshold. We define the relative abundance threshold as 'their abundance relative to the largest peak in the spectrum', and specify this using our experimental conditions. We also need to know the abundance of each fragment (which may vary with HCD setpoint). We first pull out the intensities of every substitution of every fragment from the experimental data (section 3a). We then compute the intensity of fragment x relative to the largest fragment of that experiment as follows: 

Start with fragment x, with substitutions x_1, x_2, ... x_n. Compute the observed intensity of each relative to the most abundant peak y_k of that M+N experiment, e.g. x_1 / y_k. Compare this to the predicted intensity ratio for the same set of peaks. 

The predicted version does not take into account differences in fragment intensity, while the observed version does. Thus we can find the relative fragment intensities as (x_1 / y_k)_obs / (x_1 / y_k)_pred. Do this for all x_i, then take their average. 

4) Now, we determine how many beams are available at each target abundance. For each resolution setpoint, compute a predicted spectrum. Then go through it and find the relative abundance of each peak relative to the most abundant peak of that M+N experiment. Apply a factor for the fragment intensity from (3). If the result is over the target threshold, treat that beam as observed. 

Abundance setpoints are set via targetRelAbundList

5) Generate the heatmaps and export. 
'''
deltas = [0] * 16
fragSubset = ['full','133','104','102','87','74High','74Low','61','56']
'''
(1) Precalculate the isotopologues of methionine and select only the ones we need
'''
precalc = False
if precalc:
    #precalculate isotopologues of interest
    thisMethioninePreCalc = initMet.initializeMethionine(deltas, fragSubset, smallFrags = True, printHeavy = False)
    byAtom = ci.inputToAtomDict(thisMethioninePreCalc['molecular DataFrame'], overrideIsoDict = {}, disable = False, M1Only = False)
    MN = ci.massSelections(byAtom, massThreshold = 4)

    overrideIsoDict = ci.pullOutIsotopologues(MN)
        
    with open(str(cwd) + '/Reference Files/Stored Isotopologues 8 Frags.json', 'w', encoding='utf-8') as f:
        json.dump(overrideIsoDict, f, ensure_ascii=False, indent=4)

with open(str(cwd) + "/Reference Files/Stored Isotopologues 8 Frags.json", 'r') as j:
    overrideIsoDict = json.loads(j.read())

'''
(2) Calculate a dictionary for the unresolved peaks, specifying which peaks are lost due to resolution problems.
'''
resSetpoints = [45000,60000,120000,240000,480000,1e6]
thisMethionine = initMet.initializeMethionine(deltas, fragSubset, smallFrags = True)
molecularDataFrame = thisMethionine['molecularDataFrame']
expandedFrags = thisMethionine['expandedFrags']
fragKeys = thisMethionine['fragKeys']
fragmentationDictionary = thisMethionine['fragmentationDictionary']

#Calculate all isotopologues; organize them based on M+N population
byAtom = ci.inputToAtomDict(molecularDataFrame, overrideIsoDict = overrideIsoDict)
MN = ci.massSelections(byAtom, massThreshold = 4)
MN = fas.trackMNFragments(MN, expandedFrags, fragKeys, molecularDataFrame, unresolvedDict = {})

#Two main outputs of this section: 'allUnresolvedDict' gives
#storeCloseLyingPeaks gives
allUnresolvedDict = {}
storeCloseLyingPeaks = {}

#in the isotopologuesDict, reference the full frag keys (which all have a subscript of _01, in case there are multiple fragmentation geometries leading to the same fragment.)
fullFragKeys = ['full_01','133_01','104_01','102_01','87_01','74High_01','74Low_01','61_01','56_01']
#Each resolution gets a different simulation
for thisRes in resSetpoints:
    storeCloseLyingPeaks[str(thisRes)] = {}
    unresolvedDict = {}

    #iterate through the M+N values 
    for MNKey in ['M1','M2','M3','M4']:
        storeCloseLyingPeaks[str(thisRes)][MNKey] = {}
        thisMNUnresolved = {}
        thisIsotopologues = MN[MNKey]
        siteElements = ci.strSiteElements(molecularDataFrame)

        #For each fragment, we simulate all peaks, then test for which ones are unresolved
        for fragIdx, fragKey in enumerate(fullFragKeys):
            shortFragKey = fragKey.split('_')[0]
            
            #for this fragment, we will construct a dictionary where the keys are observed substitutions and their values are dictionaries giving concentration and isotopic mass
            subByMass = {}
            #thisIso is a string giving the isotopologue, and thisIsoData is a dictionary containing information about that isotopologue
            for thisIso, thisIsoData in thisIsotopologues.items():
                thisIsoMass = fas.computeMass(thisIsoData[fragKey + ' Identity'], siteElements)
                thisIsoSubs = thisIsoData[fragKey + ' Subs']

                #if the fragment with those substitutions has not been observed, add to dictionary
                if thisIsoSubs not in subByMass:
                    subByMass[thisIsoSubs] = {'Mass':thisIsoMass,'Conc':0}

                #in either case, add its concentration
                subByMass[thisIsoSubs]['Conc'] += thisIsoData['Conc']

            #We now have all of the peaks, locations, and masses for a given fragment. We next determine which are unresolved. We begin by calculating the actual resolution at the mass of this fragment.

            #Get the mass of this fragment as a float.
            overrideMassDict = {'full':150,'74High':74,'74Low':74}
            if shortFragKey in overrideMassDict:
                thisMass = overrideMassDict[shortFragKey]
            else:
                thisMass = float(shortFragKey)

            #Compute the actual resolution at this mass
            thisResolution = (200 / thisMass)**(1/2) * thisRes

            #from RES = M / Delta M, defining deltaM as "the smallest difference delta m/z for which two ions can be separated", e.g. http://mass-spec.lsu.edu/msterms/index.php/Resolution_(mass_spectrometry)
            deltaM = thisMass / thisResolution

            #for each substituted version of this fragment, get concentration, mass, and subKey. Sort by concentrations. 
            peaksList = []
            for subKey, subInfo in subByMass.items():
                peaksList.append((subInfo['Conc'],subInfo['Mass'],subKey))
            peaksList.sort(reverse = True)
            
            #We next look at every pair of peaks; if two are close enough to be unresolved under the resolution threshold, we note the identity of the two and store in a dictionary.
            thisFragUnresolved = {}
            #iterate through tuple list, pulling out conc, mass, subKey
            for peakIdx1, peakData1 in enumerate(peaksList):
                conc, mass, sub = peakData1[0],peakData1[1],peakData1[2]
                #for each peak, check only LESS abundant peaks
                for peakIdx2, peakData2 in enumerate(peaksList):
                    if peakIdx2 > peakIdx1:
                        conc2, mass2, sub2 = peakData2[0],peakData2[1],peakData2[2]
                        #if the mass difference between them is lower than the target threshold
                        if np.abs(mass2 - mass) <= deltaM:
                            #add an entry to the output dictionary, where the key is the SMALLER substitution, keyed to a dict that records close-lying substitutions which are LARGER and close to the low abundance substitution.
                            if sub2 not in thisFragUnresolved:
                                thisFragUnresolved[sub2] = {'Sub':[],'RelConc':[],'MassDiff':[]}
                            thisFragUnresolved[sub2]['Sub'].append(sub)
                            thisFragUnresolved[sub2]['RelConc'].append(conc)
                            thisFragUnresolved[sub2]['MassDiff'].append(np.abs(mass2-mass))

            #Store the complete dictionary in case we need this information later
            storeCloseLyingPeaks[str(thisRes)][MNKey][fragKey] = copy.deepcopy(thisFragUnresolved)
                            
            #Now, as some peaks are unresolved with multiple others, pick the largest one and assign the abundance of the lost peak to that one.
            processedOutput = {}
            for lostSub, subInfo in thisFragUnresolved.items():
                #if they only lie close to 1 substitution, assign it
                if len(subInfo['Sub']) == 1:
                    processedOutput[lostSub] = subInfo['Sub'][0]
                else:
                    #if they are close-lying to multiple, assign to the most abundant substituion. ###This is a scenario where the model could be improved, i.e. to consider a combination of the mass AND concentration of close-lying substitutions. But doing so is more involved than necessary for our current applications.
                    maxConcIdx = subInfo['RelConc'].index(max(subInfo['RelConc']))
                    processedOutput[lostSub] = subInfo['Sub'][maxConcIdx]

            #fill in this information into the parent dictionaries.
            thisMNUnresolved[shortFragKey] = processedOutput

        unresolvedDict[MNKey] = copy.deepcopy(thisMNUnresolved)

    allUnresolvedDict[str(thisRes)] = copy.deepcopy(unresolvedDict)
        
with open(str(cwd) + '/Fig6_Abundance_Resolution/Fig6Unresolved_all.json', 'w', encoding='utf-8') as f:
    json.dump(allUnresolvedDict, f, ensure_ascii=False, indent=4)

with open(str(cwd) + '/Fig6_Abundance_Resolution/Fig6storedUnresolved.json', 'w', encoding='utf-8') as f:
    json.dump(storeCloseLyingPeaks, f, ensure_ascii=False, indent=4)
'''
3a) Compute the Observed Intensities of each fragment.
'''
getIntensities = False
#Takes about 6 minutes to run, so do it once and save as json
if getIntensities:
    #Folder with FT Statistic-ified files. All the files need to be processed using the same metrics.
    toRun = ['Met_M1','Met_M2','Met_M3','Met_M4']

    #"OMIT" included for peaks which were extracted but later rejected; we pass over these. Give the fragments, then isotopes of extracted peaks, in order they appear.  
    with open('C:/Users/tacse/Documents/Experimental MN Paper/Experimental_Isotomics/MasterMNDict.json') as f:
        masterMNDict = json.load(f)

    #Used for the peak drift tests
    mostAbundantDict = {'Met_M1':['13C','13C','13C','13C','13C','Unsub','Unsub','13C'],
                        'Met_M2':['34S','34S','18O','Unsub','Unsub','Unsub','Unsub','Unsub'],
                        #'Met_M3':['13C-34S','13C-34S','13C-34S','13C','34S','13C-18O','13C','34S','Unsub'],
                        'Met_M3':['13C-34S','13C-34S','13C','34S','13C-18O','13C','34S','Unsub'],
                        'Met_M4': ['36S','18O-34S','36S','18O','36S','13C','Unsub','34S','13C']}
    
    allObsIntensity = {}
    allRelIntensity = {}
    for experiment in toRun:
        totalMNIntensity = 0
        allObsIntensity[experiment] = {}
        folderPath = "C:/Users/tacse/Documents/Experimental MN Paper/Experimental_Isotomics/" + experiment

        #Source code says "gc elution", but this routine just culls based on time; so it is ok for these measurements. 
        gc_elution_on = True
        #Time frames for the GC elution; different time for each fragment, hence *9
        gcElutionTimes = [(15.00,75.00)] * 9

        metMNDict = masterMNDict[experiment]
        fragmentMostAbundant = mostAbundantDict[experiment]

        #Repackage MetMNDict as a list.
        massStr = []
        fragmentIsotopeList = []
        for i, v in metMNDict.items():
            massStr.append(i)
            fragmentIsotopeList.append(v)
            fragmentDict = metMNDict

        #Any specific properties you want to cull on
        cullOn = None
        #Multiple of SD you want to cull beyond for the cullOn property
        cull_amount = 3
        #Whether you want to cull zero scans
        cullZeroScans = False
        #Whether you want to calculate weighted averages based on NL height (specifically designed for GC elution but widely applicable!)
        weightByNLHeight = False
        #Whether you want to output each file as you process it, and where you want it to go:
        fileOutputPath = None
        #Set file extension
        fileExt = '.csv'

        #Use this to get M+N relative abundances
        MNRelativeAbundance = True

        rtnAllFilesDF, mergedList, allOutputDict = DataAnalyzerWithPeakInteg.calc_Folder_Output(folderPath, cullOn=cullOn, cullAmount=cull_amount,
                                                    gcElutionOn=gc_elution_on, gcElutionTimes = gcElutionTimes, 
                                                    fragmentIsotopeList = fragmentIsotopeList, 
                                                    fragmentMostAbundant = fragmentMostAbundant,
                                                    weightByNLHeight = False, debug = True, 
                                                    MNRelativeAbundance = MNRelativeAbundance, fileExt = fileExt, 
                                                    massStrList = list(fragmentDict.keys()),
                                                    Microscans = 1)
        
        firstFile = mergedList[0]

        thisMNFragList = list(metMNDict.keys())
        thisObsIntensity = {}

        for fragIdx, fragData in enumerate(firstFile):
            fragName = thisMNFragList[fragIdx]
            thisObsIntensity[fragName] = {}

            for subKey in metMNDict[fragName]:
                if subKey != 'OMIT':
                    thisSubAbsIntensity = fragData['absIntensity' + subKey].mean()

                    thisObsIntensity[fragName][subKey] = thisSubAbsIntensity

                    totalMNIntensity += thisSubAbsIntensity

        thisRelIntensity = {}
        for fragKey, fragData in thisObsIntensity.items():
            thisRelIntensity[fragKey] = {}
            for subKey, subData in fragData.items():
                thisRelIntensity[fragKey][subKey] = subData / totalMNIntensity

        allObsIntensity[experiment] = copy.deepcopy(thisObsIntensity)
        allRelIntensity[experiment] = copy.deepcopy(thisRelIntensity)
        

    with open(str(cwd) + '/Fig6_Abundance_Resolution/Fig6obsIntensities.json', 'w', encoding='utf-8') as f:
        json.dump(allObsIntensity, f, ensure_ascii=False, indent=4)

    with open(str(cwd) + '/Fig6_Abundance_Resolution/Fig6relIntensities.json', 'w', encoding='utf-8') as f:
        json.dump(allRelIntensity, f, ensure_ascii=False, indent=4)

with open(str(cwd) + '/Fig6_Abundance_Resolution/Fig6obsIntensities.json') as f:
    allObsIntensity = json.load(f)

with open(str(cwd) + '/Fig6_Abundance_Resolution/Fig6relIntensities.json') as f:
    allRelIntensity = json.load(f)

for MNKey, MNData in allRelIntensity.items():
    curMax = 1
    curFrag = ''
    curSub = ''
    print(MNKey)
    for fragKey, fragData in MNData.items():
        for subKey, subData in fragData.items():
            if subData <= curMax:
                curMax = subData
                curFrag = fragKey
                curSub = subKey
    print(curFrag + ' ' + curSub + ' ' + str(curMax))
'''
3b) Compute the intensities of each fragment
'''
thisMethionine = initMet.initializeMethionine(deltas, fragSubset, smallFrags = True)

#Get predicted intensities of each fragment
predictedMeasurement, MNDict, FFSmp = initMet.simulateMeasurement(thisMethionine, overrideIsoDict = overrideIsoDict,abundanceThreshold = 0, massThreshold = 4,disableProgress = True)

#Specify the intensity of the most abundant peak, as we will benchmark to this
predMostAbundantIntensity = {'M1':predictedMeasurement['M1']['56']['13C']['Abs. Abundance'],
                         'M2':predictedMeasurement['M2']['56']['Unsub']['Abs. Abundance'],
                         'M3':predictedMeasurement['M3']['56']['13C']['Abs. Abundance'],
                         'M4':predictedMeasurement['M4']['56']['Unsub']['Abs. Abundance']}

obsMostAbundantIntensity = {'M1':allObsIntensity['Met_M1']['56']['13C'],
                            'M2':allObsIntensity['Met_M2']['56']['Unsub'],
                            'M3':allObsIntensity['Met_M3']['56']['13C'],
                            'M4':allObsIntensity['Met_M4']['56']['Unsub']}

#dictionary of fragment heights
relFragHeights = {}
for MNKey in ['M1','M2','M3','M4']:
    relFragHeights[MNKey] = {}
    thisPredMostAbundant = predMostAbundantIntensity[MNKey]
    thisObsMostAbundant = obsMostAbundantIntensity[MNKey]
    for fragKey, fragData in allObsIntensity['Met_' + MNKey].items():
        subHeightChangeList = []
        for subKey, subData in fragData.items():
            #The height of the observed relative to the most abundant observed
            obsVsMost = subData / thisObsMostAbundant
            #The height of the predicted relative to the most abundant predicted
            predVsMost = predictedMeasurement[MNKey][fragKey][subKey]['Abs. Abundance'] / thisPredMostAbundant
            #The ratio of these (obs accounts for fragment height, pred does not)
            subHeightChange = obsVsMost / predVsMost
            subHeightChangeList.append(subHeightChange)

        #Take their average and output
        thisFragHeightChange = np.array(subHeightChangeList).mean()

        relFragHeights[MNKey][fragKey] = thisFragHeightChange

        #We don't have any 'full' fragments recovered for M+1 through M+3; approximate it as 5 percent
        if 'full' not in relFragHeights[MNKey]:
            relFragHeights[MNKey]['full'] = 0.05
'''
4) For each resolution, calculate the observed spectrum; for each abundance threshold, pull out beams above that threshold.
'''
targetRelAbundList = [4e-4,1e-4,1e-5,5e-6]
with open(str(cwd) + "/Reference Files/Stored Isotopologues 8 Frags.json", 'r') as j:
    overrideIsoDict = json.loads(j.read())

thisMethionine = initMet.initializeMethionine(deltas, fragSubset, smallFrags = True)

#Get predicted intensities of each fragment
predictedMeasurement, MNDict, FFSmp = initMet.simulateMeasurement(thisMethionine, overrideIsoDict = overrideIsoDict,abundanceThreshold = 0, massThreshold = 4,disableProgress = True)

totalMNAbundancePred = {}
for MNKey, MNData in allRelIntensity.items():
    shortMNKey = MNKey.split('_')[1]
    thisMNAbund = 0
    for fragKey, fragData in MNData.items():
        for subKey, subData in fragData.items():
            thisMNAbund += predictedMeasurement[shortMNKey][fragKey][subKey]['Abs. Abundance']

    totalMNAbundancePred[shortMNKey] = thisMNAbund

#allResults keys abundance to res to MNKey to a list of the beams observed
allResults = {}

#For each resolution, calculate the spectrum
for thisRes, thisUnresolvedDict in allUnresolvedDict.items():
    allResults[thisRes] = {}
    predictedMeasurement, MNDict, FFSmp = initMet.simulateMeasurement(thisMethionine, overrideIsoDict = overrideIsoDict,abundanceThreshold = 0, massThreshold = 4,disableProgress = True, unresolvedDict = thisUnresolvedDict)

    byRES = {}
    #for each abundance, calculate observed ion beams
    for targetRelAbund in targetRelAbundList:
        thisObservedBeams = {}

        for MNKey, MNData in predictedMeasurement.items():
            if MNKey in ['M1','M2','M3','M4']:
                if MNKey not in thisObservedBeams:
                    thisObservedBeams[MNKey] = {}

                for fragKey, fragData in MNData.items():
                    if fragKey not in thisObservedBeams:
                        thisObservedBeams[MNKey][fragKey] = []
                    for subKey, subData in fragData.items():
                        #Find the height of this peak relative to the most abundant peak
                        thisSubRelSpectrum = subData['Abs. Abundance'] / totalMNAbundancePred[MNKey]
                        #Apply the relative fragment height
                        thisSubRelSpectrum *= relFragHeights[MNKey][fragKey]
            
                        #if the relative height is above the threshold, add to output
                        if thisSubRelSpectrum >= targetRelAbund:
                            thisObservedBeams[MNKey][fragKey].append(subKey)

        allResults[thisRes][str(targetRelAbund)] = copy.deepcopy(thisObservedBeams)
'''
5) Output and plot
'''
fig, ax = plt.subplots(nrows = 1, ncols = 4, figsize = (12,4), sharey = True)

cMapLims = {'M1':{'vmin':0,'vmax':40},
            'M2':{'vmin':0,'vmax':80},
            'M3':{'vmin':0,'vmax':120},
            'M4':{'vmin':0,'vmax':150}}

for i in range(4):
    cAx = ax[i]
    MNKey = 'M' + str(i + 1)
    AmtMatrix = []
    #For each res and each abundance, find the number of observed beams
    for Res, ResData in allResults.items():
        cList = []
        for abundance, abundanceData in ResData.items():
            totalN = 0
            #iterate through by fragment; length of the fragData gives the number of beams
            for fragKey, fragData in abundanceData[MNKey].items():
                totalN += len(fragData)
            cList.append(totalN)
        AmtMatrix.append(copy.deepcopy(cList))
    #Generate seaborn heatmap
    cAx = sns.heatmap(np.array(AmtMatrix), ax = cAx, cmap = 'Reds', annot = True,xticklabels = ['4e-4','1e-4','1e-5','5e-6'], yticklabels = ['45k','60k','120k','240k','480k','1M'],  fmt='g', vmin = cMapLims[MNKey]['vmin'], vmax = cMapLims[MNKey]['vmax'])
    if i != 0:
        cAx.tick_params(left=False)
    else:
        cAx.set_ylabel("Nominal Resolution", fontsize = 16)

    #axes and labels etc.
    cAx.invert_yaxis()
    cAx.set_title('M+' + str(i+1))

fig.supxlabel("Detection Limit", fontsize = 16)
fig.suptitle("Number of Observed Peaks", fontsize = 16)
plt.tight_layout()

fig.savefig(str(cwd) + "/Fig6_Abundance_Resolution/Fig6AbundanceResolution.jpeg", bbox_inches = 'tight', dpi = 150)