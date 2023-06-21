#relative imports of Isotomics
import sys
import os
from pathlib import Path
cwd = Path().resolve()

sys.path.insert(1, os.path.join(cwd, 'Isotomics Scripts'))

import initializeMethionine as metInit
import fragmentAndSimulate as fas
import calcIsotopologues as ci

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

def prettyLabel(string):
    '''
    Makes nice labels for the output plots
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

#Initialize and predict measurement for methionine
deltas = [-45,-35,-30,-25,-13,2.5,10,0,0,0,0,0,0]
thisMethionine = metInit.initializeMethionine(deltas)

predictedMeasurement, MNDict, fractionationFactors = metInit.simulateMeasurement(thisMethionine, 
                                                   abundanceThreshold = 0,
                                                   massThreshold = 4,
                                                   outputPath = None,
                                                   calcFF = True)

#Select target fragment
fragKey = '104'
toShow = predictedMeasurement["M4"][fragKey]
#Experimentally observed mass error
massError = -0.001280000000
siteElements = ci.strSiteElements(thisMethionine['molecularDataFrame'])

#Pull out masses, relative abundances, and substitutions of all peaks. Store in a lsit. 
massPlot = []
relAbundPlot = []
subPlot = []
for subKey, observation in toShow.items():
    #Fill in '' to 'Unsub'
    fullSubKey = subKey
    if subKey == '':
        fullSubKey = "Unsub"
    
    #Get the isotopologues in the form of a string, so we can run fas.computeMass
    Isotopologues = pd.DataFrame.from_dict(MNDict['M4'])
    Isotopologues = Isotopologues.T
    IsotopologuesWithSub = Isotopologues[Isotopologues[fragKey + '_01 Subs'] == fullSubKey]

    IsotopologuesStr = IsotopologuesWithSub[fragKey + '_01 Identity'][0]
    
    #Compute masses
    mass = fas.computeMass(IsotopologuesStr, siteElements)
    correctedMass = mass + massError
    
    massPlot.append(correctedMass)
    relAbundPlot.append(observation['Rel. Abundance'])
    subPlot.append(fullSubKey)
    
#Zoomed in plot
fig, ax = plt.subplots(figsize = (10*0.75,2*0.75), dpi = 600)
#Get m/z limits
xLow = 108.045
xHigh = 108.060
#Get abundance limits
lowAbundanceCutOff = 0.001

#Cut off peaks below the low abundance cut off and outside of the mass window
massPlotcutOff = []
subPlotcutOff = []
relAbundcutOff = []
sumAbund = 0
for i in range(len(massPlot)):
    if relAbundPlot[i] > lowAbundanceCutOff:
        if massPlot[i] >= xLow and massPlot[i] <= xHigh:
            relAbundcutOff.append(relAbundPlot[i])
            massPlotcutOff.append(massPlot[i])
            subPlotcutOff.append(subPlot[i])

#Adjust relative abundances
relAbundAdj = np.array(relAbundcutOff) / np.array(relAbundcutOff).max()

#Plot
for i in range(len(massPlotcutOff)):
    ax.vlines(massPlotcutOff[i], 0, relAbundAdj[i], color = 'k')
    
    #Add labels
ax.set_xticks(massPlotcutOff)
labels = [format(x, '.5f') +'\n' + y for x,y in zip(massPlotcutOff,subPlotcutOff)]
labelsFix = [prettyLabel(l.split('\n')[1]) + '\n' + l.split('\n')[0] for l in labels]

ax.set_xticklabels(labelsFix,rotation = 45);
plt.ylabel("Relative Abundance", fontsize = 12)
plt.xlim(xLow,xHigh)
ax.set_xlabel("m/z", fontsize = 12)
#plt.ylim(0,1)
sns.despine()

plt.savefig("Fig2 Theoretical Spectrum M+4 Methionine.jpg", bbox_inches = 'tight', dpi = 1000)

#Next plot
fig, ax = plt.subplots(figsize = (10*0.7,2*0.7), dpi = 200)
#Get m/z limits
xLow = 104.045
xHigh = 108.060

#Get abundance limits
lowAbundanceCutOff = 0.001

#Cut off peaks outside of this range
massPlotcutOff = []
subPlotcutOff = []
relAbundcutOff = []
sumAbund = 0
for i in range(len(massPlot)):
    if relAbundPlot[i] > lowAbundanceCutOff:
        if massPlot[i] >= xLow and massPlot[i] <= xHigh:
            relAbundcutOff.append(relAbundPlot[i])
            massPlotcutOff.append(massPlot[i])
            subPlotcutOff.append(subPlot[i])

#Adjust relative abundances
relAbundAdj = np.array(relAbundcutOff) / np.array(relAbundcutOff).max()

for i in range(len(massPlotcutOff)):
    ax.vlines(massPlotcutOff[i], 0, relAbundAdj[i], color = 'k')
    
ax.set_xticks(massPlotcutOff)
labels = [format(x, '.5f') +'\n' + y for x,y in zip(massPlotcutOff,subPlotcutOff)]
labelsFix = [prettyLabel(l.split('\n')[1]) + '\n' + l.split('\n')[0] for l in labels]

ax.set_xticklabels(labelsFix,rotation = 45);
plt.ylabel("Relative Abundance", fontsize = 12)
plt.xlim(xLow,xHigh)
ax.set_xlabel("m/z", fontsize = 12)
ax.set_xlim(103.5,108.5)
#plt.ylim(0,1)
sns.despine()
ax.set_xticks([104.05,105.055,106.055,107.055,108.055])
ax.set_xticklabels(['104\nUnsub','105\nM+1','106\nM+2','107\nM+3','108\nM+4'])

plt.savefig("Fig2 Theoretical Spectrum M+4 Methionine Full.jpg", bbox_inches = 'tight', dpi = 1000)