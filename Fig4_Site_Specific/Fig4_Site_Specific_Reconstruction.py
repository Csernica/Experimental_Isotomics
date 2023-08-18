import json

#relative imports of Isotomics
import sys
import os
from pathlib import Path
cwd = Path().resolve()

import matplotlib.pyplot as plt
import matplotlib
import numpy as np

matplotlib.rcParams.update({'errorbar.capsize': 5})

MNResults = {}
AllAE = {}
AllER = {}
siteNames = ['Cmethyl','Cgamma','Calphabeta','Ccarboxyl','Ocarboxyl','Ssulfur','Namine','Hmethyl','Hgamma',
             'Halphabeta','Hamine','Hhydroxyl','Hprotonated']

for MNKey in ['M1','M2','M3','M4']:
    with open(str(cwd) + '/Read Process CSV/' + MNKey + 'ProcessedResults.json') as f:
        MNResults[MNKey] = json.load(f)

    if MNKey == 'M1':
        for idx, replicates in enumerate(MNResults[MNKey]['Mean']):
            siteName = siteNames[idx]
            AllAE[siteName] = {'AE':replicates, 'AE Error': MNResults[MNKey]['Std'][idx]}
            AllER[siteName] = {'ER':np.array(replicates).mean(),'ER Error':np.array(replicates).std()}

    else:
        for subKey, subData in MNResults[MNKey].items():
            AllER[subKey] = {'ER':np.array(subData['Means']).mean(),'ER Error':np.array(subData['Means']).std()}
            AllAE[subKey] = {'AE':subData['Means'], 'AE Error': subData['Stds']}

fig, axes = plt.subplots(nrows = 1, ncols = 2, figsize = (4,8), dpi = 300, sharey = False)
#Removed sites 'Hmethyl', 'Hgamma', 'Halphabeta', 'Hamine', 'Hhydroxyl', 'Hprotonated', 'Ocarboxyl', '13C Ccarboxyl   |   34S Ssulfur',
toPlot = ['Cmethyl','Cgamma','Calphabeta', 'Ccarboxyl', 'Ssulfur', 
'Namine',
'13C Cmethyl   |   13C Cgamma', '13C Cmethyl   |   13C Calphabeta', '18O Ocarboxyl', 
'34S Ssulfur', 
'13C Ccarboxyl   |   34S Ssulfur',
'13C Calphabeta   |   34S Ssulfur', 
'18O Ocarboxyl   |   33S Ssulfur', 
'34S Ssulfur   |   15N Namine', 
'13C Cmethyl   |   13C Cgamma   |   34S Ssulfur', 
'18O Ocarboxyl   |   18O Ocarboxyl', 
'18O Ocarboxyl   |   34S Ssulfur', 
'36S Ssulfur']

ytickLabels = []
subParams = {'Cmethyl':{'NiceLabel':'$^{13}C_{methyl}$',
                                            'marker':'D','mfc':'None','mec':'k','ms':6,'mew':1,'ecolor':'k'},
             'Cgamma':{'NiceLabel':'$^{13}C_{gamma}$',
                                            'marker':'D','mfc':'None','mec':'k','ms':6,'mew':1,'ecolor':'k', 'legend':'M+1'},
             'Calphabeta':{'NiceLabel':'$^{13}C_{alphabeta}$',
                                            'marker':'D','mfc':'None','mec':'k','ms':6,'mew':1,'ecolor':'k'},
             'Ccarboxyl':{'NiceLabel':'$^{13}C_{carboxyl}$',
                                           'marker':'D','mfc':'None','mec':'k','ms':6,'mew':1,'ecolor':'k'},
             'Ocarboxyl':{'NiceLabel':'$^{17}O}$',
                                            'marker':'D','mfc':'None','mec':'k','ms':6,'mew':1,'ecolor':'k'},
             'Ssulfur':{'NiceLabel':'$^{33}S$',
                                            'marker':'D','mfc':'None','mec':'k','ms':6,'mew':1,'ecolor':'k'},
             'Namine':{'NiceLabel':'$^{15}N$',
                                            'marker':'D','mfc':'None','mec':'k','ms':6,'mew':1,'ecolor':'k'},
             'Hmethyl':{'NiceLabel':'$^{2}H_{methyl}$',
                                            'marker':'D','mfc':'None','mec':'k','ms':6,'mew':1,'ecolor':'k'},
             'Hgamma':{'NiceLabel':'$^{2}H_{gamma}$',
                                            'marker':'D','mfc':'None','mec':'k','ms':6,'mew':1,'ecolor':'k'},
             'Halphabeta':{'NiceLabel':'$^{2}H_{alphabeta}$',
                                            'marker':'D','mfc':'None','mec':'k','ms':6,'mew':1,'ecolor':'k'},
             'Hamine':{'NiceLabel':'$^{2}H_{amine}$',
                                           'marker':'D','mfc':'None','mec':'k','ms':6,'mew':1,'ecolor':'k'}, 
             'Hhydroxyl':{'NiceLabel':'$^{2}H_{hydroxyl}$',
                                            'marker':'D','mfc':'None','mec':'k','ms':6,'mew':1,'ecolor':'k'},  
             'Hprotonated':{'NiceLabel':'$^{2}H_{protonated}$',
                                            'marker':'D','mfc':'None','mec':'k','ms':6,'mew':1,'ecolor':'k'}, 

             '13C Cmethyl   |   13C Cgamma':{'NiceLabel':'$^{13}C_{methyl}$$^{13}C_{gamma}$',
                                            'marker':'D','mfc':'None','mec':'tab:blue','ms':6,'mew':1,'ecolor':'tab:blue'},
             '13C Cmethyl   |   13C Calphabeta':{'NiceLabel':'$^{13}C_{methyl}$$^{13}C_{alphabeta}$',
                                            'marker':'D','mfc':'None','mec':'tab:blue','ms':6,'mew':1,'ecolor':'tab:blue'},
             '18O Ocarboxyl':{'NiceLabel':'$^{18}O$',
                                            'marker':'D','mfc':'None','mec':'tab:blue','ms':6,'mew':1,'ecolor':'tab:blue', 'legend':'M+2'},
             '34S Ssulfur':{'NiceLabel':'$^{34}S$',
                                            'marker':'D','mfc':'None','mec':'tab:blue','ms':6,'mew':1,'ecolor':'tab:blue'},

             '13C Cgamma   |   13C Calphabeta   |   13C Calphabeta':{'NiceLabel':'$^{13}C_{gamma}$$^{13}C_{alphabeta}$$^{13}C_{alphabeta}$',
                                            'marker':'D','mfc':'None','mec':'tab:red','ms':6,'mew':1,'ecolor':'tab:red'},
             '13C Ccarboxyl   |   34S Ssulfur':{'NiceLabel':'$^{13}C_{carboxyl}$$^{34}S$',
                                            'marker':'D','mfc':'None','mec':'tab:red','ms':6,'mew':1,'ecolor':'tab:red'},
            '13C Calphabeta   |   34S Ssulfur':{'NiceLabel':'$^{13}C_{alphabeta}$$^{34}S$',
                                            'marker':'D','mfc':'None','mec':'tab:red','ms':6,'mew':1,'ecolor':'tab:red'},
             '18O Ocarboxyl   |   33S Ssulfur':{'NiceLabel':'$^{18}O$$^{33}S$',
                                            'marker':'D','mfc':'None','mec':'tab:red','ms':6,'mew':1,'ecolor':'tab:red'},
             '34S Ssulfur   |   15N Namine':{'NiceLabel':'$^{34}S$$^{15}N$',
                                            'marker':'D','mfc':'None','mec':'tab:red','ms':6,'mew':1,'ecolor':'tab:red', 'legend':'M+3'},

            '13C Cmethyl   |   13C Cgamma   |   34S Ssulfur':{'NiceLabel':'$^{13}C_{methyl}$$^{13}C_{gamma}$$^{34}S$',
                                            'marker':'D','mfc':'None','mec':'tab:purple','ms':6,'mew':1,'ecolor':'tab:purple'},
            '18O Ocarboxyl   |   18O Ocarboxyl':{'NiceLabel':'$^{18}O$$^{18}O$',
                                            'marker':'D','mfc':'None','mec':'tab:purple','ms':6,'mew':1,'ecolor':'tab:purple'},
            '18O Ocarboxyl   |   34S Ssulfur':{'NiceLabel':'$^{18}O$$^{34}S$',
                                            'marker':'D','mfc':'None','mec':'tab:purple','ms':6,'mew':1,'ecolor':'tab:purple'},
            '36S Ssulfur':{'NiceLabel':'$^{36}S$',
                                            'marker':'D','mfc':'None','mec':'tab:purple','ms':6,'mew':1,'ecolor':'tab:purple', 'legend':'M+4'}}

for axIdx, cAx in enumerate(axes):
    for subKey in toPlot:
        
        thisParams = subParams[subKey]
        AE = AllAE[subKey]

        loc = len(toPlot) - (toPlot.index(subKey)+1)
        if 'legend' in thisParams and axIdx == 0:
            cAx.errorbar(AllER[subKey]['ER'], loc, xerr = np.array(AE['AE Error']).mean(), yerr = None,linestyle = 'None',
        marker=thisParams['marker'], mfc=thisParams['mfc'],
            mec=thisParams['mec'], ms=thisParams['ms'], mew=thisParams['mew'], ecolor = thisParams['ecolor'],
            label = thisParams['legend'])

        else:
            cAx.errorbar(AllER[subKey]['ER'], loc, xerr = np.array(AE['AE Error']).mean(), yerr = None,linestyle = "None",
        marker=thisParams['marker'], mfc=thisParams['mfc'],
            mec=thisParams['mec'], ms=thisParams['ms'], mew=thisParams['mew'], ecolor = thisParams['ecolor'])

        if axIdx == 0:
            ytickLabels.append(thisParams['NiceLabel'])

ax = axes[0]
ax2 = axes[1]

ytick = list(range(len(toPlot)))
ytick.reverse()
ax.set_yticks(ytick)
ax.set_yticklabels(ytickLabels)


ax.set_xlabel("$\delta_{STD}$")
ax.set_xlim(-15,15)
ylim = ax.get_ylim()
ax.vlines(0,ylim[0]-1,ylim[1]+1,linestyle = '--',color = 'k',label = 'Predicted Unlabeled Enrichment')
ax.set_ylim(ylim)

ax2.set_xlim(93,123)

ax2.set_xlabel("$\delta_{STD}$")

ylim = ax2.get_ylim()
ax2.vlines(100.35,ylim[0]-1,ylim[1]+1,linestyle = '--',color = 'tab:red',label = 'Predicted Labeled Enrichment')
ax2.set_ylim(ylim)

ax.spines['right'].set_visible(False)
ax2.spines['left'].set_visible(False)
ax2.set_yticks([])

d = .015  # how big to make the diagonal lines in axes coordinates
# arguments to pass to plot, just so we don't keep repeating them
kwargs = dict(transform=ax.transAxes, color='k', clip_on=False)
ax.plot((1-d, 1+d), (1-d, 1+d), **kwargs)        # top-left diagonal
ax.plot((1 - d +0.01, 1 + d +0.01), (-d, +d), **kwargs)  # top-right diagonal

kwargs.update(transform=ax2.transAxes)  # switch to the bottom axes
ax2.plot((-d, +d), (1 - d, 1 + d), **kwargs)  # bottom-left diagonal
ax2.plot((-d+0.005, d+0.005), (-d, d), **kwargs)  # bottom-right diagonal

handles1, labels1 = ax.get_legend_handles_labels()
handles2, labels2 = ax2.get_legend_handles_labels()

handles = handles2 + handles1
labels = labels2 + labels1

leg = fig.legend(handles, labels,bbox_to_anchor=(0.92,0.57), loc='upper left')
leg.get_frame().set_edgecolor('k')

plt.savefig(str(cwd) + '/Fig4_Site_Specific/Fig4_SiteSpecificReconstruction.jpeg', bbox_inches = 'tight', dpi = 1000)

#Second plot for H sites
fig, ax = plt.subplots(nrows = 1, ncols = 1, figsize = (3,2), dpi = 300)
toPlot = ['Hmethyl','Hgamma','Halphabeta','Hamine']

ytickLabels = []

for subKey in toPlot:
    thisParams = subParams[subKey]
    AE = AllAE[subKey]
    loc = len(toPlot) - (toPlot.index(subKey)+1)
    if 'legend' in thisParams:
        ax.errorbar(AllER[subKey]['ER'], loc, xerr = np.array(AE['AE Error']).mean(), yerr = None,linestyle = 'None',
    marker=thisParams['marker'], mfc=thisParams['mfc'],
        mec=thisParams['mec'], ms=thisParams['ms'], mew=thisParams['mew'], ecolor = thisParams['ecolor'],
        label = thisParams['legend'])

    else:
        ax.errorbar(AllER[subKey]['ER'], loc, xerr = np.array(AE['AE Error']).mean(), yerr = None,linestyle = "None",
    marker=thisParams['marker'], mfc=thisParams['mfc'],
        mec=thisParams['mec'], ms=thisParams['ms'], mew=thisParams['mew'], ecolor = thisParams['ecolor'])

    ytickLabels.append(thisParams['NiceLabel'])

ytick = list(range(len(toPlot)))
ytick.reverse()
ax.set_yticks(ytick)
ax.set_yticklabels(ytickLabels, fontsize = 12)

ax.set_xlabel("$\delta_{STD}$")
ax.set_xlim(-50,100)
ax.set_ylim(-0.5,3.5)
ylim = ax.get_ylim()
ax.vlines(0,ylim[0]-1,ylim[1]+1,linestyle = '--',color = 'k',label = 'Predicted Unlabeled Enrichment')
ax.set_ylim(ylim)

plt.savefig(str(cwd) + '/Fig4_Site_Specific/Fig4_SiteSpecificReconstructionB.jpeg', bbox_inches = 'tight', dpi = 1000)