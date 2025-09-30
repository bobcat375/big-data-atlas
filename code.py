import uproot
import hist
from hist import Hist
from TLorentzVector import TLorentzVector
import numpy as np
import matplotlib.pyplot as plt

def trackProgress(n,m):
	"""
    Function which prints the event loop progress every m events 
    
    Parameters
    ----------
	n : Number of events processed so far

	m : Printout event interval
    
    """
	if n == 0:
		print("Event loop tracker")
		print("------------------")
    
	if(n%m==0):
		print("%d events processed" % n)
		
# real 4lep ATLAS data
real1 = uproot.open("https://atlas-opendata.web.cern.ch/atlas-opendata/samples/2020/4lep/Data/data_A.4lep.root")
real2 = uproot.open("https://atlas-opendata.web.cern.ch/atlas-opendata/samples/2020/4lep/Data/data_B.4lep.root")
real3 = uproot.open("https://atlas-opendata.web.cern.ch/atlas-opendata/samples/2020/4lep/Data/data_C.4lep.root")
real4 = uproot.open("https://atlas-opendata.web.cern.ch/atlas-opendata/samples/2020/4lep/Data/data_D.4lep.root")

dataTree1 = real1["mini"]
dataTree2 = real2["mini"]
dataTree3 = real3["mini"]
dataTree4 = real4["mini"]
numDataEntries1 = len(dataTree1["runNumber"].array())
print("dataTree1 contains", numDataEntries1, "entries")
numDataEntries2 = len(dataTree2["runNumber"].array())
print("dataTree2 contains", numDataEntries2, "entries")
numDataEntries3 = len(dataTree3["runNumber"].array())
print("dataTree3 contains", numDataEntries3, "entries")
numDataEntries4 = len(dataTree4["runNumber"].array())
print("dataTree4 contains", numDataEntries4, "entries")

# simulated MC 4lep data
bkg = uproot.open("https://atlas-opendata.web.cern.ch/atlas-opendata/samples/2020/4lep/MC/mc_363490.llll.4lep.root")
bkgTree = bkg["mini"]
numMCEntries = len(bkgTree["runNumber"].array())
print("bkgTree contains", numMCEntries, "entries")

h_bgs = Hist(hist.axis.Regular(100, 60, 300, label = "Transverse mass m_{T}"))
h_dat = Hist(hist.axis.Regular(100, 60, 300, label = "Transverse mass m_{T}"))

bkgTree.show()

def mcWeights(data,lumi=10):
    """
    When MC simulation is compared to data the contribution of each simulated event needs to be
    scaled ('reweighted') to account for differences in how some objects behave in simulation
    vs in data, as well as the fact that there are different numbers of events in the MC tree than 
    in the data tree.
    
    Parameters
    ----------
    tree : TTree entry for this event
    """
    
    XSection = data["XSection"]
    SumWeights = data["SumWeights"]
    #These values don't change from event to event
    norm = lumi*(XSection*1000)/SumWeights
    
    scaleFactor_ELE = data["scaleFactor_ELE"]
    scaleFactor_MUON = data["scaleFactor_MUON"]
    scaleFactor_LepTRIGGER = data["scaleFactor_LepTRIGGER"]
    scaleFactor_PILEUP = data["scaleFactor_PILEUP"]
    mcWeight = data["mcWeight"]
    #These values do change from event to event
    scale_factors = scaleFactor_ELE*scaleFactor_MUON*scaleFactor_LepTRIGGER*scaleFactor_PILEUP*mcWeight
    
    weight = norm*scale_factors
    return weight


def goodLeptons(data):
	"""
	A function to return the indices of 'good leptons' (electrons or muons) in an event. This follows 
	many of the same steps as locateGoodPhotons() and photonIsolation() in Notebook 6.
	
	Parameters
	----------
	tree : TTree entry for this event
	"""

	#Initialise (set up) the variables we want to return
	goodlepton_index = [] #Indices (position in list of event's leptons) of our good leptons

	lep_n = data["lep_n"]
	##Loop through all the leptons in the event
	for j in range(0,lep_n):
		lep_isTightID = data["lep_isTightID"][j]    
		##Check lepton ID
		if(lep_isTightID):
			lep_ptcone30 = data["lep_ptcone30"][j]
			lep_pt = data["lep_pt"][j]
			lep_etcone20 = data["lep_etcone20"][j]
			#Check lepton isolation
			#Similar to photonIsolation() above, different thresholds
			if((lep_ptcone30 / lep_pt < 0.1) and 
				(lep_etcone20 / lep_pt < 0.1)):

				#Only central leptons 
				#Electrons and muons have slightly different eta requirements
				lep_type = data["lep_type"][j]
				lep_eta = data["lep_eta"][j]
				#Electrons: 'Particle type code' = 11
				if lep_type == 11:
					#Check lepton eta is in the 'central' region and not in "transition region" 
					if (np.abs(lep_eta) < 2.37) and\
						(np.abs(lep_eta) < 1.37 or np.abs(lep_eta) > 1.52): 

						goodlepton_index.append(j) #Store lepton's index

                #Muons: 'Particle type code' = 13
				elif (lep_type == 13) and (np.abs(lep_eta) < 2.5): #Check'central' region

					goodlepton_index.append(j) #Store lepton's index


	return goodlepton_index #return list of good lepton indices

def hWW(data,hist,mode):
	"""
	Function which executes the analysis flow for the Higgs production cross-section measurement in the H->WW
	decay channel.
	
	Fills a histogram with mT(llvv) of events which pass the full set of cuts 
	
	Parameters
	----------
	data : A Ttree containing data / background information
	
	hist : The name of the histogram to be filled with mT(llvv) values
	
	mode : A flag to tell the function if it is looping over 'data' or 'mc'
	"""
	
	n = 0
	for event in data:
        #############################
        ### Event-level requirements
        #############################
    
		trackProgress(n,10000)
		n += 1
	
		#If event is MC: Reweight it
		if mode.lower() == 'mc': weight = mcWeights(event)
		else: weight = 1
			
		trigE = event["trigE"]
		trigM = event["trigM"]
		#If the event passes either the electron or muon trigger
		if trigE or trigM:
            
			####Lepton preselections
			goodLeps = goodLeptons(event) #If the datafiles were not already filtered by number of leptons
		
			###################################
			### Individual lepton requirements
			###################################
		
			if len(goodLeps) >= 4: #Exactly two good leptons...
				#print("4")
				lep1 = goodLeps[0] #INDICES of the good leptons
				lep2 = goodLeps[1]
				lep3 = goodLeps[2] #INDICES of the good leptons
				lep4 = goodLeps[3]
				
				lep_type = event["lep_type"]
				# electron_count = 0
				# muon_count = 0
				#print(str(lep_type[lep1]))
				if (int(lep_type[lep1]) + int(lep_type[lep2]) + int(lep_type[lep3]) + int(lep_type[lep4]) == 48):
				#for i in range(0,3):
				#	if lep_type[goodLeps[i]] == 11:
				#		electron_count += 1
				#	elif lep_type[goodLeps[i]] == 13:
				#		muon_count += 1
				#if (electron_count == 2 and muon_count == 2): #... with same flavour
					
					lep_charge = event["lep_charge"]
					#print("bro")
					if (lep_charge[lep1] != lep_charge[lep2] and lep_charge[lep3] != lep_charge[lep4]) or (lep_charge[lep1] != lep_charge[lep3] and lep_charge[lep2] != lep_charge[lep4]) or (lep_charge[lep1] != lep_charge[lep4] and lep_charge[lep2] != lep_charge[lep3]):
						lep_pt = event["lep_pt"]
						if (lep_pt[lep1] > 25000) and (lep_pt[lep2] > 15000) and (lep_pt[lep3] > 10000) and (lep_pt[lep4] > 7000): #pT requirements
							#Note: TTrees always sort objects in descending pT order
							#print("success")
							lep_eta = event["lep_eta"]
							lep_phi = event["lep_phi"]
							lep_E = event["lep_E"]
							firstLepton  = TLorentzVector()
							secondLepton = TLorentzVector()
							thirdLepton = TLorentzVector()
							fourthLepton = TLorentzVector()
							firstLepton.SetPtEtaPhiE(lep_pt[lep1]/1000., lep_eta[lep1], lep_phi[lep1], lep_E[lep1]/1000.)
							secondLepton.SetPtEtaPhiE(lep_pt[lep2]/1000., lep_eta[lep2], lep_phi[lep2], lep_E[lep2]/1000.)
							thirdLepton.SetPtEtaPhiE(lep_pt[lep3]/1000., lep_eta[lep3], lep_phi[lep3], lep_E[lep3]/1000.)
							fourthLepton.SetPtEtaPhiE(lep_pt[lep4]/1000., lep_eta[lep4], lep_phi[lep4], lep_E[lep4]/1000.)
							
							#higgs_m = firstLepton.M() + secondLepton.M() + thirdLepton.M() + fourthLepton.M()
							higgs = firstLepton + secondLepton + thirdLepton + fourthLepton

							#higgs_E = firstLepton.E() + secondLepton.E() + thirdLepton.E() + fourthLepton.E()
							#print(str(higgs_E))
							hist.fill(higgs.Mt(), weight=weight)
                                            
#Data
data1 = dataTree1.arrays(["lep_ptcone30","lep_etcone20", "lep_isTightID", "lep_eta", "photon_phi", "lep_type", "lep_n", "photon_E", "lep_E", "lep_pt", "trigP", "XSection", "SumWeights", "trigE", "trigM", "scaleFactor_ELE", "scaleFactor_MUON", "scaleFactor_PILEUP", "scaleFactor_LepTRIGGER", "mcWeight", "lep_charge","lep_phi", "met_et", "met_phi"])
data2 = dataTree2.arrays(["lep_ptcone30","lep_etcone20", "lep_isTightID", "lep_eta", "photon_phi", "lep_type", "lep_n", "photon_E", "lep_E", "lep_pt", "trigP", "XSection", "SumWeights", "trigE", "trigM", "scaleFactor_ELE", "scaleFactor_MUON", "scaleFactor_PILEUP", "scaleFactor_LepTRIGGER", "mcWeight", "lep_charge","lep_phi", "met_et", "met_phi"])
data3 = dataTree3.arrays(["lep_ptcone30","lep_etcone20", "lep_isTightID", "lep_eta", "photon_phi", "lep_type", "lep_n", "photon_E", "lep_E", "lep_pt", "trigP", "XSection", "SumWeights", "trigE", "trigM", "scaleFactor_ELE", "scaleFactor_MUON", "scaleFactor_PILEUP", "scaleFactor_LepTRIGGER", "mcWeight", "lep_charge","lep_phi", "met_et", "met_phi"])
data4 = dataTree4.arrays(["lep_ptcone30","lep_etcone20", "lep_isTightID", "lep_eta", "photon_phi", "lep_type", "lep_n", "photon_E", "lep_E", "lep_pt", "trigP", "XSection", "SumWeights", "trigE", "trigM", "scaleFactor_ELE", "scaleFactor_MUON", "scaleFactor_PILEUP", "scaleFactor_LepTRIGGER", "mcWeight", "lep_charge","lep_phi", "met_et", "met_phi"])

#Running over the loop takes a lot of time so we will start by running over a fraction of the events.
# We will set the fraction numerator here

fNumerator = 1 # originally at 5

#If you have time and you want to run over more data, reduce the fraction numerator. fNumerator = 1 is the smallest valuie you can set
#If running over the data and MC takes too long, increase fNumerator.

#Data
#This takes a long time to run, so we will start by running over only one fifth of the events

hWW(data1,h_dat,'data')
hWW(data2,h_dat,'data')
hWW(data3,h_dat,'data')
hWW(data4,h_dat,'data')

#MC
mcSim = bkgTree.arrays(["lep_ptcone30","lep_etcone20", "lep_isTightID", "lep_eta", "photon_phi", "lep_type", "lep_n", "photon_E", "lep_E", "lep_pt", "trigP", "XSection", "SumWeights", "trigE", "trigM", "scaleFactor_ELE", "scaleFactor_MUON", "scaleFactor_PILEUP", "scaleFactor_LepTRIGGER", "mcWeight", "lep_charge","lep_phi", "met_et", "met_phi"])


#MC 
fractionOfMC = int(numMCEntries/5)
hWW(mcSim[0:fractionOfMC],h_bgs,'mc')

h_diff = h_dat - h_bgs
print("about to plot")
h_diff.plot(histtype = "fill")
plt.show()
print("plotted")

h_dat.plot(histtype="fill")

h_bgs.plot(histtype="fill")
plt.show()
