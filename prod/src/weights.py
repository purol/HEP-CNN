import numpy as np

"""
This module contains some cross section and theory param data for the xAOD
and Delphes samples. At the bottom are three helper functions to retrieve the
values.
"""

#-------------------------------------------------------------------------------
# xAOD meta data
#-------------------------------------------------------------------------------

# Dictionary of DSID -> (M_Gluino, M_Neutralino, Xsec)
# Xsec is given in pb
# TODO: put all of these things in consistent file format!!
xaodRPVMetaDict = {
    403550: (700, 450, 3.5251),
    403551: (800, 450, 1.4891),
    403552: (900, 450, 0.677478),
    403553: (1000, 50, 0.325388),
    403554: (1000, 250, 0.325388),
    403555: (1000, 450, 0.325388),
    403556: (1000, 650, 0.325388),
    403557: (1000, 850, 0.325388),
    403558: (1200, 50, 0.0856418),
    403559: (1200, 250, 0.0856418),
    403560: (1200, 450, 0.0856418),
    403561: (1200, 650, 0.0856418),
    403562: (1200, 850, 0.0856418),
    403563: (1200, 1050, 0.0856418),
    403564: (1400, 50, 0.0252977),
    403565: (1400, 250, 0.0252977),
    403566: (1400, 450, 0.0252977),
    403567: (1400, 650, 0.0252977),
    403568: (1400, 850, 0.0252977),
    403569: (1400, 1050, 0.0252977),
    403570: (1400, 1250, 0.0252977),
    403571: (1600, 50, 0.00810078),
    403572: (1600, 250, 0.00810078),
    403573: (1600, 450, 0.00810078),
    403574: (1600, 650, 0.00810078),
    403575: (1600, 850, 0.00810078),
    403576: (1600, 1050, 0.00810078),
    403577: (1600, 1250, 0.00810078),
    403578: (1600, 1450, 0.00810078),
    403579: (1700, 50, 0.00470323),
    403580: (1700, 250, 0.00470323),
    403581: (1700, 450, 0.00470323),
    403582: (1700, 650, 0.00470323),
    403583: (1700, 850, 0.00470323),
    403584: (1700, 1050, 0.00470323),
    403585: (1700, 1250, 0.00470323),
    403586: (1700, 1450, 0.00470323),
    403587: (1800, 50, 0.00276133),
    403588: (1800, 250, 0.00276133),
    403589: (1800, 450, 0.00276133),
    403590: (1800, 650, 0.00276133),
    403591: (1800, 850, 0.00276133),
    403592: (1800, 1050, 0.00276133),
    403593: (1800, 1250, 0.00276133),
    403594: (1800, 1450, 0.00276133),
    403595: (1800, 1650, 0.00276133),
    403596: (1900, 50, 0.00163547),
    403597: (1900, 250, 0.00163547),
    403598: (1900, 450, 0.00163547),
    403599: (1900, 650, 0.00163547),
    403600: (1900, 850, 0.00163547),
    403601: (1900, 1050, 0.00163547),
    403602: (1900, 1250, 0.00163547),
    403603: (1900, 1450, 0.00163547),
    403604: (1900, 1650, 0.00163547),
    403605: (900, 0, 0.677478),
    403606: (1000, 0, 0.325388),
    403607: (1100, 0, 0.163491),
    403608: (1200, 0, 0.0856418),
    403609: (1300, 0, 0.0460525),
    403610: (1400, 0, 0.0252977),
    403611: (1500, 0, 0.0141903),
    403612: (1600, 0, 0.00810078),
    403613: (1700, 0, 0.00470323),
    403614: (1800, 0, 0.00276133)
}

# Data from SUSYTools cross section file
xaodBkgXsecData = np.genfromtxt('../config/susy_crosssections_13TeV.txt',
    dtype='i4,U50,f8,f8,f8,f8',
    names=['dsid', 'sample', 'xsec', 'kfac', 'eff', 'unc'])
xaodBkgXsecDict = dict(zip(xaodBkgXsecData['dsid'],
    xaodBkgXsecData['xsec'] * xaodBkgXsecData['kfac'] * xaodBkgXsecData['eff']))

# Sumw data prepared with dump_xaod_sumw.py
xaodSumwData = np.genfromtxt('../config/sumw.txt', dtype='i4, f8',
                         names=['dsid', 'sumw'])
xaodSumwDict = dict(zip(xaodSumwData['dsid'], xaodSumwData['sumw']))

#-------------------------------------------------------------------------------
# Delphes meta data
#-------------------------------------------------------------------------------
delphesXsecData = np.genfromtxt('../config/DelphesXSec', dtype='S30, f8',
                                names=['dsid', 'xsec'])
delphesXsecDict = dict(delphesXsecData)

#-------------------------------------------------------------------------------
# Helper functions
#-------------------------------------------------------------------------------
def get_xaod_rpv_params(dsid):
    """Get the mass paramaters and theoretical cross section for a given DS ID"""
    return xaodRPVMetaDict[dsid]

def get_xaod_bkg_xsec(dsid):
    """Returns the cross section for a given DS ID"""
    return xaodBkgXsecDict[dsid]

def get_xaod_sumw(dsid):
    """Get the sum of generator weights for a sample"""
    return xaodSumwDict[dsid]

def get_delphes_xsec(dsid):
    # Convert to pb
    return 1
    #return delphesXsecDict[dsid]*1e9

def get_delphes_sumw(dsid):
    """NOT YET IMPLEMENTED; returns 1"""
    return 1
