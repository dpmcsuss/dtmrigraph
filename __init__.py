import os

import pyximport;
pyximport.install()

os.sys.path.extend([os.path.abspath('./PyGraphStat/code/'),
                 os.path.abspath('./dpUtils/src/'), 
                 os.path.abspath('./stfpvmSimulations/src/'),
                 os.path.abspath('./MR-connectome/mrcap/')])

__all__ = ['brain']