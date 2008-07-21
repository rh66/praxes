"""
Wrappers around the pytables interface to the hdf5 file.

"""

from __future__ import absolute_import

#---------------------------------------------------------------------------
# Stdlib imports
#---------------------------------------------------------------------------



#---------------------------------------------------------------------------
# Extlib imports
#---------------------------------------------------------------------------



#---------------------------------------------------------------------------
# xpaxs imports
#---------------------------------------------------------------------------

from .array import NXarray
from .registry import class_name_dict

#---------------------------------------------------------------------------
# Normal code begins
#---------------------------------------------------------------------------


class NXcarray(NXarray):

    """
    """



class_name_dict['NXcarray'] = NXcarray
