"""
File: initial_all.py
Description: Main initialization class for LICOM model, setting up namelist,
    MPP configuration, and duogrid for the cubed-sphere mosaic.

Author: Chtholly <mengleshan@mail.iap.ac.cn>
Created: 2025-09-03
Updated: 2025-09-16
"""
# Standard library imports
import sys
import os


# Local application imports
from readnamelist import Namelist
from backend import FMS_chtholly
from backend.calculation.field import Field
from duogrid.duogrid import Duogrid as Dg

import jax.numpy as jnp
import numpy as np

class Initial:

    namelist: Namelist
    
    def __init__(self):

        # namelist init
        self.namelist = Namelist.create()

        # backend.FMS_chtholly init
        FMS_chtholly.init(self.namelist)

        # backend.calculation init
        Field.init(FMS_chtholly.mp)

        # duogrid init
        Dg.init(FMS_chtholly.mp)

        a = Field.new('2d')
        for i in range(Dg.mp.xsize):
            for j in range(Dg.mp.ysize):
                #a[i,j] = self.mp.tile*100 + (i-2) + (j-2)*0.01
                a=Field.set_(a,(i,j), Dg.mp.tile*10000 + (i+Dg.mp.isd)*100 + (j+Dg.mp.jsd))
        a = FMS_chtholly.ext_scalar(a)
        print("cpu", Dg.mp.pe, "sum", f"{np.sum(a):.8f}") if (Dg.mp.pe == 6) else None
        if Dg.mp.pe == 6:
            for j in range(Dg.mp.ysize):
                for i in range(Dg.mp.xsize):
                    print("i=",Dg.mp.isd+i, "j=",Dg.mp.jsd+j, a[i,j])
        a = Field.new('2d')

    def __del__(self):
        """
        Destructor: automatically called when the object is about to be destroyed.
        Ensures FMS end is called, similar to how __init__ is used for initialization.
        """
        FMS_chtholly.end()

        # a = np.zeros((Dg.mp.xsize, Dg.mp.ysize),dtype=np.float64)
        # for i in range(Dg.mp.xsize):
        #     for j in range(Dg.mp.ysize):
        #         #a[i,j] = self.mp.tile*100 + (i-2) + (j-2)*0.01
        #         a[i,j] = Dg.mp.tile*100 + (i+Dg.mp.isd) + (j+Dg.mp.jsd)*0.01
        # a = FMS_chtholly.ext_scalar(a)

        # a = Field.new('2d')
        # for i in range(Dg.mp.xsize):
        #     for j in range(Dg.mp.ysize):
        #         #a[i,j] = self.mp.tile*100 + (i-2) + (j-2)*0.01
        #         a=Field.set_(a,(i,j), Dg.mp.tile*10000 + (i+Dg.mp.isd)*100 + (j+Dg.mp.jsd))
        # a = FMS_chtholly.ext_scalar(a)
        # print("cpu", Dg.mp.pe, "sum", f"{np.sum(a):.8f}") if (Dg.mp.pe == 6) else None
        # if Dg.mp.pe == 6:
        #     for j in range(Dg.mp.ysize):
        #         for i in range(Dg.mp.xsize):
        #             print("i=",Dg.mp.isd+i, "j=",Dg.mp.jsd+j, a[i,j])
         # a = Field.new('2d')


        # # 方法2：使用 JAX 的 full_like 重新创建数组
        # for i in range(Dg.mp.xsize):
        #     for j in range(Dg.mp.ysize):
        #         a = Field.set_(a,(i,j),Dg.mp.tile*10000 + (i+Dg.mp.isd)*100 + (j+Dg.mp.jsd))

        # a = np.array(a)
        # print(jnp.sum(a)) if (Dg.mp.pe == 6) else None
        # for i in range(100000):
        #      a = FMS_chtholly.ext_scalar(a)
        # if (Dg.mp.pe == 6):
        #     print(jnp.sum(a))
        #     for j in range(Dg.mp.ysize):
        #         for i in range(Dg.mp.xsize):
        #             print(i+Dg.mp.isd, j+Dg.mp.jsd, a[i,j])


