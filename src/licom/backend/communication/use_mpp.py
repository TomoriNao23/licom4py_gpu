# from pyfms import mpp, mpp_domains
# from licom.pyfms_mod.duogrid import Duogrid
# from licom.pyfms_mod.mp_data import MpDomain
# from licom.backend.calculation.field import BackendConfig, Field
# import numpy as np
# from typing import Tuple


# class Communication_mpp:

#     @classmethod
#     def init(cls, mp: MpDomain, backend_cfg: BackendConfig, dg: Duogrid):
#         """
#         Initialize communication handler with MPP domain and backend configuration.
        
#         Args:
#             mp: MPP domain configuration containing domain_id
#             backend_cfg: Backend configuration containing library type
#         """
#         cls.domain_id = mp.domain_id
#         cls.lib = backend_cfg.lib
#         cls.mp = mp
#         cls.dg = dg

#         # Pre-configure the update function based on backend library
#         if cls.lib == 'numpy':
#             cls._update_domains = lambda var: mpp_domains.update_domains(
#                 var,
#                 cls.domain_id,
#             )
#         elif cls.lib == 'jax':
#             cls._update_domains = lambda var: mpp_domains.update_domains(
#                 np.asarray(var),
#                 cls.domain_id,
#             )

#         # locate the domain
#         cls.rmp_w = True if cls.mp.is_ == 1 else False
#         cls.rmp_e = True if cls.mp.ie == cls.mp.nx else False
#         cls.rmp_s = True if cls.mp.js == 1 else False
#         cls.rmp_n = True if cls.mp.je == cls.mp.ny else False
#         cls.rmp_sw = cls.rmp_s and cls.rmp_w
#         cls.rmp_se = cls.rmp_s and cls.rmp_e
#         cls.rmp_nw = cls.rmp_n and cls.rmp_w
#         cls.rmp_ne = cls.rmp_n and cls.rmp_e

#     @classmethod
#     def ext_scalar(cls, var):
#         """
#         Exchange scalar values across domain boundaries.
        
#         Args:
#             var: Variable to exchange (numpy array or JAX array)
#         """
#         cls._update_domains(var)
#         return var#cls._cube_rmp(var)

#     @classmethod
#     def get_interior_wind(cls, u: Field, v: Field):
#         """
#         Get interior wind values by converting from Cartesian to local coordinates.
        
#         This function performs the matrix transformation equivalent to the Fortran code:
#         u2 = [u(i,j), v(i,j)]
#         u2 = matmul(dg%a_c2l(:,:,i,j), u2)
#         ull(i,j) = u2(1)
#         vll(i,j) = u2(2)
        
#         Args:
#             u: U-component of wind (Cartesian coordinates)
#             v: V-component of wind (Cartesian coordinates) 
#             dg: Duogrid object containing transformation matrices a_c2l
            
#         Returns:
#             Tuple of (ull, vll): Wind components in local coordinates
#         """
#         # Vectorized transformation using broadcasting (works for NumPy and JAX)
#         a = cls.dg.a_c2l  # shape: (ni, nj, 2, 2)

#         ull = u * a[:, :, 0, 0] + v * a[:, :, 0, 1]
#         vll = u * a[:, :, 1, 0] + v * a[:, :, 1, 1]
        
#         return ull, vll



#     @classmethod
#     def _cube_rmp(cls, var):
#         """
#         Port of Fortran subroutine cube_rmp(var, dg) to Python.

#         Convert halo/edge values on a cubed-sphere tile using precomputed
#         k2e mappings and coefficients stored in `dg`.

#         Args:
#             var: 2D array-like Field for which boundary remapping is applied.

#         Returns:
#             The updated `var` after remapping (for JAX it returns a new array).
#         """
#         mp = cls.mp
#         dg = cls.dg

#         # Compute domain in Python local indices: is -> ng, ie -> ni-1-ng; js -> ng, je -> nj-1-ng
#         isd = mp.isd+2;jsd = mp.jsd+2
#         ied = mp.ied+2;jed = mp.jed+2
        
#         is_ = mp.is_+2; js = mp.js+2
#         ie = mp.ie+2; je = mp.je+2

#         ng = mp.ng

#         # Work array
#         var_kik = Field.new('2d')
#         import jax.numpy as jnp
#         # Copy neighbor rings to work array (Fortran: i=isd:ied -> Python: : ; j/js/je mapped to js0/je0)
#         for ii in range(1, ng + 1):
#             for i in range(isd, ied + 1):
#                 j = js - ii
#                 var_kik = Field.set_(var_kik, (i, j), var[i, j])
#                 j = je + ii
#                 var_kik = Field.set_(var_kik, (i, j), var[i, j])
#             for j in range(jsd, jed + 1):
#                 i = is_ - ii
#                 var_kik = Field.set_(var_kik, (i, j), var[i, j])
#                 i = ie + ii
#                 var_kik = Field.set_(var_kik, (i, j), var[i, j])

#         # k2e parameters
#         coef = dg.k2e_coef  # (ni, nj, nord)
#         loc_arr = dg.k2e_loc  # (ni, nj) Fortran physical indices
#         nord = 2
#         offset = 1

#         # if mpp.pe() == 0:
#         #     for j in range(var_kik.shape[1]):
#         #         for i in range(var_kik.shape[0]):
#         #             print(i-2,j-2,f"{var_kik[i,j]:.8f}")

#         # South boundary
#         if cls.rmp_s:
#             for ii in range(1, ng + 1):
#                 j = js - ii
#                 for i in range(is_, ie + 1):
#                     loc = int(loc_arr[i, j])
#                     lo = loc - offset
#                     var = Field.set_(var, (i, j), 0)
#                     for n in range(1, nord+1):
#                         var = Field.set_(var, (i, j), var[i,j] + var_kik[lo+n+2,j]*coef[i,j,n-1])
#                         #if mpp.pe() == 0:
#                             #print(i-2,j-2,lo+n, var_kik[lo+n+2,j],coef[i,j,n-1])

#         # North boundary
#         if cls.rmp_n:
#             for ii in range(1, ng + 1):
#                 j = je + ii
#                 for i in range(is_, ie + 1):
#                     loc = int(loc_arr[i, j])
#                     lo = loc - offset
#                     var = Field.set_(var, (i, j), 0)
#                     for n in range(1, nord+1):
#                         var = Field.set_(var, (i, j), var[i,j] + var_kik[lo+n+2,j]*coef[i,j,n-1])
#         # West boundary
#         if cls.rmp_w:
#             for ii in range(1, ng + 1):
#                 i = is_ - ii
#                 for j in range(js, je + 1):
#                     loc = int(loc_arr[i, j])
#                     lo = loc - offset
#                     var = Field.set_(var, (i, j), 0)
#                     for n in range(1, nord+1):
#                         var = Field.set_(var, (i, j), var[i,j] + var_kik[i,lo+n+2]*coef[i,j,n-1])
#         # East boundary
#         if cls.rmp_e:
#             for ii in range(1, ng + 1):
#                 i = ie + ii
#                 for j in range(js, je + 1):
#                     loc = int(loc_arr[i, j])
#                     lo = loc - offset
#                     var = Field.set_(var, (i, j), 0)
#                     for n in range(1, nord+1):
#                         var = Field.set_(var, (i, j), var[i,j] + var_kik[i,lo+n+2]*coef[i,j,n-1])

#         # Copy corners (need to check if rmp)
#         # sw
#         for i in range(isd, is_-1 + 1):
#             for j in range(jsd, js-1 + 1):
#                 var = Field.set_(var, (i, j), var[is_, j])

#         # se
#         for i in range(ie+1, ied + 1):
#             for j in range(jsd, js-1 + 1):
#                 var = Field.set_(var, (i, j), var[ie, j])

#         # ne
#         for i in range(ie+1, ied + 1):
#             for j in range(je+1, jed + 1):
#                 var = Field.set_(var, (i, j), var[ie, j])

#         # nw
#         for i in range(isd, is_-1 + 1):
#             for j in range(je+1, jed + 1):
#                 var = Field.set_(var, (i, j), var[is_, j])

#         return var

