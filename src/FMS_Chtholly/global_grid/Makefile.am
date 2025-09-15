#***********************************************************************
#*                   GNU Lesser General Public License
#*
#* This file is part of the GFDL Flexible Modeling System (FMS).
#*
#* FMS is free software: you can redistribute it and/or modify it under
#* the terms of the GNU Lesser General Public License as published by
#* the Free Software Foundation, either version 3 of the License, or (at
#* your option) any later version.
#*
#* FMS is distributed in the hope that it will be useful, but WITHOUT
#* ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
#* FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License
#* for more details.
#*
#* You should have received a copy of the GNU Lesser General Public
#* License along with FMS.  If not, see <http://www.gnu.org/licenses/>.
#***********************************************************************

# This is an automake file for the global_grid directory of the FMS
# package.

# Include .mod files.
AM_CPPFLAGS = -I$(top_srcdir)/include
AM_FCFLAGS = $(FC_MODINC). $(FC_MODOUT)$(MODDIR)

# Build an uninstalled convenience library for global_grid
noinst_LTLIBRARIES = libglobal_grid.la

# Each convenience library depends on its source.
libglobal_grid_la_SOURCES = \
  lib_grid.F90 \
  global_grid_data.F90 \
  global_grid_alloc.F90 \
  global_grid_util.F90 \
  global_grid_gen_lonlat.F90 \
  global_grid_gen_coords.F90 \
  global_grid_gen_cell.F90 \
  global_grid_gen_vec.F90 \
  global_grid_gen_k2e.F90 \
  global_grid_base.F90 \
  global_grid.F90

# Mod files are built and then installed as headers.
MODFILES = \
  lib_grid_mod.$(FC_MODEXT) \
  global_grid_data_mod.$(FC_MODEXT) \
  global_grid_alloc_mod.$(FC_MODEXT) \
  global_grid_util_mod.$(FC_MODEXT) \
  global_grid_gen_lonlat_mod.$(FC_MODEXT) \
  global_grid_gen_coords_mod.$(FC_MODEXT) \
  global_grid_gen_cell_mod.$(FC_MODEXT) \
  global_grid_gen_vec_mod.$(FC_MODEXT) \
  global_grid_gen_k2e_mod.$(FC_MODEXT) \
  global_grid_base_mod.$(FC_MODEXT) \
  global_grid_mod.$(FC_MODEXT)

nodist_include_HEADERS = $(MODFILES)
BUILT_SOURCES = $(MODFILES)

include $(top_srcdir)/mkmods.mk


