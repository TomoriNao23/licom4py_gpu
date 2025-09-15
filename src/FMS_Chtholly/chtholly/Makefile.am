# This is an automake file for the chtholly directory of the FMS
# package.

# Include .h and .mod files.
AM_CPPFLAGS = -I${top_srcdir}/include
AM_CPPFLAGS += -I${top_builddir}/platform
AM_CPPFLAGS += -I${top_builddir}/mpp
AM_CPPFLAGS += -I${top_builddir}/fms

# Build this uninstalled convenience library.
noinst_LTLIBRARIES = libchtholly.la

# The convenience library depends on its source.
libchtholly_la_SOURCES = wrapper.F90

# Mod file depends on its o file, is built and then installed.
chtholly_c_rapper_mod.mod: wrapper.$(OBJEXT)
BUILT_SOURCES = chtholly_c_rapper_mod.mod
include_HEADERS = chtholly_c_rapper_mod.mod

CLEANFILES = *.mod
