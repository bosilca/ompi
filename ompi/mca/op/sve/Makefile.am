#
# Copyright (c) 2019      The University of Tennessee and The University
#                         of Tennessee Research Foundation.  All rights
#                         reserved.
# $COPYRIGHT$
#
# Additional copyrights may follow
#
# $HEADER$
#

# This is an sve op component.  This Makefile.am is a typical
# sve of how to integrate into Open MPI's Automake-based build
# system.
#
# See https://github.com/open-mpi/ompi/wiki/devel-CreateComponent
# for more details on how to make Open MPI components.

# First, list all .h and .c sources.  It is necessary to list all .h
# files so that they will be picked up in the distribution tarball.

sources = \
    op_sve.h \
    op_sve_component.c \
    op_sve_functions.h \
    op_sve_functions.c

# Open MPI components can be compiled two ways:
#
# 1. As a standalone dynamic shared object (DSO), sometimes called a
# dynamically loadable library (DLL).
#
# 2. As a static library that is slurped up into the upper-level
# libmpi library (regardless of whether libmpi is a static or dynamic
# library).  This is called a "Libtool convenience library".
#
# The component needs to create an output library in this top-level
# component directory, and named either mca_<type>_<name>.la (for DSO
# builds) or libmca_<type>_<name>.la (for static builds).  The OMPI
# build system will have set the
# MCA_BUILD_ompi_<framework>_<component>_DSO AM_CONDITIONAL to indicate
# which way this component should be built.

if MCA_BUILD_ompi_op_sve_DSO
component_noinst  =
component_install = mca_op_sve.la
else
component_install =
component_noinst  = libmca_op_sve.la
endif

# Specific information for DSO builds.
#
# The DSO should install itself in $(ompilibdir) (by default,
# $prefix/lib/openmpi).

mcacomponentdir = $(ompilibdir)
mcacomponent_LTLIBRARIES = $(component_install)
mca_op_sve_la_SOURCES = $(sources)
mca_op_sve_la_LDFLAGS = -module -avoid-version
mca_op_sve_la_LIBADD = $(top_builddir)/ompi/lib@OMPI_LIBMPI_NAME@.la

# Specific information for static builds.
#
# Note that we *must* "noinst"; the upper-layer Makefile.am's will
# slurp in the resulting .la library into libmpi.

noinst_LTLIBRARIES = $(component_noinst)
libmca_op_sve_la_SOURCES = $(sources)
libmca_op_sve_la_LDFLAGS = -module -avoid-version
