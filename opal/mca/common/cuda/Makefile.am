#
# Copyright (c) 2026      NVIDIA Corporation.  All rights reserved.
# $COPYRIGHT$
#
# Additional copyrights may follow
#
# $HEADER$
#

# This common component is configured only when CUDA support is enabled and
# usable.  It intentionally links against the CUDA driver library, not cudart.

AM_CPPFLAGS = $(common_cuda_CPPFLAGS)

headers = \
        common_cuda_cmdq.h

sources = \
        common_cuda_cmdq.c

EXTRA_DIST = \
        common_cuda_cmdq_test.cu

lib_LTLIBRARIES =
noinst_LTLIBRARIES =
comp_inst = lib@OPAL_LIB_NAME@mca_common_cuda.la
comp_noinst = lib@OPAL_LIB_NAME@mca_common_cuda_noinst.la

if MCA_BUILD_opal_common_cuda_DSO
lib_LTLIBRARIES += $(comp_inst)
else
noinst_LTLIBRARIES += $(comp_noinst)
endif

lib@OPAL_LIB_NAME@mca_common_cuda_la_SOURCES = \
        $(headers) $(sources)
lib@OPAL_LIB_NAME@mca_common_cuda_la_LDFLAGS = \
        $(common_cuda_LDFLAGS) \
        -version-info $(libmca_opal_common_cuda_so_version)
lib@OPAL_LIB_NAME@mca_common_cuda_la_LIBADD = \
        $(common_cuda_LIBS)

lib@OPAL_LIB_NAME@mca_common_cuda_noinst_la_SOURCES = \
        $(headers) $(sources)
lib@OPAL_LIB_NAME@mca_common_cuda_noinst_la_LDFLAGS = \
        $(common_cuda_LDFLAGS)
lib@OPAL_LIB_NAME@mca_common_cuda_noinst_la_LIBADD = \
        $(common_cuda_LIBS)

if WANT_INSTALL_HEADERS
opaldir = $(opalincludedir)/$(subdir)
opal_HEADERS = $(headers)
endif

V=0
OMPI_V_LN_SCOMP = $(ompi__v_LN_SCOMP_$V)
ompi__v_LN_SCOMP_ = $(ompi__v_LN_SCOMP_$AM_DEFAULT_VERBOSITY)
ompi__v_LN_SCOMP_0 = @echo "  LN_S    " `basename $(comp_inst)`;

all-local:
	$(OMPI_V_LN_SCOMP) if test -z "$(lib_LTLIBRARIES)"; then \
	  rm -f "$(comp_inst)"; \
	  $(LN_S) "$(comp_noinst)" "$(comp_inst)"; \
	fi

clean-local:
	if test -z "$(lib_LTLIBRARIES)"; then \
	  rm -f "$(comp_inst)"; \
	fi
