# -*- shell-script -*-
#
# Copyright (c) 2018      Mellanox Technologies.  All rights reserved.
# Copyright (c) 2022      Amazon.com, Inc. or its affiliates.  All Rights reserved.
# Copyright (c) 2026      NVIDIA Corporation.  All rights reserved.
# $COPYRIGHT$
#
# Additional copyrights may follow
#
# $HEADER$
# SPDX-License-Identifier: BSD-3-Clause-Open-MPI
#

# MCA_opal_common_ucx_CONFIG([action-if-can-compile],
#                           [action-if-cant-compile])
# ------------------------------------------------
AC_DEFUN([MCA_opal_common_ucx_CONFIG],[
    AC_CONFIG_FILES([opal/mca/common/ucx/Makefile])

    OMPI_CHECK_UCX([common_ucx],
               [common_ucx_happy="yes"],
               [common_ucx_happy="no"])

    dnl opal_common_ucx_support_level() inventories the node's transports
    dnl through UCT.  OMPI_CHECK_UCX already requires UCX >= 1.9, so the
    dnl uct_component_h API (UCT 1.7) is always available here.
    AS_IF([test "$common_ucx_happy" = "yes"],
          [$1
           common_ucx_LIBS="$common_ucx_LIBS -luct"
          ],
          [$2])

    # substitute in the things needed to build common_ucx
    AC_SUBST([common_ucx_CPPFLAGS])
    AC_SUBST([common_ucx_LDFLAGS])
    AC_SUBST([common_ucx_LIBS])
])dnl


