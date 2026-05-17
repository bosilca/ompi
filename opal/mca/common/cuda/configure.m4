# -*- shell-script -*-
#
# Copyright (c) 2026      NVIDIA Corporation.  All rights reserved.
# $COPYRIGHT$
#
# Additional copyrights may follow
#
# $HEADER$
#

# MCA_opal_common_cuda_CONFIG([action-if-can-compile],
#                            [action-if-cant-compile])
# --------------------------------------------------
AC_DEFUN([MCA_opal_common_cuda_CONFIG],[
    OPAL_VAR_SCOPE_PUSH([common_cuda_happy])

    AC_CONFIG_FILES([opal/mca/common/cuda/Makefile])

    OPAL_CHECK_CUDA([common_cuda])

    AS_IF([test "x$CUDA_SUPPORT" = "x1" && test "x$CUDA_VERSION_60_OR_GREATER" = "x1"],
          [common_cuda_happy=yes],
          [common_cuda_happy=no])

    AS_IF([test "$common_cuda_happy" = "yes"],
          [$1],
          [$2])

    AC_SUBST([common_cuda_CPPFLAGS])
    AC_SUBST([common_cuda_LDFLAGS])
    AC_SUBST([common_cuda_LIBS])

    OPAL_VAR_SCOPE_POP
])dnl
