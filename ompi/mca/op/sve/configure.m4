# -*- shell-script -*-
#
# Copyright (c) 2019-2020 The University of Tennessee and The University
#                         of Tennessee Research Foundation.  All rights
#                         reserved.
#
# $COPYRIGHT$
#
# Additional copyrights may follow
#
# $HEADER$
#

# MCA_ompi_op_sve_CONFIG([action-if-can-compile],
#		         [action-if-cant-compile])
# ------------------------------------------------
# We can always build, unless we were explicitly disabled.
AC_DEFUN([MCA_ompi_op_sve_CONFIG],[
    AC_CONFIG_FILES([ompi/mca/op/sve/Makefile])
    case "${host}" in
        aarch64)
            op_sve_support="yes";;
        *)
            op_sve_support="no";;
    esac
    [$1]
    AM_CONDITIONAL([MCA_BUILD_ompi_op_has_sve_support],
                   [test "$op_sve_support" = "yes"])
    AC_SUBST(MCA_BUILD_ompi_op_has_sve_support)
    
    AS_IF([test $op_sve_support == "yes"],
          [$1],
          [$2])
])dnl
