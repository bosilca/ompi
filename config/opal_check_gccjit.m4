dnl -*- shell-script -*-
dnl
dnl Copyright (c) 2023      The University of Tennessee and The University
dnl                         of Tennessee Research Foundation.  All rights
dnl                         reserved.
dnl $COPYRIGHT$
dnl
dnl Additional copyrights may follow
dnl
dnl $HEADER$
dnl

# OPAL_CHECK_GCCJIT(prefix, [action-if-found], [action-if-not-found])
# --------------------------------------------------------
# check if gccjit support can be found.  sets prefix_{CPPFLAGS,
# LDFLAGS, LIBS} as needed and runs action-if-found if there is
# support, otherwise executes action-if-not-found
AC_DEFUN([OPAL_CHECK_GCCJIT],[
    OPAL_VAR_SCOPE_PUSH([opal_check_gccjit_happy opal_check_gccjit_CPPFLAGS_save opal_check_gccjit_LDFLAGS_save opal_check_gccjit_LIBS_save])

    m4_ifblank([$1], [m4_fatal([First argument to OPAL_CHECK_GCCJIT cannot be blank])])

    AC_ARG_WITH([gccjit],
                [AS_HELP_STRING([--with-gccjit(=DIR)],
                                [Build with libgccjit library support])])
    AC_ARG_WITH([gccjit-libdir],
                [AS_HELP_STRING([--with-gccjit-libdir=DIR],
                                [Search for gccjit libraries in DIR])])
    AC_ARG_WITH([gccjit-incdir],
                [AS_HELP_STRING([--with-gccjit-incdir=DIR],
                                [Search for gccjit includes in DIR])])

    OAC_CHECK_PACKAGE([gccjit],
                      [$1],
                      [libgccjit.h],
                      [gccjit],
                      [gcc_jit_context_acquire],
                      [opal_check_gccjit_happy="yes"],
                      [opal_check_gccjit_happy="no"])

    OPAL_SUMMARY_ADD([Accelerators], [libgccjit], [], [$opal_check_gccjit_happy])

    AS_IF([test ! -z "$with_gccjit" && test "$with_gccjit" != "no" && test "$opal_check_gccjit_happy" != "yes"],
                 [AC_MSG_ERROR([libgccjit support requested but not found.  Aborting])])
    AS_IF([test "$opal_check_gccjit_happy" == "yes"],
          [AC_DEFINE(OPAL_HAVE_LIBGCCJIT, ,
                     [Whether your compiler has support for libgccjit])])
    # substitute in the things needed to build gccjit support
    AC_SUBST([gccjit_CPPFLAGS])
    AC_SUBST([gccjit_LDFLAGS])
    AC_SUBST([gccjit_LIBS]) 
    OPAL_VAR_SCOPE_POP
])

