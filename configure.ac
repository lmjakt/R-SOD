AC_INIT(SODConfig, version-1.0)

AC_PROG_CC
AC_PROG_CXX

AC_LANG(C++)

dnl To find RHOME do 
: ${R_HOME=`R RHOME`}
if test -z "${R_HOME}"; then
   echo "could not determine R_HOME"
   exit 1
fi

CC=`"${R_HOME}/bin/R" CMD config CC`
CFLAGS=`"${R_HOME}/bin/R" CMD config CFLAGS`
CPPFLAGS=`"${R_HOME}/bin/R" CMD config CPPFLAGS`
PKG_LIBS=`${R_HOME}/bin/Rscript -e 'Rcpp:::LdFlags()'` 

dnl The simplest way may be to simply have two different templates
dnl and then to simply rename the correct ones having inserted the
dnl correct variables. Not elegant, but, better than having ifdefs
dnl all over the source code.

dnl Determine if we have openCL installed somewhere
AC_CHECK_HEADERS( CL/cl.h OpenCL/opencl.h )

HAVE_CL=yes

if test "${ac_cv_header_CL_cl_h}" = no -a "${ac_cv_header_OpenCL_opencl_h}" = no; then
   AC_MSG_ERROR("Could not find CL/cl.h or OpenCL/opencl.h")
   HAVE_CL=no
fi

dnl We should also check for the library
AC_CHECK_LIB(OpenCL, clGetDeviceInfo)

echo "ac_cv_lib_OpenCL $ac_cv_lib_OpenCL_clGetDeviceInfo"

if test "$ac_cv_lib_OpenCL_clGetDeviceInfo" = no; then
   HAVE_CL=no
else
   echo "We have the OpenCL libs"
fi


SOD_CPPFLAGS=$CPPFLAGS
SOD_PKG_LIBS=$PKG_LIBS

SOD_CXXFLAGS=""

dnl The following is particularly ugly, but not sure how to improve the readability

SOD_HEADERS="\
R_DimSqueezer.h \
sod/node_set.h \
sod/stressInfo.h \
sod/mappingInfo.h \
sod/DistanceMapper.h"

SOD_CL_HEADERS="\
sod/oCL_DistanceMapper.h \
sod/oCL_DistanceMapperManager.h \
sod/open_cl/oCL_base.h \
sod/open_cl/clError.h \
sod/oCL_Kernels/oCL_Kernels.h"

SOD_SOURCES="\
R_DimSqueezer.cpp \
sod/node_set.cpp \
sod/DistanceMapper.cpp"

SOD_CL_SOURCES="\
sod/oCL_DistanceMapper.cpp \
sod/oCL_DistanceMapperManager.cpp \
sod/open_cl/oCL_base.cpp \
sod/open_cl/clError.cpp"

SOD_OBJECTS="\
R_DimSqueezer.o \
sod/node_set.o \
sod/DistanceMapper.o"

SOD_CL_OBJECTS="\
sod/oCL_DistanceMapper.o \
sod/oCL_DistanceMapperManager.o \
sod/open_cl/oCL_base.o \
sod/open_cl/clError.o"

dnl This should be conditional, but for testing purposes

if test ${HAVE_CL} = yes  ; then
   SOD_PKG_LIBS="${PKG_LIBS} -lOpenCL"
   SOD_CXXFLAGS="-DHAVE_CL"
   SOD_HEADERS="$SOD_HEADERS $SOD_CL_HEADERS"
   SOD_SOURCES="$SOD_SOURCES $SOD_CL_SOURCES"
   SOD_OBJECTS="$SOD_OBJECTS $SOD_CL_OBJECTS"
fi


echo $CC
echo $CFLAGS
echo $CPPFLAGS
echo $PKG_LIBS

AC_SUBST(SOD_CPPFLAGS)
AC_SUBST(SOD_PKG_LIBS)
AC_SUBST(SOD_CXXFLAGS)
AC_SUBST(SOD_HEADERS)
AC_SUBST(SOD_SOURCES)
AC_SUBST(SOD_OBJECTS)
AC_CONFIG_FILES([src/Makevars])
AC_OUTPUT
