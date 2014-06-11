AC_INIT(SODConfig, version-1.0)

AC_PROG_CC
AC_PROG_CXX

AC_LANG(C++)

dnl To find the host os do
dnl AC_CANONICAL_HOST
dnl AC_CANONICAL_HOST fails on the OSX.
dnl we can try uname -s instead. Not as good, but
host_os=`uname -s`
echo "Host is $host_os"

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

dnl Determine if we have openCL installed somewhere
dnl CL/cl.h on Linux, OpenCL/opencl.h on MacOS

AC_CHECK_HEADERS( CL/cl.h OpenCL/opencl.h )

HAVE_CL=yes

if test "${ac_cv_header_CL_cl_h}" = no -a "${ac_cv_header_OpenCL_opencl_h}" = no; then
#   AC_MSG_ERROR("Could not find CL/cl.h or OpenCL/opencl.h") ## no need to fail 
   echo "Could not find OpenCL headers will not compile openCL dependancies"
   HAVE_CL=no
fi

dnl We should also check for the library
dnl but the way we do this varies from OS to OS

# if host is not Darwin assume it is Linux
if test $host_os != Darwin; then
   echo "Host is not Darwin, assuming Linux: checking for shared OpenCL libraries"
   AC_CHECK_LIB(OpenCL, clGetDeviceInfo)

   if test "$ac_cv_lib_OpenCL_clGetDeviceInfo" = no; then
      echo "We couldn't find the openCL library, but will ignore it"
      HAVE_CL=no
   else
      echo "We have the OpenCL libs"
   fi
fi

dnl If we are on the MacOS we need to check for a framework. Unfortunately this isn't built
dnl in, and we need to define our own macro. I got the following from:
dnl http://lists.apple.com/archives/unix-porting/2009/Jan/msg00026.html
dnl courtesy of: Peter O'Gorman
dnl
dnl unfortunately I don't really understand it as I don't know m4
dnl but it seems to work.

m4_define([MY_CHECK_FRAMEWORK],
  [AC_CACHE_CHECK([if -framework $1 works],[my_cv_framework_$1],
     [save_LIBS="$LIBS"
     LIBS="$LIBS -framework $1"
     AC_LINK_IFELSE([AC_LANG_PROGRAM([],[])],
             [my_cv_framework_$1=yes],
            [my_cv_framework_$1=no])
     LIBS="$save_LIBS"
    ])
   if test "$my_cv_framework_$1"="yes"; then
       AC_DEFINE(AS_TR_CPP([HAVE_FRAMEWORK_$1]),1,
            [Define if you have the  $1 framework])
   AS_TR_CPP([FRAMEWORK_$1])="-framework $1"
   AC_SUBST(AS_TR_CPP([FRAMEWORK_$1]))
   fi]
)

if test $host_os = Darwin; then
   echo "Host is Darwin checking for OpenCL framework"
   MY_CHECK_FRAMEWORK([OpenCL])
   echo "my_cv_framework_opencl $my_cv_framework_OpenCL"
   if test $my_cv_framework_OpenCL = no; then
      HAVE_CL=no
   fi
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

## here we test if the system is Darwin, and assume if it is not
## that we're on Linux or other sane platform. If not
if test ${HAVE_CL} = yes; then
   if test $host_os = Darwin; then
      	SOD_PKG_LIBS="${PKG_LIBS} -framework OpenCL"
   else
	SOD_PKG_LIBS="${PKG_LIBS} -lOpenCL"
   fi	
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
