AM_CPPFLAGS = @common_cuda_CPPFLAGS@

NVCC       = @NVCC@
ARCH       = @AR@
ARCHFLAGS  = cr
STLIB     ?= opal_datatype_cuda_kernel.a
DYLIB     ?= opal_datatype_cuda_kernel.so

#CFLAGS  = -I$(top_builddir)/opal/include -I$(top_srcdir)/opal/include -I$(top_builddir) -I$(top_srcdir)  --compiler-options '-fPIC -g'
#LDFLAGS = -shared --compiler-options '-fPIC @LDFLAGS@'

cu_sources = opal_datatype_cuda.cu \
    opal_datatype_pack_cuda_kernel.cu \
    opal_datatype_pack_cuda_wrapper.cu \
    opal_datatype_unpack_cuda_kernel.cu \
    opal_datatype_unpack_cuda_wrapper.cu

lib_LTLIBRARIES = libdatatype_cuda_kernels.la

libdatatype_cuda_kernels_la_SOURCES = cu_sources
libdatatype_cuda_kernels_la_CFLAGS = --std c++17 --gpu-architecture=compute_$(OPAL_NVCC_COMPUTE_ARCH)
libdatatype_cuda_kernels_la_LDFLAGS = 

.cu.l$(OBJEXT):
	$(LIBTOOL) $(AM_V_lt) --tag=CC $(AM_LIBTOOLFLAGS) \
	$(LIBTOOLFLAGS) --mode=compile $(OPAL_NVCC) $(libdatatype_cuda_kernels_la_CPPFLAGS) $(PIC_FLAGS) -c $<

