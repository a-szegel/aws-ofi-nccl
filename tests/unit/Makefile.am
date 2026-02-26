#
# Copyright (c) 2018-2025, Amazon.com, Inc. or its affiliates. All rights reserved.
#
# See LICENSE.txt for license information
#

# Please remember to update .gitignore in this directory.

if ENABLE_UNIT_TESTS
AM_CPPFLAGS = -I$(top_srcdir)/include
AM_CPPFLAGS += -isystem $(abs_top_srcdir)/3rd-party
AM_CPPFLAGS += -isystem $(abs_top_srcdir)/3rd-party/nccl/$(DEVICE_INTERFACE)/include
LDADD = $(top_builddir)/src/libinternal_plugin.la
noinst_HEADERS = test-logger.h

if ENABLE_GTEST
AM_CPPFLAGS += $(GTEST_CPPFLAGS)
noinst_HEADERS += libfabric_mock.h
endif

noinst_PROGRAMS = \
	assert \
	environ \
	freelist \
	msgbuff \
	scheduler \
	idpool \
	ep_addr_list \
	mr \
	histogram_binner \
	histogram \
	dlopen_c_test \
	param \
	platform_manager

if ENABLE_GTEST
# Linker flags to wrap all mocked inline libfabric functions.
# Add new --wrap entries here when extending the mock.
GTEST_MOCK_WRAP_FLAGS = \
	-Wl,--wrap=fi_allocinfo,--wrap=fi_fabric,--wrap=fi_domain,--wrap=fi_endpoint \
	-Wl,--wrap=fi_close,--wrap=fi_enable,--wrap=fi_ep_bind \
	-Wl,--wrap=fi_mr_bind,--wrap=fi_mr_desc,--wrap=fi_mr_enable,--wrap=fi_mr_key,--wrap=fi_mr_regattr \
	-Wl,--wrap=fi_av_insert,--wrap=fi_av_open \
	-Wl,--wrap=fi_cq_open,--wrap=fi_cq_read,--wrap=fi_cq_readfrom,--wrap=fi_cq_readerr,--wrap=fi_cq_strerror \
	-Wl,--wrap=fi_send,--wrap=fi_recv,--wrap=fi_senddata,--wrap=fi_recvmsg \
	-Wl,--wrap=fi_tsend,--wrap=fi_trecv \
	-Wl,--wrap=fi_read,--wrap=fi_write,--wrap=fi_writedata,--wrap=fi_writemsg \
	-Wl,--wrap=fi_setopt,--wrap=fi_getopt,--wrap=fi_getname

noinst_PROGRAMS += libfabric_mock_test
libfabric_mock_test_SOURCES = libfabric_mock_test.cpp libfabric_mock_impl.cpp
libfabric_mock_test_CPPFLAGS = $(AM_CPPFLAGS) $(GTEST_CPPFLAGS)
libfabric_mock_test_CXXFLAGS = -Wno-error
libfabric_mock_test_LDADD = $(GTEST_LIBS)
libfabric_mock_test_LDFLAGS = $(GTEST_LDFLAGS) $(GTEST_MOCK_WRAP_FLAGS)

noinst_PROGRAMS += ofiutils_get_providers_test
ofiutils_get_providers_test_SOURCES = ofiutils_get_providers_test.cpp libfabric_mock_impl.cpp \
	$(top_srcdir)/src/nccl_ofi_ofiutils.cpp
ofiutils_get_providers_test_CPPFLAGS = $(AM_CPPFLAGS) $(GTEST_CPPFLAGS)
ofiutils_get_providers_test_CXXFLAGS = -Wno-error
ofiutils_get_providers_test_LDADD = $(GTEST_LIBS)
ofiutils_get_providers_test_LDFLAGS = $(GTEST_LDFLAGS) $(GTEST_MOCK_WRAP_FLAGS)

noinst_PROGRAMS += nccl_ofi_math_test
nccl_ofi_math_test_SOURCES = nccl_ofi_math_test.cpp
nccl_ofi_math_test_CPPFLAGS = $(AM_CPPFLAGS) $(GTEST_CPPFLAGS)
nccl_ofi_math_test_CXXFLAGS = -Wno-error
nccl_ofi_math_test_LDADD = $(GTEST_LIBS)
nccl_ofi_math_test_LDFLAGS = $(GTEST_LDFLAGS)

noinst_PROGRAMS += msgbuff_gtest
msgbuff_gtest_SOURCES = msgbuff_test.cpp
msgbuff_gtest_CPPFLAGS = $(AM_CPPFLAGS) $(GTEST_CPPFLAGS)
msgbuff_gtest_CXXFLAGS = -Wno-error
msgbuff_gtest_LDADD = $(GTEST_LIBS) $(LDADD)
msgbuff_gtest_LDFLAGS = $(GTEST_LDFLAGS)

noinst_PROGRAMS += idpool_gtest
idpool_gtest_SOURCES = idpool_test.cpp
idpool_gtest_CPPFLAGS = $(AM_CPPFLAGS) $(GTEST_CPPFLAGS)
idpool_gtest_CXXFLAGS = -Wno-error
idpool_gtest_LDADD = $(GTEST_LIBS) $(LDADD)
idpool_gtest_LDFLAGS = $(GTEST_LDFLAGS)

noinst_PROGRAMS += ep_addr_list_gtest
ep_addr_list_gtest_SOURCES = ep_addr_list_test.cpp
ep_addr_list_gtest_CPPFLAGS = $(AM_CPPFLAGS) $(GTEST_CPPFLAGS)
ep_addr_list_gtest_CXXFLAGS = -Wno-error
ep_addr_list_gtest_LDADD = $(GTEST_LIBS) $(LDADD)
ep_addr_list_gtest_LDFLAGS = $(GTEST_LDFLAGS)

noinst_PROGRAMS += scheduler_gtest
scheduler_gtest_SOURCES = scheduler_test.cpp
scheduler_gtest_CPPFLAGS = $(AM_CPPFLAGS) $(GTEST_CPPFLAGS)
scheduler_gtest_CXXFLAGS = -Wno-error
scheduler_gtest_LDADD = $(GTEST_LIBS) $(LDADD)
scheduler_gtest_LDFLAGS = $(GTEST_LDFLAGS)

noinst_PROGRAMS += freelist_gtest
freelist_gtest_SOURCES = freelist_test.cpp
freelist_gtest_CPPFLAGS = $(AM_CPPFLAGS) $(GTEST_CPPFLAGS)
freelist_gtest_CXXFLAGS = -Wno-error
freelist_gtest_LDADD = $(GTEST_LIBS) $(LDADD)
freelist_gtest_LDFLAGS = $(GTEST_LDFLAGS)

noinst_PROGRAMS += mr_cache_gtest
mr_cache_gtest_SOURCES = mr_cache_test.cpp
mr_cache_gtest_CPPFLAGS = $(AM_CPPFLAGS) $(GTEST_CPPFLAGS)
mr_cache_gtest_CXXFLAGS = -Wno-error
mr_cache_gtest_LDADD = $(GTEST_LIBS) $(LDADD)
mr_cache_gtest_LDFLAGS = $(GTEST_LDFLAGS)

noinst_PROGRAMS += param_gtest
param_gtest_SOURCES = param_test.cpp
param_gtest_CPPFLAGS = $(AM_CPPFLAGS) $(GTEST_CPPFLAGS)
param_gtest_CXXFLAGS = -Wno-error
param_gtest_LDADD = $(GTEST_LIBS) $(LDADD)
param_gtest_LDFLAGS = $(GTEST_LDFLAGS)

noinst_PROGRAMS += histogram_gtest
histogram_gtest_SOURCES = histogram_gtest.cpp
histogram_gtest_CPPFLAGS = $(AM_CPPFLAGS) $(GTEST_CPPFLAGS)
histogram_gtest_CXXFLAGS = -Wno-error
histogram_gtest_LDADD = $(GTEST_LIBS) $(LDADD)
histogram_gtest_LDFLAGS = $(GTEST_LDFLAGS)

noinst_PROGRAMS += pthread_wrapper_gtest
pthread_wrapper_gtest_SOURCES = pthread_wrapper_test.cpp
pthread_wrapper_gtest_CPPFLAGS = $(AM_CPPFLAGS) $(GTEST_CPPFLAGS)
pthread_wrapper_gtest_CXXFLAGS = -Wno-error
pthread_wrapper_gtest_LDADD = $(GTEST_LIBS) $(LDADD)
pthread_wrapper_gtest_LDFLAGS = $(GTEST_LDFLAGS)
endif

if WANT_PLATFORM_AWS
noinst_PROGRAMS += aws_platform_mapper
endif

if !ENABLE_NEURON
  AM_LDFLAGS = $(CUDA_LDFLAGS)
  AM_CPPFLAGS += $(CUDA_CPPFLAGS)
  LDADD += $(CUDA_LIBS)
  noinst_PROGRAMS += gdrcopy
  gdrcopy_SOURCES = gdrcopy.cpp
if WANT_PLATFORM_AWS
  noinst_PROGRAMS += region_based_tuner
  region_based_tuner_SOURCES = region_based_tuner.cpp
endif
endif

assert_SOURCES = assert.cpp
environ_SOURCES = environ.cpp
idpool_SOURCES = idpool.cpp
freelist_SOURCES = freelist.cpp
msgbuff_SOURCES = msgbuff.cpp
scheduler_SOURCES = scheduler.cpp
ep_addr_list_SOURCES = ep_addr_list.cpp
mr_SOURCES = mr.cpp
aws_platform_mapper_SOURCES = aws_platform_mapper.cpp
histogram_binner_SOURCES = histogram_binner.cpp
histogram_SOURCES = histogram.cpp
dlopen_c_test_SOURCES = dlopen_c_test.c
param_SOURCES = param.cpp
platform_manager_SOURCES = platform_manager.cpp

TESTS = $(noinst_PROGRAMS)
endif
