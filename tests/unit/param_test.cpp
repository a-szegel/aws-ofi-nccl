/*
 * Copyright (c) 2025 Amazon.com, Inc. or its affiliates. All rights reserved.
 */
#include "config.h"
#include <gtest/gtest.h>
#include <thread>
#include <vector>
#include <climits>

#include "nccl_ofi_param_impl.h"
#include "nccl_ofi_log.h"
#include "test-logger.h"

class ParamTest : public ::testing::Test {
protected:
	void SetUp() override { ofi_log_function = logger; }
};

/* --- string_to_value tests --- */

TEST_F(ParamTest, BoolTrue) {
	EXPECT_EQ(true, *ofi_nccl_param_string_to_value<bool>("true"));
	EXPECT_EQ(true, *ofi_nccl_param_string_to_value<bool>("True"));
	EXPECT_EQ(true, *ofi_nccl_param_string_to_value<bool>("TRUE"));
	EXPECT_EQ(true, *ofi_nccl_param_string_to_value<bool>("1"));
}

TEST_F(ParamTest, BoolFalse) {
	EXPECT_EQ(false, *ofi_nccl_param_string_to_value<bool>("false"));
	EXPECT_EQ(false, *ofi_nccl_param_string_to_value<bool>("False"));
	EXPECT_EQ(false, *ofi_nccl_param_string_to_value<bool>("0"));
}

TEST_F(ParamTest, BoolInvalid) {
	EXPECT_FALSE(ofi_nccl_param_string_to_value<bool>("maybe"));
	EXPECT_EQ(true, *ofi_nccl_param_string_to_value<bool>("2")); /* any nonzero int is true */
}

TEST_F(ParamTest, IntBasic) {
	EXPECT_EQ(42, *ofi_nccl_param_string_to_value<int>("42"));
	EXPECT_EQ(-7, *ofi_nccl_param_string_to_value<int>("-7"));
	EXPECT_EQ(0, *ofi_nccl_param_string_to_value<int>("0"));
}

TEST_F(ParamTest, IntInvalid) {
	EXPECT_FALSE(ofi_nccl_param_string_to_value<int>("abc"));
	EXPECT_FALSE(ofi_nccl_param_string_to_value<int>("1.5"));
	EXPECT_FALSE(ofi_nccl_param_string_to_value<int>(""));
}

TEST_F(ParamTest, UnsignedRejectsNegative) {
	EXPECT_FALSE(ofi_nccl_param_string_to_value<unsigned int>("-1"));
	EXPECT_FALSE(ofi_nccl_param_string_to_value<unsigned long>("-55"));
}

TEST_F(ParamTest, UnsignedAcceptsLeadingSpaces) {
	EXPECT_EQ(5u, *ofi_nccl_param_string_to_value<unsigned int>("  5"));
}

TEST_F(ParamTest, UnsignedRejectsNegativeWithSpaces) {
	EXPECT_FALSE(ofi_nccl_param_string_to_value<unsigned int>("  -1"));
}

TEST_F(ParamTest, FloatBasic) {
	auto v = ofi_nccl_param_string_to_value<float>("3.14");
	ASSERT_TRUE(v.has_value());
	EXPECT_NEAR(3.14f, *v, 0.001f);
}

TEST_F(ParamTest, FloatInvalid) {
	EXPECT_FALSE(ofi_nccl_param_string_to_value<float>("abc"));
}

TEST_F(ParamTest, Uint16Boundary) {
	EXPECT_EQ(65535, *ofi_nccl_param_string_to_value<uint16_t>("65535"));
	EXPECT_FALSE(ofi_nccl_param_string_to_value<uint16_t>("65536"));
}

/* --- value_to_string tests --- */

TEST_F(ParamTest, BoolToString) {
	EXPECT_EQ("true", ofi_nccl_param_value_to_string<bool>(true));
	EXPECT_EQ("false", ofi_nccl_param_value_to_string<bool>(false));
}

TEST_F(ParamTest, IntToString) {
	EXPECT_EQ("42", ofi_nccl_param_value_to_string<int>(42));
}

/* --- ofi_nccl_param_impl tests --- */

TEST_F(ParamTest, DefaultValue) {
	ofi_nccl_param_impl<int> p("OFI_NCCL_TEST_DEFAULT_UNUSED", 99);
	EXPECT_EQ(99, p.get());
	EXPECT_EQ(ParamSource::DEFAULT, p.get_source());
}

TEST_F(ParamTest, SetBeforeGet) {
	ofi_nccl_param_impl<int> p("OFI_NCCL_TEST_SET_UNUSED", 1);
	EXPECT_EQ(0, p.set(42));
	EXPECT_EQ(42, p.get());
	EXPECT_EQ(ParamSource::API, p.get_source());
}

TEST_F(ParamTest, SetAfterGetFails) {
	ofi_nccl_param_impl<int> p("OFI_NCCL_TEST_LATE_SET_UNUSED", 1);
	p.get();
	EXPECT_EQ(-EINVAL, p.set(42));
}

TEST_F(ParamTest, EnvironmentOverridesDefault) {
	setenv("OFI_NCCL_TEST_ENV_PARAM", "77", 1);
	ofi_nccl_param_impl<int> p("OFI_NCCL_TEST_ENV_PARAM", 1);
	EXPECT_EQ(77, p.get());
	EXPECT_EQ(ParamSource::ENVIRONMENT, p.get_source());
	unsetenv("OFI_NCCL_TEST_ENV_PARAM");
}

TEST_F(ParamTest, InvalidEnvThrows) {
	setenv("OFI_NCCL_TEST_BAD_ENV", "notanumber", 1);
	EXPECT_THROW(
		ofi_nccl_param_impl<int>("OFI_NCCL_TEST_BAD_ENV", 0),
		std::runtime_error
	);
	unsetenv("OFI_NCCL_TEST_BAD_ENV");
}

TEST_F(ParamTest, GetStringReturnsConsistentPointer) {
	ofi_nccl_param_impl<int> p("OFI_NCCL_TEST_STR_UNUSED", 42);
	const char *s1 = p.get_string();
	const char *s2 = p.get_string();
	EXPECT_EQ(s1, s2);
	EXPECT_STREQ("42", s1);
}

TEST_F(ParamTest, OperatorParens) {
	ofi_nccl_param_impl<int> p("OFI_NCCL_TEST_PARENS_UNUSED", 5);
	EXPECT_EQ(5, p());
}

TEST_F(ParamTest, ConcurrentGet) {
	ofi_nccl_param_impl<int> p("OFI_NCCL_TEST_CONC_UNUSED", 42);
	std::atomic<int> errors{0};
	std::vector<std::thread> threads;
	for (int i = 0; i < 8; i++) {
		threads.emplace_back([&]() {
			for (int j = 0; j < 1000; j++) {
				if (p.get() != 42) errors++;
			}
		});
	}
	for (auto &t : threads) t.join();
	EXPECT_EQ(0, errors.load());
}

TEST_F(ParamTest, BoolParam) {
	ofi_nccl_param_impl<bool> p("OFI_NCCL_TEST_BOOL_UNUSED", false);
	EXPECT_EQ(false, p.get());
	EXPECT_STREQ("false", p.get_string());
}

TEST_F(ParamTest, StringParam) {
	ofi_nccl_param_impl<std::string> p("OFI_NCCL_TEST_STRING_UNUSED", "hello");
	EXPECT_STREQ("hello", p.get_string());
}


TEST_F(ParamTest, StringParamFromEnv) {
	setenv("OFI_NCCL_TEST_STR_ENV", "hello_world", 1);
	ofi_nccl_param_impl<std::string> p("OFI_NCCL_TEST_STR_ENV", "default");
	EXPECT_STREQ("hello_world", p.get_string());
	EXPECT_EQ(ParamSource::ENVIRONMENT, p.get_source());
	unsetenv("OFI_NCCL_TEST_STR_ENV");
}

TEST_F(ParamTest, UnsignedLongBoundary) {
	EXPECT_EQ(0UL, *ofi_nccl_param_string_to_value<unsigned long>("0"));
	auto v = ofi_nccl_param_string_to_value<unsigned long>("18446744073709551615");
	ASSERT_TRUE(v.has_value());
	EXPECT_EQ(ULONG_MAX, *v);
}

TEST_F(ParamTest, IntOverflow) {
	EXPECT_FALSE(ofi_nccl_param_string_to_value<int>("2147483648"));
}

TEST_F(ParamTest, DoubleSetBeforeGet) {
	ofi_nccl_param_impl<int> p("OFI_NCCL_TEST_DBLSET_UNUSED", 1);
	EXPECT_EQ(0, p.set(10));
	EXPECT_EQ(0, p.set(20));
	EXPECT_EQ(20, p.get());
}

TEST_F(ParamTest, EmptyStringEnvThrows) {
	setenv("OFI_NCCL_TEST_EMPTY_STR", "", 1);
	EXPECT_THROW(
		ofi_nccl_param_impl<std::string>("OFI_NCCL_TEST_EMPTY_STR", "default"),
		std::runtime_error
	);
	unsetenv("OFI_NCCL_TEST_EMPTY_STR");
}

int main(int argc, char **argv)
{
	::testing::InitGoogleTest(&argc, argv);
	return RUN_ALL_TESTS();
}
