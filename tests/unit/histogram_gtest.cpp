/*
 * Copyright (c) 2025 Amazon.com, Inc. or its affiliates. All rights reserved.
 */
#include "config.h"
#include <gtest/gtest.h>

#include "nccl_ofi_log.h"
#include "stats/histogram.h"
#include "stats/histogram_binner.h"
#include "test-logger.h"

class HistogramBinnerTest : public ::testing::Test {
protected:
	void SetUp() override { ofi_log_function = logger; }
};

/* --- Linear binner --- */

TEST_F(HistogramBinnerTest, LinearBasic) {
	histogram_linear_binner<size_t> b(0, 10, 5);
	EXPECT_EQ(5u, b.get_num_bins());
	EXPECT_EQ(0u, b.get_bin(0));
	EXPECT_EQ(0u, b.get_bin(9));
	EXPECT_EQ(1u, b.get_bin(10));
	EXPECT_EQ(4u, b.get_bin(40));
}

TEST_F(HistogramBinnerTest, LinearOverflow) {
	histogram_linear_binner<size_t> b(0, 10, 5);
	/* Values beyond last bin should clamp to last bin */
	EXPECT_EQ(4u, b.get_bin(100));
	EXPECT_EQ(4u, b.get_bin(999));
}

TEST_F(HistogramBinnerTest, LinearNonZeroMin) {
	histogram_linear_binner<size_t> b(100, 50, 3);
	EXPECT_EQ(0u, b.get_bin(100));
	EXPECT_EQ(0u, b.get_bin(149));
	EXPECT_EQ(1u, b.get_bin(150));
	EXPECT_EQ(2u, b.get_bin(200));
	EXPECT_EQ(2u, b.get_bin(999));
}

TEST_F(HistogramBinnerTest, LinearRangeLabels) {
	histogram_linear_binner<size_t> b(0, 10, 3);
	auto &labels = b.get_bin_ranges();
	ASSERT_EQ(3u, labels.size());
	EXPECT_EQ(0u, labels[0]);
	EXPECT_EQ(10u, labels[1]);
	EXPECT_EQ(20u, labels[2]);
}

/* --- Custom binner --- */

TEST_F(HistogramBinnerTest, CustomBasic) {
	std::vector<size_t> ranges = {0, 10, 100, 1000};
	histogram_custom_binner<size_t> b(ranges);
	EXPECT_EQ(4u, b.get_num_bins());
	EXPECT_EQ(0u, b.get_bin(0));
	EXPECT_EQ(0u, b.get_bin(9));
	EXPECT_EQ(1u, b.get_bin(10));
	EXPECT_EQ(1u, b.get_bin(99));
	EXPECT_EQ(2u, b.get_bin(100));
	EXPECT_EQ(3u, b.get_bin(1000));
	EXPECT_EQ(3u, b.get_bin(9999));
}

/* --- Histogram --- */

TEST_F(HistogramBinnerTest, HistogramInsert) {
	histogram_linear_binner<size_t> b(0, 10, 5);
	histogram<size_t, histogram_linear_binner<size_t>> h("test", b);
	h.insert(5);
	h.insert(15);
	h.insert(25);
	/* Just verify it doesn't crash; internal state is protected */
}

TEST_F(HistogramBinnerTest, HistogramMinMax) {
	histogram_linear_binner<size_t> b(0, 10, 5);
	histogram<size_t, histogram_linear_binner<size_t>> h("test", b);
	h.insert(42);
	h.insert(7);
	h.insert(99);
	/* print_stats exercises min/max tracking */
	h.print_stats();
}

TEST_F(HistogramBinnerTest, HistogramSingleInsert) {
	histogram_linear_binner<size_t> b(0, 10, 5);
	histogram<size_t, histogram_linear_binner<size_t>> h("test", b);
	h.insert(5);
	h.print_stats();
}

TEST_F(HistogramBinnerTest, LinearBinSizeOne) {
	histogram_linear_binner<size_t> b(0, 1, 10);
	EXPECT_EQ(0u, b.get_bin(0));
	EXPECT_EQ(1u, b.get_bin(1));
	EXPECT_EQ(9u, b.get_bin(9));
	EXPECT_EQ(9u, b.get_bin(100));
}

TEST_F(HistogramBinnerTest, CustomSingleBin) {
	std::vector<size_t> ranges = {0};
	histogram_custom_binner<size_t> b(ranges);
	EXPECT_EQ(1u, b.get_num_bins());
	EXPECT_EQ(0u, b.get_bin(0));
	EXPECT_EQ(0u, b.get_bin(999));
}

int main(int argc, char **argv)
{
	::testing::InitGoogleTest(&argc, argv);
	return RUN_ALL_TESTS();
}
