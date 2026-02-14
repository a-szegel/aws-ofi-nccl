/*
 * Copyright (c) 2026 Amazon.com, Inc. or its affiliates. All rights reserved.
 *
 * Unit tests for nccl_ofi_math.h utility functions.
 * These are pure template functions with no external dependencies.
 */

#include <gtest/gtest.h>
#include <cstdint>
#include <climits>

/* Provide minimal stubs so we can include the header */
#include "config.h"
#include "nccl_ofi_math.h"

/* ===== NCCL_OFI_DIV_CEIL tests ===== */

/* Verify basic ceiling division: exact multiples produce exact quotient. */
TEST(DivCeil, ExactMultiple)
{
	EXPECT_EQ(NCCL_OFI_DIV_CEIL(10, 5), 2);
	EXPECT_EQ(NCCL_OFI_DIV_CEIL(64, 64), 1);
	EXPECT_EQ(NCCL_OFI_DIV_CEIL(128, 8), 16);
}

/* Verify ceiling division rounds up for non-exact multiples. */
TEST(DivCeil, RoundsUp)
{
	EXPECT_EQ(NCCL_OFI_DIV_CEIL(1, 2), 1);
	EXPECT_EQ(NCCL_OFI_DIV_CEIL(7, 3), 3);
	EXPECT_EQ(NCCL_OFI_DIV_CEIL(65, 64), 2);
}

/* Verify x=0 returns 0 regardless of y. */
TEST(DivCeil, ZeroNumerator)
{
	EXPECT_EQ(NCCL_OFI_DIV_CEIL(0, 1), 0);
	EXPECT_EQ(NCCL_OFI_DIV_CEIL(0, 100), 0);
	EXPECT_EQ(NCCL_OFI_DIV_CEIL(0UL, 64UL), 0UL);
}

/* Verify x=1 always produces 1. */
TEST(DivCeil, OneNumerator)
{
	EXPECT_EQ(NCCL_OFI_DIV_CEIL(1, 1), 1);
	EXPECT_EQ(NCCL_OFI_DIV_CEIL(1, 1000), 1);
}

/* Verify large values don't overflow for size_t. */
TEST(DivCeil, LargeValues)
{
	size_t big = SIZE_MAX;
	EXPECT_EQ(NCCL_OFI_DIV_CEIL(big, big), 1UL);
	EXPECT_EQ(NCCL_OFI_DIV_CEIL(big, 1UL), big);
}

/* ===== NCCL_OFI_IS_POWER_OF_TWO tests ===== */

/* Verify known powers of two. */
TEST(IsPowerOfTwo, KnownPowers)
{
	EXPECT_TRUE(NCCL_OFI_IS_POWER_OF_TWO(1));
	EXPECT_TRUE(NCCL_OFI_IS_POWER_OF_TWO(2));
	EXPECT_TRUE(NCCL_OFI_IS_POWER_OF_TWO(4));
	EXPECT_TRUE(NCCL_OFI_IS_POWER_OF_TWO(1024));
	EXPECT_TRUE(NCCL_OFI_IS_POWER_OF_TWO(1UL << 63));
}

/* Verify non-powers of two. */
TEST(IsPowerOfTwo, NonPowers)
{
	EXPECT_FALSE(NCCL_OFI_IS_POWER_OF_TWO(0));
	EXPECT_FALSE(NCCL_OFI_IS_POWER_OF_TWO(3));
	EXPECT_FALSE(NCCL_OFI_IS_POWER_OF_TWO(6));
	EXPECT_FALSE(NCCL_OFI_IS_POWER_OF_TWO(100));
}

/* ===== NCCL_OFI_ROUND_UP_TO_POWER_OF_TWO tests ===== */

/* Verify values already a power of two are unchanged. */
TEST(RoundUpToPowerOfTwo, AlreadyPower)
{
	EXPECT_EQ(NCCL_OFI_ROUND_UP_TO_POWER_OF_TWO(1), 1);
	EXPECT_EQ(NCCL_OFI_ROUND_UP_TO_POWER_OF_TWO(2), 2);
	EXPECT_EQ(NCCL_OFI_ROUND_UP_TO_POWER_OF_TWO(64), 64);
}

/* Verify rounding up to next power of two. */
TEST(RoundUpToPowerOfTwo, RoundsUp)
{
	EXPECT_EQ(NCCL_OFI_ROUND_UP_TO_POWER_OF_TWO(3), 4);
	EXPECT_EQ(NCCL_OFI_ROUND_UP_TO_POWER_OF_TWO(5), 8);
	EXPECT_EQ(NCCL_OFI_ROUND_UP_TO_POWER_OF_TWO(65), 128);
	EXPECT_EQ(NCCL_OFI_ROUND_UP_TO_POWER_OF_TWO(100), 128);
}

/* Verify zero and one edge cases. */
TEST(RoundUpToPowerOfTwo, EdgeCases)
{
	EXPECT_EQ(NCCL_OFI_ROUND_UP_TO_POWER_OF_TWO(0), 1);
	EXPECT_EQ(NCCL_OFI_ROUND_UP_TO_POWER_OF_TWO(1), 1);
}

/* ===== NCCL_OFI_IS_ALIGNED tests ===== */

/* Verify alignment checks with power-of-two alignments. */
TEST(IsAligned, BasicAlignment)
{
	EXPECT_TRUE(NCCL_OFI_IS_ALIGNED(0, 8));
	EXPECT_TRUE(NCCL_OFI_IS_ALIGNED(8, 8));
	EXPECT_TRUE(NCCL_OFI_IS_ALIGNED(128, 64));
	EXPECT_FALSE(NCCL_OFI_IS_ALIGNED(1, 8));
	EXPECT_FALSE(NCCL_OFI_IS_ALIGNED(7, 4));
	EXPECT_FALSE(NCCL_OFI_IS_ALIGNED(129, 64));
}

/* ===== NCCL_OFI_ROUND_DOWN tests ===== */

/* Verify rounding down to alignment boundary. */
TEST(RoundDown, Basic)
{
	EXPECT_EQ(NCCL_OFI_ROUND_DOWN((size_t)0, (size_t)8), 0UL);
	EXPECT_EQ(NCCL_OFI_ROUND_DOWN((size_t)7, (size_t)8), 0UL);
	EXPECT_EQ(NCCL_OFI_ROUND_DOWN((size_t)8, (size_t)8), 8UL);
	EXPECT_EQ(NCCL_OFI_ROUND_DOWN((size_t)15, (size_t)8), 8UL);
	EXPECT_EQ(NCCL_OFI_ROUND_DOWN((size_t)130, (size_t)64), 128UL);
}

/* ===== NCCL_OFI_ROUND_UP tests ===== */

/* Verify rounding up to alignment boundary. */
TEST(RoundUp, Basic)
{
	EXPECT_EQ(NCCL_OFI_ROUND_UP((size_t)0, (size_t)8), 0UL);
	EXPECT_EQ(NCCL_OFI_ROUND_UP((size_t)1, (size_t)8), 8UL);
	EXPECT_EQ(NCCL_OFI_ROUND_UP((size_t)8, (size_t)8), 8UL);
	EXPECT_EQ(NCCL_OFI_ROUND_UP((size_t)9, (size_t)8), 16UL);
	EXPECT_EQ(NCCL_OFI_ROUND_UP((size_t)65, (size_t)64), 128UL);
}

int main(int argc, char **argv)
{
	::testing::InitGoogleTest(&argc, argv);
	return RUN_ALL_TESTS();
}
