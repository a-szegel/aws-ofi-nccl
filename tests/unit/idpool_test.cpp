/*
 * Copyright (c) 2026 Amazon.com, Inc. or its affiliates. All rights reserved.
 *
 * GoogleTest unit tests for nccl_ofi_idpool_t.
 * Tests the ID pool used for communicator IDs and MR keys.
 */

#include <gtest/gtest.h>
#include <stdexcept>
#include <thread>
#include <vector>
#include <atomic>
#include "config.h"
#include "nccl_ofi.h"
#include "test-logger.h"
#include "nccl_ofi_idpool.h"
#include "nccl_ofi_math.h"

/* Subclass to access protected members for verification. */
class TestableIdpool : public nccl_ofi_idpool_t {
public:
	TestableIdpool(size_t s) : nccl_ofi_idpool_t(s) {}
	uint64_t get_element(size_t i) { std::lock_guard l(lock); return idpool.at(i); }
	size_t get_vector_size() { std::lock_guard l(lock); return idpool.size(); }
};

class IdpoolTest : public ::testing::Test {
protected:
	void SetUp() override { ofi_log_function = logger; }
};

/* Verify pool of size 1 works: allocate, exhaust, free, reallocate. */
TEST_F(IdpoolTest, SizeOne)
{
	TestableIdpool pool(1);
	EXPECT_EQ(pool.get_size(), 1UL);
	EXPECT_EQ(pool.get_vector_size(), 1UL);
	/* Only bit 0 should be set */
	EXPECT_EQ(pool.get_element(0), 1ULL);

	size_t id = pool.allocate_id();
	EXPECT_EQ(id, 0UL);

	/* Pool exhausted */
	EXPECT_EQ(pool.allocate_id(), FI_KEY_NOTAVAIL);

	pool.free_id(0);
	id = pool.allocate_id();
	EXPECT_EQ(id, 0UL);
}

/* Verify pool of size 64 (exact uint64_t boundary). */
TEST_F(IdpoolTest, ExactBoundary64)
{
	TestableIdpool pool(64);
	EXPECT_EQ(pool.get_vector_size(), 1UL);
	EXPECT_EQ(pool.get_element(0), 0xFFFFFFFFFFFFFFFFULL);

	for (size_t i = 0; i < 64; i++) {
		EXPECT_EQ(pool.allocate_id(), i);
	}
	EXPECT_EQ(pool.allocate_id(), FI_KEY_NOTAVAIL);
}

/* Verify pool of size 65 (one past uint64_t boundary). */
TEST_F(IdpoolTest, OnePastBoundary65)
{
	TestableIdpool pool(65);
	EXPECT_EQ(pool.get_vector_size(), 2UL);
	EXPECT_EQ(pool.get_element(0), 0xFFFFFFFFFFFFFFFFULL);
	/* Only bit 0 of second element should be set */
	EXPECT_EQ(pool.get_element(1), 1ULL);

	for (size_t i = 0; i < 65; i++) {
		EXPECT_EQ(pool.allocate_id(), i);
	}
	EXPECT_EQ(pool.allocate_id(), FI_KEY_NOTAVAIL);
}

/* Verify allocate from empty (size=0) pool throws. */
TEST_F(IdpoolTest, AllocateFromEmptyThrows)
{
	nccl_ofi_idpool_t pool(0);
	EXPECT_THROW(pool.allocate_id(), std::runtime_error);
}

/* Verify free from empty (size=0) pool throws. */
TEST_F(IdpoolTest, FreeFromEmptyThrows)
{
	nccl_ofi_idpool_t pool(0);
	EXPECT_THROW(pool.free_id(0), std::runtime_error);
}

/* Verify freeing an out-of-range ID throws. */
TEST_F(IdpoolTest, FreeOutOfRangeThrows)
{
	nccl_ofi_idpool_t pool(10);
	EXPECT_THROW(pool.free_id(10), std::runtime_error);
	EXPECT_THROW(pool.free_id(100), std::runtime_error);
}

/* Verify double-free throws. */
TEST_F(IdpoolTest, DoubleFreeThrows)
{
	nccl_ofi_idpool_t pool(10);
	/* ID 0 is available (never allocated), freeing it is a double-free */
	EXPECT_THROW(pool.free_id(0), std::runtime_error);
}

/* Verify IDs are allocated in order (lowest first). */
TEST_F(IdpoolTest, AllocatesInOrder)
{
	nccl_ofi_idpool_t pool(128);
	for (size_t i = 0; i < 128; i++) {
		EXPECT_EQ(pool.allocate_id(), i);
	}
}

/* Verify free-then-allocate returns the freed ID. */
TEST_F(IdpoolTest, FreeAndReallocate)
{
	nccl_ofi_idpool_t pool(10);

	/* Allocate all */
	for (size_t i = 0; i < 10; i++)
		pool.allocate_id();

	/* Free ID 5, then allocate should return 5 */
	pool.free_id(5);
	EXPECT_EQ(pool.allocate_id(), 5UL);
}

/* Verify freeing multiple IDs and reallocating returns them in order. */
TEST_F(IdpoolTest, FreeMultipleReallocateInOrder)
{
	nccl_ofi_idpool_t pool(10);

	for (size_t i = 0; i < 10; i++)
		pool.allocate_id();

	/* Free 3, 7, 1 */
	pool.free_id(3);
	pool.free_id(7);
	pool.free_id(1);

	/* Should get them back in order: 1, 3, 7 */
	EXPECT_EQ(pool.allocate_id(), 1UL);
	EXPECT_EQ(pool.allocate_id(), 3UL);
	EXPECT_EQ(pool.allocate_id(), 7UL);
	EXPECT_EQ(pool.allocate_id(), FI_KEY_NOTAVAIL);
}

/* Verify large pool (1000 IDs). */
TEST_F(IdpoolTest, LargePool)
{
	nccl_ofi_idpool_t pool(1000);
	EXPECT_EQ(pool.get_size(), 1000UL);

	for (size_t i = 0; i < 1000; i++) {
		EXPECT_EQ(pool.allocate_id(), i);
	}
	EXPECT_EQ(pool.allocate_id(), FI_KEY_NOTAVAIL);

	for (size_t i = 0; i < 1000; i++)
		pool.free_id(i);

	/* All freed, allocate again */
	for (size_t i = 0; i < 1000; i++) {
		EXPECT_EQ(pool.allocate_id(), i);
	}
}


TEST_F(IdpoolTest, ConcurrentAllocFree)
{
	nccl_ofi_idpool_t pool(256);
	std::atomic<int> errors{0};
	auto worker = [&]() {
		for (int i = 0; i < 500; i++) {
			size_t id = pool.allocate_id();
			if (id == FI_KEY_NOTAVAIL) continue;
			if (id >= 256) { errors++; continue; }
			pool.free_id(id);
		}
	};
	std::vector<std::thread> threads;
	for (int i = 0; i < 8; i++)
		threads.emplace_back(worker);
	for (auto &t : threads) t.join();
	EXPECT_EQ(0, errors.load());
}

TEST_F(IdpoolTest, GetSizeIsCapacity)
{
	nccl_ofi_idpool_t pool(100);
	EXPECT_EQ(pool.get_size(), 100UL);
	pool.allocate_id();
	EXPECT_EQ(pool.get_size(), 100UL);
}

TEST_F(IdpoolTest, SizeBelowBoundary63)
{
	TestableIdpool pool(63);
	EXPECT_EQ(pool.get_vector_size(), 1UL);
	EXPECT_EQ(pool.get_element(0), 0x7FFFFFFFFFFFFFFFULL);
	for (size_t i = 0; i < 63; i++)
		EXPECT_EQ(pool.allocate_id(), i);
	EXPECT_EQ(pool.allocate_id(), FI_KEY_NOTAVAIL);
}

TEST_F(IdpoolTest, ExactBoundary128)
{
	TestableIdpool pool(128);
	EXPECT_EQ(pool.get_vector_size(), 2UL);
	EXPECT_EQ(pool.get_element(0), 0xFFFFFFFFFFFFFFFFULL);
	EXPECT_EQ(pool.get_element(1), 0xFFFFFFFFFFFFFFFFULL);
}

int main(int argc, char **argv)
{
	ofi_log_function = logger;
	::testing::InitGoogleTest(&argc, argv);
	return RUN_ALL_TESTS();
}
