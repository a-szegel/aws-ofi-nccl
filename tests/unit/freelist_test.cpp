/*
 * Copyright (c) 2025 Amazon.com, Inc. or its affiliates. All rights reserved.
 */
#include "config.h"
#include <gtest/gtest.h>
#include <thread>
#include <vector>
#include <set>
#include <atomic>

#include "nccl_ofi.h"
#include "nccl_ofi_freelist.h"
#include "test-logger.h"

class FreelistTest : public ::testing::Test {
protected:
	void SetUp() override {
		ofi_log_function = logger;
		system_page_size = 4096;
	}
};

TEST_F(FreelistTest, BasicAllocFree) {
	nccl_ofi_freelist fl(64, 8, 8, 0, nullptr, nullptr, "test", false);
	auto *e = fl.entry_alloc();
	ASSERT_NE(nullptr, e);
	ASSERT_NE(nullptr, e->ptr);
	fl.entry_free(e);
}

TEST_F(FreelistTest, MaxEntryCountEnforced) {
	nccl_ofi_freelist fl(8, 4, 4, 8, nullptr, nullptr, "test", false);
	std::vector<nccl_ofi_freelist::fl_entry *> entries;
	for (int i = 0; i < 8; i++) {
		auto *e = fl.entry_alloc();
		ASSERT_NE(nullptr, e) << "alloc failed at i=" << i;
		entries.push_back(e);
	}
	/* 9th should fail */
	EXPECT_EQ(nullptr, fl.entry_alloc());
	for (auto *e : entries) fl.entry_free(e);
}

TEST_F(FreelistTest, GrowOnDemand) {
	nccl_ofi_freelist fl(64, 4, 4, 0, nullptr, nullptr, "test", false);
	std::vector<nccl_ofi_freelist::fl_entry *> entries;
	/* Allocate well beyond initial count */
	for (int i = 0; i < 100; i++) {
		auto *e = fl.entry_alloc();
		ASSERT_NE(nullptr, e) << "alloc failed at i=" << i;
		entries.push_back(e);
	}
	for (auto *e : entries) fl.entry_free(e);
}

TEST_F(FreelistTest, UniquePointers) {
	nccl_ofi_freelist fl(64, 16, 16, 0, nullptr, nullptr, "test", false);
	std::set<void *> ptrs;
	std::vector<nccl_ofi_freelist::fl_entry *> entries;
	for (int i = 0; i < 16; i++) {
		auto *e = fl.entry_alloc();
		ASSERT_NE(nullptr, e);
		EXPECT_TRUE(ptrs.insert(e->ptr).second) << "duplicate ptr at i=" << i;
		entries.push_back(e);
	}
	for (auto *e : entries) fl.entry_free(e);
}

static int init_count = 0;
static int fini_count = 0;

static int test_init_fn(void *entry) {
	*(uint8_t *)entry = 0xAB;
	init_count++;
	return 0;
}

static void test_fini_fn(void *entry) {
	EXPECT_EQ(0xAB, *(uint8_t *)entry);
	fini_count++;
}

TEST_F(FreelistTest, InitFiniFunctions) {
	init_count = fini_count = 0;
	{
		nccl_ofi_freelist fl(8, 16, 16, 0, test_init_fn, test_fini_fn,
				     "test", false);
		EXPECT_GT(init_count, 0);
		auto *e = fl.entry_alloc();
		ASSERT_NE(nullptr, e);
		EXPECT_EQ(0xAB, *(uint8_t *)e->ptr);
		fl.entry_free(e);
	}
	EXPECT_EQ(init_count, fini_count);
}

TEST_F(FreelistTest, ZeroEntrySize) {
	/* entry_size=0 should be bumped to 8 internally */
	nccl_ofi_freelist fl(0, 4, 4, 0, nullptr, nullptr, "test", false);
	auto *e = fl.entry_alloc();
	ASSERT_NE(nullptr, e);
	ASSERT_NE(nullptr, e->ptr);
	fl.entry_free(e);
}

TEST_F(FreelistTest, AllocFreeReuse) {
	nccl_ofi_freelist fl(64, 4, 4, 4, nullptr, nullptr, "test", false);
	auto *e1 = fl.entry_alloc();
	ASSERT_NE(nullptr, e1);
	void *ptr1 = e1->ptr;
	fl.entry_free(e1);
	auto *e2 = fl.entry_alloc();
	ASSERT_NE(nullptr, e2);
	/* Should reuse the same entry */
	EXPECT_EQ(ptr1, e2->ptr);
	fl.entry_free(e2);
}

TEST_F(FreelistTest, ConcurrentAllocFree) {
	nccl_ofi_freelist fl(64, 16, 16, 0, nullptr, nullptr, "test", false);
	std::atomic<int> errors{0};
	auto worker = [&]() {
		for (int i = 0; i < 200; i++) {
			auto *e = fl.entry_alloc();
			if (!e) { errors++; continue; }
			/* Touch the memory */
			memset(e->ptr, 0x55, 64);
			fl.entry_free(e);
		}
	};
	std::vector<std::thread> threads;
	for (int i = 0; i < 8; i++)
		threads.emplace_back(worker);
	for (auto &t : threads) t.join();
	EXPECT_EQ(0, errors.load());
}

TEST_F(FreelistTest, LargeEntrySize) {
	nccl_ofi_freelist fl(4096, 4, 4, 0, nullptr, nullptr, "test", false);
	auto *e = fl.entry_alloc();
	ASSERT_NE(nullptr, e);
	/* Should be able to write full entry */
	memset(e->ptr, 0xCC, 4096);
	fl.entry_free(e);
}

TEST_F(FreelistTest, EntrySpacing) {
	nccl_ofi_freelist fl(1024, 8, 8, 8, nullptr, nullptr, "test", false);
	std::vector<nccl_ofi_freelist::fl_entry *> entries;
	for (int i = 0; i < 4; i++) {
		auto *e = fl.entry_alloc();
		ASSERT_NE(nullptr, e);
		entries.push_back(e);
	}
	/* Entries from same block should be spaced by entry_size */
	for (size_t i = 1; i < entries.size(); i++) {
		ptrdiff_t diff = std::abs((char *)entries[i]->ptr -
					  (char *)entries[i-1]->ptr);
		EXPECT_GE(diff, 1024) << "entries too close together";
	}
	for (auto *e : entries) fl.entry_free(e);
}


TEST_F(FreelistTest, InitCallbackFailure) {
	auto bad_init = [](void *entry) -> int { return -ENOMEM; };
	EXPECT_THROW(
		nccl_ofi_freelist(64, 4, 4, 0, bad_init, nullptr, "test", false),
		std::runtime_error
	);
}

TEST_F(FreelistTest, MultipleGrowCycles) {
	nccl_ofi_freelist fl(64, 2, 2, 0, nullptr, nullptr, "test", false);
	std::vector<nccl_ofi_freelist::fl_entry *> entries;
	for (int i = 0; i < 20; i++) {
		auto *e = fl.entry_alloc();
		ASSERT_NE(nullptr, e) << "alloc failed at i=" << i;
		entries.push_back(e);
	}
	for (auto *e : entries) fl.entry_free(e);
}

int main(int argc, char **argv)
{
	::testing::InitGoogleTest(&argc, argv);
	return RUN_ALL_TESTS();
}
