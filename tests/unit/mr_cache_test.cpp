/*
 * Copyright (c) 2025 Amazon.com, Inc. or its affiliates. All rights reserved.
 */
#include "config.h"
#include <gtest/gtest.h>
#include <vector>
#include <thread>
#include <atomic>

#include "nccl_ofi.h"
#include "nccl_ofi_mr.h"
#include "test-logger.h"

class MrCacheTest : public ::testing::Test {
protected:
	static constexpr size_t PAGE = 1024;
	nccl_ofi_mr_cache_t *cache = nullptr;

	void SetUp() override {
		ofi_log_function = logger;
		mr_cache_alignment = PAGE;
		cache = nccl_ofi_mr_cache_init(16, PAGE);
		ASSERT_NE(nullptr, cache);
	}

	void TearDown() override {
		if (cache) nccl_ofi_mr_cache_finalize(cache);
	}

	void *lookup(uintptr_t addr, size_t size) {
		nccl_ofi_mr_ckey_t k = nccl_ofi_mr_ckey_mk_vec((void *)addr, size, nullptr);
		return nccl_ofi_mr_cache_lookup_entry(cache, &k, false);
	}

	int insert(uintptr_t addr, size_t size, void *handle) {
		nccl_ofi_mr_ckey_t k = nccl_ofi_mr_ckey_mk_vec((void *)addr, size, nullptr);
		return nccl_ofi_mr_cache_insert_entry(cache, &k, false, handle);
	}
};

TEST_F(MrCacheTest, InitWithZeroEntries) {
	auto *c = nccl_ofi_mr_cache_init(0, PAGE);
	EXPECT_EQ(nullptr, c);
}

TEST_F(MrCacheTest, InitWithZeroPageSize) {
	auto *c = nccl_ofi_mr_cache_init(16, 0);
	EXPECT_EQ(nullptr, c);
}

TEST_F(MrCacheTest, InsertAndLookup) {
	EXPECT_EQ(0, insert(0, 1, (void *)0x100));
	EXPECT_EQ((void *)0x100, lookup(0, 1));
}

TEST_F(MrCacheTest, LookupMiss) {
	EXPECT_EQ(nullptr, lookup(0, 1));
}

TEST_F(MrCacheTest, InsertDuplicate) {
	EXPECT_EQ(0, insert(0, 1, (void *)0x100));
	EXPECT_EQ(-EEXIST, insert(0, 1, (void *)0x200));
}

TEST_F(MrCacheTest, DeleteDecreasesRefcount) {
	EXPECT_EQ(0, insert(0, 1, (void *)0x100));
	/* lookup bumps refcnt to 2 */
	EXPECT_NE(nullptr, lookup(0, 1));
	/* first delete: refcnt 2->1 */
	EXPECT_EQ(0, nccl_ofi_mr_cache_del_entry(cache, (void *)0x100));
	/* second delete: refcnt 1->0, entry removed */
	EXPECT_EQ(1, nccl_ofi_mr_cache_del_entry(cache, (void *)0x100));
	/* third delete: not found */
	EXPECT_EQ(-ENOENT, nccl_ofi_mr_cache_del_entry(cache, (void *)0x100));
}

TEST_F(MrCacheTest, GrowBeyondInitialSize) {
	for (size_t i = 0; i < 64; i++) {
		EXPECT_EQ(0, insert(i * PAGE, 1, (void *)(i + 1)));
	}
	/* Verify all entries are findable */
	for (size_t i = 0; i < 64; i++) {
		EXPECT_EQ((void *)(i + 1), lookup(i * PAGE, 1));
	}
}

TEST_F(MrCacheTest, SubPageLookupHit) {
	/* Insert a full page, lookup a sub-range within it */
	EXPECT_EQ(0, insert(PAGE, PAGE, (void *)0x42));
	EXPECT_EQ((void *)0x42, lookup(PAGE + 100, 50));
}

TEST_F(MrCacheTest, CrossPageLookupMiss) {
	EXPECT_EQ(0, insert(PAGE, PAGE, (void *)0x42));
	/* Lookup spanning beyond the registered page should miss */
	EXPECT_EQ(nullptr, lookup(PAGE + PAGE - 1, 2));
}

TEST_F(MrCacheTest, DeleteMiddleEntry) {
	for (int i = 0; i < 5; i++)
		EXPECT_EQ(0, insert(i * PAGE, 1, (void *)(uintptr_t)(i + 1)));
	/* Delete middle entry (handle=3, addr=2*PAGE) */
	EXPECT_EQ(1, nccl_ofi_mr_cache_del_entry(cache, (void *)3));
	/* Verify others still work */
	EXPECT_NE(nullptr, lookup(0, 1));
	EXPECT_NE(nullptr, lookup(PAGE, 1));
	EXPECT_EQ(nullptr, lookup(2 * PAGE, 1));
	EXPECT_NE(nullptr, lookup(3 * PAGE, 1));
	EXPECT_NE(nullptr, lookup(4 * PAGE, 1));
}

TEST_F(MrCacheTest, DeleteNonexistent) {
	EXPECT_EQ(-ENOENT, nccl_ofi_mr_cache_del_entry(cache, (void *)0xDEAD));
}

TEST_F(MrCacheTest, PageAlignment) {
	/* Insert at non-page-aligned address; should be rounded down */
	EXPECT_EQ(0, insert(PAGE + 100, 50, (void *)0x42));
	/* Lookup at page start should hit */
	EXPECT_EQ((void *)0x42, lookup(PAGE, 1));
}

TEST_F(MrCacheTest, InsertAndDeleteMany) {
	/* Stress test: insert many, delete all */
	for (size_t i = 0; i < 100; i++)
		EXPECT_EQ(0, insert(i * PAGE, 1, (void *)(i + 1)));
	for (size_t i = 0; i < 100; i++)
		EXPECT_EQ(1, nccl_ofi_mr_cache_del_entry(cache, (void *)(i + 1)));
	EXPECT_EQ(0u, cache->used);
}

int main(int argc, char **argv)
{
	::testing::InitGoogleTest(&argc, argv);
	return RUN_ALL_TESTS();
}
