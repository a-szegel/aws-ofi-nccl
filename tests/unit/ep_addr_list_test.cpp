/*
 * Copyright (c) 2026 Amazon.com, Inc. or its affiliates. All rights reserved.
 *
 * GoogleTest unit tests for nccl_ofi_ep_addr_list_t.
 * Tests the endpoint-address mapping used to avoid duplicate connections.
 */

#include <gtest/gtest.h>
#include "config.h"
#include "nccl_ofi.h"
#include "test-logger.h"
#include "nccl_ofi_ep_addr_list.h"

/* Fake endpoint pointers - we only need distinct addresses. */
static nccl_net_ofi_ep_t *ep1 = reinterpret_cast<nccl_net_ofi_ep_t *>(0x1000);
static nccl_net_ofi_ep_t *ep2 = reinterpret_cast<nccl_net_ofi_ep_t *>(0x2000);

class EpAddrListTest : public ::testing::Test {
protected:
	void SetUp() override { ofi_log_function = logger; }
	nccl_ofi_ep_addr_list_t list;
};

/* Verify insert and get basic flow. */
TEST_F(EpAddrListTest, InsertAndGet)
{
	uint32_t addr1 = 0xAAAA;
	EXPECT_EQ(list.insert(ep1, &addr1, sizeof(addr1)), 0);

	/* Get with a different address should return ep1 */
	uint32_t addr2 = 0xBBBB;
	nccl_net_ofi_ep_t *result = nullptr;
	EXPECT_EQ(list.get(&addr2, sizeof(addr2), &result), 0);
	EXPECT_EQ(result, ep1);
}

/* Verify get returns NULL when all endpoints are connected to the address. */
TEST_F(EpAddrListTest, GetReturnsNullWhenAllConnected)
{
	uint32_t addr = 0xAAAA;
	EXPECT_EQ(list.insert(ep1, &addr, sizeof(addr)), 0);

	/* ep1 is already connected to addr, so get should return NULL */
	nccl_net_ofi_ep_t *result = nullptr;
	EXPECT_EQ(list.get(&addr, sizeof(addr), &result), 0);
	EXPECT_EQ(result, nullptr);
}

/* Verify get picks an endpoint not yet connected to the address. */
TEST_F(EpAddrListTest, GetPicksUnconnectedEndpoint)
{
	uint32_t addr1 = 0xAAAA;
	uint32_t addr2 = 0xBBBB;

	EXPECT_EQ(list.insert(ep1, &addr1, sizeof(addr1)), 0);
	EXPECT_EQ(list.insert(ep2, &addr2, sizeof(addr2)), 0);

	/* Get with addr1: ep1 is connected to addr1, ep2 is not */
	nccl_net_ofi_ep_t *result = nullptr;
	EXPECT_EQ(list.get(&addr1, sizeof(addr1), &result), 0);
	EXPECT_EQ(result, ep2);
}

/* Verify duplicate insert returns -EINVAL. */
TEST_F(EpAddrListTest, DuplicateInsertFails)
{
	uint32_t addr = 0xAAAA;
	EXPECT_EQ(list.insert(ep1, &addr, sizeof(addr)), 0);
	EXPECT_EQ(list.insert(ep1, &addr, sizeof(addr)), -EINVAL);
}

/* Verify remove works. */
TEST_F(EpAddrListTest, RemoveEndpoint)
{
	uint32_t addr = 0xAAAA;
	EXPECT_EQ(list.insert(ep1, &addr, sizeof(addr)), 0);
	EXPECT_EQ(list.remove(ep1), 0);

	/* After removal, get should return NULL (no endpoints) */
	nccl_net_ofi_ep_t *result = nullptr;
	EXPECT_EQ(list.get(&addr, sizeof(addr), &result), 0);
	EXPECT_EQ(result, nullptr);
}

/* Verify removing non-existent endpoint returns -ENOENT. */
TEST_F(EpAddrListTest, RemoveNonExistent)
{
	EXPECT_EQ(list.remove(ep1), -ENOENT);
}

/* Verify get adds the address to the endpoint's connection set. */
TEST_F(EpAddrListTest, GetAddsAddressToConnectionSet)
{
	uint32_t addr1 = 0xAAAA;
	uint32_t addr2 = 0xBBBB;

	EXPECT_EQ(list.insert(ep1, &addr1, sizeof(addr1)), 0);

	/* Get with addr2 should return ep1 and add addr2 to ep1's set */
	nccl_net_ofi_ep_t *result = nullptr;
	EXPECT_EQ(list.get(&addr2, sizeof(addr2), &result), 0);
	EXPECT_EQ(result, ep1);

	/* Now ep1 is connected to both addr1 and addr2.
	 * Get with either should return NULL. */
	result = nullptr;
	EXPECT_EQ(list.get(&addr1, sizeof(addr1), &result), 0);
	EXPECT_EQ(result, nullptr);

	result = nullptr;
	EXPECT_EQ(list.get(&addr2, sizeof(addr2), &result), 0);
	EXPECT_EQ(result, nullptr);
}

/* Verify address comparison uses content, not pointer identity. */
TEST_F(EpAddrListTest, AddressComparisonByContent)
{
	char addr_a[4] = {1, 2, 3, 4};
	char addr_b[4] = {1, 2, 3, 4}; /* Same content, different pointer */

	EXPECT_EQ(list.insert(ep1, addr_a, sizeof(addr_a)), 0);

	/* Get with addr_b (same content) should see ep1 as already connected */
	nccl_net_ofi_ep_t *result = nullptr;
	EXPECT_EQ(list.get(addr_b, sizeof(addr_b), &result), 0);
	EXPECT_EQ(result, nullptr);
}

/* Verify empty list get returns NULL. */
TEST_F(EpAddrListTest, GetFromEmptyList)
{
	uint32_t addr = 0xAAAA;
	nccl_net_ofi_ep_t *result = nullptr;
	EXPECT_EQ(list.get(&addr, sizeof(addr), &result), 0);
	EXPECT_EQ(result, nullptr);
}

int main(int argc, char **argv)
{
	ofi_log_function = logger;
	::testing::InitGoogleTest(&argc, argv);
	return RUN_ALL_TESTS();
}
