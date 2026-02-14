/*
 * Copyright (c) 2025 Amazon.com, Inc. or its affiliates. All rights reserved.
 */
#include "config.h"
#include <gtest/gtest.h>
#include <thread>
#include <vector>
#include <atomic>

#include "nccl_ofi.h"
#include "nccl_ofi_scheduler.h"
#include "nccl_ofi_param.h"
#include "test-logger.h"

class SchedulerTest : public ::testing::Test {
protected:
	void SetUp() override {
		ofi_log_function = logger;
		system_page_size = 4096;
	}

	nccl_net_ofi_scheduler_t *create_scheduler(int num_rails,
						    size_t min_stripe = 4096,
						    size_t max_small = 64) {
		ofi_nccl_min_stripe_size.set(min_stripe);
		ofi_nccl_sched_max_small_msg_size.set(max_small);
		nccl_net_ofi_scheduler_t *s = nullptr;
		EXPECT_EQ(0, nccl_net_ofi_threshold_scheduler_init(num_rails, &s));
		return s;
	}
};

TEST_F(SchedulerTest, InitAndFini) {
	auto *s = create_scheduler(4);
	ASSERT_NE(nullptr, s);
	EXPECT_EQ(0, s->fini(s));
}

TEST_F(SchedulerTest, ZeroSizeMessage) {
	auto *s = create_scheduler(4);
	auto *sched = s->get_schedule(s, 0, 4);
	ASSERT_NE(nullptr, sched);
	EXPECT_EQ(1u, sched->num_xfer_infos);
	EXPECT_EQ(0u, sched->rail_xfer_infos[0].msg_size);
	EXPECT_EQ(0u, sched->rail_xfer_infos[0].offset);
	nccl_net_ofi_release_schedule(s, sched);
	s->fini(s);
}

TEST_F(SchedulerTest, SmallMessageRoundRobin) {
	auto *s = create_scheduler(4, 4096, 64);
	std::vector<int> rails;
	for (int i = 0; i < 8; i++) {
		auto *sched = s->get_schedule(s, 32, 4);
		ASSERT_NE(nullptr, sched);
		EXPECT_EQ(1u, sched->num_xfer_infos);
		EXPECT_EQ(32u, sched->rail_xfer_infos[0].msg_size);
		rails.push_back(sched->rail_xfer_infos[0].rail_id);
		nccl_net_ofi_release_schedule(s, sched);
	}
	for (int i = 0; i < 8; i++) {
		EXPECT_EQ(i % 4, rails[i]);
	}
	s->fini(s);
}

TEST_F(SchedulerTest, LargeMessageStriping) {
	auto *s = create_scheduler(4, 4096, 64);
	size_t msg_size = 4 * 4096;
	auto *sched = s->get_schedule(s, msg_size, 4);
	ASSERT_NE(nullptr, sched);
	EXPECT_EQ(4u, sched->num_xfer_infos);
	size_t total = 0;
	for (size_t i = 0; i < sched->num_xfer_infos; i++)
		total += sched->rail_xfer_infos[i].msg_size;
	EXPECT_EQ(msg_size, total);
	nccl_net_ofi_release_schedule(s, sched);
	s->fini(s);
}

TEST_F(SchedulerTest, StripeSizesAreAligned) {
	auto *s = create_scheduler(4, 4096, 64);
	size_t msg_size = 4096 + 1;
	auto *sched = s->get_schedule(s, msg_size, 4);
	ASSERT_NE(nullptr, sched);
	EXPECT_EQ(2u, sched->num_xfer_infos);
	EXPECT_EQ(0u, sched->rail_xfer_infos[0].msg_size % 128);
	nccl_net_ofi_release_schedule(s, sched);
	s->fini(s);
}

TEST_F(SchedulerTest, SingleRail) {
	auto *s = create_scheduler(1, 4096, 64);
	for (size_t sz : {0ul, 32ul, 100ul, 8192ul, 65536ul}) {
		auto *sched = s->get_schedule(s, sz, 1);
		ASSERT_NE(nullptr, sched);
		EXPECT_EQ(1u, sched->num_xfer_infos);
		EXPECT_EQ(0, sched->rail_xfer_infos[0].rail_id);
		EXPECT_EQ(sz, sched->rail_xfer_infos[0].msg_size);
		nccl_net_ofi_release_schedule(s, sched);
	}
	s->fini(s);
}

TEST_F(SchedulerTest, OffsetsAreContiguous) {
	auto *s = create_scheduler(4, 1024, 64);
	size_t msg_size = 5000;
	auto *sched = s->get_schedule(s, msg_size, 4);
	ASSERT_NE(nullptr, sched);
	size_t expected_offset = 0;
	for (size_t i = 0; i < sched->num_xfer_infos; i++) {
		EXPECT_EQ(expected_offset, sched->rail_xfer_infos[i].offset);
		expected_offset += sched->rail_xfer_infos[i].msg_size;
	}
	EXPECT_EQ(msg_size, expected_offset);
	nccl_net_ofi_release_schedule(s, sched);
	s->fini(s);
}

TEST_F(SchedulerTest, ConcurrentScheduling) {
	auto *s = create_scheduler(4, 4096, 64);
	std::atomic<int> errors{0};
	auto worker = [&](size_t msg_size) {
		for (int i = 0; i < 100; i++) {
			auto *sched = s->get_schedule(s, msg_size, 4);
			if (!sched) { errors++; continue; }
			size_t total = 0;
			for (size_t j = 0; j < sched->num_xfer_infos; j++)
				total += sched->rail_xfer_infos[j].msg_size;
			if (total != msg_size) errors++;
			nccl_net_ofi_release_schedule(s, sched);
		}
	};
	std::vector<std::thread> threads;
	for (int i = 0; i < 4; i++) {
		threads.emplace_back(worker, 32);
		threads.emplace_back(worker, 16384);
	}
	for (auto &t : threads) t.join();
	EXPECT_EQ(0, errors.load());
	s->fini(s);
}

TEST_F(SchedulerTest, LargeMessageRoundRobinAdvances) {
	auto *s = create_scheduler(4, 4096, 64);
	auto *s3 = s->get_schedule(s, 4096 + 1, 4);
	auto *s4 = s->get_schedule(s, 4096 + 1, 4);
	ASSERT_NE(nullptr, s3);
	ASSERT_NE(nullptr, s4);
	int start3 = s3->rail_xfer_infos[0].rail_id;
	int start4 = s4->rail_xfer_infos[0].rail_id;
	EXPECT_EQ((start3 + 2) % 4, start4);
	nccl_net_ofi_release_schedule(s, s3);
	nccl_net_ofi_release_schedule(s, s4);
	s->fini(s);
}

TEST_F(SchedulerTest, ExactThresholdBoundary) {
	auto *s = create_scheduler(4, 4096, 64);
	/* Exactly at max_small_msg_size boundary: 63 is small, 64 is large */
	auto *small = s->get_schedule(s, 63, 4);
	ASSERT_NE(nullptr, small);
	EXPECT_EQ(1u, small->num_xfer_infos);
	nccl_net_ofi_release_schedule(s, small);

	auto *large = s->get_schedule(s, 64, 4);
	ASSERT_NE(nullptr, large);
	/* 64 bytes < min_stripe_size, so still 1 stripe but via large path */
	EXPECT_EQ(1u, large->num_xfer_infos);
	EXPECT_EQ(64u, large->rail_xfer_infos[0].msg_size);
	nccl_net_ofi_release_schedule(s, large);
	s->fini(s);
}

int main(int argc, char **argv)
{
	::testing::InitGoogleTest(&argc, argv);
	return RUN_ALL_TESTS();
}
