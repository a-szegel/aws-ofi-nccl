/*
 * Copyright (c) 2026 Amazon.com, Inc. or its affiliates. All rights reserved.
 *
 * GoogleTest unit tests for nccl_ofi_msgbuff.
 * Tests the circular message buffer used to track in-flight messages.
 */

#include <gtest/gtest.h>
#include "config.h"
#include "nccl_ofi.h"
#include "test-logger.h"
#include "nccl_ofi_msgbuff.h"

class MsgbuffTest : public ::testing::Test {
protected:
	void SetUp() override { ofi_log_function = logger; }

	void TearDown() override
	{
		if (msgbuff) {
			nccl_ofi_msgbuff_destroy(msgbuff);
			msgbuff = nullptr;
		}
	}

	nccl_ofi_msgbuff_t *msgbuff = nullptr;
};

/* Verify that init rejects max_inprogress=0. */
TEST_F(MsgbuffTest, InitRejectsZeroMaxInprogress)
{
	msgbuff = nccl_ofi_msgbuff_init(0, 4, 0);
	EXPECT_EQ(msgbuff, nullptr);
}

/* Verify that init rejects bit_width too small for max_inprogress.
 * The requirement is: (1 << bit_width) > 2 * max_inprogress.
 * With bit_width=2 (field_size=4) and max_inprogress=2, 4 <= 4 fails. */
TEST_F(MsgbuffTest, InitRejectsBitWidthTooSmall)
{
	msgbuff = nccl_ofi_msgbuff_init(2, 2, 0);
	EXPECT_EQ(msgbuff, nullptr);
}

/* Verify that init succeeds with valid parameters. */
TEST_F(MsgbuffTest, InitSucceeds)
{
	msgbuff = nccl_ofi_msgbuff_init(4, 4, 0);
	ASSERT_NE(msgbuff, nullptr);
}

/*
 * Verify bit_width=16 is rejected.
 *
 * The validation check (uint16_t)(1 << bit_width) <= 2 * max_inprogress
 * has a uint16_t overflow when bit_width=16: (uint16_t)(65536) wraps to 0.
 * This accidentally still rejects (0 <= 8 is true, triggering the error path),
 * but the logic is fragile and relies on the overflow producing 0.
 * Note: 1 << 16 is also undefined behavior when int is 16-bit.
 */
TEST_F(MsgbuffTest, BitWidth16Rejected)
{
	msgbuff = nccl_ofi_msgbuff_init(4, 16, 0);
	EXPECT_EQ(msgbuff, nullptr)
		<< "bit_width=16 should be rejected (field_size would be 0)";
}


/* Verify basic insert-retrieve-complete cycle. */
TEST_F(MsgbuffTest, BasicInsertRetrieveComplete)
{
	msgbuff = nccl_ofi_msgbuff_init(4, 4, 0);
	ASSERT_NE(msgbuff, nullptr);

	int data = 42;
	nccl_ofi_msgbuff_status_t stat;
	nccl_ofi_msgbuff_elemtype_t type = NCCL_OFI_MSGBUFF_REQ;

	EXPECT_EQ(nccl_ofi_msgbuff_insert(msgbuff, 0, &data, type, &stat),
		  NCCL_OFI_MSGBUFF_SUCCESS);

	void *elem = nullptr;
	nccl_ofi_msgbuff_elemtype_t out_type;
	EXPECT_EQ(nccl_ofi_msgbuff_retrieve(msgbuff, 0, &elem, &out_type, &stat),
		  NCCL_OFI_MSGBUFF_SUCCESS);
	EXPECT_EQ(elem, &data);
	EXPECT_EQ(out_type, NCCL_OFI_MSGBUFF_REQ);

	EXPECT_EQ(nccl_ofi_msgbuff_complete(msgbuff, 0, &stat),
		  NCCL_OFI_MSGBUFF_SUCCESS);
}

/* Verify that inserting beyond capacity returns UNAVAILABLE. */
TEST_F(MsgbuffTest, InsertBeyondCapacity)
{
	msgbuff = nccl_ofi_msgbuff_init(2, 4, 0);
	ASSERT_NE(msgbuff, nullptr);

	int data = 1;
	nccl_ofi_msgbuff_status_t stat;

	EXPECT_EQ(nccl_ofi_msgbuff_insert(msgbuff, 0, &data, NCCL_OFI_MSGBUFF_REQ, &stat),
		  NCCL_OFI_MSGBUFF_SUCCESS);
	EXPECT_EQ(nccl_ofi_msgbuff_insert(msgbuff, 1, &data, NCCL_OFI_MSGBUFF_REQ, &stat),
		  NCCL_OFI_MSGBUFF_SUCCESS);
	/* Buffer is full (max_inprogress=2), next insert should fail */
	EXPECT_EQ(nccl_ofi_msgbuff_insert(msgbuff, 2, &data, NCCL_OFI_MSGBUFF_REQ, &stat),
		  NCCL_OFI_MSGBUFF_INVALID_IDX);
	EXPECT_EQ(stat, NCCL_OFI_MSGBUFF_UNAVAILABLE);
}

/* Verify duplicate insert returns INPROGRESS status. */
TEST_F(MsgbuffTest, DuplicateInsert)
{
	msgbuff = nccl_ofi_msgbuff_init(4, 4, 0);
	ASSERT_NE(msgbuff, nullptr);

	int data = 1;
	nccl_ofi_msgbuff_status_t stat;

	EXPECT_EQ(nccl_ofi_msgbuff_insert(msgbuff, 0, &data, NCCL_OFI_MSGBUFF_REQ, &stat),
		  NCCL_OFI_MSGBUFF_SUCCESS);
	EXPECT_EQ(nccl_ofi_msgbuff_insert(msgbuff, 0, &data, NCCL_OFI_MSGBUFF_REQ, &stat),
		  NCCL_OFI_MSGBUFF_INVALID_IDX);
	EXPECT_EQ(stat, NCCL_OFI_MSGBUFF_INPROGRESS);
}

/* Verify out-of-order completion advances msg_last_incomplete correctly. */
TEST_F(MsgbuffTest, OutOfOrderCompletion)
{
	msgbuff = nccl_ofi_msgbuff_init(4, 4, 0);
	ASSERT_NE(msgbuff, nullptr);

	int data[4] = {0, 1, 2, 3};
	nccl_ofi_msgbuff_status_t stat;

	for (int i = 0; i < 4; i++)
		nccl_ofi_msgbuff_insert(msgbuff, i, &data[i], NCCL_OFI_MSGBUFF_REQ, &stat);

	/* Complete out of order: 2, 0, 1, 3 */
	EXPECT_EQ(nccl_ofi_msgbuff_complete(msgbuff, 2, &stat), NCCL_OFI_MSGBUFF_SUCCESS);
	/* msg_last_incomplete should still be 0 */
	EXPECT_EQ(nccl_ofi_msgbuff_complete(msgbuff, 0, &stat), NCCL_OFI_MSGBUFF_SUCCESS);
	/* Now 0 is complete, but 1 is not, so msg_last_incomplete = 1 */
	EXPECT_EQ(nccl_ofi_msgbuff_complete(msgbuff, 1, &stat), NCCL_OFI_MSGBUFF_SUCCESS);
	/* Now 0,1,2 are complete, msg_last_incomplete jumps to 3 */
	EXPECT_EQ(nccl_ofi_msgbuff_complete(msgbuff, 3, &stat), NCCL_OFI_MSGBUFF_SUCCESS);

	/* All complete, should be able to insert new messages */
	EXPECT_EQ(nccl_ofi_msgbuff_insert(msgbuff, 4, &data[0], NCCL_OFI_MSGBUFF_REQ, &stat),
		  NCCL_OFI_MSGBUFF_SUCCESS);
}

/* Verify replace works on in-progress messages. */
TEST_F(MsgbuffTest, ReplaceInProgress)
{
	msgbuff = nccl_ofi_msgbuff_init(4, 4, 0);
	ASSERT_NE(msgbuff, nullptr);

	int data1 = 1, data2 = 2;
	nccl_ofi_msgbuff_status_t stat;

	nccl_ofi_msgbuff_insert(msgbuff, 0, &data1, NCCL_OFI_MSGBUFF_REQ, &stat);

	EXPECT_EQ(nccl_ofi_msgbuff_replace(msgbuff, 0, &data2, NCCL_OFI_MSGBUFF_BUFF, &stat),
		  NCCL_OFI_MSGBUFF_SUCCESS);

	void *elem;
	nccl_ofi_msgbuff_elemtype_t type;
	nccl_ofi_msgbuff_retrieve(msgbuff, 0, &elem, &type, &stat);
	EXPECT_EQ(elem, &data2);
	EXPECT_EQ(type, NCCL_OFI_MSGBUFF_BUFF);
}

/* Verify replace fails on not-started messages. */
TEST_F(MsgbuffTest, ReplaceNotStarted)
{
	msgbuff = nccl_ofi_msgbuff_init(4, 4, 0);
	ASSERT_NE(msgbuff, nullptr);

	int data = 1;
	nccl_ofi_msgbuff_status_t stat;

	EXPECT_EQ(nccl_ofi_msgbuff_replace(msgbuff, 0, &data, NCCL_OFI_MSGBUFF_REQ, &stat),
		  NCCL_OFI_MSGBUFF_INVALID_IDX);
	EXPECT_EQ(stat, NCCL_OFI_MSGBUFF_NOTSTARTED);
}

/* Verify retrieve returns NULL elem pointer for null-elem input. */
TEST_F(MsgbuffTest, RetrieveNullElemPointer)
{
	msgbuff = nccl_ofi_msgbuff_init(4, 4, 0);
	ASSERT_NE(msgbuff, nullptr);

	nccl_ofi_msgbuff_elemtype_t type;
	nccl_ofi_msgbuff_status_t stat;

	EXPECT_EQ(nccl_ofi_msgbuff_retrieve(msgbuff, 0, nullptr, &type, &stat),
		  NCCL_OFI_MSGBUFF_ERROR);
}

/* Verify sequence number wraparound works correctly. */
TEST_F(MsgbuffTest, SequenceWraparound)
{
	/* bit_width=4 means field_size=16, start near the wrap point */
	uint16_t start = 14;
	msgbuff = nccl_ofi_msgbuff_init(2, 4, start);
	ASSERT_NE(msgbuff, nullptr);

	int data = 99;
	nccl_ofi_msgbuff_status_t stat;

	/* Insert at 14, 15 (wraps to 0 internally) */
	EXPECT_EQ(nccl_ofi_msgbuff_insert(msgbuff, 14, &data, NCCL_OFI_MSGBUFF_REQ, &stat),
		  NCCL_OFI_MSGBUFF_SUCCESS);
	EXPECT_EQ(nccl_ofi_msgbuff_insert(msgbuff, 15, &data, NCCL_OFI_MSGBUFF_REQ, &stat),
		  NCCL_OFI_MSGBUFF_SUCCESS);

	/* Complete both */
	EXPECT_EQ(nccl_ofi_msgbuff_complete(msgbuff, 14, &stat), NCCL_OFI_MSGBUFF_SUCCESS);
	EXPECT_EQ(nccl_ofi_msgbuff_complete(msgbuff, 15, &stat), NCCL_OFI_MSGBUFF_SUCCESS);

	/* Now insert at 0 (wrapped) */
	EXPECT_EQ(nccl_ofi_msgbuff_insert(msgbuff, 0, &data, NCCL_OFI_MSGBUFF_REQ, &stat),
		  NCCL_OFI_MSGBUFF_SUCCESS);
}

/* Verify destroy handles NULL gracefully. */
TEST_F(MsgbuffTest, DestroyNull)
{
	EXPECT_FALSE(nccl_ofi_msgbuff_destroy(nullptr));
	msgbuff = nullptr; /* prevent double-destroy in TearDown */
}

/* Verify non-zero start_seq works. */
TEST_F(MsgbuffTest, NonZeroStartSeq)
{
	msgbuff = nccl_ofi_msgbuff_init(4, 4, 5);
	ASSERT_NE(msgbuff, nullptr);

	int data = 1;
	nccl_ofi_msgbuff_status_t stat;

	/* Insert at start_seq=5 should work */
	EXPECT_EQ(nccl_ofi_msgbuff_insert(msgbuff, 5, &data, NCCL_OFI_MSGBUFF_REQ, &stat),
		  NCCL_OFI_MSGBUFF_SUCCESS);
}

int main(int argc, char **argv)
{
	::testing::InitGoogleTest(&argc, argv);
	return RUN_ALL_TESTS();
}
