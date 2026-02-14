#include "config.h"
#include <gtest/gtest.h>
#include "nccl_ofi.h"
#include "nccl_ofi_pthread.h"

nccl_ofi_logger_t ofi_log_function = nullptr;

class PthreadWrapperTest : public ::testing::Test {
protected:
    pthread_mutex_t mutex;
    void SetUp() override { pthread_mutex_init(&mutex, nullptr); }
    void TearDown() override { pthread_mutex_destroy(&mutex); }
};

TEST_F(PthreadWrapperTest, LockUnlockOnDestruction) {
    { pthread_wrapper w(&mutex); }
    EXPECT_EQ(0, pthread_mutex_trylock(&mutex));
    pthread_mutex_unlock(&mutex);
}

TEST_F(PthreadWrapperTest, ManualUnlock) {
    pthread_wrapper w(&mutex);
    w.unlock();
    EXPECT_EQ(0, pthread_mutex_trylock(&mutex));
    pthread_mutex_unlock(&mutex);
}

TEST_F(PthreadWrapperTest, LockIsHeld) {
    pthread_wrapper w(&mutex);
    EXPECT_EQ(EBUSY, pthread_mutex_trylock(&mutex));
}

TEST_F(PthreadWrapperTest, MutexInitDestroy) {
    pthread_mutex_t m;
    EXPECT_EQ(0, nccl_net_ofi_mutex_init(&m, nullptr));
    EXPECT_EQ(0, nccl_net_ofi_mutex_destroy(&m));
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
