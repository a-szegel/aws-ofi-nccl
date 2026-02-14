# Bugs Found by Unit Testing

## Bug 1: freelist increase_entry_count page-padding overwritten

**Location:** `src/nccl_ofi_freelist.cpp`, `init_internal()`, lines ~59 and ~64

**Cause:** The `increase_entry_count` member is assigned twice:
```cpp
// Line ~59: correctly page-pads the value
this->increase_entry_count = freelist_page_padded_entry_count(increase_entry_count_arg);
// ...
// Line ~64: overwrites with the original unpadded value
this->increase_entry_count = increase_entry_count_arg;
```

The second assignment undoes the page-padding optimization. The comment above
line 59 explicitly states the intent is to pad both `initial_entry_count` and
`increase_entry_count` to full pages, but only `initial_entry_count_arg` (a
local variable) retains its padded value.

**Impact:** Subsequent freelist growth allocations (when the freelist needs to
expand beyond its initial size) use the raw `increase_entry_count_arg` instead
of the page-padded value. This means growth allocations may not fill complete
memory pages, wasting the tail of each page. The initial allocation is
unaffected (it uses the local `initial_entry_count_arg` which is correctly
padded).

**Test that caught it:** Code review during test development (visual inspection
of `init_internal()`).

**Fix:** Remove the second assignment on line ~64. The page-padded value from
line ~59 is the intended one.
```diff
-	this->increase_entry_count = increase_entry_count_arg;
```

---

## Bug 2: freelist entry_free uses wrong redzone size constant

**Location:** `include/nccl_ofi_freelist.h`, `entry_free()` line ~199 and
`entry_set_undefined()` line ~218

**Cause:** These functions compute user entry size as:
```cpp
size_t user_entry_size = this->entry_size - MEMCHECK_REDZONE_SIZE;
```
But `add()` in `src/nccl_ofi_freelist.cpp` uses:
```cpp
size_t user_entry_size = this->entry_size - this->memcheck_redzone_size;
```
The member `this->memcheck_redzone_size` is `NCCL_OFI_ROUND_UP(MEMCHECK_REDZONE_SIZE, entry_alignment)`,
which can be larger than the raw `MEMCHECK_REDZONE_SIZE` when `entry_alignment > 1`.

**Impact:** When ASAN or Valgrind memcheck is enabled (`MEMCHECK_REDZONE_SIZE > 0`)
AND the freelist uses entry alignment > 1 (the "complex" constructor), the
`entry_free()` and `entry_set_undefined()` functions mark a larger region as
noaccess/undefined than was actually allocated for user data. This could cause
false-positive memcheck errors or mask real out-of-bounds accesses.

In the default build (`MEMCHECK_REDZONE_SIZE = 0`), both values are 0 so the
bug is dormant.

**Test that caught it:** Code review during test development.

**Fix:** Replace `MEMCHECK_REDZONE_SIZE` with `this->memcheck_redzone_size` in
both `entry_free()` and `entry_set_undefined()`.

---

## Bug 3: freelist add() error path corrupts state when entry_init_fn fails

**Location:** `src/nccl_ofi_freelist.cpp`, `add()`, error path starting at
line ~310

**Cause:** When `entry_init_fn` fails partway through the entry initialization
loop, the error path has three problems:

1. **Dangling `this->blocks` pointer:** `this->blocks = block` is set before
   the loop (line ~274), but the error path `free(block)` without removing it
   from `this->blocks`, leaving a dangling pointer.

2. **Wrong deallocation pointer:** The local `buffer` pointer has been advanced
   past the start of the allocation during the loop. The error path calls
   `nccl_net_ofi_dealloc_mr_buffer(buffer, block_mem_size)` with this advanced
   pointer instead of the original `block->memory`.

3. **Orphaned entries:** Entries already added to `this->entries` linked list
   during the loop point into the freed memory block.

**Impact:** If `entry_init_fn` fails (which is a rare error path), the freelist
is left in a corrupted state. The dangling `this->blocks` pointer will cause a
use-after-free when the freelist destructor iterates the block list. The wrong
deallocation pointer may cause the memory allocator to corrupt its internal
state or crash.

**Test that caught it:** Code review during test development.

**Fix:** Save the original buffer pointer before the loop and use it in the
error path. Remove the block from `this->blocks` and unlink entries from
`this->entries` in the error path.

---

## Bug 4: msgbuff_init validation has uint16_t overflow (latent)

**Location:** `src/nccl_ofi_msgbuff.cpp`, `nccl_ofi_msgbuff_init()`, line ~21

**Cause:** The validation check:
```cpp
(uint16_t)(1 << bit_width) <= 2 * max_inprogress
```
When `bit_width >= 16`, `(1 << bit_width)` overflows `uint16_t` to 0. The
check `0 <= 2 * max_inprogress` is always true, so the function accidentally
rejects these values (the condition triggers the error path).

Additionally, `1 << 16` is undefined behavior when `int` is 16-bit (though
this is not the case on any current target platform).

**Impact:** Currently benign — the overflow accidentally produces the correct
rejection behavior. However, the logic is fragile and confusing. A future
refactor could break it.

**Test that caught it:** `MsgbuffTest.BitWidth16Rejected`

**Fix:** Add an explicit check for `bit_width >= 16` before the shift, or use
`uint32_t` for the computation:
```cpp
if (bit_width >= 16 || (uint32_t)(1 << bit_width) <= 2 * max_inprogress) {
```

---

## Previously Found (Commit 2)

## Bug 5: filter_provider_list leaks removed fi_info nodes

**Location:** `src/nccl_ofi_ofiutils.cpp`, `filter_provider_list()`

**Cause:** When `filter_provider_list()` removes all providers from the list,
it sets `*providers = NULL`. The error path in `get_providers()` only calls
`fi_freeinfo()` when the provider list is non-NULL, so the removed nodes are
leaked.

**Impact:** Memory leak on the error path when all providers are filtered out.

**Test that caught it:** `GetProvidersTest.ProvIncludeFiltersOutAllProviders`
(documented in test comments in commit 2).

---

## Bug 6: MR cache grow updates size before realloc succeeds

**Location:** `src/nccl_ofi_mr.cpp`, `nccl_ofi_mr_cache_grow()`, line ~88

**Cause:** The function doubles `cache->size` before calling `realloc()`:
```cpp
cache->size *= 2;                    // size updated first
ptr = realloc(cache->slots, ...);    // realloc may fail
```
If `realloc()` fails, `cache->size` has been doubled but the actual allocation
remains at the old size. Subsequent operations (insert, lookup) use the inflated
`cache->size` and may access memory beyond the allocation.

**Impact:** On realloc failure (out of memory), the MR cache has an inconsistent
size field. Any subsequent `memmove()` in `insert_entry` or `del_entry` could
write past the end of the slots array, causing heap corruption. This is a
memory safety bug triggered only under memory pressure.

**Test that caught it:** Code review during test development.

**Fix:** Use a local variable for the new size and only update `cache->size`
after `realloc()` succeeds.

---

## Bug 7: CM listener accept() uses "throw new" instead of "throw"

**Location:** `src/cm/nccl_ofi_cm.cpp`, `nccl_ofi_cm_listener::accept()`, line 71

**Cause:** The code uses:
```cpp
throw new std::runtime_error("Failed to process pending reqs");
```
instead of:
```cpp
throw std::runtime_error("Failed to process pending reqs");
```
`throw new` allocates the exception on the heap and throws a pointer
(`std::runtime_error*`), not the exception object itself.

**Impact:** All catch blocks in the codebase catch `const std::exception &e`
(by reference). A thrown pointer will not match any of these handlers, causing
`std::terminate()` to be called and the process to crash. Additionally, the
heap-allocated exception object is never deleted, causing a memory leak (though
the crash makes this moot). Every other `throw` in the codebase (40+ instances)
correctly throws by value.

**Test that caught it:** Code review during systematic examination of
`src/cm/nccl_ofi_cm.cpp`.

**Fix:** Remove `new` to throw by value, matching all other throw statements.
```diff
-		throw new std::runtime_error("Failed to process pending reqs");
+		throw std::runtime_error("Failed to process pending reqs");
```

---

## Bug 8: Inverted `first_error` condition in cleanup functions (FIXED)

**Location:** `src/nccl_ofi_net.cpp`, lines 783, 799, 808 (`release_all_domain_and_ep`) and lines 962, 971, 979 (`release_all_ep`)

**Cause:** Both `release_all_domain_and_ep()` and `release_all_ep()` use a `first_error` variable initialized to 0 to capture the first error during cleanup. However, the conditions guarding the assignment are inverted:

```cpp
int ret, first_error = 0;
...
if (ret != 0) {
    if (first_error != 0) {   // BUG: should be == 0
        first_error = ret;     // Never reached on first error!
    }
}
```

Since `first_error` starts at 0, the condition `first_error != 0` is false when the first error occurs, so the error is silently swallowed. All 6 instances across both functions have this same inverted logic.

**Impact:** During device/domain/endpoint cleanup, if any `release_ep()` or `release_domain()` call fails, the error is silently discarded and the function returns 0 (success). This means:
- Callers never learn that cleanup failed
- Resource leaks during shutdown go unreported
- The `FI_EBUSY` fallback for non-empty tables is also never set

**Fix:** Changed all 6 instances of `first_error != 0` to `first_error == 0` so the first error is properly captured and returned.

**Test that caught it:** Code review during systematic audit of error-handling patterns.

---

## Bug 9: Memory leak in `rdma_comm_alloc_flush_req` flush buffer error path (FIXED)

**Location:** `src/nccl_ofi_rdma.cpp`, function `rdma_comm_alloc_flush_req()`, line ~3894

**Cause:** When `flush_buff_fl->entry_alloc()` fails (returns NULL), the function returns `-ENOMEM` immediately without freeing the `req` that was already allocated from the freelist on line ~3877. The caller (`flush()`) checks `if (req) req->free(req, false)` in its error path, but `*ret_req` was set to NULL at the top of `rdma_comm_alloc_flush_req`, so the caller's `req` is NULL and the freelist entry is leaked.

**Impact:** Under memory pressure, each failed flush buffer allocation leaks one request from the freelist. Over time this exhausts the request freelist, causing all subsequent flush operations to fail with `-ENOMEM`.

**Test that caught it:** Manual code review of error paths in RDMA flush allocation.

**Fix:** Call `req->free(req, false)` before returning `-ENOMEM` when flush buffer allocation fails. The `free_flush_req` function already handles a NULL `flush_fl_elem` safely.

---

## Bug 10: Freelist leak in `create_send_comm` error path (FIXED)

**Location:** `src/nccl_ofi_rdma.cpp`, function `nccl_net_ofi_rdma_ep_t::create_send_comm()`, error label (~line 6130)

**Cause:** When `reg_internal_mr()` fails after `nccl_ofi_reqs_fl` has been allocated with `new`, the error path calls `free_rdma_send_comm()` which only frees the `calloc`'d members (rails, control_rails, ctrl_mailbox) and the comm struct itself. It does NOT `delete` the `nccl_ofi_reqs_fl` freelist. The analogous error path in `prepare_recv_comm()` correctly deletes `nccl_ofi_reqs_fl` before calling `free_rdma_recv_comm()`.

**Impact:** If MR registration fails during send communicator creation, the request freelist (and all its internal allocations) is leaked. This can happen under memory pressure or when the provider rejects the registration.

**Test that caught it:** Manual code review comparing error paths of `create_send_comm` vs `prepare_recv_comm`.

**Fix:** Add `delete ret_s_comm->nccl_ofi_reqs_fl;` before `free_rdma_send_comm(ret_s_comm)` in the error path.

---

## Bug 11: Two memory leaks in `sendrecv_recv_comm_prepare` (FIXED)

**Location:** `src/nccl_ofi_sendrecv.cpp`, function `sendrecv_recv_comm_prepare()`

**Cause (leak 1, line ~1349):** When the tag limit check fails (`ep->tag + 1 >= device->max_tag`), the function returns `nullptr` without freeing `r_comm` which was `calloc`'d earlier at line ~1323.

**Cause (leak 2, line ~1372):** When `sendrecv_recv_comm_alloc_and_reg_flush_buff()` fails, the error path calls `free(r_comm)` but does NOT `delete r_comm->nccl_ofi_reqs_fl` which was allocated with `new` at line ~1358.

**Impact:** Leak 1: Each failed connection attempt when at the tag limit leaks a `sendrecv_recv_comm_t` struct. Leak 2: Each failed flush buffer registration leaks the request freelist and all its internal allocations.

**Test that caught it:** Manual code review of error paths in sendrecv communicator preparation.

**Fix:** (1) Add `free(r_comm)` before `return nullptr` in the tag limit check. (2) Add `delete r_comm->nccl_ofi_reqs_fl` before `free(r_comm)` in the flush buffer error path.
