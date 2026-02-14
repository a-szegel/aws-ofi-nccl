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
