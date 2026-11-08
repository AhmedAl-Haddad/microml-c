#define WIN32_LEAN_AND_MEAN
#include "base.h"
#include "platform.h"
#include <windows.h>

#ifdef _WIN32

u32 plat_get_pagesize(void) {
    SYSTEM_INFO sysinfo = {0};
    GetSystemInfo(&sysinfo);

    return sysinfo.dwPageSize;
}

void *plat_mem_reserve(u64 size) {
    return VirtualAlloc(NULL, size, MEM_RESERVE, PAGE_READWRITE);
}

b32 plat_mem_commit(void *ptr, u64 size) {
    void *ret = VirtualAlloc(ptr, size, MEM_COMMIT, PAGE_READWRITE);
    return ret != NULL;
}

b32 plat_mem_decommit(void *ptr, u64 size) {
    return VirtualFree(ptr, size, MEM_DECOMMIT);
}

b32 plat_mem_release(void *ptr, u64 size) {
    return VirtualFree(ptr, size, MEM_RELEASE);
}

#endif
