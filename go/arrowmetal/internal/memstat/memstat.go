// Package memstat reads the process's current physical memory footprint.
//
// It exists because the binding's leak test needs a *current* measure, not a high-water mark:
// syscall.Getrusage's ru_maxrss only ever rises, so once the rest of the test suite has touched a
// few hundred megabytes a leak has to exceed that peak before it shows up at all. macOS's
// TASK_VM_INFO.phys_footprint is the number Activity Monitor calls "Memory" — current, and it counts
// the Metal buffers ArrowMetal allocates as well as the Go heap.
//
// cgo is not allowed in _test.go files, which is why this is a package rather than a test helper.
package memstat

/*
#include <mach/mach.h>
#include <mach/task_info.h>

// Returns the current phys_footprint in bytes, or 0 with *err set to the kern_return_t.
static unsigned long long am_phys_footprint(int *err) {
    task_vm_info_data_t info;
    mach_msg_type_number_t count = TASK_VM_INFO_COUNT;
    kern_return_t kr = task_info(mach_task_self(), TASK_VM_INFO, (task_info_t)&info, &count);
    if (kr != KERN_SUCCESS) {
        *err = (int)kr;
        return 0;
    }
    // phys_footprint arrived in TASK_VM_INFO rev1; a shorter reply does not carry it.
    if (count < TASK_VM_INFO_REV1_COUNT) {
        *err = -1;
        return 0;
    }
    *err = 0;
    return (unsigned long long)info.phys_footprint;
}
*/
import "C"

import "fmt"

// PhysFootprint is the process's current physical memory footprint in bytes.
func PhysFootprint() (uint64, error) {
	var cerr C.int
	v := C.am_phys_footprint(&cerr)
	if cerr != 0 {
		return 0, fmt.Errorf("memstat: task_info(TASK_VM_INFO) failed: %d", int(cerr))
	}
	return uint64(v), nil
}
