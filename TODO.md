# TODO: Project Roadmap (4 Weeks)

## Week 1: Core Implementation ✓
- [x] Project setup and build system
- [x] Basic hash map structure (fixed-size)
- [x] Core operations: Insert, Search, Delete, Count
- [x] Simple test programs

## Week 2: Testing & Validation
- [ ] Verify correctness of all operations
- [ ] Test edge cases (empty, full table, collisions)
- [ ] Test concurrent operations
- [ ] Fix bugs and improve stability
- [ ] Add basic performance timing

## Week 3: SlabAlloc Integration
- [ ] Study SlabHash's SlabAlloc implementation
  - Read `SlabHash/SlabAlloc/src/slab_alloc.cuh`
  - Understand warp-level allocator interface

- [ ] Integrate allocator into our code
  - Add allocator context to `GpuHashMapContext`
  - Update insert to use dynamic allocation
  - Change from open addressing to chaining

- [ ] Test dynamic allocation version
  - Compare with fixed-size version
  - Verify no memory leaks

## Week 4: Polish & Documentation
- [ ] Performance comparison (fixed vs dynamic)
- [ ] Clean up code and comments
- [ ] Update documentation with results
- [ ] Prepare project presentation
- [ ] Final testing and bug fixes

## Known Issues to Fix

1. **Tombstone accumulation**: Add cleanup or switch to chaining
2. **Full table handling**: Return errors gracefully
3. **High load factor**: Document performance degradation >0.7

## Stretch Goals (If Time Permits)

- [ ] Simple Python benchmark script
- [ ] Better hash function options
- [ ] Table resizing support
- [ ] More usage examples

## Quick Reference: SlabAlloc Files

Key files to study for Week 3:
- `SlabHash/SlabAlloc/src/slab_alloc.cuh` - Allocator API
- `SlabHash/src/concurrent_map/cmap_class.cuh` - How context uses allocator
- `SlabHash/src/concurrent_map/warp/insert.cuh` - Dynamic insertion example
