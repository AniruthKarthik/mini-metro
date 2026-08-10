- [x] implement tunnel segment gating — spending a tunnel token unlocks a specific line segment
- [x] implement interchange routing effect — IsInterchange flag gives transfer priority or capacity bonus
- [x] add line shortening action — remove the last station from a line endpoint
- [x] fix station base capacity to 6 passengers (currently 8)
- [x] fix interchange capacity to 18 passengers (3x base, not 2x) and reduce dwell time for faster boarding
- [x] fix weekly reward to offer 2 bonus choices not 3
- [x] implement closed loop lines — last station connects to first, trains travel one-way with no bounce
- [x] implement A* passenger routing with transfer penalty and capacity reservation replacing BFS
- [x] implement accelerating passenger spawn rate that increases over simulation time
- [x] add train repositioning action — move a train to a specific segment on its line
- [x] enforce max trains per line limit configurable per map
- [x] replace distance-threshold tunnel proxy with actual river/water body geometry
- [x] expand station types beyond 5 base kinds to include rare shapes (Gem, Sector, Cross, etc.)
- [ ] weight passenger destination demand toward rare station shapes instead of uniform random sampling
- [ ] support map/city geography definitions (e.g., London, NYC, Tokyo) with custom river layouts and line limits
- [ ] implement smooth train acceleration/deceleration curves and track-curve slowdown physics

TO BE DONE ONLY AFTER COMPLETING THE SIMULATOR ENGINE
- [ ] create gym-style reset and step api for ml agent integration
- [ ] expose JSON or C-Go IPC bindings for external python RL frameworks
