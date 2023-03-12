# zalias

efficient repeated weighted random choice

## Purpose

The std library already offers weighted random choice, but if you want to sample from the same weighted distribution many (>3ish) times, data structures exist to reduce the workload to `O(1)` with a cheap `O(N)` initialization. This is a basic implementation of Vose's Alias Method -- a combination of fair dice and weighted coins, accounting for floating point errors.

## Installation

Zig has a package manager!!! Do something like the following.

```zig
// build.zig.zon
.{
    .name = "foo",
    .version = "0.0.0",

    .dependencies = .{
        .zalias = .{
            .name = "zalias",
	   .url = "https://github.com/hmusgrave/zalias/archive/refs/tags/z11-0.0.0.tar.gz",
            .hash = "12201bb45374f0ac377810370d3736bf95fcab688a6d6be47f0dd5d4f622cd0e23a6",
        },
    },
}
```

```zig
// build.zig
const zalias_pkg = b.dependency("zalias", .{
    .target = target,
    .optimize = optimize,
});
const zalias_mod = zalias_pkg.module("zalias");
exe.addModule("zalias", zalias_mod);
exe_tests.addModule("zalias", zalias_mod);
```

## Examples
```zig
const std = @import("std");
const RndGen = std.rand.DefaultPrng;

const zalias = @import("zalias");

test "something" {
    var rnd = RndGen.init(0);
    const allocator = std.testing.allocator;

    const F = f32;
    var probs = [_]F{ 1, 2, 3, 4 };
    var table = try Alias(F).init(allocator, probs[0..], .{
    	// - all of these arguments are optional
	// - all default to false
	// - if the values are compile-time known then generated code is
 	//   suitably smaller/faster
    	.can_mutate = false,
	.pre_normalized = false,
    	.pre_scaled = false,
	.weights_are_validated = false,
    });
    defer table.deinit();

    var total = [_]usize{ 0, 0, 0, 0 };
    const N: usize = 10000;
    for (0..N) |_|
        total[table.generate(rnd.random())] += 1;

    var freq: [total.len]F = undefined;
    for (freq[0..], total) |*f, t|
        f.* = @intToFloat(F, t) / @intToFloat(F, N);

    try std.testing.expectApproxEqAbs(@as(F, 0.1), freq[0], 0.01);
    try std.testing.expectApproxEqAbs(@as(F, 0.2), freq[1], 0.01);
    try std.testing.expectApproxEqAbs(@as(F, 0.3), freq[2], 0.01);
    try std.testing.expectApproxEqAbs(@as(F, 0.4), freq[3], 0.01);
}
```

## Status
Work has me pretty busy lately, so I might have a few weeks lead time on any responses. Targets Zig 0.11.
