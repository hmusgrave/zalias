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
            .url = "https://github.com/hmusgrave/zalias/archive/refs/tags/z11-0.2.0.tar.gz",
            .hash = "1220bb80981d83ffae53dd8d0636ceba9d621850292a301d84411120bfa38ec0ae4b",
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
    const weights = [_]F{ 1, 2, 3, 4 };

    const Table = Alias(F, u8)

    // We're using the type system to ensure that we only try to construct
    // a table if this is a valid probability distribution (weights are non-negative
    // and can be scaled by a positive scalar to sum to 1). If you know your
    // weights satisfy those constraints then you can choose to bypass
    // the check.
    // 
    // const validated_weights = Table.pinky_promise_weights_are_valid(weights[0..])
    const validated_weights = try Table.validate_weights(weights[0..]);

    var table = try Alias(F, u8).init(allocator, validated_weights, .{
    	// - all of these arguments are optional
	// - all default to false
	// - if the values are compile-time known then generated code is
 	//   suitably smaller/faster
	.pre_normalized = false,  // weights sum to 1 (ignored if pre_scaled is true)
    	.pre_scaled = false,  // weights sum to N
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
Targets Zig 0.11. Note that the minimum size in the current implementation is approximately a cache line, which might not be suitable for holding many small distributions.
