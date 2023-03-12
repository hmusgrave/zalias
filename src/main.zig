const std = @import("std");
const Allocator = std.mem.Allocator;
const Random = std.rand.Random;
const RndGen = std.rand.DefaultPrng;

const zkwargs = @import("zkwargs");

fn kahan(comptime F: type, data: []F) F {
    var total: F = 0;
    var comp: F = 0;

    for (data) |x| {
        const corrected = x - comp;
        const temp_total = total + corrected;
        comp = (temp_total - total) - corrected;
        total = temp_total;
    }

    return total;
}

const AliasOpt = struct {
    pub fn can_mutate(comptime _: ?type) type {
        return zkwargs.Default(false);
    }

    pub fn pre_normalized(comptime _: ?type) type {
        return zkwargs.Default(false);
    }

    pub fn pre_scaled(comptime _: ?type) type {
        return zkwargs.Default(false);
    }

    pub fn weights_are_validated(comptime _: ?type) type {
        return zkwargs.Default(false);
    }
};

fn Alias(comptime F: type) type {
    // Great resource for Vose's Alias Method:
    // https://www.keithschwarz.com/darts-dice-coins/
    return struct {
        alias: []usize,
        prob: []F,
        allocator: Allocator,

        pub fn init(allocator: Allocator, weights: []F, _kwargs: anytype) !@This() {
            // weights:
            //   - w[i] == 0 implies i will not be selected
            //   - w[i] == w[k] * z implies i is z times more likely to be selected than k
            //   - at least one entry must be positive
            //   - no entries may be negative
            //
            // _kwargs:
            // ========
            // .can_mutate (false):
            //   - can we mutate the weights, or must we create an auxilliary buffer?
            // .pre_normalized (false):
            //   - (only used if !pre_scaled) do the weights sum to 1.0, or do we need to
            //     estimate the sum?
            // .pre_scaled (false):
            //   - are the (normalized) weights already multiplied by weights.len?
            // .weights_are_validated(false):
            //   - are the weights already guaranteed to be non-empty, non-negative, and
            //     contain at least one positive member?
            var kwargs = zkwargs.Options(AliasOpt).parse(_kwargs);

            const N = weights.len;

            if (!kwargs.weights_are_validated) {
                if (N == 0)
                    return error.NoWeights;

                var any_pos = false;
                for (weights) |w| {
                    any_pos = any_pos or w > 0;
                    if (w < 0)
                        return error.NegativeWeight;
                }

                if (!any_pos)
                    return error.CannotNormalizeToOne;
            }

            // Fragmentation tends to be better if we allocate
            // all the stuff we want to keep before all the
            // temporary intermediates
            var alias = try allocator.alloc(usize, N);
            errdefer allocator.free(alias);

            var prob = try allocator.alloc(F, N);
            errdefer allocator.free(prob);

            // Now all the stuff we're going to immediately
            // throw away
            var _small = try allocator.alloc(usize, N);
            defer allocator.free(_small);
            var small: []usize = _small[0..0];

            var _large = try allocator.alloc(usize, N);
            defer allocator.free(_large);
            var large: []usize = _large[0..0];

            var p = weights;
            if (!kwargs.can_mutate) {
                p = try allocator.alloc(F, N);
                for (p, weights) |*_p, w|
                    _p.* = w;
            }
            defer {
                if (!kwargs.can_mutate) allocator.free(p);
            }

            // turn `p` into scaled weights, so that they
            // all sum to `N`
            if (!kwargs.pre_scaled) {
                var scalar = @intToFloat(F, p.len);
                if (!kwargs.pre_normalized) {
                    const total = kahan(F, p);
                    if (total <= 0)
                        return error.InvalidWeights;
                    scalar /= total;
                }
                for (p) |*_p|
                    _p.* *= scalar;
            }

            // split apart weights that fill a bucket vs those
            // which need aliases
            //
            // note that we simply update the slice into the
            // same _small/_large backing memory to model elements
            // being included/deleted
            for (p, 0..) |_p, i| {
                if (_p < 1) {
                    const L = small.len;
                    small = _small[0 .. L + 1];
                    small[L] = i;
                } else {
                    const L = large.len;
                    large = _large[0 .. L + 1];
                    large[L] = i;
                }
            }

            // main work loop, pop beginning and choose how to alias
            // and what to work on next
            while (small.len != 0 and large.len != 0) {
                var l = small[0];
                var g = large[0];
                prob[l] = p[l];
                alias[l] = g;
                p[g] = (p[l] + p[g]) - 1;
                if (p[g] < 1) {
                    small[0] = g;
                    large = large[1..];
                } else {
                    large[0] = g;
                    small = small[1..];
                }
            }

            // cleanup
            for (large) |l|
                prob[l] = 1.0;

            // more cleanup, only needed due to floating-point
            // instabilities
            for (small) |s|
                prob[s] = 1.0;

            return @This(){
                .alias = alias,
                .prob = prob,
                .allocator = allocator,
            };
        }

        pub fn deinit(self: *@This()) void {
            self.allocator.free(self.alias);
            self.allocator.free(self.prob);
        }

        pub fn generate(self: *@This(), rand: Random) usize {
            const i = rand.uintLessThan(usize, self.prob.len);
            const f = rand.float(F);
            return if (f < self.prob[i]) i else self.alias[i];
        }
    };
}

test "can mutate" {
    var rnd = RndGen.init(0);
    const allocator = std.testing.allocator;
    const F = f32;
    var probs = [_]F{ 1, 2, 3, 4 };
    var table = try Alias(F).init(allocator, probs[0..], .{ .can_mutate = true });
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

test "works normalized" {
    var rnd = RndGen.init(0);
    const allocator = std.testing.allocator;
    const F = f32;
    var probs = [_]F{ 0.1, 0.2, 0.3, 0.4 };
    var table = try Alias(F).init(allocator, probs[0..], .{ .pre_normalized = true });
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

test "works scaled" {
    var rnd = RndGen.init(0);
    const allocator = std.testing.allocator;
    const F = f32;
    var probs = [_]F{ 0.4, 0.8, 1.2, 1.6 };
    var table = try Alias(F).init(allocator, probs[0..], .{ .pre_scaled = true });
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

test "valiation not needed" {
    var rnd = RndGen.init(0);
    const allocator = std.testing.allocator;
    const F = f32;
    var probs = [_]F{ 0.4, 0.8, 1.2, 1.6 };
    var table = try Alias(F).init(allocator, probs[0..], .{ .weights_are_validated = true });
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
