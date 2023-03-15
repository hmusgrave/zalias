const std = @import("std");
const Allocator = std.mem.Allocator;
const Random = std.rand.Random;
const RndGen = std.rand.DefaultPrng;

const zkwargs = @import("zkwargs");

fn kahan_scalar(comptime F: type, data: []F, _total: F, _comp: F) F {
    // naive kahan summation, directly transcribing
    // any blog on the topic
    var total = _total;
    var comp = _comp;

    for (data) |x| {
        const corrected = x - comp;
        const temp_total = total + corrected;
        comp = (temp_total - total) - corrected;
        total = temp_total;
    }

    return total;
}

fn kahan_simd(comptime F: type, data: []F, comptime V: usize) F {
    // naive kahan summation, applied to each "lane" of data
    // then use scalar methods for stragglers at the end
    var total = @splat(V, @as(F, 0));
    var comp = @splat(V, @as(F, 0));
    var i: usize = 0;
    while (i + V < data.len) : (i += V) {
        const _x: [V]F = data[i..][0..V].*;
        const x: @Vector(V, F) = _x;
        const corrected = x - comp;
        const temp_total = total + corrected;
        comp = (temp_total - total) - corrected;
        total = temp_total;
    }

    return kahan_scalar(F, data[@max(V, data.len) - V ..], @reduce(.Add, total), @reduce(.Add, comp));
}

fn kahan(comptime F: type, data: []F) F {
    // TODO: more principled approach to vector lengths, maybe
    // requiring a different function signature to make that
    // happen
    //
    // Big enough for f16 on avx512 though, and not a huge amount
    // of overhead so that smallish inputs aren't too slow
    return kahan_simd(F, data, 32);
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

            var alias = try allocator.alloc(usize, N);
            errdefer allocator.free(alias);

            var prob = try allocator.alloc(F, N);
            errdefer allocator.free(prob);

            // TODO: pdv on the fact that we're taking up
            // a sentinel value
            const U: usize = std.math.maxInt(usize);
            var less_head = U;
            var more_head = U;

            // compute adjusted probability scalar, and when .pre_scaled
            // is comptime-known this should get inlined into no-ops with
            // the multiply by a constant 1 later in this function
            var scalar: F = 1;
            if (!kwargs.pre_scaled) {
                if (!kwargs.pre_normalized)
                    scalar /= kahan(F, weights[0..]);
                scalar *= @intToFloat(F, weights.len);
            }

            // initialize small/large buffers -- in-place in the result
            // buffer, abusing the `alias` as a backward-pointing linked
            // list, to be overwritten with the real alias later
            for (weights, 0..) |w, i| {
                const p = w * scalar;
                prob[i] = p;
                var ptr = if (p < 1) &less_head else &more_head;
                alias[i] = ptr.*;
                ptr.* = i;
            }

            // main work loop, walk linked lists and fill alias
            // table as you empty where they list used to be
            while (less_head != U and more_head != U) {
                const l = less_head;
                const g = more_head;
                less_head = alias[less_head];
                more_head = alias[more_head];
                alias[l] = g;
                prob[g] = (prob[g] + prob[l]) - 1;
                var ptr = if (prob[g] < 1) &less_head else &more_head;
                alias[g] = ptr.*;
                ptr.* = g;
            }

            // stragglers from the main work loop
            while (more_head != U) {
                defer more_head = alias[more_head];
                prob[more_head] = 1;
            }

            // more stragglers, only due to floating point
            // error
            while (less_head != U) {
                defer less_head = alias[less_head];
                prob[less_head] = 1;
            }

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
