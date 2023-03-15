const std = @import("std");
const Allocator = std.mem.Allocator;
const Random = std.rand.Random;
const RndGen = std.rand.DefaultPrng;

const zkwargs = @import("zkwargs");
const pdv = @import("pdv");

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
    pub fn pre_normalized(comptime _: ?type) type {
        return zkwargs.Default(false);
    }

    pub fn pre_scaled(comptime _: ?type) type {
        return zkwargs.Default(false);
    }
};

fn PackedEntry(comptime U: type, comptime F: type) type {
    const pair_bytes = @max(@sizeOf(U), @alignOf(U)) + @max(@sizeOf(F), @alignOf(F));
    const max_in_cache_line = 64 / pair_bytes;
    return struct {
        prob: [max_in_cache_line]F,
        alias: [max_in_cache_line]U,
    };
}

fn Alias(comptime F: type, comptime I: type) type {
    // Great resource for Vose's Alias Method:
    // https://www.keithschwarz.com/darts-dice-coins/

    const PE = PackedEntry(I, F);
    const NLine: usize = @typeInfo(@TypeOf(@as(PE, undefined).alias)).Array.len;

    const NonEmpty = struct {};
    const NonNegative = struct {};
    const SomePositive = struct {};
    const FitsInIndex = struct {};

    const ValidWeights = pdv.Constraint([]F, .{ NonEmpty, NonNegative, SomePositive, FitsInIndex });

    return struct {
        entries: []PE,
        allocator: Allocator,
        n: usize,

        // TODO: kind of a weird pair of methods to be public
        pub inline fn ith_alias(self: *@This(), i: usize) *I {
            return &self.entries[i / NLine].alias[i % NLine];
        }

        pub inline fn ith_prob(self: *@This(), i: usize) *F {
            return &self.entries[i / NLine].prob[i % NLine];
        }

        pub fn pinky_promise_weights_are_valid(weights: []F) ValidWeights {
            return ValidWeights{ .val = weights };
        }

        pub fn validate_weights(weights: []F) !ValidWeights {
            if (weights.len == 0)
                return error.Empty;

            var some_positive = false;
            for (weights) |w| {
                if (w < 0)
                    return error.Negative;
                some_positive = some_positive or w > 0;
            }

            if (!some_positive)
                return error.NoPositive;

            if (weights.len - 1 > @intCast(usize, std.math.maxInt(I)))
                return error.TooBigForIndexType;

            return ValidWeights{ .val = weights };
        }

        pub fn init(allocator: Allocator, _weights: ValidWeights, _kwargs: anytype) !@This() {
            // _weights:
            //   - w[i] == 0 implies i will not be selected
            //   - w[i] == w[k] * z implies i is z times more likely to be selected than k
            //   - at least one entry must be positive
            //   - no entries may be negative
            //
            // _kwargs:
            // ========
            // .pre_normalized (false):
            //   - (only used if !pre_scaled) do the weights sum to 1.0, or do we need to
            //     estimate the sum?
            // .pre_scaled (false):
            //   - are the (normalized) weights already multiplied by weights.len?
            var weights = _weights.extract(.{ NonEmpty, NonNegative, SomePositive, FitsInIndex });
            var kwargs = zkwargs.Options(AliasOpt).parse(_kwargs);

            var n_packed_entries = weights.len / NLine + @boolToInt(weights.len % NLine != 0);

            var entries = try allocator.alloc(PE, n_packed_entries);
            errdefer allocator.free(entries);

            var less_head: I = 0;
            var more_head: I = 0;
            var n_small: usize = 0;
            var n_large: usize = 0;

            // compute adjusted probability scalar, and when .pre_scaled
            // is comptime-known this should get inlined into no-ops with
            // the multiply by a constant 1 later in this function
            var scalar: F = 1;
            if (!kwargs.pre_scaled) {
                if (!kwargs.pre_normalized)
                    scalar /= kahan(F, weights[0..]);
                scalar *= @intToFloat(F, weights.len);
            }

            var rtn = @This(){
                .entries = entries,
                .allocator = allocator,
                .n = weights.len,
            };

            // initialize small/large buffers -- in-place in the result
            // buffer, abusing the `alias` as a backward-pointing linked
            // list, to be overwritten with the real alias later
            for (weights, 0..) |w, i| {
                const p = w * scalar;
                rtn.ith_prob(i).* = p;
                if (p < 1) {
                    rtn.ith_alias(i).* = less_head;
                    less_head = @intCast(I, i);
                    n_small += 1;
                } else {
                    rtn.ith_alias(i).* = more_head;
                    more_head = @intCast(I, i);
                    n_large += 1;
                }
            }

            // main work loop, walk linked lists and fill alias
            // table as you empty where they list used to be
            while (n_small > 0 and n_large > 0) {
                const l = less_head;
                const g = more_head;
                less_head = rtn.ith_alias(less_head).*;
                more_head = rtn.ith_alias(more_head).*;
                rtn.ith_alias(l).* = g;
                rtn.ith_prob(g).* = (rtn.ith_prob(g).* + rtn.ith_prob(l).*) - 1;
                if (rtn.ith_prob(g).* < 1) {
                    rtn.ith_alias(g).* = less_head;
                    less_head = g;
                    n_large -= 1;
                } else {
                    rtn.ith_alias(g).* = more_head;
                    more_head = g;
                    n_small -= 1;
                }
            }

            // stragglers from the main work loop
            while (n_large > 0) : (n_large -= 1) {
                defer more_head = rtn.ith_alias(more_head).*;
                rtn.ith_prob(more_head).* = 1;
            }

            // more stragglers, only due to floating point
            // error
            while (n_small > 0) : (n_small -= 1) {
                defer less_head = rtn.ith_alias(less_head).*;
                rtn.ith_prob(less_head).* = 1;
            }

            return rtn;
        }

        pub fn deinit(self: *@This()) void {
            self.allocator.free(self.entries);
        }

        pub fn generate(self: *@This(), rand: Random) usize {
            const i = rand.uintLessThan(usize, self.n);
            const f = rand.float(F);
            return if (f <= self.ith_prob(i).*) i else self.ith_alias(i).*;
        }
    };
}

test "works normalized" {
    var rnd = RndGen.init(0);
    const allocator = std.testing.allocator;
    const F = f32;
    var probs = [_]F{ 0.1, 0.2, 0.3, 0.4 };
    var validated_probs = try Alias(F, usize).validate_weights(probs[0..]);
    var table = try Alias(F, usize).init(allocator, validated_probs, .{ .pre_normalized = true });
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
    var validated_probs = try Alias(F, usize).validate_weights(probs[0..]);
    var table = try Alias(F, usize).init(allocator, validated_probs, .{ .pre_scaled = true });
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
    var validated_probs = Alias(F, usize).pinky_promise_weights_are_valid(probs[0..]);
    var table = try Alias(F, usize).init(allocator, validated_probs, .{});
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

test "other index types" {
    var rnd = RndGen.init(0);
    const allocator = std.testing.allocator;
    const F = f32;
    var probs = [_]F{ 0.4, 0.8, 1.2, 1.6 };
    var validated_probs = try Alias(F, u2).validate_weights(probs[0..]);
    var table = try Alias(F, u2).init(allocator, validated_probs, .{});
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
