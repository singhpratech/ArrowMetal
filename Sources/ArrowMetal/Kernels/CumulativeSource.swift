import Foundation

/// MSL for Arrow's cumulative functions as a two-level GPU scan.
///
/// The shape is the one `StringSource`'s offset scan uses, generalised from "add" to any associative
/// combine so that `cumulative_min` and `cumulative_max` share it, and made *inclusive* rather than
/// exclusive:
///
///   1. `cum_block_*` — one threadgroup per 256 elements. Each thread loads its value (or the op's
///      neutral element when the slot is null or past the end), the threadgroup runs a Hillis-Steele
///      inclusive scan in threadgroup memory, every element writes its block-local running value, and
///      the last thread writes the block total.
///   2. `cum_totals_*` — a single threadgroup turns the block totals into *exclusive* prefixes, each
///      thread folding a strided slice first so the pass covers any number of blocks.
///   3. `cum_add_*` — folds each block's exclusive prefix back into every element of that block.
///
/// **Nulls.** A null slot contributes the neutral element and stays null in the output; the running
/// value carries straight through it. That is Arrow's `skip_nulls` behaviour for the cumulative
/// functions: output null where input null, the accumulation unbroken.
///
/// **Float reassociation.** The block scan reassociates additions, so a `float32` or `float64`
/// `cumulative_sum` need not be bit-identical to a strictly sequential one. Integers wrap and are
/// associative, so integer results are exact; so are min and max on every type.
enum CumulativeSource {
    /// `body` is an expression over `a` and `b` returning the value type; `identity` is a literal of it.
    typealias Op = (name: String, identity: String, body: String)

    static func source(V: String, extraPrelude: String, ops: [Op]) -> String {
        var s = KernelSource.prelude + extraPrelude + "\n"
        for op in ops {
            let ident = op.identity
            s += """
            inline \(V) cum_\(op.name)(\(V) a, \(V) b) { return \(op.body); }
            kernel void cum_block_\(op.name)(device const \(V)* vals [[buffer(0)]],
                                             device const uchar* validity [[buffer(1)]],
                                             device const uint* nPtr [[buffer(2)]],
                                             constant uint& hasValidity [[buffer(3)]],
                                             device \(V)* out [[buffer(4)]],
                                             device \(V)* blockTotals [[buffer(5)]],
                                             uint i [[thread_position_in_grid]],
                                             uint lid [[thread_index_in_threadgroup]],
                                             uint tgid [[threadgroup_position_in_grid]]) {
                threadgroup \(V) sdata[TG];
                uint n = *nPtr;
                \(V) acc = \(ident);
                if (i < n && (hasValidity == 0u || bit_get(validity, i))) acc = vals[i];
                sdata[lid] = acc;
                threadgroup_barrier(mem_flags::mem_threadgroup);
                // Every thread runs every step: the barriers must be reached uniformly.
                for (uint off = 1u; off < TG; off <<= 1) {
                    \(V) t = (lid >= off) ? sdata[lid - off] : (\(V))(\(ident));
                    threadgroup_barrier(mem_flags::mem_threadgroup);
                    acc = cum_\(op.name)(t, acc);
                    sdata[lid] = acc;
                    threadgroup_barrier(mem_flags::mem_threadgroup);
                }
                if (i < n) out[i] = acc;
                if (lid == TG - 1u) blockTotals[tgid] = acc;
            }
            // Exclusive scan of the block totals, in place, in one threadgroup.
            kernel void cum_totals_\(op.name)(device \(V)* totals [[buffer(0)]],
                                              constant uint& blocks [[buffer(1)]],
                                              uint lid [[thread_index_in_threadgroup]]) {
                threadgroup \(V) sdata[TG];
                uint per = (blocks + TG - 1u) / TG;
                uint lo = lid * per, hi = min(blocks, lo + per);
                \(V) acc = \(ident);
                for (uint b = lo; b < hi; b++) acc = cum_\(op.name)(acc, totals[b]);
                sdata[lid] = acc;
                threadgroup_barrier(mem_flags::mem_threadgroup);
                for (uint off = 1u; off < TG; off <<= 1) {
                    \(V) t = (lid >= off) ? sdata[lid - off] : (\(V))(\(ident));
                    threadgroup_barrier(mem_flags::mem_threadgroup);
                    acc = cum_\(op.name)(t, acc);
                    sdata[lid] = acc;
                    threadgroup_barrier(mem_flags::mem_threadgroup);
                }
                \(V) run = (lid == 0u) ? (\(V))(\(ident)) : sdata[lid - 1u];
                for (uint b = lo; b < hi; b++) { \(V) c = totals[b]; totals[b] = run; run = cum_\(op.name)(run, c); }
            }
            kernel void cum_add_\(op.name)(device \(V)* out [[buffer(0)]],
                                           device const \(V)* blockTotals [[buffer(1)]],
                                           device const uint* nPtr [[buffer(2)]],
                                           uint i [[thread_position_in_grid]],
                                           uint tgid [[threadgroup_position_in_grid]]) {
                if (i < *nPtr) out[i] = cum_\(op.name)(blockTotals[tgid], out[i]);
            }

            """
        }
        return s
    }
}
