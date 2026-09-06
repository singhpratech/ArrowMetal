import Foundation
import CArrowABI
import ArrowMetal

// C ABI for the temporal functions beyond `am_temporal_extract` and `am_temporal_math`: the
// option-carrying week numbers, the struct-valued extractors, `subsecond`, `is_dst` and every
// `*_between` difference. One entry point with an op table, following ArrowMetalC_Text.swift; the
// numbering below is the table documented in include/arrowmetal.h.

/// Op codes accepted by `am_temporal_extra`.
enum TemporalExtraOp: Int32 {
    case week = 0, usWeek = 1, usYear = 2, isoCalendar = 3, yearMonthDay = 4
    case isDST = 5, dayOfWeek = 6, subsecond = 7
    case yearsBetween = 8, quartersBetween = 9, monthsBetween = 10, weeksBetween = 11
    case hoursBetween = 12, minutesBetween = 13, secondsBetween = 14
    case millisecondsBetween = 15, microsecondsBetween = 16, nanosecondsBetween = 17
}

private let errorKeyExtra = "ArrowMetalC.lastError"
private func setExtraError(_ e: Error) { Thread.current.threadDictionary[errorKeyExtra] = "\(e)" }

@inline(__always) private func extraTemporal(_ p: OpaquePointer?) throws -> MetalTemporalArray {
    guard let p else { throw ArrowMetalError.invalidArrowArray("null array handle") }
    let a = Unmanaged<Box>.fromOpaque(UnsafeRawPointer(p)).takeUnretainedValue().a
    guard case .temporal(let t) = a else {
        throw ArrowMetalError.unsupportedType("expected a temporal array, got \(a.arrowFormat)")
    }
    return t
}

private func runExtra(_ out: UnsafeMutablePointer<OpaquePointer?>?, _ body: () throws -> AnyMetalArray) -> Int32 {
    do {
        out?.pointee = OpaquePointer(Unmanaged.passRetained(Box(try body())).toOpaque())
        return 0
    } catch { setExtraError(error); return 1 }
}

/// The remaining Arrow temporal functions, behind one entry point.
///
/// `p1` and `p2` carry the options an op takes (see the table in `arrowmetal.h`); `b` is the second
/// column for the `*_between` ops and is ignored otherwise. Ops 3 and 4 return a struct array whose
/// fields are read with `am_struct_field`.
@_cdecl("am_temporal_extra")
public func am_temporal_extra(_ a: OpaquePointer?, _ op: Int32, _ p1: Int64, _ p2: Int64,
                              _ b: OpaquePointer?,
                              _ out: UnsafeMutablePointer<OpaquePointer?>?) -> Int32 {
    guard a != nil, out != nil else { return 2 }
    return runExtra(out) {
        let t = try extraTemporal(a)
        guard let kind = TemporalExtraOp(rawValue: op) else {
            throw ArrowMetalError.invalidArrowArray("unknown temporal extra op \(op)")
        }
        func other() throws -> MetalTemporalArray {
            guard b != nil else {
                throw ArrowMetalError.invalidArrowArray("op \(op) needs a second temporal column")
            }
            return try extraTemporal(b)
        }
        switch kind {
        case .week:
            return .int64(try t.week(weekStartsMonday: (p1 & 1) != 0,
                                     countFromZero: (p1 & 2) != 0,
                                     firstWeekIsFullyInYear: (p1 & 4) != 0))
        case .usWeek: return .int64(try t.usWeek())
        case .usYear: return .int64(try t.usYear())
        case .isoCalendar: return .structure(try t.isoCalendar())
        case .yearMonthDay: return .structure(try t.yearMonthDay())
        case .isDST: return .boolean(try t.isDST())
        case .dayOfWeek:
            return .int64(try t.dayOfWeek(countFromZero: p1 != 0, weekStart: p2 == 0 ? 1 : Int(p2)))
        case .subsecond: return .float64(try t.subsecond())
        case .yearsBetween: return .int64(try t.yearsBetween(try other()))
        case .quartersBetween: return .int64(try t.quartersBetween(try other()))
        case .monthsBetween: return .int64(try t.monthsBetween(try other()))
        case .weeksBetween:
            return .int64(try t.weeksBetween(try other(), countFromZero: p1 != 0,
                                             weekStart: p2 == 0 ? 1 : Int(p2)))
        case .hoursBetween: return .int64(try t.hoursBetween(try other()))
        case .minutesBetween: return .int64(try t.minutesBetween(try other()))
        case .secondsBetween: return .int64(try t.secondsBetween(try other()))
        case .millisecondsBetween: return .int64(try t.millisecondsBetween(try other()))
        case .microsecondsBetween: return .int64(try t.microsecondsBetween(try other()))
        case .nanosecondsBetween: return .int64(try t.nanosecondsBetween(try other()))
        }
    }
}
