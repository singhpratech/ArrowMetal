import XCTest
@testable import ArrowMetal

/// `partition_nth_indices` with `nullPlacement: .atStart` over a float column containing NaN.
///
/// Arrow's `PartitionNthToIndices` runs `PartitionNulls`, which moves **nulls and NaNs together** to
/// `null_placement`'s end and takes the nth over what is left; `MetalArray.argsort` does the same
/// ("NaN travels with the nulls, not with the values", `Kernels/Sort.swift`), and `PartitionNth`'s own
/// doc says its `nullPlacement` works "exactly as in `argsort`". Only `.atEnd` agreed: there the
/// ascending keys already leave the NaNs at the tail of the value block, just before the nulls.
///
/// The expectations below were read off pyarrow 25.0.1 `pc.partition_nth_indices`.
final class PartitionNthNaNTests: XCTestCase {

    /// The partition contract: position `pivot` holds the row a sorted order would put there, nothing
    /// before it is greater and nothing after it is smaller. Checked against the ordering
    /// `argsort(nullPlacement:)` itself produces, which is the one this entry point documents.
    private func assertPartitionAgreesWithArgsort<T: ArrowPrimitive & BinaryFloatingPoint>(
        _ vals: [T?], pivot: Int, placement: NullPlacement,
        file: StaticString = #filePath, line: UInt = #line) throws {
        let a = try MetalArray<T>(vals, context: .shared)
        let sortedIdx = try a.argsort(nullPlacement: placement).toRawArray()
        let part = try a.partitionNthIndices(pivot, nullPlacement: placement).toRawArray()
        XCTAssertEqual(Set(part), Set(sortedIdx), "not a permutation", file: file, line: line)
        guard pivot < vals.count else { return }
        // Rank of each row in the argsort order; ties (equal values, all the NaNs, all the nulls) share
        // the rank of the first of their run, so the comparison below is well defined.
        var rankOf = [Int](repeating: 0, count: vals.count)
        for (r, row) in sortedIdx.enumerated() { rankOf[Int(row)] = r }
        let pivotRank = rankOf[Int(sortedIdx[pivot])]
        XCTAssertEqual(rankOf[Int(part[pivot])], pivotRank,
                       "position \(pivot) holds row \(part[pivot]), but the sorted order puts row \(sortedIdx[pivot]) there",
                       file: file, line: line)
    }

    func testFloat32NaNTravelsWithTheNullsAtStart() throws {
        try requireRealGPU()
        let vals: [Float?] = [1.0, nil, 3.0, Float.nan, 2.0]
        for pivot in 0..<vals.count {
            try assertPartitionAgreesWithArgsort(vals, pivot: pivot, placement: .atStart)
            try assertPartitionAgreesWithArgsort(vals, pivot: pivot, placement: .atEnd)
        }
    }

    /// pyarrow, verbatim: `pc.partition_nth_indices(pa.array([1.0, 2.0, 3.0, nan]), pivot=1,
    /// null_placement="at_start")` puts the NaN first, so position 1 holds the smallest value.
    func testNoNullsJustNaNAtStart() throws {
        try requireRealGPU()
        let a = try MetalArray<Float>([1.0, 2.0, 3.0, Float.nan], context: .shared)
        let p0 = try a.partitionNthIndices(0, nullPlacement: .atStart).toRawArray()
        XCTAssertEqual(p0[0], 3, "with .atStart the NaN sorts first, so position 0 is the NaN row")
        let p1 = try a.partitionNthIndices(1, nullPlacement: .atStart).toRawArray()
        XCTAssertEqual(p1[1], 0, "position 1 is then the smallest value")
    }

    func testFloat64AndAllNaNColumn() throws {
        try requireRealGPU()
        let vals: [Double?] = [Double.nan, nil, Double.nan, 5.0, Double.nan]
        for pivot in 0..<vals.count {
            try assertPartitionAgreesWithArgsort(vals, pivot: pivot, placement: .atStart)
            try assertPartitionAgreesWithArgsort(vals, pivot: pivot, placement: .atEnd)
        }
        let allNaN: [Double?] = [Double.nan, Double.nan, Double.nan]
        for pivot in 0..<3 {
            try assertPartitionAgreesWithArgsort(allNaN, pivot: pivot, placement: .atStart)
        }
    }

    /// The integer and no-NaN paths must be untouched.
    func testNonFloatAndNaNFreeColumnsAreUnchanged() throws {
        try requireRealGPU()
        let ints: [Int32?] = [5, nil, 1, 4, nil, 3]
        let a = try MetalArray<Int32>(ints, context: .shared)
        for pivot in 0..<ints.count {
            for placement in [NullPlacement.atEnd, .atStart] {
                let sortedIdx = try a.argsort(nullPlacement: placement).toRawArray()
                let part = try a.partitionNthIndices(pivot, nullPlacement: placement).toRawArray()
                XCTAssertEqual(Set(part), Set(sortedIdx))
                if pivot < ints.count {
                    XCTAssertEqual(ints[Int(part[pivot])], ints[Int(sortedIdx[pivot])],
                                   "int32 pivot \(pivot) \(placement)")
                }
            }
        }
        let floats: [Float?] = [5.0, nil, 1.0, 4.0, 3.0]
        for pivot in 0..<floats.count {
            try assertPartitionAgreesWithArgsort(floats, pivot: pivot, placement: .atStart)
        }
    }
}
