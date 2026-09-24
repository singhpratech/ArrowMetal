import XCTest
@testable import ArrowMetal

/// Regressions for the type an **untyped literal** takes in a fused expression.
///
/// The compiler used to hand the literal the other operand's type unconditionally, so a literal that
/// did not fit was truncated to that width and the kernel silently answered a different question:
/// `int8_column > 200` compared against `(char)200 == -56` and was true for every row, and
/// `int64_column >= 2.5` compared against `2`. Rule 1 of `docs/EXPR.md` ("float64 with anything
/// numeric → float64") did not hold for a float literal either.
final class ExprLiteralTypeTests: XCTestCase {

    func batch(_ pairs: [(String, AnyMetalArray)]) throws -> MetalRecordBatch {
        try MetalRecordBatch(names: pairs.map(\.0), columns: pairs.map(\.1))
    }

    func booleans(_ a: AnyMetalArray?) -> [Bool?] {
        guard case .boolean(let b)? = a else { XCTFail("expected a boolean column"); return [] }
        return b.toArray()
    }

    func doubles(_ a: AnyMetalArray?) -> [Double?] {
        guard case .float64(let d)? = a else {
            XCTFail("expected a float64 column, got \(a?.arrowFormat ?? "nil")"); return []
        }
        return d.toArray()
    }

    // MARK: - an integer literal wider than the column

    func testInt8ColumnComparedWithAnOutOfRangeLiteral() throws {
        try requireRealGPU()
        let a = try MetalArray<Int8>([-56, 0, 1, 100, 127])
        let rb = try batch([("a", .int8(a))])
        let r = try rb.query(query().project([
            ("gt", col("a") > 200), ("eq", col("a") == 1000), ("lt", col("a") < -900),
        ]))
        // Every value of an int8 column is below 200, none is 1000, none is below -900.
        XCTAssertEqual(booleans(r["gt"]), [false, false, false, false, false])
        XCTAssertEqual(booleans(r["eq"]), [false, false, false, false, false])
        XCTAssertEqual(booleans(r["lt"]), [false, false, false, false, false])
    }

    func testUnsignedColumnComparedWithANegativeLiteral() throws {
        try requireRealGPU()
        let a = try MetalArray<UInt8>([0, 1, 254, 255])
        let rb = try batch([("a", .uint8(a))])
        let r = try rb.query(query().project([("eq", col("a") == -1), ("big", col("a") == 256)]))
        XCTAssertEqual(booleans(r["eq"]), [false, false, false, false])
        XCTAssertEqual(booleans(r["big"]), [false, false, false, false])
    }

    func testInt8ColumnPlusAnOutOfRangeLiteralWidens() throws {
        try requireRealGPU()
        let a = try MetalArray<Int8>([1, 2, 127])
        let rb = try batch([("a", .int8(a))])
        let r = try rb.query(query().project([("sum", col("a") + 1000)]))
        guard case .int16(let out)? = r["sum"] else {
            XCTFail("int8 + 1000 should widen to int16, got \(r["sum"]?.arrowFormat ?? "nil")"); return
        }
        XCTAssertEqual(out.toArray(), [1001, 1002, 1127])
    }

    /// A literal that does fit must *not* widen: `uint64_column == 5` stays in uint64.
    func testLiteralThatFitsKeepsTheColumnType() throws {
        try requireRealGPU()
        let a = try MetalArray<UInt64>([5, UInt64(Int64.max) + 9, 0])
        let rb = try batch([("a", .uint64(a))])
        let r = try rb.query(query().project([("eq", col("a") == 5), ("sum", col("a") + 1)]))
        XCTAssertEqual(booleans(r["eq"]), [true, false, false])
        guard case .uint64(let out)? = r["sum"] else {
            XCTFail("uint64 + 1 should stay uint64, got \(r["sum"]?.arrowFormat ?? "nil")"); return
        }
        XCTAssertEqual(out.toArray(), [6, UInt64(Int64.max) + 10, 1])
    }

    func testIsInWidensAnOutOfRangeValue() throws {
        try requireRealGPU()
        let a = try MetalArray<Int8>([-56, 1, 100])
        let rb = try batch([("a", .int8(a))])
        let r = try rb.query(query().project([("in", col("a").isIn([200, 1] as [Int64]))]))
        XCTAssertEqual(booleans(r["in"]), [false, true, false])
    }

    // MARK: - a floating literal against an integer column

    func testIntegerColumnComparedWithAFloatLiteral() throws {
        try requireRealGPU()
        let a = try MetalArray<Int64>([1, 2, 3])
        let rb = try batch([("a", .int64(a))])
        let r = try rb.query(query().project([
            ("ge", col("a") >= 2.5), ("eq", col("a") == 2.5), ("lt", col("a") < 2.5),
        ]))
        XCTAssertEqual(booleans(r["ge"]), [false, false, true])
        XCTAssertEqual(booleans(r["eq"]), [false, false, false])
        XCTAssertEqual(booleans(r["lt"]), [true, true, false])
    }

    func testIntegerColumnPlusAFloatLiteralPromotesToFloat64() throws {
        try requireRealGPU()
        let a = try MetalArray<Int64>([1, 2])
        let rb = try batch([("a", .int64(a))])
        let r = try rb.query(query().project([("sum", col("a") + 0.5)]))
        XCTAssertEqual(doubles(r["sum"]), [1.5, 2.5])
    }

    func testFillNullAndIfElseWithAFloatLiteral() throws {
        try requireRealGPU()
        let a = try MetalArray<Int64>([1, nil, 3])
        let rb = try batch([("a", .int64(a))])
        let r = try rb.query(query().project([
            ("fill", col("a").fillNull(.double(0.5))),
            ("pick", .ifElse(col("a") > 1, col("a"), .double(-1.5))),
        ]))
        XCTAssertEqual(doubles(r["fill"]), [1.0, 0.5, 3.0])
        XCTAssertEqual(doubles(r["pick"]), [-1.5, nil, 3.0])
    }

    /// A float32 column with an integer literal must stay float32 — the fix must not widen that.
    func testFloat32ColumnWithAnIntegerLiteralStaysFloat32() throws {
        try requireRealGPU()
        let a = try MetalArray<Float>([1.5, 2.5])
        let rb = try batch([("a", .float32(a))])
        let r = try rb.query(query().project([("sum", col("a") + 100)]))
        guard case .float32(let out)? = r["sum"] else {
            XCTFail("float32 + 100 should stay float32, got \(r["sum"]?.arrowFormat ?? "nil")"); return
        }
        XCTAssertEqual(out.toArray(), [101.5, 102.5])
    }

    // MARK: - a float literal beyond the 64-bit integer range

    /// `Int64(Double)` traps outside [-2^63, 2^63), and the compiler used to compute that integer for
    /// every float literal, even when the target type was a float. So `int64_column >= 1e19`,
    /// `x * 4.49e307` or a float32 literal of 1e30 ended the host process (found 2026-09-24 by the
    /// review of the Polars engine lane). These must compile and answer.
    func testFloatLiteralsBeyondTheInt64RangeDoNotTrap() throws {
        try requireRealGPU()
        let i = try MetalArray<Int64>([Int64.min, -1, 0, Int64.max])
        let d = try MetalArray<Double>([1.0, -2.0, 1.0, -2.0])
        let f = try MetalArray<Float>([1.5, 3e38, 1.5, 3e38])
        let rb = try batch([("i", .int64(i)), ("d", .float64(d)), ("f", .float32(f))])
        let r = try rb.query(query().project([
            ("ge", col("i") >= 1e19), ("gt", col("i") > -1e19),
            ("big", col("d") * .typedDouble(4.49423283715579e+307, .float64)),
            ("f32", col("f") > .typedDouble(1e30, .float32)),
        ]))
        // Every int64 is below 1e19 and above -1e19.
        XCTAssertEqual(booleans(r["ge"]), [false, false, false, false])
        XCTAssertEqual(booleans(r["gt"]), [true, true, true, true])
        XCTAssertEqual(doubles(r["big"]), [4.49423283715579e+307, -8.98846567431158e+307, 4.49423283715579e+307, -8.98846567431158e+307])
        XCTAssertEqual(booleans(r["f32"]), [false, true, false, true])
    }

    /// A float literal typed as an integer truncates toward zero like an Arrow cast; one that does not
    /// fit 64 bits is a clear error, not a trap.
    func testAFloatLiteralTypedAsAnIntegerOutsideTheRangeIsAnError() throws {
        try requireRealGPU()
        let i = try MetalArray<Int64>([1, 2, 3])
        let rb = try batch([("i", .int64(i))])
        let ok = try rb.query(query().project([("ge", col("i") >= .typedDouble(2.9, .int64))]))
        XCTAssertEqual(booleans(ok["ge"]), [false, true, true])
        XCTAssertThrowsError(try rb.query(query().project([("ge", col("i") >= .typedDouble(1e19, .int64))]))) { e in
            XCTAssertTrue("\(e)".contains("does not fit a 64-bit integer"), "\(e)")
        }
    }
}
