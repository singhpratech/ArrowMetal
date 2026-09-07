import XCTest
import CArrowABI
@testable import ArrowMetal

// Nested types: list, large_list, fixed_size_list, struct, map and dense/sparse union.
//
// The interop tests build the same C Data Interface structs pyarrow exports — a parent with a validity
// and an offsets buffer plus a moved-in child, each node owning its own memory and releasing it through
// its own callback — so the importer is exercised against a foreign-shaped producer rather than against
// ArrowMetal's own exports. The kernel tests build arrays in Swift and check every GPU result against a
// plain Swift oracle at 0, 1, 33, 4097 and 100_003 rows, with nulls and empty lists throughout.

// MARK: - Hand-built C Data Interface nodes

/// One node of a hand-built array tree. Owns its buffers and its children's structs; the `release`
/// callback below drops exactly one node, so a child that the importer moves out outlives its parent.
final class CNode {
    var blocks: [UnsafeMutableRawPointer] = []
    let bufferPtrs: UnsafeMutablePointer<UnsafeRawPointer?>
    let childPtrs: UnsafeMutablePointer<UnsafeMutablePointer<ArrowArray>?>
    let childStructs: UnsafeMutablePointer<ArrowArray>
    let nChildren: Int

    init(nBuffers: Int, nChildren: Int) {
        bufferPtrs = .allocate(capacity: max(nBuffers, 1))
        for i in 0..<max(nBuffers, 1) { bufferPtrs[i] = nil }
        self.nChildren = nChildren
        childPtrs = .allocate(capacity: max(nChildren, 1))
        childStructs = .allocate(capacity: max(nChildren, 1))
        childStructs.initialize(repeating: ArrowArray(), count: max(nChildren, 1))
        for i in 0..<nChildren { childPtrs[i] = childStructs + i }
    }

    /// A page-aligned copy of `bytes`, so the importer can take its zero-copy path.
    func alloc(_ bytes: [UInt8]) -> UnsafeRawPointer {
        let page = Int(getpagesize())
        let padded = max((max(bytes.count, 1) + page - 1) / page * page, page)
        var raw: UnsafeMutableRawPointer? = nil
        precondition(posix_memalign(&raw, page, padded) == 0)
        memset(raw!, 0, padded)
        bytes.withUnsafeBytes { if $0.count > 0 { memcpy(raw!, $0.baseAddress!, $0.count) } }
        blocks.append(raw!)
        return UnsafeRawPointer(raw!)
    }

    deinit {
        blocks.forEach { free($0) }
        bufferPtrs.deallocate(); childPtrs.deallocate(); childStructs.deallocate()
    }
}

func releaseTestArray(_ p: UnsafeMutablePointer<ArrowArray>?) {
    guard let p, let pd = p.pointee.private_data else { return }
    let node = Unmanaged<CNode>.fromOpaque(pd).takeUnretainedValue()
    for i in 0..<node.nChildren { if let r = node.childStructs[i].release { r(node.childStructs + i) } }
    Unmanaged<CNode>.fromOpaque(pd).release()
    p.pointee.release = nil
    p.pointee.private_data = nil
}

final class CSchemaNode {
    let formatC: UnsafeMutablePointer<CChar>
    let nameC: UnsafeMutablePointer<CChar>
    let childPtrs: UnsafeMutablePointer<UnsafeMutablePointer<ArrowSchema>?>
    let childStructs: UnsafeMutablePointer<ArrowSchema>
    let nChildren: Int

    init(format: String, name: String, nChildren: Int) {
        formatC = strdup(format)!
        nameC = strdup(name)!
        self.nChildren = nChildren
        childPtrs = .allocate(capacity: max(nChildren, 1))
        childStructs = .allocate(capacity: max(nChildren, 1))
        childStructs.initialize(repeating: ArrowSchema(), count: max(nChildren, 1))
        for i in 0..<nChildren { childPtrs[i] = childStructs + i }
    }
    deinit {
        free(formatC); free(nameC)
        childPtrs.deallocate(); childStructs.deallocate()
    }
}

func releaseTestSchema(_ p: UnsafeMutablePointer<ArrowSchema>?) {
    guard let p, let pd = p.pointee.private_data else { return }
    let node = Unmanaged<CSchemaNode>.fromOpaque(pd).takeUnretainedValue()
    for i in 0..<node.nChildren { if let r = node.childStructs[i].release { r(node.childStructs + i) } }
    Unmanaged<CSchemaNode>.fromOpaque(pd).release()
    p.pointee.release = nil
    p.pointee.private_data = nil
}

func cSchema(_ format: String, name: String = "", flags: Int64 = Int64(ARROW_FLAG_NULLABLE),
             children: [ArrowSchema] = []) -> ArrowSchema {
    let node = CSchemaNode(format: format, name: name, nChildren: children.count)
    for (i, c) in children.enumerated() { node.childStructs[i] = c }
    var s = ArrowSchema()
    s.format = UnsafePointer(node.formatC)
    s.name = UnsafePointer(node.nameC)
    s.metadata = nil
    s.flags = flags
    s.n_children = Int64(children.count)
    s.children = children.isEmpty ? nil : node.childPtrs
    s.dictionary = nil
    s.release = releaseTestSchema
    s.private_data = Unmanaged.passRetained(node).toOpaque()
    return s
}

func cArray(length: Int, nullCount: Int = -1, offset: Int = 0,
            buffers: [[UInt8]?], children: [ArrowArray] = []) -> ArrowArray {
    let node = CNode(nBuffers: buffers.count, nChildren: children.count)
    for (i, b) in buffers.enumerated() { node.bufferPtrs[i] = b.map { node.alloc($0) } }
    for (i, c) in children.enumerated() { node.childStructs[i] = c }
    var a = ArrowArray()
    a.length = Int64(length); a.null_count = Int64(nullCount); a.offset = Int64(offset)
    a.n_buffers = Int64(buffers.count); a.n_children = Int64(children.count)
    a.buffers = node.bufferPtrs
    a.children = children.isEmpty ? nil : node.childPtrs
    a.dictionary = nil
    a.release = releaseTestArray
    a.private_data = Unmanaged.passRetained(node).toOpaque()
    return a
}

func rawBytes<T>(_ v: [T]) -> [UInt8] { v.withUnsafeBytes { Array($0) } }

func bitmapBytes(_ valid: [Bool]) -> [UInt8] {
    var b = [UInt8](repeating: 0, count: max((valid.count + 7) / 8, 1))
    for (i, v) in valid.enumerated() where v { b[i >> 3] |= UInt8(1 << (i & 7)) }
    return b
}

/// The three buffers of a utf8 array: validity (nil when nothing is null), int32 offsets and bytes.
func utf8Buffers(_ strings: [String?]) -> (validity: [UInt8]?, offsets: [UInt8], data: [UInt8]) {
    var offsets: [Int32] = [0]
    var data: [UInt8] = []
    for s in strings {
        if let s { data.append(contentsOf: Array(s.utf8)) }
        offsets.append(Int32(data.count))
    }
    let valid = strings.map { $0 != nil }
    return (valid.contains(false) ? bitmapBytes(valid) : nil, rawBytes(offsets), data)
}

func utf8CArray(_ strings: [String?]) -> ArrowArray {
    let b = utf8Buffers(strings)
    return cArray(length: strings.count, buffers: [b.validity, b.offsets, b.data])
}

/// A primitive child array from optionals.
func primitiveCArray<T>(_ values: [T?], zero: T) -> ArrowArray {
    let valid = values.map { $0 != nil }
    return cArray(length: values.count,
                  buffers: [valid.contains(false) ? bitmapBytes(valid) : nil, rawBytes(values.map { $0 ?? zero })])
}

// MARK: - Interop round trips

final class NestedTests: XCTestCase {

    /// Reads a list<int64> back as Swift values.
    func readLists(_ l: MetalListArray) throws -> [[Int64?]?] {
        let child = try XCTUnwrap(l.values.asInt64)
        let vals = child.toArray()
        return (0..<l.length).map { i in l.valueRange(i).map { r in r.map { vals[$0] } } }
    }

    // MARK: list<int64>

    func listInt64Input() -> (schema: ArrowSchema, array: ArrowArray) {
        let child = primitiveCArray([1, 2, 3, 4, 5, 6].map { Int64?($0) }, zero: 0)
        let offsets: [Int32] = [0, 3, 3, 3, 4, 6]
        let arr = cArray(length: 5, nullCount: 1,
                         buffers: [bitmapBytes([true, true, false, true, true]), rawBytes(offsets)],
                         children: [child])
        return (cSchema("+l", name: "l", children: [cSchema("l", name: "item")]), arr)
    }

    func testListInt64RoundTrip() throws {
        try requireRealGPU()
        var (schema, arr) = listInt64Input()
        let r = try importArrowArray(schema: &schema, array: &arr)
        XCTAssertNil(arr.release, "import must move the array")
        let list = try XCTUnwrap(r.array.asList)
        XCTAssertEqual(list.arrowFormat, "+l")
        XCTAssertEqual(list.length, 5)
        XCTAssertEqual(list.nullCount, 1)
        XCTAssertEqual(try readLists(list), [[1, 2, 3], [], nil, [4], [5, 6]])

        // Export as pyarrow-shaped C structs and import them back.
        var outSchema = ArrowSchema(); var outArray = ArrowArray()
        r.array.exportArrowSchema(name: "l", into: &outSchema)
        r.array.exportArrowArray(into: &outArray)
        XCTAssertEqual(String(cString: outSchema.format), "+l")
        XCTAssertEqual(outSchema.n_children, 1)
        XCTAssertEqual(String(cString: outSchema.children[0]!.pointee.format), "l")
        XCTAssertEqual(String(cString: outSchema.children[0]!.pointee.name), "item")
        XCTAssertEqual(outArray.n_buffers, 2)
        XCTAssertEqual(outArray.n_children, 1)
        XCTAssertEqual(outArray.length, 5)
        XCTAssertEqual(outArray.null_count, 1)
        let back = try importArrowArray(schema: &outSchema, array: &outArray)
        XCTAssertEqual(try readLists(try XCTUnwrap(back.array.asList)), [[1, 2, 3], [], nil, [4], [5, 6]])
        outSchema.release?(&outSchema)
        schema.release?(&schema)
    }

    func testLargeListNarrowsOffsets() throws {
        try requireRealGPU()
        let child = primitiveCArray([10, 20, 30, 40].map { Int64?($0) }, zero: 0)
        let offsets: [Int64] = [0, 2, 2, 4]
        var arr = cArray(length: 3, nullCount: 0, buffers: [nil, rawBytes(offsets)], children: [child])
        var schema = cSchema("+L", name: "l", children: [cSchema("l", name: "item")])
        let r = try importArrowArray(schema: &schema, array: &arr)
        let list = try XCTUnwrap(r.array.asList)
        XCTAssertEqual(list.kind, ArrowListKind.variable)
        XCTAssertEqual(try readLists(list), [[10, 20], [], [30, 40]])
        // large_list comes back out as list, exactly as large_utf8 comes back out as utf8.
        XCTAssertEqual(list.arrowFormat, "+l")
        schema.release?(&schema)
    }

    func testListOfListInt32RoundTrip() throws {
        try requireRealGPU()
        // [[[1,2],[3]], [], null, [[],[4,5,6]]]
        let leaf = primitiveCArray([1, 2, 3, 4, 5, 6].map { Int32?($0) }, zero: 0)
        let inner = cArray(length: 4, nullCount: 0, buffers: [nil, rawBytes([0, 2, 3, 3, 6] as [Int32])],
                           children: [leaf])
        var arr = cArray(length: 4, nullCount: 1,
                         buffers: [bitmapBytes([true, true, false, true]), rawBytes([0, 2, 2, 2, 4] as [Int32])],
                         children: [inner])
        var schema = cSchema("+l", name: "outer",
                             children: [cSchema("+l", name: "item", children: [cSchema("i", name: "item")])])
        let r = try importArrowArray(schema: &schema, array: &arr)
        let outer = try XCTUnwrap(r.array.asList)
        XCTAssertEqual(outer.length, 4)
        let innerList = try XCTUnwrap(outer.values.asList)
        XCTAssertEqual(innerList.length, 4)

        func read(_ l: MetalListArray) throws -> [[[Int32?]?]?] {
            let inner = try XCTUnwrap(l.values.asList)
            let leafVals = try XCTUnwrap(inner.values.asInt32).toArray()
            return (0..<l.length).map { i in
                l.valueRange(i).map { r in r.map { j in inner.valueRange(j).map { rr in rr.map { leafVals[$0] } } } }
            }
        }
        XCTAssertEqual(try read(outer), [[[1, 2], [3]], [], nil, [[], [4, 5, 6]]])

        // A take over the outer list must gather the inner list *and* its leaf, recursively.
        let taken = try outer.take(try MetalArray<Int32>([3, 0, 2]))
        XCTAssertEqual(try read(taken), [[[], [4, 5, 6]], [[1, 2], [3]], nil])

        var s2 = ArrowSchema(); var a2 = ArrowArray()
        r.array.exportArrowSchema(name: "outer", into: &s2)
        r.array.exportArrowArray(into: &a2)
        XCTAssertEqual(String(cString: s2.children[0]!.pointee.format), "+l")
        XCTAssertEqual(String(cString: s2.children[0]!.pointee.children[0]!.pointee.format), "i")
        let back = try importArrowArray(schema: &s2, array: &a2)
        XCTAssertEqual(try read(try XCTUnwrap(back.array.asList)), [[[1, 2], [3]], [], nil, [[], [4, 5, 6]]])
        s2.release?(&s2)
        schema.release?(&schema)
    }

    func testListUtf8RoundTrip() throws {
        try requireRealGPU()
        let child = utf8CArray(["alpha", "beta", nil, "delta", "eps"])
        var arr = cArray(length: 4, nullCount: 1,
                         buffers: [bitmapBytes([true, false, true, true]), rawBytes([0, 2, 2, 2, 5] as [Int32])],
                         children: [child])
        var schema = cSchema("+l", name: "words", children: [cSchema("u", name: "item")])
        let r = try importArrowArray(schema: &schema, array: &arr)
        let list = try XCTUnwrap(r.array.asList)

        func read(_ l: MetalListArray) throws -> [[String?]?] {
            let vals = try XCTUnwrap(l.values.asString).toArray()
            return (0..<l.length).map { i in l.valueRange(i).map { r in r.map { vals[$0] } } }
        }
        XCTAssertEqual(try read(list), [["alpha", "beta"], nil, [], [nil, "delta", "eps"]])
        XCTAssertEqual(try list.listValueLength().toArray(), [2, nil, 0, 3])
        XCTAssertEqual(try XCTUnwrap(try list.listFlatten().asString).toArray(),
                       ["alpha", "beta", nil, "delta", "eps"])
        XCTAssertEqual(try XCTUnwrap(try list.listElement(0).asString).toArray(), ["alpha", nil, nil, nil])
        XCTAssertEqual(try XCTUnwrap(try list.listElement(1).asString).toArray(), ["beta", nil, nil, "delta"])

        let f = try list.filter(try MetalBooleanArray([true, true, false, true]))
        XCTAssertEqual(try read(f), [["alpha", "beta"], nil, [nil, "delta", "eps"]])
        var s2 = ArrowSchema(); var a2 = ArrowArray()
        r.array.exportArrowSchema(into: &s2); r.array.exportArrowArray(into: &a2)
        let back = try importArrowArray(schema: &s2, array: &a2)
        XCTAssertEqual(try read(try XCTUnwrap(back.array.asList)), [["alpha", "beta"], nil, [], [nil, "delta", "eps"]])
        s2.release?(&s2)
        schema.release?(&schema)
    }

    func testFixedSizeListFloat32RoundTrip() throws {
        try requireRealGPU()
        let child = primitiveCArray((0..<12).map { Float?(Float($0)) }, zero: 0)
        var arr = cArray(length: 4, nullCount: 1, buffers: [bitmapBytes([true, false, true, true])],
                         children: [child])
        var schema = cSchema("+w:3", name: "vec", children: [cSchema("f", name: "item")])
        let r = try importArrowArray(schema: &schema, array: &arr)
        let list = try XCTUnwrap(r.array.asList)
        XCTAssertEqual(list.kind, ArrowListKind.fixedSize(3))
        XCTAssertEqual(list.arrowFormat, "+w:3")
        XCTAssertEqual(list.length, 4)
        XCTAssertEqual(list.nullCount, 1)
        XCTAssertEqual(try list.listValueLength().toArray(), [3, nil, 3, 3])

        func read(_ l: MetalListArray) throws -> [[Float?]?] {
            let vals = try XCTUnwrap(l.values.asFloat32).toArray()
            return (0..<l.length).map { i in l.valueRange(i).map { r in r.map { vals[$0] } } }
        }
        XCTAssertEqual(try read(list), [[0, 1, 2], nil, [6, 7, 8], [9, 10, 11]])
        XCTAssertEqual(try XCTUnwrap(try list.listElement(2).asFloat32).toArray(), [2, nil, 8, 11])
        XCTAssertNil(try XCTUnwrap(try list.listElement(3).asFloat32).toArray().compactMap { $0 }.first)

        // take keeps the i * N invariant, including through a null index.
        let taken = try list.take(try MetalArray<Int32>([3, nil, 0] as [Int32?]))
        XCTAssertEqual(taken.kind, ArrowListKind.fixedSize(3))
        XCTAssertEqual(taken.offsets.typed(Int32.self)[0], 0)
        XCTAssertEqual(taken.offsets.typed(Int32.self)[1], 3)
        XCTAssertEqual(taken.offsets.typed(Int32.self)[2], 6)
        XCTAssertEqual(taken.offsets.typed(Int32.self)[3], 9)
        XCTAssertEqual(try read(taken), [[9, 10, 11], nil, [0, 1, 2]])

        var s2 = ArrowSchema(); var a2 = ArrowArray()
        r.array.exportArrowSchema(name: "vec", into: &s2); r.array.exportArrowArray(into: &a2)
        XCTAssertEqual(String(cString: s2.format), "+w:3")
        XCTAssertEqual(a2.n_buffers, 1, "a fixed_size_list has no offsets buffer")
        let back = try importArrowArray(schema: &s2, array: &a2)
        XCTAssertEqual(try read(try XCTUnwrap(back.array.asList)), [[0, 1, 2], nil, [6, 7, 8], [9, 10, 11]])
        s2.release?(&s2)
        schema.release?(&schema)
    }

    func testStructRoundTrip() throws {
        try requireRealGPU()
        let a = primitiveCArray([10, 20, 30, 40].map { Int64?($0) }, zero: 0)
        let b = utf8CArray(["w", nil, "y", "z"])
        var arr = cArray(length: 4, nullCount: 1, buffers: [bitmapBytes([true, true, false, true])],
                         children: [a, b])
        var schema = cSchema("+s", name: "s", children: [cSchema("l", name: "a"), cSchema("u", name: "b")])
        let r = try importArrowArray(schema: &schema, array: &arr)
        let s = try XCTUnwrap(r.array.asStruct)
        XCTAssertEqual(s.names, ["a", "b"])
        XCTAssertEqual(s.length, 4)
        XCTAssertEqual(s.nullCount, 1)
        XCTAssertEqual([s.isValid(0), s.isValid(1), s.isValid(2), s.isValid(3)], [true, true, false, true])
        // struct_field propagates the struct's own nulls into the field, as Arrow's does.
        XCTAssertEqual(try XCTUnwrap(try s.structField("a").asInt64).toArray(), [10, 20, nil, 40])
        XCTAssertEqual(try XCTUnwrap(s.children[0].asInt64).toArray(), [10, 20, 30, 40])
        XCTAssertEqual(try XCTUnwrap(try s.structField("b").asString).toArray(), ["w", nil, nil, "z"])
        XCTAssertThrowsError(try s.structField("nope"))

        let t = try s.take(try MetalArray<Int32>([3, 2, nil, 0] as [Int32?]))
        XCTAssertEqual(t.nullCount, 2)
        XCTAssertEqual([t.isValid(0), t.isValid(1), t.isValid(2), t.isValid(3)], [true, false, false, true])
        XCTAssertEqual(try XCTUnwrap(try t.structField("a").asInt64).toArray(), [40, nil, nil, 10])
        XCTAssertEqual(try XCTUnwrap(try t.structField("b").asString).toArray(), ["z", nil, nil, "w"])

        let f = try s.filter(try MetalBooleanArray([false, true, true, true]))
        XCTAssertEqual(f.length, 3)
        XCTAssertEqual(f.nullCount, 1)
        XCTAssertEqual(try XCTUnwrap(try f.structField("a").asInt64).toArray(), [20, nil, 40])

        let sl = try s.slice(offset: 1, length: 2)
        XCTAssertEqual(sl.length, 2)
        XCTAssertEqual(sl.nullCount, 1)
        XCTAssertEqual(try XCTUnwrap(try sl.structField("a").asInt64).toArray(), [20, nil])

        var s2 = ArrowSchema(); var a2 = ArrowArray()
        r.array.exportArrowSchema(name: "s", into: &s2); r.array.exportArrowArray(into: &a2)
        XCTAssertEqual(String(cString: s2.format), "+s")
        XCTAssertEqual(s2.n_children, 2)
        XCTAssertEqual(a2.n_buffers, 1)
        XCTAssertEqual(a2.null_count, 1)
        let back = try importArrowArray(schema: &s2, array: &a2)
        let bs = try XCTUnwrap(back.array.asStruct)
        XCTAssertEqual(bs.nullCount, 1)
        XCTAssertEqual(try XCTUnwrap(try bs.structField("b").asString).toArray(), ["w", nil, nil, "z"])
        s2.release?(&s2)
        schema.release?(&schema)
    }

    func testStructInsideListAndListInsideStruct() throws {
        try requireRealGPU()
        // struct<id: int64, tags: list<utf8>>
        let ids = primitiveCArray([1, 2, 3].map { Int64?($0) }, zero: 0)
        let tagValues = utf8CArray(["a", "b", "c"])
        let tags = cArray(length: 3, nullCount: 0, buffers: [nil, rawBytes([0, 2, 2, 3] as [Int32])],
                          children: [tagValues])
        var arr = cArray(length: 3, nullCount: 0, buffers: [nil], children: [ids, tags])
        var schema = cSchema("+s", name: "row", children: [
            cSchema("l", name: "id"),
            cSchema("+l", name: "tags", children: [cSchema("u", name: "item")]),
        ])
        let r = try importArrowArray(schema: &schema, array: &arr)
        let s = try XCTUnwrap(r.array.asStruct)
        let tagList = try XCTUnwrap(try s.structField("tags").asList)
        XCTAssertEqual(try tagList.listValueLength().toArray(), [2, 0, 1])

        let t = try s.take(try MetalArray<Int32>([2, 0]))
        let tl = try XCTUnwrap(try t.structField("tags").asList)
        XCTAssertEqual(try tl.listValueLength().toArray(), [1, 2])
        let vals = try XCTUnwrap(tl.values.asString).toArray()
        XCTAssertEqual((0..<tl.length).map { i in tl.valueRange(i).map { r in r.map { vals[$0] } } },
                       [["c"], ["a", "b"]])
        schema.release?(&schema)
    }

    func testMapRoundTrip() throws {
        try requireRealGPU()
        // map<utf8, int32>: [{"a": 1, "b": 2}, {}, null, {"c": 3}]
        let keys = utf8CArray(["a", "b", "c"])
        let values = primitiveCArray([1, 2, 3].map { Int32?($0) }, zero: 0)
        let entries = cArray(length: 3, nullCount: 0, buffers: [nil], children: [keys, values])
        var arr = cArray(length: 4, nullCount: 1,
                         buffers: [bitmapBytes([true, true, false, true]), rawBytes([0, 2, 2, 2, 3] as [Int32])],
                         children: [entries])
        var schema = cSchema("+m", name: "m", children: [
            cSchema("+s", name: "entries", flags: 0,
                    children: [cSchema("u", name: "key", flags: 0), cSchema("i", name: "value")]),
        ])
        let r = try importArrowArray(schema: &schema, array: &arr)
        let m = try XCTUnwrap(r.array.asMap)
        XCTAssertEqual(m.length, 4)
        XCTAssertEqual(m.nullCount, 1)
        XCTAssertFalse(m.keysSorted)
        XCTAssertEqual(try XCTUnwrap(m.keys.asString).toArray(), ["a", "b", "c"])
        XCTAssertEqual(try XCTUnwrap(m.items.asInt32).toArray(), [1, 2, 3])
        XCTAssertEqual([m.valueRange(0), m.valueRange(1), m.valueRange(2), m.valueRange(3)],
                       [0..<2, 2..<2, nil, 2..<3])
        XCTAssertEqual(try r.array.listValueLength().toArray(), [2, 0, nil, 1])

        let t = try m.take(try MetalArray<Int32>([3, 2, 0]))
        XCTAssertEqual(t.nullCount, 1)
        XCTAssertEqual(try XCTUnwrap(t.keys.asString).toArray(), ["c", "a", "b"])
        XCTAssertEqual(try XCTUnwrap(t.items.asInt32).toArray(), [3, 1, 2])
        XCTAssertEqual([t.valueRange(0), t.valueRange(1), t.valueRange(2)], [0..<1, nil, 1..<3])

        var s2 = ArrowSchema(); var a2 = ArrowArray()
        r.array.exportArrowSchema(name: "m", into: &s2); r.array.exportArrowArray(into: &a2)
        XCTAssertEqual(String(cString: s2.format), "+m")
        XCTAssertEqual(String(cString: s2.children[0]!.pointee.format), "+s")
        XCTAssertEqual(String(cString: s2.children[0]!.pointee.name), "entries")
        XCTAssertEqual(s2.children[0]!.pointee.flags, 0, "the entries struct must be non-nullable")
        XCTAssertEqual(s2.children[0]!.pointee.children[0]!.pointee.flags, 0, "the key field must be non-nullable")
        XCTAssertEqual(a2.n_buffers, 2)
        let back = try importArrowArray(schema: &s2, array: &a2)
        let bm = try XCTUnwrap(back.array.asMap)
        XCTAssertEqual(bm.nullCount, 1)
        XCTAssertEqual(try XCTUnwrap(bm.keys.asString).toArray(), ["a", "b", "c"])
        s2.release?(&s2)
        schema.release?(&schema)
    }

    func testDenseUnionRoundTrip() throws {
        try requireRealGPU()
        // dense union<num: int64, txt: utf8>, values: 10, "a", 20, "b", 30
        let nums = primitiveCArray([10, 20, 30].map { Int64?($0) }, zero: 0)
        let txt = utf8CArray(["a", "b"])
        var arr = cArray(length: 5, nullCount: 0,
                         buffers: [rawBytes([0, 1, 0, 1, 0] as [Int8]), rawBytes([0, 0, 1, 1, 2] as [Int32])],
                         children: [nums, txt])
        var schema = cSchema("+ud:0,1", name: "u", children: [cSchema("l", name: "num"), cSchema("u", name: "txt")])
        let r = try importArrowArray(schema: &schema, array: &arr)
        let u = try XCTUnwrap(r.array.asUnion)
        XCTAssertEqual(u.mode, .dense)
        XCTAssertEqual(u.arrowFormat, "+ud:0,1")
        XCTAssertEqual(u.length, 5)
        XCTAssertEqual(u.names, ["num", "txt"])

        func read(_ u: MetalUnionArray) throws -> [String] {
            let nums = try XCTUnwrap(u.children[0].asInt64).toArray()
            let txts = try XCTUnwrap(u.children[1].asString).toArray()
            return (0..<u.length).map { i in
                guard let (c, j) = u.location(i) else { return "?" }
                return c == 0 ? "\(nums[j] ?? -1)" : (txts[j] ?? "?")
            }
        }
        XCTAssertEqual(try read(u), ["10", "a", "20", "b", "30"])
        XCTAssertEqual(try read(try u.take(try MetalArray<Int32>([4, 1, 0]))), ["30", "a", "10"])
        XCTAssertEqual(try read(try u.slice(offset: 1, length: 3)), ["a", "20", "b"])
        XCTAssertEqual(try read(try u.filter(try MetalBooleanArray([true, false, true, false, true]))),
                       ["10", "20", "30"])

        var s2 = ArrowSchema(); var a2 = ArrowArray()
        r.array.exportArrowSchema(name: "u", into: &s2); r.array.exportArrowArray(into: &a2)
        XCTAssertEqual(String(cString: s2.format), "+ud:0,1")
        XCTAssertEqual(a2.n_buffers, 2)
        XCTAssertEqual(a2.n_children, 2)
        let back = try importArrowArray(schema: &s2, array: &a2)
        XCTAssertEqual(try read(try XCTUnwrap(back.array.asUnion)), ["10", "a", "20", "b", "30"])
        s2.release?(&s2)
        schema.release?(&schema)
    }

    func testSparseUnionRoundTrip() throws {
        try requireRealGPU()
        let nums = primitiveCArray([10, 0, 20, 0, 30].map { Int64?($0) }, zero: 0)
        let txt = utf8CArray(["", "a", "", "b", ""])
        var arr = cArray(length: 5, nullCount: 0, buffers: [rawBytes([0, 1, 0, 1, 0] as [Int8])],
                         children: [nums, txt])
        var schema = cSchema("+us:0,1", name: "u", children: [cSchema("l", name: "num"), cSchema("u", name: "txt")])
        let r = try importArrowArray(schema: &schema, array: &arr)
        let u = try XCTUnwrap(r.array.asUnion)
        XCTAssertEqual(u.mode, .sparse)
        XCTAssertEqual(u.arrowFormat, "+us:0,1")
        XCTAssertNil(u.offsets)
        XCTAssertEqual(u.children.map(\.length), [5, 5])

        func read(_ u: MetalUnionArray) throws -> [String] {
            let nums = try XCTUnwrap(u.children[0].asInt64).toArray()
            let txts = try XCTUnwrap(u.children[1].asString).toArray()
            return (0..<u.length).map { i in
                guard let (c, j) = u.location(i) else { return "?" }
                return c == 0 ? "\(nums[j] ?? -1)" : (txts[j] ?? "?")
            }
        }
        XCTAssertEqual(try read(u), ["10", "a", "20", "b", "30"])
        // A sparse union's children move with the selection, so they stay the union's length.
        let t = try u.take(try MetalArray<Int32>([3, 0]))
        XCTAssertEqual(t.children.map(\.length), [2, 2])
        XCTAssertEqual(try read(t), ["b", "10"])

        var s2 = ArrowSchema(); var a2 = ArrowArray()
        r.array.exportArrowSchema(name: "u", into: &s2); r.array.exportArrowArray(into: &a2)
        XCTAssertEqual(String(cString: s2.format), "+us:0,1")
        XCTAssertEqual(a2.n_buffers, 1)
        let back = try importArrowArray(schema: &s2, array: &a2)
        XCTAssertEqual(try read(try XCTUnwrap(back.array.asUnion)), ["10", "a", "20", "b", "30"])
        s2.release?(&s2)
        schema.release?(&schema)
    }

    /// A struct child of a record batch used to be rejected outright; now it imports as a nested column
    /// while `importArrowRecordBatch` keeps its top-level meaning.
    func testRecordBatchWithNestedColumns() throws {
        try requireRealGPU()
        let ids = primitiveCArray([1, 2, 3].map { Int64?($0) }, zero: 0)
        let leaf = primitiveCArray([7, 8, 9, 10].map { Int32?($0) }, zero: 0)
        let lists = cArray(length: 3, nullCount: 0, buffers: [nil, rawBytes([0, 2, 3, 4] as [Int32])],
                           children: [leaf])
        let inner = primitiveCArray([1.5, 2.5, 3.5].map { Float?($0) }, zero: 0)
        let nested = cArray(length: 3, nullCount: 0, buffers: [nil], children: [inner])
        var arr = cArray(length: 3, nullCount: 0, buffers: [nil], children: [ids, lists, nested])
        var schema = cSchema("+s", name: "batch", children: [
            cSchema("l", name: "id"),
            cSchema("+l", name: "vals", children: [cSchema("i", name: "item")]),
            cSchema("+s", name: "inner", children: [cSchema("f", name: "x")]),
        ])
        let (batch, _) = try importArrowRecordBatch(schema: &schema, array: &arr)
        XCTAssertEqual(batch.names, ["id", "vals", "inner"])
        XCTAssertEqual(batch.length, 3)
        XCTAssertEqual(try XCTUnwrap(batch["vals"]?.asList).listValueLength().toArray(), [2, 1, 1])
        XCTAssertEqual(try XCTUnwrap(batch["inner"]?.asStruct).structField("x").asFloat32?.toArray(), [1.5, 2.5, 3.5])
        // Filtering a batch filters the nested columns with everything else.
        let f = try batch.filter(try MetalBooleanArray([true, false, true]))
        XCTAssertEqual(f.length, 2)
        XCTAssertEqual(try XCTUnwrap(f["vals"]?.asList).listValueLength().toArray(), [2, 1])
        XCTAssertEqual(try XCTUnwrap(f["id"]?.asInt64).toArray(), [1, 3])
        schema.release?(&schema)
    }

    func testMalformedNestedArraysThrow() throws {
        // "+l" with no children is still rejected, as it was before lists existed.
        var s1 = ArrowSchema(); exportArrowSchema(format: "+l", into: &s1)
        var a1 = ArrowArray(); a1.n_buffers = 2; a1.n_children = 1; a1.release = { _ in }
        XCTAssertThrowsError(try importArrowArray(schema: &s1, array: &a1))
        s1.release?(&s1)

        // Offsets that reach past the child are a data error, not a crash.
        let child = primitiveCArray([1, 2].map { Int64?($0) }, zero: 0)
        var a2 = cArray(length: 2, nullCount: 0, buffers: [nil, rawBytes([0, 2, 9] as [Int32])], children: [child])
        var s2 = cSchema("+l", children: [cSchema("l", name: "item")])
        XCTAssertThrowsError(try importArrowArray(schema: &s2, array: &a2))
        s2.release?(&s2)

        // A fixed_size_list whose child is too short is rejected too.
        let short = primitiveCArray([1, 2].map { Int32?($0) }, zero: 0)
        var a3 = cArray(length: 3, nullCount: 0, buffers: [nil], children: [short])
        var s3 = cSchema("+w:4", children: [cSchema("i", name: "item")])
        XCTAssertThrowsError(try importArrowArray(schema: &s3, array: &a3))
        s3.release?(&s3)
    }
}

// MARK: - GPU kernels against a Swift oracle

final class NestedKernelTests: XCTestCase {
    static let sizes = [0, 1, 33, 4097, 100_003]

    /// Rows with nulls, empty lists and varying lengths.
    func model(_ n: Int) -> [[Int64?]?] {
        (0..<n).map { i in
            if i % 7 == 3 { return nil }
            let len = i % 4
            return (0..<len).map { k in (i + k) % 5 == 0 ? nil : Int64(i * 10 + k) }
        }
    }

    func build(_ lists: [[Int64?]?]) throws -> MetalListArray {
        let flat = lists.flatMap { $0 ?? [] }
        return try MetalListArray(counts: lists.map { $0?.count },
                                  values: .int64(try MetalArray<Int64>(flat)))
    }

    func read(_ l: MetalListArray) throws -> [[Int64?]?] {
        let vals = try XCTUnwrap(l.values.asInt64).toArray()
        return (0..<l.length).map { i in l.valueRange(i).map { r in r.map { vals[$0] } } }
    }

    func testBuildAndReadMatchesModel() throws {
        try requireRealGPU()
        for n in Self.sizes {
            let m = model(n)
            XCTAssertEqual(try read(try build(m)), m, "n = \(n)")
        }
    }

    func testListValueLength() throws {
        try requireRealGPU()
        for n in Self.sizes {
            let m = model(n)
            let got = try build(m).listValueLength().toArray()
            XCTAssertEqual(got, m.map { $0.map { Int32($0.count) } }, "n = \(n)")
        }
    }

    func testListFlatten() throws {
        try requireRealGPU()
        for n in Self.sizes {
            let m = model(n)
            let flat = try XCTUnwrap(try build(m).listFlatten().asInt64).toArray()
            XCTAssertEqual(flat, m.flatMap { $0 ?? [] }, "n = \(n)")
        }
    }

    func testListFlattenOnASlicedList() throws {
        try requireRealGPU()
        // A slice keeps the child and moves the offsets, so flatten must cut the child to the range.
        let m = model(4097)
        let list = try build(m)
        let sliced = try list.slice(offset: 1000, length: 500)
        let expected = m[1000..<1500].flatMap { $0 ?? [] }
        XCTAssertEqual(try XCTUnwrap(try sliced.listFlatten().asInt64).toArray(), expected)
        XCTAssertEqual(try read(sliced), Array(m[1000..<1500]))
    }

    func testListElement() throws {
        try requireRealGPU()
        for n in Self.sizes {
            let m = model(n)
            let list = try build(m)
            for k in [0, 1, 2, 3, 7] {
                let got = try XCTUnwrap(try list.listElement(k).asInt64).toArray()
                let want = m.map { l -> Int64? in
                    guard let l, k < l.count else { return nil }
                    return l[k]
                }
                XCTAssertEqual(got, want, "n = \(n), element \(k)")
            }
        }
    }

    func testTake() throws {
        try requireRealGPU()
        for n in Self.sizes {
            let m = model(n)
            let list = try build(m)
            guard n > 0 else {
                XCTAssertEqual(try read(try list.take(try MetalArray<Int32>([]))), [])
                continue
            }
            var idx: [Int32?] = (0..<min(n, 500)).map { i in Int32((i * 7919) % n) }
            idx.append(nil)
            idx.append(Int32(n - 1))
            let taken = try list.take(try MetalArray<Int32>(idx))
            let want = idx.map { i -> [Int64?]? in i.map { m[Int($0)] } ?? nil }
            XCTAssertEqual(try read(taken), want, "n = \(n)")
        }
    }

    func testFilter() throws {
        try requireRealGPU()
        for n in Self.sizes {
            let m = model(n)
            let keep = (0..<n).map { $0 % 3 != 1 }
            let got = try read(try build(m).filter(try MetalBooleanArray(keep)))
            XCTAssertEqual(got, zip(m, keep).filter { $0.1 }.map { $0.0 }, "n = \(n)")
        }
    }

    func testSlice() throws {
        try requireRealGPU()
        for n in Self.sizes {
            let m = model(n)
            let list = try build(m)
            let off = n / 3, len = n / 4
            XCTAssertEqual(try read(try list.slice(offset: off, length: len)),
                           Array(m[off..<(off + len)]), "n = \(n)")
            XCTAssertEqual(try read(try list.slice(offset: 0, length: n)), m, "n = \(n)")
            XCTAssertThrowsError(try list.slice(offset: 0, length: n + 1))
        }
    }

    func testFixedSizeListKernels() throws {
        try requireRealGPU()
        let width = 3
        for n in Self.sizes {
            let valid = (0..<n).map { $0 % 11 != 4 }
            let child = try MetalArray<Float>((0..<(n * width)).map { Float($0) })
            let list = try MetalListArray(counts: (0..<n).map { valid[$0] ? width : nil },
                                          values: .float32(child), kind: .fixedSize(width))
            XCTAssertEqual(try list.listValueLength().toArray(),
                           valid.map { $0 ? Int32(width) : nil }, "n = \(n)")
            let got = try XCTUnwrap(try list.listElement(1).asFloat32).toArray()
            XCTAssertEqual(got, (0..<n).map { valid[$0] ? Float($0 * width + 1) : nil }, "n = \(n)")
            let kept = (0..<n).map { $0 % 5 == 0 }
            let f = try list.filter(try MetalBooleanArray(kept))
            XCTAssertEqual(f.length, kept.filter { $0 }.count)
            let o = f.offsets.typed(Int32.self)
            for i in 0...f.length { XCTAssertEqual(o[i], Int32(i * width), "n = \(n), offset \(i)") }
            let fv = try XCTUnwrap(f.values.asFloat32).toArray()
            let wantRows = (0..<n).filter { kept[$0] }
            XCTAssertEqual(fv.count, wantRows.count * width)
            for (j, i) in wantRows.enumerated() {
                XCTAssertEqual(fv[j * width], Float(i * width), "n = \(n), row \(i)")
            }
        }
    }

    func testStructSelectionMatchesOracle() throws {
        try requireRealGPU()
        for n in Self.sizes {
            let valid = (0..<n).map { $0 % 9 != 2 }
            let a = (0..<n).map { $0 % 6 == 1 ? nil : Int64($0) }
            let b = (0..<n).map { "s\($0)" }
            let s = try MetalStructArray(names: ["a", "b"],
                                         children: [.int64(try MetalArray<Int64>(a)),
                                                    .string(try MetalStringArray(b))],
                                         valid: valid)
            XCTAssertEqual(s.nullCount, valid.filter { !$0 }.count, "n = \(n)")
            let keep = (0..<n).map { $0 % 4 == 3 }
            let f = try s.filter(try MetalBooleanArray(keep))
            let wantRows = (0..<n).filter { keep[$0] }
            XCTAssertEqual(f.length, wantRows.count)
            XCTAssertEqual((0..<f.length).map { f.isValid($0) }, wantRows.map { valid[$0] }, "n = \(n)")
            XCTAssertEqual(try XCTUnwrap(try f.structField("a").asInt64).toArray(),
                           wantRows.map { valid[$0] ? a[$0] : nil }, "n = \(n)")
            XCTAssertEqual(try XCTUnwrap(try f.structField("b").asString).toArray(),
                           wantRows.map { valid[$0] ? b[$0] : nil }, "n = \(n)")
            let off = n / 3, len = n / 5
            let sl = try s.slice(offset: off, length: len)
            XCTAssertEqual((0..<sl.length).map { sl.isValid($0) }, Array(valid[off..<(off + len)]), "n = \(n)")
            XCTAssertEqual(try XCTUnwrap(try sl.structField("a").asInt64).toArray(),
                           (off..<(off + len)).map { valid[$0] ? a[$0] : nil }, "n = \(n)")
        }
    }

    func testTakeOfListOfStringMatchesOracle() throws {
        try requireRealGPU()
        // The child is a string array, so `take` gathers offsets *and* bytes through the expanded indices.
        let n = 4097
        let model: [[String]?] = (0..<n).map { i in
            i % 13 == 5 ? nil : (0..<(i % 3)).map { "r\(i)_\($0)" }
        }
        let flat = model.flatMap { $0 ?? [] }
        let list = try MetalListArray(counts: model.map { $0?.count },
                                      values: .string(try MetalStringArray(flat.map { Optional($0) })))
        let idx = (0..<1000).map { Int32(($0 * 4099) % n) }
        let taken = try list.take(try MetalArray<Int32>(idx))
        let vals = try XCTUnwrap(taken.values.asString).toArray()
        let got: [[String]?] = (0..<taken.length).map { i in
            taken.valueRange(i).map { r in r.map { vals[$0]! } }
        }
        XCTAssertEqual(got, idx.map { model[Int($0)] })
    }
}

// MARK: - list_value_length, wide and narrow

extension NestedTests {
    /// `list_value_length` writes the same bytes whether it takes the eight-rows-per-thread kernel or
    /// the one-row one. The wide kernel needs the offsets pointer 16-byte aligned, which a slice at a
    /// row that is not a multiple of four breaks, so both paths are exercised here at every length
    /// around a multiple of eight, plus the 0-row, 1-row and all-null cases.
    func testListValueLengthWideAndSliced() throws {
        try requireRealGPU()
        for n in [0, 1, 2, 7, 8, 9, 33, 4097, 100_003] {
            let counts: [Int?] = (0..<n).map { i in i % 11 == 3 ? nil : (i % 5) }
            let total = counts.reduce(0) { $0 + ($1 ?? 0) }
            let child = try MetalArray<Int32>((0..<total).map { Int32($0) })
            let list = try MetalListArray(counts: counts, values: .int32(child))
            XCTAssertEqual(try list.listValueLength().toArray(), counts.map { $0.map(Int32.init) },
                           "whole array, \(n) rows")
            // Every slice offset 0...5 covers both the aligned (0, 4) and the unaligned cases.
            for off in 0..<Swift.min(n, 6) {
                let s = try list.slice(offset: off, length: n - off)
                XCTAssertEqual(try s.listValueLength().toArray(),
                               counts[off...].map { $0.map(Int32.init) },
                               "slice from \(off) of \(n) rows")
            }
        }
    }

    /// An all-null list column and a single-row one: the lengths are still the offsets difference, and
    /// the validity bitmap is the input's.
    func testListValueLengthAllNullAndSingleRow() throws {
        try requireRealGPU()
        let allNull = try MetalListArray(counts: [Int?](repeating: nil, count: 40),
                                         values: .int32(try MetalArray<Int32>([Int32]())))
        let lens = try allNull.listValueLength()
        XCTAssertEqual(lens.nullCount, 40)
        XCTAssertEqual(lens.toArray(), [Int32?](repeating: nil, count: 40))

        let one = try MetalListArray(counts: [3], values: .int32(try MetalArray<Int32>([1, 2, 3])))
        XCTAssertEqual(try one.listValueLength().toArray(), [3])
    }
}
