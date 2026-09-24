import Foundation

// MARK: - arrow.fixed_shape_tensor

/// Arrow's canonical `arrow.fixed_shape_tensor` extension type: every row is one tensor of the same
/// shape, stored as a `fixed_size_list` of `product(shape)` values in row-major order.
///
/// The extension metadata is JSON with a required `shape` and optional `dim_names` and `permutation`
/// (see the Arrow canonical extensions page). ArrowMetal keeps the metadata bytes exactly as they were
/// read and checks them against the storage; compute runs on the storage (`storageArray`, a list
/// array whose child holds every element), so the engine's list and numeric kernels apply unchanged.
///
/// IPC `Tensor` and `SparseTensor` messages are a different thing (an n-dimensional array sent as a
/// message of its own, outside any record batch) and are refused by `ArrowIPCReader`.
public struct ArrowFixedShapeTensorType: Equatable, Sendable {
    public static let extensionName = "arrow.fixed_shape_tensor"

    /// The logical shape of each tensor.
    public let shape: [Int]
    /// Optional names of the logical dimensions, one per entry of `shape`.
    public let dimNames: [String]?
    /// Optional permutation of `0 ..< shape.count` giving the physical order of the dimensions.
    public let permutation: [Int]?

    /// Elements per tensor: the product of `shape` (and the storage list's fixed size).
    public var elementCount: Int { shape.reduce(1, *) }

    public init(shape: [Int], dimNames: [String]? = nil, permutation: [Int]? = nil) throws {
        func fail(_ s: String) -> ArrowMetalError { .invalidArrowArray("arrow.fixed_shape_tensor: " + s) }
        guard shape.allSatisfy({ $0 >= 0 }) else { throw fail("shape \(shape) has a negative dimension") }
        if let dimNames, dimNames.count != shape.count {
            throw fail("\(dimNames.count) dim_names for a \(shape.count)-dimensional shape")
        }
        if let permutation, permutation.sorted() != Array(0..<shape.count) {
            throw fail("permutation \(permutation) is not a permutation of 0..<\(shape.count)")
        }
        self.shape = shape
        self.dimNames = dimNames
        self.permutation = permutation
    }

    /// Parses `ARROW:extension:metadata`, e.g. `{"shape":[2,3],"dim_names":["H","W"]}`.
    public init(metadata: [UInt8]) throws {
        func fail(_ s: String) -> ArrowMetalError { .invalidArrowArray("arrow.fixed_shape_tensor metadata: " + s) }
        guard let object = try? JSONSerialization.jsonObject(with: Data(metadata)) as? [String: Any] else {
            throw fail("not a JSON object")
        }
        func ints(_ key: String) throws -> [Int]? {
            guard let v = object[key] else { return nil }
            guard let a = v as? [NSNumber] else { throw fail("\(key) is not a list of integers") }
            return a.map(\.intValue)
        }
        guard let shape = try ints("shape") else { throw fail("no shape") }
        var names: [String]? = nil
        if let v = object["dim_names"] {
            guard let a = v as? [String] else { throw fail("dim_names is not a list of strings") }
            names = a
        }
        try self.init(shape: shape, dimNames: names, permutation: try ints("permutation"))
    }

    /// The metadata JSON, in the key order pyarrow writes: `shape`, then `permutation`, then `dim_names`.
    public var metadata: [UInt8] {
        func list<T>(_ xs: [T], _ f: (T) -> String) -> String { "[" + xs.map(f).joined(separator: ",") + "]" }
        func quoted(_ s: String) -> String {
            let data = (try? JSONSerialization.data(withJSONObject: [s], options: [.fragmentsAllowed])) ?? Data("[\"\"]".utf8)
            let text = String(decoding: data, as: UTF8.self)
            return String(text.dropFirst().dropLast())
        }
        var parts = ["\"shape\":" + list(shape) { "\($0)" }]
        if let permutation { parts.append("\"permutation\":" + list(permutation) { "\($0)" }) }
        if let dimNames { parts.append("\"dim_names\":" + list(dimNames, quoted)) }
        return Array(("{" + parts.joined(separator: ",") + "}").utf8)
    }

    /// Parses the metadata a column carries and checks it against the column's storage: a fixed-size
    /// list of exactly `product(shape)` elements.
    init(metadata: [UInt8], storage: AnyMetalArray, column: String) throws {
        do { try self.init(metadata: metadata) } catch {
            throw ArrowIPCError.malformed("column '\(column)': \(error)")
        }
        guard case .list(let list) = storage, case .fixedSize(let n) = list.kind else {
            throw ArrowIPCError.malformed(
                "column '\(column)' is arrow.fixed_shape_tensor but its storage is \(storage.ipcType), not a fixed_size_list")
        }
        guard n == elementCount else {
            throw ArrowIPCError.malformed(
                "column '\(column)' is arrow.fixed_shape_tensor of shape \(shape) (\(elementCount) elements) "
                + "over a fixed_size_list of \(n)")
        }
    }
}

extension AnyMetalArray {
    /// Wraps a `fixed_size_list` column as an `arrow.fixed_shape_tensor` extension column of `type`.
    /// The list's fixed size must be `type.elementCount`.
    public static func fixedShapeTensor(_ storage: MetalListArray,
                                        type: ArrowFixedShapeTensorType) throws -> AnyMetalArray {
        guard case .fixedSize(let n) = storage.kind, n == type.elementCount else {
            throw ArrowMetalError.invalidArrowArray(
                "arrow.fixed_shape_tensor of shape \(type.shape) needs a fixed_size_list of \(type.elementCount), "
                + "not \(storage.kind.arrowFormat)")
        }
        return .extended(MetalExtensionArray(storage: .list(storage), name: ArrowFixedShapeTensorType.extensionName,
                                             metadata: type.metadata))
    }

    /// The tensor type of an `arrow.fixed_shape_tensor` column, or nil for every other column (and for
    /// one whose metadata does not parse).
    public var fixedShapeTensorType: ArrowFixedShapeTensorType? {
        guard let e = asExtension, e.name == ArrowFixedShapeTensorType.extensionName else { return nil }
        return try? ArrowFixedShapeTensorType(metadata: e.metadata ?? [])
    }
}
