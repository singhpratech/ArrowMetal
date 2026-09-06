import Foundation
import CArrowABI
import ArrowMetal

// End-to-end scenarios. Run: swift run -c release arrowmetal-examples
setvbuf(stdout, nil, _IONBF, 0)
func ms(_ t0: DispatchTime) -> String { String(format: "%.2f ms", Double(DispatchTime.now().uptimeNanoseconds - t0.uptimeNanoseconds) / 1e6) }
let n = 10_000_000
print("(timings include first-use shader compilation; run a scenario twice to see steady state)\n")
var g = SystemRandomNumberGenerator()

// ---------------------------------------------------------------------------------------------
print("Scenario 1: analytics on a record batch (orders table, \(n) rows)")
// Columns arrive from anywhere that speaks Arrow; here we build them directly in Metal shared memory.
let orders = try MetalRecordBatch(names: ["order_id", "region", "amount", "qty", "returned"], columns: [
    .int64(try MetalArray<Int64>((0..<n).map { Int64($0) })),
    .int32(try MetalArray<Int32>((0..<n).map { _ in Int32.random(in: 0..<5, using: &g) })),
    .float32(try MetalArray<Float>((0..<n).map { _ in Int.random(in: 0..<50, using: &g) == 0 ? nil : Float.random(in: 1...500, using: &g) })),
    .int32(try MetalArray<Int32>((0..<n).map { _ in Int32.random(in: 1...20, using: &g) })),
    .boolean(try MetalBooleanArray((0..<n).map { _ in Int.random(in: 0..<10, using: &g) == 0 })),
])
var t0 = DispatchTime.now()
// SELECT sum(amount), avg(amount), max(qty), count(*) FROM orders WHERE region = 2 AND amount > 100 AND NOT returned
let region = orders["region"]!.asInt32!, amount = orders["amount"]!.asFloat32!, qty = orders["qty"]!.asInt32!
let mask = try region.compare(.eq, 2).and(try amount.compare(.gt, 100)).and(try orders["returned"]!.asBoolean!.not())
let hits = try orders.filter(mask)
let sum = try hits["amount"]!.asFloat32!.sum()!.asDouble
let avg = try hits["amount"]!.asFloat32!.mean()!
let maxQty = try hits["qty"]!.asInt32!.max()!
print("  rows kept: \(hits.length)  sum(amount)=\(String(format: "%.1f", sum))  avg=\(String(format: "%.2f", avg))  max(qty)=\(maxQty)   [\(ms(t0))]")
_ = qty

// ---------------------------------------------------------------------------------------------
print("\nScenario 2: feature preparation (sample, cast, normalise) for a model")
t0 = DispatchTime.now()
// Random sample of 1M rows by index, then cast qty to Float and scale amount into [0,1].
let sample = try MetalArray<Int32>((0..<1_000_000).map { _ in Int32.random(in: 0..<Int32(n), using: &g) })
let feat = try orders.selecting(["amount", "qty"]).take(sample)
let qtyF = try feat["qty"]!.asInt32!.cast(to: Float.self)
let amt = feat["amount"]!.asFloat32!
let scaled = try amt.subtract(try amt.min()!).divide(try amt.max()! - (try amt.min()!))
print("  sampled \(feat.length) rows, qty->Float \(qtyF.length), amount scaled: min=\(try scaled.min()!) max=\(try scaled.max()!) nulls=\(scaled.nullCount)   [\(ms(t0))]")

// ---------------------------------------------------------------------------------------------
print("\nScenario 3: Float64 prices with NaN (exact GPU compare/min/max/filter, CPU sum)")
t0 = DispatchTime.now()
let prices = try MetalArray<Double>((0..<n).map { i in i % 1000 == 0 ? .nan : Double.random(in: 10...1000, using: &g) })
let cheap = try prices.filter(try prices.compare(.lt, 100))
print("  below 100: \(cheap.length) rows, min=\(String(format: "%.3f", try prices.min()!)) max=\(String(format: "%.3f", try prices.max()!)) (NaN skipped), sum=\(String(format: "%.1f", try cheap.sum()!.asDouble))   [\(ms(t0))]")

// ---------------------------------------------------------------------------------------------
print("\nScenario 4: interop through the Arrow C Data Interface (zero-copy both ways)")
t0 = DispatchTime.now()
var schema = ArrowSchema(); var carr = ArrowArray()
hits.exportArrowSchema(name: "hits", into: &schema)      // hand the filtered batch to any Arrow library...
hits.exportArrowArray(into: &carr)
let back = try importArrowRecordBatch(schema: &schema, array: &carr)   // ...and get one back
print("  struct(+s) with \(schema.n_children) children exported and re-imported, zeroCopy=\(back.zeroCopy), rows=\(back.batch.length)   [\(ms(t0))]")
var dev = ArrowDeviceArray(); hits.exportArrowDeviceArray(into: &dev)
print("  device export: device_type=\(dev.device_type) (ARROW_DEVICE_METAL=\(ARROW_DEVICE_METAL))")
dev.array.release?(&dev.array); schema.release?(&schema)

// ---------------------------------------------------------------------------------------------
print("\nScenario 5: sliced windows (zero-copy views) with per-window aggregates")
t0 = DispatchTime.now()
let window = 1_000_000
var line = "  "
for w in 0..<5 {
    let s = try orders["amount"]!.asFloat32!.slice(offset: w * window, length: window)
    line += String(format: "w%d=%.1f ", w, try s.mean()!)
}
print(line + "  [\(ms(t0))]")
print("\ndone")
