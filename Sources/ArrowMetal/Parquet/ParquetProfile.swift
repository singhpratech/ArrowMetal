import Foundation

// Phase timing for the Parquet reader, off unless ARROWMETAL_PARQUET_PROFILE is set. With it set, every
// lap first waits for the GPU work already encoded (so a phase's kernels are charged to that phase),
// then records wall time and the process's minor page faults since the previous lap; the read prints
// one line per phase to stderr when it returns.
enum ParquetProfile {
    static let enabled = ProcessInfo.processInfo.environment["ARROWMETAL_PARQUET_PROFILE"] != nil

    private static let lock = NSLock()
    private static var last: UInt64 = 0
    private static var lastFaults: Int = 0
    private static var rows: [(String, Double, Int)] = []

    private static func faults() -> Int {
        var u = rusage()
        getrusage(RUSAGE_SELF, &u)
        return Int(u.ru_minflt)
    }

    private static let timebase: mach_timebase_info_data_t = {
        var t = mach_timebase_info_data_t()
        mach_timebase_info(&t)
        return t
    }()

    static func start() {
        guard enabled else { return }
        lock.lock(); defer { lock.unlock() }
        rows.removeAll()
        last = mach_absolute_time()
        lastFaults = faults()
    }

    /// Charges the time since the previous lap to `name`. With `sync`, waits for the GPU first.
    static func lap(_ name: @autoclosure () -> String, sync ctx: MetalContext? = nil) {
        guard enabled else { return }
        if let ctx { try? ctx.syncPoint() }
        lock.lock(); defer { lock.unlock() }
        let now = mach_absolute_time()
        let f = faults()
        let ms = Double(now &- last) * Double(timebase.numer) / Double(timebase.denom) / 1e6
        rows.append((name(), ms, f - lastFaults))
        last = now
        lastFaults = f
    }

    static func report(_ title: String) {
        guard enabled else { return }
        lock.lock(); defer { lock.unlock() }
        var merged: [(String, Double, Int)] = []
        var index: [String: Int] = [:]
        for r in rows {
            if let i = index[r.0] { merged[i].1 += r.1; merged[i].2 += r.2 }
            else { index[r.0] = merged.count; merged.append(r) }
        }
        let total = merged.reduce(0.0) { $0 + $1.1 }
        var s = "parquet-profile \(title) total=\(String(format: "%.2f", total)) ms\n"
        for (n, ms, f) in merged {
            s += String(format: "  %-28@ %9.2f ms %9d faults\n", n as NSString, ms, f)
        }
        FileHandle.standardError.write(s.data(using: .utf8)!)
        rows.removeAll()
    }
}
