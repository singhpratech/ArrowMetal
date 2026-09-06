import Foundation

// The serialised expression grammar. It is the exact text `Expr.description` and `ExprQuery.canonical`
// produce, so a tree round-trips, and it is what the C ABI (`am_query`) and the Python package speak.
// The grammar is documented in include/arrowmetal.h and docs/EXPR.md.

private enum Token: Equatable {
    case open, close
    case atom(String)
    case text(String)
}

private struct Lexer {
    let s: [Character]
    var i = 0
    init(_ text: String) { s = Array(text) }

    mutating func next() throws -> Token? {
        while i < s.count, s[i] == " " || s[i] == "\n" || s[i] == "\t" || s[i] == "\r" { i += 1 }
        guard i < s.count else { return nil }
        let c = s[i]
        if c == "(" { i += 1; return .open }
        if c == ")" { i += 1; return .close }
        if c == "\"" {
            i += 1
            var out = ""
            while i < s.count, s[i] != "\"" {
                if s[i] == "\\", i + 1 < s.count {
                    i += 1
                    switch s[i] {
                    case "n": out.append("\n")
                    case "t": out.append("\t")
                    default: out.append(s[i])
                    }
                } else { out.append(s[i]) }
                i += 1
            }
            guard i < s.count else { throw ExprError.invalid("unterminated string in expression text") }
            i += 1
            return .text(out)
        }
        var out = ""
        while i < s.count, !" \n\t\r()".contains(s[i]) { out.append(s[i]); i += 1 }
        return .atom(out)
    }
}

private struct Node {
    var head: String
    var args: [Arg]
    enum Arg { case node(Node); case atom(String); case text(String) }
}

private struct Parser {
    var lexer: Lexer
    var peeked: Token?

    init(_ text: String) { lexer = Lexer(text) }

    mutating func take() throws -> Token? {
        if let p = peeked { peeked = nil; return p }
        return try lexer.next()
    }
    mutating func peek() throws -> Token? {
        if peeked == nil { peeked = try lexer.next() }
        return peeked
    }

    mutating func parseNode() throws -> Node {
        guard let t = try take() else { throw ExprError.invalid("unexpected end of expression text") }
        guard t == .open else { throw ExprError.invalid("expected '(' in expression text") }
        guard case .atom(let head)? = try take() else { throw ExprError.invalid("expected an operator name after '('") }
        var args: [Node.Arg] = []
        while let n = try peek() {
            if n == .close { _ = try take(); return Node(head: head, args: args) }
            if n == .open { args.append(.node(try parseNode())); continue }
            _ = try take()
            switch n {
            case .atom(let a): args.append(.atom(a))
            case .text(let s): args.append(.text(s))
            default: break
            }
        }
        throw ExprError.invalid("missing ')' in expression text")
    }
}

private func buildExpr(_ n: Node) throws -> Expr {
    func sub(_ i: Int) throws -> Expr {
        guard i < n.args.count, case .node(let m) = n.args[i] else {
            throw ExprError.invalid("\(n.head) expects a nested expression at position \(i)")
        }
        return try buildExpr(m)
    }
    func str(_ i: Int) throws -> String {
        guard i < n.args.count else { throw ExprError.invalid("\(n.head) is missing an argument") }
        switch n.args[i] {
        case .text(let s): return s
        case .atom(let s): return s
        case .node: throw ExprError.invalid("\(n.head) expects a literal at position \(i)")
        }
    }
    func allSubs(from k: Int) throws -> [Expr] {
        var out: [Expr] = []
        for a in n.args.dropFirst(k) {
            guard case .node(let m) = a else { throw ExprError.invalid("\(n.head) expects nested expressions") }
            out.append(try buildExpr(m))
        }
        return out
    }

    switch n.head {
    case "col": return .column(try str(0))
    case "int":
        guard let v = Int64(try str(0)) else { throw ExprError.invalid("bad integer literal") }
        return .int(v)
    case "float":
        guard let v = Double(try str(0)) else { throw ExprError.invalid("bad float literal") }
        return .double(v)
    case "bool": return .bool(try str(0) == "true")
    case "str": return .string(try str(0))
    case "null":
        guard let t = ExprType.fromToken(try str(0)) else { throw ExprError.invalid("unknown type in (null ...)") }
        return .nullLiteral(t)
    case "cast":
        guard let t = ExprType.fromToken(try str(1)) else { throw ExprError.invalid("unknown cast target type") }
        return .cast(try sub(0), t)
    case "if_else": return .ifElse(try sub(0), try sub(1), try sub(2))
    case "coalesce": return .coalesce(try allSubs(from: 0))
    case "fill_null": return .fillNull(try sub(0), try sub(1))
    case "is_null": return .isNull(try sub(0))
    case "is_valid": return .isValid(try sub(0))
    case "is_in": return .isIn(try sub(0), try allSubs(from: 1))
    case "str_eq", "starts_with", "contains":
        return .stringMatch(ExprStringPredicate(rawValue: n.head)!, try sub(0), try str(1))
    default:
        if let t = ExprType.fromToken(n.head) {
            let text = try str(0)
            if t.isFloat { guard let v = Double(text) else { throw ExprError.invalid("bad \(n.head) literal") }
                           return .typedDouble(v, t) }
            if t == .boolean { return .bool(text == "true") }
            if t == .utf8 { return .string(text) }
            guard let v = Int64(text) else { throw ExprError.invalid("bad \(n.head) literal") }
            return .typedInt(v, t)
        }
        if let op = ExprBinaryOp(rawValue: n.head) { return .binary(op, try sub(0), try sub(1)) }
        if let op = ExprUnaryOp(rawValue: n.head) { return .unary(op, try sub(0)) }
        throw ExprError.invalid("unknown expression operator \"\(n.head)\"")
    }
}

extension Expr {
    /// Parses the s-expression form produced by `description`.
    public init(text: String) throws {
        var p = Parser(text)
        let n = try p.parseNode()
        self = try buildExpr(n)
    }
}

extension ExprQuery {
    /// Parses the serialised query form:
    ///
    ///     (query (filter PRED)? (group_by KEYCOUNT "keyname" KEYEXPR)?
    ///            ( (project (as "name" EXPR)...) | (aggregate (sum "name" EXPR) | (count "name")...) ))
    public init(text: String) throws {
        var p = Parser(text)
        let root = try p.parseNode()
        guard root.head == "query" else { throw ExprError.invalid("a query must start with (query ...)") }
        var filter: Expr? = nil, groupKey: Expr? = nil, keyCount = 0, keyName = "key"
        var terminal: ExprQuery.Terminal? = nil
        for a in root.args {
            guard case .node(let n) = a else { throw ExprError.invalid("unexpected literal in (query ...)") }
            switch n.head {
            case "filter":
                guard case .node(let e)? = n.args.first else { throw ExprError.invalid("(filter ...) needs an expression") }
                filter = try buildExpr(e)
            case "group_by":
                guard n.args.count >= 3, case .atom(let kc) = n.args[0], let k = Int(kc) else {
                    throw ExprError.invalid("(group_by KEYCOUNT \"name\" KEYEXPR) is malformed")
                }
                keyCount = k
                if case .text(let name) = n.args[1] { keyName = name }
                guard case .node(let e) = n.args[2] else { throw ExprError.invalid("(group_by ...) needs a key expression") }
                groupKey = try buildExpr(e)
            case "project":
                var ps: [ExprQuery.Projection] = []
                for a2 in n.args {
                    guard case .node(let m) = a2, m.head == "as", m.args.count >= 2,
                          case .text(let name) = m.args[0], case .node(let e) = m.args[1] else {
                        throw ExprError.invalid("(project (as \"name\" EXPR) ...) is malformed")
                    }
                    ps.append(ExprQuery.Projection(name: name, expr: try buildExpr(e)))
                }
                terminal = .project(ps)
            case "aggregate":
                var aggs: [ExprAggregate] = []
                for a2 in n.args {
                    guard case .node(let m) = a2, let op = ExprAggregate.Op(rawValue: m.head), !m.args.isEmpty,
                          case .text(let name) = m.args[0] else {
                        throw ExprError.invalid("(aggregate (sum \"name\" EXPR) ...) is malformed")
                    }
                    var e: Expr? = nil
                    if m.args.count > 1, case .node(let en) = m.args[1] { e = try buildExpr(en) }
                    aggs.append(ExprAggregate(op, e, name: name))
                }
                terminal = .aggregate(aggs)
            default:
                throw ExprError.invalid("unknown clause \"\(n.head)\" in (query ...)")
            }
        }
        guard let t = terminal else { throw ExprError.invalid("a query needs a (project ...) or (aggregate ...) terminal") }
        self.init(filter: filter, groupKey: groupKey, keyCount: keyCount, keyName: keyName, terminal: t)
    }
}
