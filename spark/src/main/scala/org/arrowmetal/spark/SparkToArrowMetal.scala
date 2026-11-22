package org.arrowmetal.spark

import org.apache.spark.sql.catalyst.expressions._
import org.apache.spark.sql.catalyst.expressions.aggregate._
import org.apache.spark.sql.types._
import org.apache.spark.unsafe.types.UTF8String

/**
 * Spark (Catalyst) expressions to ArrowMetal's expression grammar (include/arrowmetal.h, am_query), with
 * Spark's semantics where they differ from Arrow's. Anything that cannot be written with exactly Spark's
 * answer returns None, and the operator stays on Spark:
 *
 *  - Float comparisons follow Spark: NaN equals NaN and is above every other value; null in, null out.
 *  - AND / OR are three-valued (Kleene), as in SQL.
 *  - IF / CASE WHEN take the ELSE branch when the condition is null; Arrow's if_else would return null.
 *  - x IN (...) is null when x is null; Arrow's is_in is false.
 *  - Integer + - * and unary minus only with spark.sql.ansi.enabled=false: ANSI raises on overflow, the
 *    GPU wraps. Division only by a non-zero literal: Spark returns null or raises on zero, the GPU does not.
 *  - Casts only where every value converts exactly (widening).
 */
object SparkToArrowMetal {
  def quote(s: String): String =
    "\"" + s.replace("\\", "\\\\").replace("\"", "\\\"").replace("\n", "\\n").replace("\t", "\\t") + "\""

  /** The grammar's TYPE for a Spark type, as the column is stored on the GPU. */
  def amType(dt: DataType): Option[String] = dt match {
    case ByteType => Some("i8")
    case ShortType => Some("i16")
    case IntegerType | DateType => Some("i32")
    case LongType | TimestampType | TimestampNTZType => Some("i64")
    case FloatType => Some("f32")
    case DoubleType => Some("f64")
    case BooleanType => Some("bool")
    case StringType => Some("str")
    case _ => None
  }

  def isFloat(dt: DataType): Boolean = dt == DoubleType || dt == FloatType
  def isIntegral(dt: DataType): Boolean = dt match {
    case ByteType | ShortType | IntegerType | LongType => true
    case _ => false
  }

  private def plainDouble(d: Double): String = new java.math.BigDecimal(d).toPlainString

  /** Widening casts: every value of `from` converts exactly to `to`. */
  private def widening(from: DataType, to: DataType): Boolean = (from, to) match {
    case (a, b) if a == b => true
    case (ByteType, ShortType | IntegerType | LongType | FloatType | DoubleType) => true
    case (ShortType, IntegerType | LongType | FloatType | DoubleType) => true
    case (IntegerType, LongType | DoubleType) => true
    case (FloatType, DoubleType) => true
    case _ => false
  }

  private def likeToMatch(pattern: String, escape: Char): Option[(String, String)] = {
    if (pattern.exists(c => c == '_' || c == escape)) return None
    val core = pattern.stripPrefix("%").stripSuffix("%")
    if (core.contains("%")) return None
    (pattern.startsWith("%") && pattern.length > 1, pattern.endsWith("%")) match {
      case (false, false) => Some(("str_eq", core))
      case (false, true) => Some(("starts_with", core))
      case (true, true) => Some(("contains", core))
      case _ => None                                   // '%abc' (ends_with) is not in the grammar
    }
  }
}

/** Translates expressions over attributes that `env` resolves to grammar text. */
class SparkToArrowMetal(env: ExprId => Option[String], ansi: Boolean) {
  import SparkToArrowMetal._

  def apply(e: Expression): Option[String] = tr(e)

  private def bin(op: String, l: Expression, r: Expression): Option[String] =
    for (a <- tr(l); b <- tr(r)) yield s"($op $a $b)"

  private def un(op: String, c: Expression): Option[String] = tr(c).map(a => s"($op $a)")

  private def cmp(op: String, l: Expression, r: Expression): Option[String] = {
    val dt = l.dataType
    if (dt != r.dataType) return None
    if (dt == StringType) {
      // equality with a literal only; string ordering is not in the grammar
      (l, r, op) match {
        case (_, Literal(v: UTF8String, _), "eq") => tr(l).map(a => s"(str_eq $a ${quote(v.toString)})")
        case (Literal(v: UTF8String, _), _, "eq") => tr(r).map(a => s"(str_eq $a ${quote(v.toString)})")
        case _ => None
      }
    } else if (isFloat(dt)) {
      for (a <- tr(l); b <- tr(r)) yield {
        def eq = s"(or (and (is_nan $a) (is_nan $b)) (eq $a $b))"
        def gt(x: String, y: String) = s"(or (and (is_nan $x) (not (is_nan $y))) (gt $x $y))"
        def ge(x: String, y: String) = s"(or (is_nan $x) (ge $x $y))"
        val body = op match {
          case "eq" => eq
          case "gt" => gt(a, b)
          case "ge" => ge(a, b)
          case "lt" => gt(b, a)
          case "le" => ge(b, a)
        }
        s"(if_else (or (is_null $a) (is_null $b)) (null bool) $body)"
      }
    } else if (isIntegral(dt) || dt == DateType || dt == TimestampType || dt == TimestampNTZType || dt == BooleanType) {
      bin(op, l, r)
    } else None
  }

  private def arith(op: String, e: BinaryArithmetic): Option[String] = {
    val dt = e.dataType
    if (isFloat(dt) || (isIntegral(dt) && !ansi)) bin(op, e.left, e.right) else None
  }

  private def tr(e: Expression): Option[String] = e match {
    case a: AttributeReference => env(a.exprId)
    case Literal(null, dt) => amType(dt).map(t => s"(null $t)")
    case Literal(v, dt) => dt match {
      case ByteType => Some(s"(i8 $v)")
      case ShortType => Some(s"(i16 $v)")
      case IntegerType | DateType => Some(s"(i32 $v)")
      case LongType | TimestampType | TimestampNTZType => Some(s"(i64 $v)")
      case DoubleType =>
        val d = v.asInstanceOf[Double]
        if (d.isNaN || d.isInfinite) None else Some(s"(f64 ${plainDouble(d)})")
      case FloatType =>
        val f = v.asInstanceOf[Float]
        if (f.isNaN || f.isInfinite) None else Some(s"(f32 ${plainDouble(f.toDouble)})")
      case BooleanType => Some(s"(bool $v)")
      case StringType => Some(s"(str ${quote(v.toString)})")
      case _ => None
    }
    case And(l, r) => bin("and_kleene", l, r)
    case Or(l, r) => bin("or_kleene", l, r)
    case Not(c) => un("not", c)
    case IsNull(c) => tr(c).map(a => s"(is_null $a)")
    case IsNotNull(c) => tr(c).map(a => s"(is_valid $a)")
    case EqualTo(l, r) => cmp("eq", l, r)
    case LessThan(l, r) => cmp("lt", l, r)
    case LessThanOrEqual(l, r) => cmp("le", l, r)
    case GreaterThan(l, r) => cmp("gt", l, r)
    case GreaterThanOrEqual(l, r) => cmp("ge", l, r)
    case a: Add => arith("add", a)
    case s: Subtract => arith("sub", s)
    case m: Multiply => arith("mul", m)
    case d: Divide if d.dataType == DoubleType && d.right.foldable && {
      val v = d.right.eval(); v != null && v.asInstanceOf[Double] != 0.0 && !v.asInstanceOf[Double].isNaN
    } => bin("div", d.left, d.right)
    case u: UnaryMinus if isFloat(u.dataType) || (isIntegral(u.dataType) && !ansi) => un("negate", u.child)
    case a: Abs if isFloat(a.dataType) || (isIntegral(a.dataType) && !ansi) => un("abs", a.child)
    case c: Cast if widening(c.child.dataType, c.dataType) =>
      if (c.child.dataType == c.dataType) tr(c.child)
      else for (x <- tr(c.child); t <- amType(c.dataType)) yield s"(cast $x $t)"
    case Coalesce(cs) if cs.nonEmpty =>
      val xs = cs.map(tr)
      if (xs.forall(_.isDefined)) Some(s"(coalesce ${xs.flatten.mkString(" ")})") else None
    case If(p, a, b) =>
      for (c <- tr(p); x <- tr(a); y <- tr(b)) yield s"(if_else (fill_null $c (bool false)) $x $y)"
    case CaseWhen(branches, elseValue) =>
      val otherwise = elseValue match {
        case Some(v) => tr(v)
        case None => amType(e.dataType).map(t => s"(null $t)")
      }
      branches.foldRight(otherwise) { case ((cond, value), acc) =>
        for (c <- tr(cond); v <- tr(value); rest <- acc) yield s"(if_else (fill_null $c (bool false)) $v $rest)"
      }
    case In(v, list) if v.dataType == StringType && list.nonEmpty &&
        list.forall(x => x.isInstanceOf[Literal] && x.asInstanceOf[Literal].value != null) =>
      // is_in takes no strings: an OR of equalities, which is null when v is null, as IN is
      tr(v).map(x => list.map(l => s"(str_eq $x ${quote(l.asInstanceOf[Literal].value.toString)})")
        .reduce((a, b) => s"(or_kleene $a $b)"))
    case In(v, list) if (isIntegral(v.dataType) || v.dataType == DateType) &&
        list.nonEmpty && list.forall(x => x.isInstanceOf[Literal] && x.asInstanceOf[Literal].value != null) =>
      val lits = list.map(tr)
      for (x <- tr(v) if lits.forall(_.isDefined)) yield
        s"(if_else (is_null $x) (null bool) (is_in $x ${lits.flatten.mkString(" ")}))"
    case s: StartsWith => s.right match {
      case Literal(p: UTF8String, StringType) => tr(s.left).map(a => s"(starts_with $a ${quote(p.toString)})")
      case _ => None
    }
    case c: Contains => c.right match {
      case Literal(p: UTF8String, StringType) => tr(c.left).map(a => s"(contains $a ${quote(p.toString)})")
      case _ => None
    }
    case l: Like => l.right match {
      case Literal(p: UTF8String, StringType) =>
        likeToMatch(p.toString, l.escapeChar).flatMap { case (op, lit) =>
          tr(l.left).map(a => s"($op $a ${quote(lit)})")
        }
      case _ => None
    }
    case _ => None
  }
}

/** A partial aggregate as ArrowMetal plan aggregates: (op, input) pairs, one per Spark buffer slot. */
object PartialAggs {
  /** For each buffer slot: the plan aggregate op, its input expression (None: count rows) and the value
   *  that stands for "no row" (Spark's initial buffer value) when the GPU answers null. */
  case class Slot(op: String, input: Option[String], nullAs: Option[Any])

  def slots(f: AggregateFunction, tr: Expression => Option[String], ansi: Boolean): Option[Seq[Slot]] = f match {
    case s: Sum if s.aggBufferAttributes.size == 1 && (SparkToArrowMetal.isFloat(s.child.dataType) ||
        (SparkToArrowMetal.isIntegral(s.child.dataType) && (s.child.dataType != LongType || !ansi))) =>
      val t = if (SparkToArrowMetal.isFloat(s.child.dataType)) "f64" else "i64"
      tr(s.child).map(x => Seq(Slot("sum", Some(s"(cast $x $t)"), None)))
    case c: Count if c.children.size == 1 && c.children.head.foldable && c.children.head.eval() != null =>
      Some(Seq(Slot("count", None, Some(0L))))
    case c: Count if c.children.size == 1 =>
      tr(c.children.head).map(x => Seq(Slot("count", Some(x), Some(0L))))
    case m: Min if SparkToArrowMetal.isIntegral(m.child.dataType) || m.child.dataType == DateType =>
      tr(m.child).map(x => Seq(Slot("min", Some(x), None)))
    case m: Max if SparkToArrowMetal.isIntegral(m.child.dataType) || m.child.dataType == DateType =>
      tr(m.child).map(x => Seq(Slot("max", Some(x), None)))
    case a: Average if a.aggBufferAttributes.size == 2 &&
        (SparkToArrowMetal.isFloat(a.child.dataType) || SparkToArrowMetal.isIntegral(a.child.dataType)) =>
      tr(a.child).map(x => Seq(Slot("sum", Some(s"(cast $x f64)"), Some(0.0)), Slot("count", Some(x), Some(0L))))
    case _ => None
  }
}
