package org.arrowmetal.spark

import java.lang.foreign.{Arena, MemorySegment, ValueLayout}

import scala.collection.mutable
import scala.collection.mutable.ArrayBuffer

import org.apache.spark.rdd.RDD
import org.apache.spark.sql.catalyst.InternalRow
import org.apache.spark.sql.catalyst.expressions._
import org.apache.spark.sql.catalyst.rules.Rule
import org.apache.spark.sql.execution.{ColumnarToRowExec, FilterExec, ProjectExec, SparkPlan, UnaryExecNode}
import org.apache.spark.sql.execution.aggregate.HashAggregateExec
import org.apache.spark.sql.execution.metric.{SQLMetric, SQLMetrics}
import org.apache.spark.sql.internal.SQLConf
import org.apache.spark.sql.types._
import org.apache.spark.sql.vectorized.ColumnarBatch
import org.apache.spark.unsafe.types.UTF8String

/**
 * Finds the operators Spark runs directly above a columnar scan (any chain of FilterExec and ProjectExec,
 * optionally topped by the partial HashAggregateExec) and replaces the chain with one MetalStageExec when
 * every expression in it translates (SparkToArrowMetal). Runs after Spark has inserted ColumnarToRowExec,
 * which the stage removes: it takes the scan's column batches itself.
 */
object MetalStageRule extends Rule[SparkPlan] {
  private def conf(k: String, d: String) = SQLConf.get.getConfString(k, d)

  override def apply(plan: SparkPlan): SparkPlan = {
    if (!conf("spark.arrowmetal.enabled", "true").toBoolean || !conf("spark.arrowmetal.stage.enabled", "true").toBoolean) return plan
    val ansi = SQLConf.get.ansiEnabled
    val rowsToo = conf("spark.arrowmetal.stage.rows", "true").toBoolean
    plan.transformDown {
      case agg: HashAggregateExec if MetalStage.build(agg, ansi).isDefined => MetalStage.build(agg, ansi).get
      case p @ (_: FilterExec | _: ProjectExec) if rowsToo && MetalStage.build(p, ansi).isDefined => MetalStage.build(p, ansi).get
    }
  }
}

/** One GPU stage: `json` runs over source "t" made of the scan columns `inputs` (scan ordinal, name, type). */
case class MetalStageExec(output: Seq[Attribute], child: SparkPlan, json: String,
                          inputs: Seq[(Int, String, DataType)], emptyRow: Option[Seq[Any]],
                          nullAs: Seq[Option[Any]], passthrough: Map[Int, Int], label: String) extends UnaryExecNode {
  override protected def withNewChildInternal(newChild: SparkPlan): MetalStageExec = copy(child = newChild)
  override def nodeName: String = "MetalStage"
  override def simpleString(maxFields: Int): String = s"MetalStage [$label] ${output.mkString("[", ", ", "]")}"

  override lazy val metrics: Map[String, SQLMetric] = Map(
    "inputRows" -> SQLMetrics.createMetric(sparkContext, "rows into the GPU"),
    "outputRows" -> SQLMetrics.createMetric(sparkContext, "number of output rows"),
    "copyTime" -> SQLMetrics.createNanoTimingMetric(sparkContext, "column copy time"),
    "gpuTime" -> SQLMetrics.createNanoTimingMetric(sparkContext, "GPU plan time"))

  override protected def doExecute(): RDD[InternalRow] = {
    val (j, ins, out, empty, na, pt, m) = (json, inputs, output.map(_.dataType), emptyRow, nullAs, passthrough, metrics)
    child.executeColumnar().mapPartitions(batches => MetalStage.run(batches, j, ins, out, empty, na, pt, m))
  }
}

object MetalStage {
  private def supported(dt: DataType): Boolean = dt match {
    case ByteType | ShortType | IntegerType | LongType | FloatType | DoubleType | BooleanType | StringType |
         DateType | TimestampType | TimestampNTZType => true
    case _ => false
  }

  /** The Filter/Project chain under `top` down to ColumnarToRowExec(scan), bottom first, and the scan. */
  private def chain(top: SparkPlan): Option[(List[SparkPlan], SparkPlan)] = top match {
    case ColumnarToRowExec(scan) if scan.supportsColumnar => Some((Nil, scan))
    case f: FilterExec => chain(f.child).map { case (ns, s) => (ns :+ f, s) }
    case p: ProjectExec => chain(p.child).map { case (ns, s) => (ns :+ p, s) }
    case _ => None
  }

  private def q(s: String) = SparkToArrowMetal.quote(s)

  def build(top: SparkPlan, ansi: Boolean): Option[MetalStageExec] = {
    val under = top match {
      case agg: HashAggregateExec => agg.child
      case other => other
    }
    val (chainNodes, scan) = chain(under).getOrElse(return None)
    if (!top.isInstanceOf[HashAggregateExec] && chainNodes.isEmpty) return None
    // Scan attributes become columns c0, c1, ...; only the referenced ones are imported.
    val used = mutable.LinkedHashMap[Int, (String, DataType)]()
    val env = new mutable.HashMap[ExprId, String]()
    val byId = scan.output.zipWithIndex.map { case (a, i) => a.exprId -> (i, a.dataType) }.toMap
    def scanCol(k: ExprId): Option[String] = byId.get(k).flatMap { case (i, dt) =>
      if (!supported(dt)) None
      else { used.getOrElseUpdate(i, (s"c$i", dt)); Some(s"(col ${q(s"c$i")})") }
    }
    val resolver: ExprId => Option[String] = k => env.get(k).orElse(scanCol(k))
    val preds = ArrayBuffer[String]()
    val passthrough = mutable.LinkedHashMap[Int, Int]()
    for (n <- chainNodes) n match {
      case f: FilterExec =>
        preds += new SparkToArrowMetal(resolver, ansi)(f.condition).getOrElse(return None)
      case p: ProjectExec =>
        val next = p.projectList.map {
          case a: AttributeReference => a.exprId -> resolver(a.exprId)
          case al: Alias => al.exprId -> new SparkToArrowMetal(resolver, ansi)(al.child)
          case _ => return None
        }
        if (next.exists(_._2.isEmpty)) return None
        env.clear()
        next.foreach { case (id, s) => env(id) = s.get }
    }
    val tr = new SparkToArrowMetal(resolver, ansi)
    var input = """{"op":"scan","source":"t"}"""
    if (preds.nonEmpty) {
      val p = preds.reduce((a, b) => s"(and_kleene $a $b)")
      input = s"""{"op":"filter","predicate":${q(p)},"input":$input}"""
    }
    val (json, emptyRow, nullAs, label) = top match {
      case agg: HashAggregateExec =>
        if (agg.aggregateExpressions.isEmpty) return None
        if (!agg.aggregateExpressions.forall(e => e.mode == org.apache.spark.sql.catalyst.expressions.aggregate.Partial &&
            !e.isDistinct && e.filter.isEmpty)) return None
        if (!agg.groupingExpressions.forall(g => supported(g.dataType))) return None
        // several keys are folded into one inside the engine, which does not take a string among them
        if (agg.groupingExpressions.size > 1 && agg.groupingExpressions.exists(_.dataType == StringType)) return None
        val keys = agg.groupingExpressions.zipWithIndex.map { case (g, i) =>
          val e = g match { case al: Alias => al.child; case other => other }
          val t = SparkToArrowMetal.amType(g.dataType).getOrElse(return None)
          val x = tr(e).getOrElse(return None)
          // a plain column goes in as itself: the engine groups a utf8 column but not a utf8 expression
          val key = if (e.isInstanceOf[AttributeReference]) x else if (g.dataType == StringType) return None else s"(cast $x $t)"
          s"""[${q(s"k$i")},${q(key)}]"""
        }
        val slots = agg.aggregateExpressions.map(ae => PartialAggs.slots(ae.aggregateFunction, x => tr(x), ansi).getOrElse(return None)).flatten
        if (agg.output.size != keys.size + slots.size) return None
        if (!agg.output.forall(a => supported(a.dataType))) return None
        val aggs = slots.zipWithIndex.map { case (s, i) =>
          s.input match {
            case Some(x) => s"""[${q(s.op)},${q(s"a$i")},${q(x)}]"""
            case None => s"""[${q(s.op)},${q(s"a$i")}]"""
          }
        }
        if (keys.nonEmpty) {
          (s"""{"op":"group_by","keys":[${keys.mkString(",")}],"aggs":[${aggs.mkString(",")}],"input":$input}""",
            None, Seq.fill(keys.size)(None) ++ slots.map(_.nullAs), s"partial aggregate, ${keys.size} key(s)")
        } else {
          // A partial aggregate without keys emits Spark's initial buffer even for an empty partition.
          (s"""{"op":"aggregate","aggs":[${aggs.mkString(",")}],"input":$input}""",
            Some(slots.map(_.nullAs.orNull)), slots.map(_.nullAs), "partial aggregate, no keys")
        }
      case _ =>
        if (!top.output.forall(a => supported(a.dataType))) return None
        // The plan engine does not materialise string outputs: a string output that is a scan column is
        // carried on the JVM side and joined back by row number (__row); a computed one stays on Spark.
        val ColRef = """\(col "c(\d+)"\)""".r
        val exprs = top.output.zipWithIndex.map { case (a, i) =>
          val t = SparkToArrowMetal.amType(a.dataType).getOrElse(return None)
          val x = resolver(a.exprId).getOrElse(return None)
          if (a.dataType == StringType) x match {
            case ColRef(c) => passthrough(i) = c.toInt; s"""[${q(s"o$i")},${q("(col \"__row\")")}]"""
            case _ => return None
          } else s"""[${q(s"o$i")},${q(s"(cast $x $t)")}]"""
        }
        (s"""{"op":"select","exprs":[${exprs.mkString(",")}],"input":$input}""", None,
          Seq.fill(exprs.size)(None), s"${chainNodes.map(_.nodeName.toLowerCase).mkString(" + ")}")
    }
    // Import only the scan columns the plan text names, plus the row number when strings ride alongside.
    val named = """col \\"c(\d+)\\"""".r.findAllMatchIn(json).map(_.group(1).toInt).toSet
    val inputs0 = used.toSeq.filter { case (i, _) => named.contains(i) }.map { case (i, (n, dt)) => (i, n, dt) }
    var inputs = if (passthrough.nonEmpty) inputs0 :+ ((-1, "__row", LongType)) else inputs0
    // Every plan needs one column to know the row count; take the first scan column when none is used.
    if (inputs.isEmpty) {
      val firstOk = scan.output.zipWithIndex.find { case (a, _) => supported(a.dataType) && a.dataType != StringType }
        .getOrElse(return None)
      inputs = Seq((firstOk._2, s"c${firstOk._2}", firstOk._1.dataType))
    }
    Some(MetalStageExec(top.output, scan, json, inputs, emptyRow, nullAs, passthrough.toMap, label))
  }

  /** One input column gathered from a partition's batches. */
  private final class Gather(dt: DataType) {
    private val chunks = new ArrayBuffer[AnyRef]
    private val nullRows = new ArrayBuffer[Long]
    private var bytes = 0L
    var n = 0L
    def add(b: ColumnarBatch, idx: Int): Unit = {
      val v = b.column(idx)
      val rows = b.numRows()
      dt match {
        case ByteType => chunks += v.getBytes(0, rows)
        case ShortType => chunks += v.getShorts(0, rows)
        case IntegerType | DateType => chunks += v.getInts(0, rows)
        case LongType | TimestampType | TimestampNTZType => chunks += v.getLongs(0, rows)
        case FloatType => chunks += v.getFloats(0, rows)
        case DoubleType => chunks += v.getDoubles(0, rows)
        case BooleanType => chunks += v.getBooleans(0, rows)
        case StringType =>
          val strs = new Array[Array[Byte]](rows)
          var i = 0
          while (i < rows) {
            if (!v.isNullAt(i)) { val s = v.getUTF8String(i).getBytes; strs(i) = s; bytes += s.length }
            i += 1
          }
          chunks += strs
      }
      if (v.hasNull) { var i = 0; while (i < rows) { if (v.isNullAt(i)) nullRows += n + i; i += 1 } }
      n += rows
    }

    def importInto(arena: Arena): MemorySegment = {
      var validity = MemorySegment.NULL
      if (nullRows.nonEmpty) {
        validity = arena.allocate(math.max(8L, ((n + 63) / 64) * 8), 64)
        validity.fill(0xFF.toByte)
        nullRows.foreach { r =>
          val b = r >>> 3
          validity.set(ValueLayout.JAVA_BYTE, b, (validity.get(ValueLayout.JAVA_BYTE, b) & ~(1 << (r & 7)).toInt).toByte)
        }
      }
      val nulls = nullRows.size.toLong
      dt match {
        case StringType =>
          if (bytes > Int.MaxValue) throw new IllegalStateException(
            s"MetalStage: $bytes bytes of strings in one partition do not fit 32-bit offsets; use more partitions")
          val offs = arena.allocate(4L * (n + 1), 64)
          val data = arena.allocate(math.max(1L, bytes), 64)
          var row = 0L
          var at = 0L
          chunks.foreach { c =>
            c.asInstanceOf[Array[Array[Byte]]].foreach { s =>
              offs.setAtIndex(ValueLayout.JAVA_INT, row, at.toInt)
              if (s != null) { MemorySegment.copy(s, 0, data, ValueLayout.JAVA_BYTE, at, s.length); at += s.length }
              row += 1
            }
          }
          offs.setAtIndex(ValueLayout.JAVA_INT, row, at.toInt)
          chunks.clear()
          ArrowMetalNative.importColumn(arena, "u", offs, data, validity, n, nulls)
        case BooleanType =>
          val bits = arena.allocate(math.max(8L, ((n + 63) / 64) * 8), 64)
          var row = 0L
          chunks.foreach { c =>
            c.asInstanceOf[Array[Boolean]].foreach { x =>
              if (x) bits.set(ValueLayout.JAVA_BYTE, row >>> 3, (bits.get(ValueLayout.JAVA_BYTE, row >>> 3) | (1 << (row & 7)).toInt).toByte)
              row += 1
            }
          }
          chunks.clear()
          ArrowMetalNative.importColumn(arena, "b", bits, validity, n, nulls)
        case _ =>
          val (w, fmt) = dt match {
            case ByteType => (1, "c")
            case ShortType => (2, "s")
            case IntegerType | DateType => (4, "i")
            case LongType | TimestampType | TimestampNTZType => (8, "l")
            case FloatType => (4, "f")
            case DoubleType => (8, "g")
          }
          val values = arena.allocate(math.max(8L, w * n), 64)
          var at = 0L
          chunks.foreach {
            case a: Array[Byte] => MemorySegment.copy(a, 0, values, ValueLayout.JAVA_BYTE, at, a.length); at += a.length
            case a: Array[Short] => MemorySegment.copy(a, 0, values, ValueLayout.JAVA_SHORT, at, a.length); at += 2L * a.length
            case a: Array[Int] => MemorySegment.copy(a, 0, values, ValueLayout.JAVA_INT, at, a.length); at += 4L * a.length
            case a: Array[Long] => MemorySegment.copy(a, 0, values, ValueLayout.JAVA_LONG, at, a.length); at += 8L * a.length
            case a: Array[Float] => MemorySegment.copy(a, 0, values, ValueLayout.JAVA_FLOAT, at, a.length); at += 4L * a.length
            case a: Array[Double] => MemorySegment.copy(a, 0, values, ValueLayout.JAVA_DOUBLE, at, a.length); at += 8L * a.length
          }
          chunks.clear()
          ArrowMetalNative.importColumn(arena, fmt, values, validity, n, nulls)
      }
    }
  }

  private def set(row: SpecificInternalRow, j: Int, dt: DataType, c: ArrowMetalNative.Column, g: Int,
                  nullAs: Option[Any]): Unit = {
    if (c.isNull(g)) {
      nullAs match {
        case Some(v) => setValue(row, j, dt, v)
        case None => row.setNullAt(j)
      }
      return
    }
    def asLong: Long = if (c.longs != null) c.longs(g) else if (c.ints != null) c.ints(g).toLong
      else if (c.doubles != null) c.doubles(g).toLong else c.floats(g).toLong
    def asDouble: Double = if (c.doubles != null) c.doubles(g) else if (c.floats != null) c.floats(g).toDouble
      else if (c.longs != null) c.longs(g).toDouble else c.ints(g).toDouble
    dt match {
      case ByteType => row.setByte(j, asLong.toByte)
      case ShortType => row.setShort(j, asLong.toShort)
      case IntegerType | DateType => row.setInt(j, asLong.toInt)
      case LongType | TimestampType | TimestampNTZType => row.setLong(j, asLong)
      case DoubleType => row.setDouble(j, asDouble)
      case FloatType => row.setFloat(j, if (c.floats != null) c.floats(g) else asDouble.toFloat)
      case BooleanType => row.setBoolean(j, c.booleans(g))
      case StringType => row.update(j, UTF8String.fromBytes(c.utf8, c.offsets(g), c.offsets(g + 1) - c.offsets(g)))
    }
  }

  private def setValue(row: SpecificInternalRow, j: Int, dt: DataType, v: Any): Unit = (dt, v) match {
    case (_, null) => row.setNullAt(j)
    case (LongType, x: Long) => row.setLong(j, x)
    case (DoubleType, x: Double) => row.setDouble(j, x)
    case (_, x) => row.update(j, x)
  }

  def run(batches: Iterator[ColumnarBatch], json: String, inputs: Seq[(Int, String, DataType)],
          outTypes: Seq[DataType], emptyRow: Option[Seq[Any]], nullAs: Seq[Option[Any]],
          passthrough: Map[Int, Int], m: Map[String, SQLMetric]): Iterator[InternalRow] = {
    val t0 = System.nanoTime()
    val real = inputs.filter(_._1 >= 0)
    val gathers = real.map { case (_, _, dt) => new Gather(dt) }
    val ptCols = passthrough.values.toSeq.distinct
    val strings = ptCols.map(c => c -> new ArrayBuffer[UTF8String]).toMap
    var n = 0L
    while (batches.hasNext) {
      val b = batches.next()
      var k = 0
      while (k < real.size) { gathers(k).add(b, real(k)._1); k += 1 }
      ptCols.foreach { c =>
        val v = b.column(c); val buf = strings(c); var i = 0
        while (i < b.numRows()) { buf += (if (v.isNullAt(i)) null else v.getUTF8String(i).clone()); i += 1 }
      }
      n += b.numRows()
    }
    if (n > Int.MaxValue) throw new IllegalStateException(s"MetalStage: $n rows in one partition; use more partitions")
    val proj = UnsafeProjection.create(outTypes.toArray)
    val row = new SpecificInternalRow(outTypes)
    if (n == 0) {
      return emptyRow match {
        case Some(vals) =>
          vals.zipWithIndex.foreach { case (v, j) => setValue(row, j, outTypes(j), v) }
          m("outputRows") += 1
          Iterator.single(proj(row).copy())
        case None => Iterator.empty
      }
    }
    val arena = Arena.ofConfined()
    var cols: Array[ArrowMetalNative.Column] = null
    val t1 = System.nanoTime()
    var t2 = 0L
    try {
      val handles = inputs.map { case (i, _, _) =>
        if (i >= 0) gathers(real.indexWhere(_._1 == i)).importInto(arena)
        else {
          val ids = arena.allocate(8L * n, 64)
          var r = 0L
          while (r < n) { ids.setAtIndex(ValueLayout.JAVA_LONG, r, r); r += 1 }
          ArrowMetalNative.importColumn(arena, "l", ids, MemorySegment.NULL, n, 0L)
        }
      }
      t2 = System.nanoTime()
      try cols = ArrowMetalNative.runPlan(json, handles.toArray, inputs.map(_._2).toArray)
      finally handles.foreach(ArrowMetalNative.release)
    } finally arena.close()
    val t3 = System.nanoTime()
    m("copyTime") += t2 - t0
    m("gpuTime") += t3 - t2
    m("inputRows") += n
    if (cols.length != outTypes.size)
      throw new IllegalStateException(s"MetalStage: the plan returned ${cols.length} columns, Spark expects ${outTypes.size}")
    val rows = if (cols.isEmpty) 0 else cols(0).length
    m("outputRows") += rows
    Iterator.range(0, rows).map { g =>
      var j = 0
      while (j < outTypes.size) {
        passthrough.get(j) match {
          case Some(c) =>
            val v = strings(c)(cols(j).longs(g).toInt)
            if (v == null) row.setNullAt(j) else row.update(j, v)
          case None => set(row, j, outTypes(j), cols(j), g, nullAs(j))
        }
        j += 1
      }
      proj(row)
    }
  }
}
