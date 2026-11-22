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
    val rowsToo = conf("spark.arrowmetal.stage.rows", "false").toBoolean
    val ungrouped = conf("spark.arrowmetal.stage.ungrouped", "false").toBoolean
    plan.transformDown {
      case agg: HashAggregateExec if (ungrouped || agg.groupingExpressions.nonEmpty) && MetalStage.build(agg, ansi).isDefined => MetalStage.build(agg, ansi).get
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
    val batchRows = math.max(1L, conf.getConfString("spark.arrowmetal.stage.batchRows", "16000000").toLong)
    child.executeColumnar().mapPartitions(batches => MetalStage.run(batches, j, ins, out, empty, na, pt, batchRows, m))
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

  /** One input column of a chunk, copied batch by batch into off-heap memory that grows by doubling. */
  private final class Gather(dt: DataType, arena: Arena) {
    private val (w, fmt) = dt match {
      case ByteType => (1, "c")
      case ShortType => (2, "s")
      case IntegerType | DateType => (4, "i")
      case LongType | TimestampType | TimestampNTZType => (8, "l")
      case FloatType => (4, "f")
      case DoubleType => (8, "g")
      case BooleanType => (0, "b")
      case StringType => (4, "u")
    }
    private var values = arena.allocate(1L << 16, 64)
    private var data = if (dt == StringType) arena.allocate(1L << 16, 64) else null
    private var validity = arena.allocate(1L << 12, 64)          // set bits are valid
    private var nulls = 0L
    var bytes = 0L
    var n = 0L
    def held: Long = values.byteSize() + validity.byteSize() + (if (data == null) 0L else data.byteSize())

    private def grow(seg: MemorySegment, need: Long): MemorySegment =
      if (need <= seg.byteSize()) seg
      else {
        val bigger = arena.allocate(math.max(need, seg.byteSize() * 2), 64)
        MemorySegment.copy(seg, 0L, bigger, 0L, seg.byteSize())
        bigger
      }
    private def setBit(seg: MemorySegment, r: Long): Unit =
      seg.set(ValueLayout.JAVA_BYTE, r >>> 3, (seg.get(ValueLayout.JAVA_BYTE, r >>> 3) | (1 << (r & 7).toInt)).toByte)

    def add(b: ColumnarBatch, idx: Int): Unit = {
      val v = b.column(idx)
      val rows = b.numRows()
      val end = n + rows
      values = grow(values, if (dt == StringType) 4L * (end + 1) else if (dt == BooleanType) (end + 7) / 8 + 8 else w * end)
      validity = grow(validity, (end + 7) / 8 + 8)
      val at = w * n
      dt match {
        case ByteType => MemorySegment.copy(v.getBytes(0, rows), 0, values, ValueLayout.JAVA_BYTE, at, rows)
        case ShortType => MemorySegment.copy(v.getShorts(0, rows), 0, values, ValueLayout.JAVA_SHORT, at, rows)
        case IntegerType | DateType => MemorySegment.copy(v.getInts(0, rows), 0, values, ValueLayout.JAVA_INT, at, rows)
        case LongType | TimestampType | TimestampNTZType => MemorySegment.copy(v.getLongs(0, rows), 0, values, ValueLayout.JAVA_LONG, at, rows)
        case FloatType => MemorySegment.copy(v.getFloats(0, rows), 0, values, ValueLayout.JAVA_FLOAT, at, rows)
        case DoubleType => MemorySegment.copy(v.getDoubles(0, rows), 0, values, ValueLayout.JAVA_DOUBLE, at, rows)
        case BooleanType =>
          val x = v.getBooleans(0, rows)
          var i = 0
          while (i < rows) { if (x(i)) setBit(values, n + i); i += 1 }
        case StringType =>
          var i = 0
          while (i < rows) {
            values.setAtIndex(ValueLayout.JAVA_INT, n + i, bytes.toInt)
            if (!v.isNullAt(i)) {
              val x = v.getUTF8String(i).getBytes
              if (bytes + x.length > Int.MaxValue) throw new IllegalStateException("MetalStage: one chunk's strings do not fit 32-bit offsets; lower spark.arrowmetal.stage.batchRows")
              data = grow(data, bytes + x.length)
              MemorySegment.copy(x, 0, data, ValueLayout.JAVA_BYTE, bytes, x.length)
              bytes += x.length
            }
            i += 1
          }
      }
      var i = 0
      if (v.hasNull) while (i < rows) { if (v.isNullAt(i)) nulls += 1 else setBit(validity, n + i); i += 1 }
      else {
        while (i < rows && ((n + i) & 7) != 0) { setBit(validity, n + i); i += 1 }
        while (i + 8 <= rows) { validity.set(ValueLayout.JAVA_BYTE, (n + i) >>> 3, 0xFF.toByte); i += 8 }
        while (i < rows) { setBit(validity, n + i); i += 1 }
      }
      n = end
    }

    def importInto(): MemorySegment = {
      if (dt == StringType) values.setAtIndex(ValueLayout.JAVA_INT, n, bytes.toInt)
      ArrowMetalNative.importColumn(arena, fmt, values, data, validity, n, nulls)
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

  /** Runs the plan over the partition in chunks of about `batchRows` rows. A chunk of a partial aggregate
   *  emits its own partial groups; Spark's final aggregate merges them as it merges partitions'. */
  def run(batches: Iterator[ColumnarBatch], json: String, inputs: Seq[(Int, String, DataType)],
          outTypes: Seq[DataType], emptyRow: Option[Seq[Any]], nullAs: Seq[Option[Any]],
          passthrough: Map[Int, Int], batchRows: Long, m: Map[String, SQLMetric]): Iterator[InternalRow] = {
    val proj = UnsafeProjection.create(outTypes.toArray)
    val row = new SpecificInternalRow(outTypes)
    val real = inputs.filter(_._1 >= 0)
    val ptCols = passthrough.values.toSeq.distinct
    var total = 0L

    def chunk(): Iterator[InternalRow] = {
      val t0 = System.nanoTime()
      val arena = Arena.ofConfined()
      var cols: Array[ArrowMetalNative.Column] = null
      val strings = ptCols.map(c => c -> new ArrayBuffer[UTF8String]).toMap
      try {
        val gathers = real.map { case (_, _, dt) => new Gather(dt, arena) }
        var n = 0L
        while (batches.hasNext && n < batchRows && gathers.forall(_.bytes < (512L << 20)) && gathers.map(_.held).sum < (1L << 30)) {
          val b = batches.next()
          var k = 0
          while (k < real.size) { gathers(k).add(b, real(k)._1); k += 1 }
          ptCols.foreach { c =>
            val v = b.column(c); val buf = strings(c); var i = 0
            while (i < b.numRows()) { buf += (if (v.isNullAt(i)) null else v.getUTF8String(i).clone()); i += 1 }
          }
          n += b.numRows()
        }
        total += n
        if (n == 0) return Iterator.empty
        val handles = inputs.map { case (i, _, _) =>
          if (i >= 0) gathers(real.indexWhere(_._1 == i)).importInto()
          else {
            val ids = arena.allocate(8L * n, 64)
            var r = 0L
            while (r < n) { ids.setAtIndex(ValueLayout.JAVA_LONG, r, r); r += 1 }
            ArrowMetalNative.importColumn(arena, "l", ids, MemorySegment.NULL, n, 0L)
          }
        }
        val t1 = System.nanoTime()
        try cols = ArrowMetalNative.runPlan(json, handles.toArray, inputs.map(_._2).toArray)
        finally handles.foreach(ArrowMetalNative.release)
        m("copyTime") += t1 - t0
        m("gpuTime") += System.nanoTime() - t1
        m("inputRows") += n
      } finally arena.close()
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

    val body = Iterator.continually(()).takeWhile(_ => batches.hasNext).flatMap(_ => chunk())
    def initial: Iterator[InternalRow] = emptyRow match {
      case Some(vals) if total == 0 =>
        vals.zipWithIndex.foreach { case (v, j) => setValue(row, j, outTypes(j), v) }
        m("outputRows") += 1
        Iterator.single(proj(row))
      case _ => Iterator.empty
    }
    body ++ initial
  }
}
