// ArrowMetal for Node.js: Apache Arrow arrays on the Apple silicon GPU.
//
//   import { MetalArray } from 'arrowmetal';
//   import { vectorFromArray, Int64 } from 'apache-arrow';
//
//   const col = MetalArray.fromArrow(vectorFromArray([1n, 2n, 3n], new Int64()));
//   col.sum();                        // 6n
//   col.filter(col.gt(1n)).toArrow(); // Vector<Int64> [2n, 3n]
//
// The browser is out of scope: there is no Metal there. This package is macOS/arm64 only.

import { native, info, ArrayHandle, GroupByHandle, PlanSourceHandle, PlanResultHandle } from './native';
import type { Data, DataType, Vector } from 'apache-arrow';

export { info };

/** Comparison operators accepted by {@link MetalArray.compare}. */
export type CompareOp = '==' | '!=' | '<' | '<=' | '>' | '>=';
/** Arithmetic operators accepted by {@link MetalArray.arith}. */
export type ArithOp = '+' | '-' | '*' | '/';

const CMP: Record<CompareOp, number> = { '==': 0, '!=': 1, '<': 2, '<=': 3, '>': 4, '>=': 5 };
const ARITH: Record<ArithOp, number> = { '+': 0, '-': 1, '*': 2, '/': 3 };

// am_reduce op numbering.
const REDUCE = { sum: 0, min: 1, max: 2, mean: 3 } as const;
// am_group_agg_ex op numbering.
const HASH_AGG = { sum: 0, count: 1, countValues: 2, mean: 3, min: 4, max: 5 } as const;

// ---------------------------------------------------------------------------------------------
// Arrow JS interop
// ---------------------------------------------------------------------------------------------

// Arrow JS is a peer dependency: everything except fromArrow/toArrow works without it.
let arrowModule: typeof import('apache-arrow') | null = null;
function arrow(): typeof import('apache-arrow') {
  if (arrowModule === null) {
    try {
      // eslint-disable-next-line @typescript-eslint/no-var-requires
      arrowModule = require('apache-arrow') as typeof import('apache-arrow');
    } catch {
      throw new Error(
        "ArrowMetal: 'apache-arrow' is not installed. It is a peer dependency, needed only by " +
          'MetalArray.fromArrow and MetalArray.toArrow. Install it with `npm i apache-arrow`.',
      );
    }
  }
  return arrowModule;
}

/** The Arrow format strings this binding carries across the C Data Interface. */
export type Format = 'c' | 'C' | 's' | 'S' | 'i' | 'I' | 'l' | 'L' | 'f' | 'g' | 'b' | 'u';

const FIXED_WIDTH: Partial<Record<Format, number>> = {
  c: 1, C: 1, s: 2, S: 2, i: 4, I: 4, f: 4, l: 8, L: 8, g: 8,
};

/** Maps an Arrow JS type to its C Data Interface format string. */
function typeToFormat(type: DataType): Format {
  const A = arrow();
  const t = type as unknown as { typeId: number; bitWidth?: number; isSigned?: boolean; precision?: number };
  switch (t.typeId) {
    case A.Type.Int: {
      const signed = t.isSigned !== false;
      switch (t.bitWidth) {
        case 8: return signed ? 'c' : 'C';
        case 16: return signed ? 's' : 'S';
        case 32: return signed ? 'i' : 'I';
        case 64: return signed ? 'l' : 'L';
      }
      break;
    }
    case A.Type.Float:
      if (t.precision === A.Precision.SINGLE) return 'f';
      if (t.precision === A.Precision.DOUBLE) return 'g';
      break;
    case A.Type.Bool:
      return 'b';
    case A.Type.Utf8:
      return 'u';
  }
  throw new Error(
    `ArrowMetal (Node): Arrow type ${String(type)} is not carried by this binding. ` +
      'Supported: Int8/16/32/64, Uint8/16/32/64, Float32, Float64, Bool, Utf8.',
  );
}

/** Maps a C Data Interface format string back to an Arrow JS type. */
function formatToType(format: string): DataType {
  const A = arrow();
  switch (format) {
    case 'c': return new A.Int8();
    case 'C': return new A.Uint8();
    case 's': return new A.Int16();
    case 'S': return new A.Uint16();
    case 'i': return new A.Int32();
    case 'I': return new A.Uint32();
    case 'l': return new A.Int64();
    case 'L': return new A.Uint64();
    case 'f': return new A.Float32();
    case 'g': return new A.Float64();
    case 'b': return new A.Bool();
    case 'u': return new A.Utf8();
  }
  throw new Error(
    `ArrowMetal (Node): result has Arrow format "${format}", which this binding cannot turn ` +
      'into an Arrow JS vector.',
  );
}

// Arrow JS's Data.slice advances the numeric data buffer and the utf8 offsets buffer but leaves the
// validity bitmap alone, keeping the row offset in Data.offset. The C Data Interface applies one
// offset to every buffer, so we rewind the advanced buffers back to their origin and let the C
// offset do the work. No bytes move.
function rewind<T extends ArrayBufferView>(view: T, offset: number, totalElements: number): T {
  if (offset === 0) return view;
  const bytesPerElement = (view as unknown as { BYTES_PER_ELEMENT: number }).BYTES_PER_ELEMENT;
  const back = offset * bytesPerElement;
  if (view.byteOffset < back) {
    throw new Error(
      'ArrowMetal (Node): this Arrow JS Data has offset ' + offset + ' but its buffer starts at ' +
        'byteOffset ' + view.byteOffset + ', so it cannot be rewound to the row-0 origin the C ' +
        'Data Interface needs. Materialise it first, e.g. with ' +
        '`vectorFromArray([...vector], vector.type)`.',
    );
  }
  const Ctor = (view as unknown as { constructor: new (b: ArrayBufferLike, o: number, n: number) => T })
    .constructor;
  return new Ctor(view.buffer, view.byteOffset - back, totalElements);
}

// ---------------------------------------------------------------------------------------------
// MetalArray
// ---------------------------------------------------------------------------------------------

/**
 * One Arrow array living on the GPU. Every method runs a Metal kernel and returns either a scalar
 * or a new MetalArray; nothing is copied back to the host until you call `toArrow` or
 * `toTypedArray`, and even those wrap ArrowMetal's own buffers rather than copying them.
 *
 * Handles are freed by the garbage collector. Call `release()` when you want it to happen now.
 */
export class MetalArray {
  /** @internal */
  readonly handle: ArrayHandle;
  #released = false;

  /** @internal */
  constructor(handle: ArrayHandle) {
    this.handle = handle;
  }

  // -- construction ----------------------------------------------------------------------------

  /**
   * Imports an Arrow JS `Vector` or `Data` across the C Data Interface.
   *
   * No bytes are copied here: the ArrowArray we hand to ArrowMetal points straight at the V8
   * backing store, and this package keeps a reference to it for the life of the handle. ArrowMetal
   * itself wraps those pages when they are page aligned and copies once when they are not; see
   * `wrappedProducerBuffers`.
   *
   * A chunked Vector (more than one `Data`) is rejected rather than silently concatenated.
   */
  static fromArrow<T extends DataType>(input: Vector<T> | Data<T>): MetalArray {
    const data = 'data' in input && Array.isArray((input as Vector<T>).data)
      ? (() => {
          const chunks = (input as Vector<T>).data;
          if (chunks.length !== 1) {
            throw new Error(
              `ArrowMetal (Node): expected a single-chunk Vector, got ${chunks.length} chunks. ` +
                'Concatenate it first, e.g. with `vectorFromArray([...vector], vector.type)`.',
            );
          }
          return chunks[0];
        })()
      : (input as Data<T>);

    const format = typeToFormat(data.type);
    const offset = data.offset;
    const length = data.length;
    const nullCount = data.nullCount;

    const bitmap = data.nullBitmap;
    const validity = nullCount > 0 && bitmap != null && bitmap.length > 0 ? bitmap : null;

    if (format === 'u') {
      const valueOffsets = rewind(data.valueOffsets as Int32Array, offset, offset + length + 1);
      const bytes = data.values as Uint8Array;
      return new MetalArray(
        native.importArray('u', length, offset, nullCount, validity, bytes, valueOffsets),
      );
    }
    if (format === 'b') {
      // Bool data is a bitmap; Arrow JS does not advance it, so the C offset applies as-is.
      return new MetalArray(
        native.importArray('b', length, offset, nullCount, validity, data.values as Uint8Array, null),
      );
    }
    const values = rewind(data.values as ArrayBufferView, offset, offset + length);
    return new MetalArray(
      native.importArray(format, length, offset, nullCount, validity, values, null),
    );
  }

  /**
   * Imports a plain typed array, with an optional Arrow validity bitmap or boolean null mask.
   * `BigInt64Array` becomes Int64, `Float64Array` becomes Float64, and so on.
   */
  static fromTypedArray(
    values: ArrayBufferView,
    options: { validity?: Uint8Array | null; nullCount?: number } = {},
  ): MetalArray {
    const name = values.constructor.name;
    const format = (
      {
        Int8Array: 'c', Uint8Array: 'C', Int16Array: 's', Uint16Array: 'S',
        Int32Array: 'i', Uint32Array: 'I', BigInt64Array: 'l', BigUint64Array: 'L',
        Float32Array: 'f', Float64Array: 'g',
      } as Record<string, Format>
    )[name];
    if (format === undefined) {
      throw new Error(`ArrowMetal (Node): ${name} has no Arrow equivalent in this binding.`);
    }
    const length = (values as unknown as { length: number }).length;
    const validity = options.validity ?? null;
    const nullCount = options.nullCount ?? (validity === null ? 0 : -1);
    return new MetalArray(native.importArray(format, length, 0, nullCount, validity, values, null));
  }

  // -- introspection ---------------------------------------------------------------------------

  /** Number of rows. */
  get length(): number {
    return native.length(this.handle);
  }

  /** Number of null rows. */
  get nullCount(): number {
    return native.nullCount(this.handle);
  }

  /** The C Data Interface format string of the element type. */
  get format(): string {
    return native.format(this.handle);
  }

  /**
   * True when ArrowMetal kept the producer's buffers rather than copying them at import: it did not
   * hand them back during `am_import`. Only meaningful for arrays built by `fromArrow` /
   * `fromTypedArray`; false for every array ArrowMetal computed itself.
   */
  get wrappedProducerBuffers(): boolean {
    return native.importRetained(this.handle);
  }

  /** Frees the GPU handle now instead of at the next garbage collection. */
  release(): void {
    if (!this.#released) {
      native.release(this.handle);
      this.#released = true;
    }
  }

  // -- reductions ------------------------------------------------------------------------------

  /** Sum. Int64 and Uint64 columns answer with a BigInt; float columns with a number. */
  sum(): bigint | number | null {
    return native.reduce(this.handle, REDUCE.sum);
  }
  /** Minimum, skipping nulls and NaN. */
  min(): bigint | number | null {
    return native.reduce(this.handle, REDUCE.min);
  }
  /** Maximum, skipping nulls and NaN. */
  max(): bigint | number | null {
    return native.reduce(this.handle, REDUCE.max);
  }
  /** Arithmetic mean, as a number. */
  mean(): number | null {
    return native.reduce(this.handle, REDUCE.mean) as number | null;
  }

  // -- element-wise ----------------------------------------------------------------------------

  /** Compares every row against a scalar, giving a Boolean array. */
  compare(op: CompareOp, scalar: number | bigint | boolean): MetalArray {
    return new MetalArray(native.compareScalar(this.handle, CMP[op], scalar));
  }
  /** Compares row by row against another array of the same type. */
  compareWith(op: CompareOp, other: MetalArray): MetalArray {
    return new MetalArray(native.compareArray(this.handle, CMP[op], other.handle));
  }
  eq(v: number | bigint | boolean): MetalArray { return this.compare('==', v); }
  ne(v: number | bigint | boolean): MetalArray { return this.compare('!=', v); }
  lt(v: number | bigint | boolean): MetalArray { return this.compare('<', v); }
  le(v: number | bigint | boolean): MetalArray { return this.compare('<=', v); }
  gt(v: number | bigint | boolean): MetalArray { return this.compare('>', v); }
  ge(v: number | bigint | boolean): MetalArray { return this.compare('>=', v); }

  /** Element-wise arithmetic against a scalar of the column's own type. */
  arith(op: ArithOp, scalar: number | bigint): MetalArray {
    return new MetalArray(native.arithScalar(this.handle, ARITH[op], scalar));
  }

  /** Casts to another Arrow format string, e.g. `'g'` for Float64. */
  cast(format: string): MetalArray {
    return new MetalArray(native.cast(this.handle, format));
  }

  // -- selection -------------------------------------------------------------------------------

  /** Keeps the rows where `mask` is true. A null in the mask drops the row. */
  filter(mask: MetalArray): MetalArray {
    return new MetalArray(native.filter(this.handle, mask.handle));
  }
  /** Gathers rows by an Int32 index array. */
  take(indices: MetalArray): MetalArray {
    return new MetalArray(native.take(this.handle, indices.handle));
  }
  /** A zero-copy view of `length` rows starting at `offset`. */
  slice(offset: number, length: number): MetalArray {
    return new MetalArray(native.slice(this.handle, offset, length));
  }

  // -- sorting ---------------------------------------------------------------------------------

  /** Int32 indices that put the array in order. Stable; nulls last, NaN after +Infinity. */
  argsort(descending = false): MetalArray {
    return new MetalArray(native.argsort(this.handle, descending));
  }
  /** A sorted copy, same type. */
  sort(descending = false): MetalArray {
    return new MetalArray(native.sort(this.handle, descending));
  }

  // -- output ----------------------------------------------------------------------------------

  /**
   * Exports to an Arrow JS `Vector` across the C Data Interface, wrapping ArrowMetal's buffers as
   * external ArrayBuffers. No bytes are copied; the GPU allocation is freed when the last buffer
   * of the vector is garbage collected.
   */
  toArrow(): Vector {
    const A = arrow();
    const e = native.exportArray(this.handle);
    const type = formatToType(e.format);
    const total = e.offset + e.length;

    const nullBitmap = e.validity === null ? undefined : new Uint8Array(e.validity);
    if (e.format === 'u') {
      const valueOffsets = new Int32Array(e.offsets!, 0, total + 1);
      const values = e.data === null ? new Uint8Array(0) : new Uint8Array(e.data);
      return new A.Vector([
        A.makeData({
          type: type as import('apache-arrow').Utf8,
          length: e.length,
          offset: e.offset,
          nullCount: e.nullCount,
          nullBitmap,
          valueOffsets,
          data: values,
        }),
      ]);
    }
    if (e.format === 'b') {
      return new A.Vector([
        A.makeData({
          type: type as import('apache-arrow').Bool,
          length: e.length,
          offset: e.offset,
          nullCount: e.nullCount,
          nullBitmap,
          data: e.data === null ? new Uint8Array(0) : new Uint8Array(e.data),
        }),
      ]);
    }
    const Ctor = TYPED_ARRAY[e.format as Format]!;
    const values = e.data === null ? new Ctor(0) : new Ctor(e.data, 0, total);
    return new A.Vector([
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      A.makeData({ type, length: e.length, offset: e.offset, nullCount: e.nullCount, nullBitmap, data: values } as any),
    ]);
  }

  /**
   * The values as a plain typed array, wrapping ArrowMetal's buffer. Nulls are not represented;
   * use `toArrow()` when the column has any. Boolean and utf8 columns are rejected: they are not
   * one value per element.
   */
  toTypedArray(): ArrayBufferView {
    const e = native.exportArray(this.handle);
    const Ctor = TYPED_ARRAY[e.format as Format];
    if (Ctor === undefined) {
      throw new Error(
        `ArrowMetal (Node): format "${e.format}" has no plain typed-array form; use toArrow().`,
      );
    }
    if (e.data === null) return new Ctor(0);
    return new Ctor(e.data, e.offset * (FIXED_WIDTH[e.format as Format] ?? 1), e.length);
  }

  /** The values as a JS array, nulls included. Convenience over `toArrow()`; this one copies. */
  toArray(): unknown[] {
    return [...this.toArrow()];
  }
}

type TypedArrayCtor = new (b?: ArrayBufferLike | number, o?: number, n?: number) => ArrayBufferView;
const TYPED_ARRAY: Partial<Record<Format, TypedArrayCtor>> = {
  c: Int8Array as unknown as TypedArrayCtor,
  C: Uint8Array as unknown as TypedArrayCtor,
  s: Int16Array as unknown as TypedArrayCtor,
  S: Uint16Array as unknown as TypedArrayCtor,
  i: Int32Array as unknown as TypedArrayCtor,
  I: Uint32Array as unknown as TypedArrayCtor,
  l: BigInt64Array as unknown as TypedArrayCtor,
  L: BigUint64Array as unknown as TypedArrayCtor,
  f: Float32Array as unknown as TypedArrayCtor,
  g: Float64Array as unknown as TypedArrayCtor,
};

// ---------------------------------------------------------------------------------------------
// Group-by
// ---------------------------------------------------------------------------------------------

/**
 * A dense group-id mapping over one or more key columns, produced by {@link groupBy}. Group order
 * is deterministic but is not first-seen: ascending by key for numeric, boolean and temporal
 * columns (nulls last), first-seen for utf8, lexicographic in column order for several columns.
 */
export class GroupBy {
  /** @internal */
  readonly handle: GroupByHandle;
  readonly keyCount: number;

  /** @internal */
  constructor(handle: GroupByHandle, keyCount: number) {
    this.handle = handle;
    this.keyCount = keyCount;
  }

  /** Number of distinct groups. */
  get groups(): number {
    return native.groupCount(this.handle);
  }

  /** The i-th key column, one row per group, in group order. */
  keys(i = 0): MetalArray {
    return new MetalArray(native.groupKeysResult(this.handle, i));
  }

  /** Sum of `values` per group. */
  sum(values: MetalArray): MetalArray {
    return this.agg(values, HASH_AGG.sum);
  }
  /** Mean of `values` per group, as Float64. */
  mean(values: MetalArray): MetalArray {
    return this.agg(values, HASH_AGG.mean);
  }
  /** Minimum of `values` per group. */
  min(values: MetalArray): MetalArray {
    return this.agg(values, HASH_AGG.min);
  }
  /** Maximum of `values` per group. */
  max(values: MetalArray): MetalArray {
    return this.agg(values, HASH_AGG.max);
  }
  /** Rows per group, nulls included, as Int64. */
  count(): MetalArray {
    return new MetalArray(native.groupAgg(this.handle, null, HASH_AGG.count, 0));
  }

  private agg(values: MetalArray, op: number): MetalArray {
    return new MetalArray(native.groupAgg(this.handle, values.handle, op, 0));
  }
}

/** Builds a group-id mapping over one or more key columns. */
export function groupBy(keys: MetalArray | MetalArray[]): GroupBy {
  const cols = Array.isArray(keys) ? keys : [keys];
  if (cols.length === 0) throw new Error('ArrowMetal (Node): groupBy needs at least one key column.');
  return new GroupBy(native.groupByKeys(cols.map((c) => c.handle)), cols.length);
}

/** Int32 indices ordering the rows by each column in turn, the first column most significant. */
export function lexsort(columns: MetalArray[], descending?: boolean[]): MetalArray {
  const desc = descending ?? columns.map(() => false);
  return new MetalArray(native.lexsort(columns.map((c) => c.handle), desc));
}

// ---------------------------------------------------------------------------------------------
// The JSON plan runner
// ---------------------------------------------------------------------------------------------

/** One registered table a plan can `scan`. */
export class PlanSource {
  /** @internal */
  readonly handle: PlanSourceHandle;
  readonly name: string;

  private constructor(handle: PlanSourceHandle, name: string) {
    this.handle = handle;
    this.name = name;
  }

  /** Registers `columns` under `name`; ArrowMetal retains the handles. */
  static create(name: string, columns: Record<string, MetalArray>): PlanSource {
    const names = Object.keys(columns);
    return new PlanSource(
      native.planSourceCreate(name, names.map((n) => columns[n].handle), names),
      name,
    );
  }
}

/** The columns a plan produced. */
export class PlanResult {
  /** @internal */
  readonly handle: PlanResultHandle;
  readonly names: string[];
  readonly rows: number;

  /** @internal */
  constructor(handle: PlanResultHandle) {
    this.handle = handle;
    const i = native.planResultInfo(handle);
    this.names = i.names;
    this.rows = i.rows;
  }

  /** A result column, by index or by name. */
  column(which: number | string): MetalArray {
    const i = typeof which === 'number' ? which : this.names.indexOf(which);
    if (i < 0) {
      throw new Error(
        `ArrowMetal (Node): no column "${String(which)}" in the result; have ${this.names.join(', ')}.`,
      );
    }
    return new MetalArray(native.planColumn(this.handle, i));
  }
}

/**
 * Runs a whole query plan (see `docs/ENGINE.md` for the JSON grammar) in one call. The plan is
 * type-checked, optimized and lowered to fused Metal kernels inside ArrowMetal.
 */
export function runPlan(
  plan: object | string,
  sources: PlanSource[],
  options: { optimize?: boolean } = {},
): PlanResult {
  const json = typeof plan === 'string' ? plan : JSON.stringify(plan);
  return new PlanResult(native.planRun(json, sources.map((s) => s.handle), options.optimize ?? true));
}

/** The optimized logical plan and the physical plan it lowers to, as text. */
export function explainPlan(
  plan: object | string,
  sources: PlanSource[],
  options: { optimize?: boolean } = {},
): string {
  const json = typeof plan === 'string' ? plan : JSON.stringify(plan);
  return native.planExplain(json, sources.map((s) => s.handle), options.optimize ?? true);
}

// ---------------------------------------------------------------------------------------------
// Diagnostics
// ---------------------------------------------------------------------------------------------

/** The address of a typed array's first byte, for alignment measurements. */
export function bufferAddress(view: ArrayBufferView | ArrayBuffer): bigint {
  return native.bufferAddress(view);
}

/** True when the buffer starts on a page boundary, which is what makes ArrowMetal's import free. */
export function isPageAligned(view: ArrayBufferView | ArrayBuffer): boolean {
  return native.bufferAddress(view) % BigInt(info.pageSize) === 0n;
}
