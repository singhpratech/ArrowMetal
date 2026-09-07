// Loads the compiled addon and, through it, libArrowMetalC.dylib.
//
// Search order for the dylib, enforced in C (src/addon.cc):
//   1. $ARROWMETAL_LIB, taken as a full path to the dylib
//   2. <package>/../.build/release/libArrowMetalC.dylib
// Neither found is an error naming both paths.

import * as path from 'node:path';

/** Opaque handle to a Metal-resident Arrow array. */
export type ArrayHandle = { readonly __brand: 'am_array' };
/** Opaque handle to a group-by key mapping. */
export type GroupByHandle = { readonly __brand: 'am_groupby' };
/** Opaque handle to a registered plan source (one table). */
export type PlanSourceHandle = { readonly __brand: 'am_plan_source' };
/** Opaque handle to the result of one plan run. */
export type PlanResultHandle = { readonly __brand: 'am_plan_result' };

/** The raw buffers of an exported array, wrapped (never copied) over ArrowMetal memory. */
export interface ExportedBuffers {
  format: string;
  length: number;
  offset: number;
  nullCount: number;
  validity: ArrayBuffer | null;
  offsets: ArrayBuffer | null;
  data: ArrayBuffer | null;
}

export interface LoadInfo {
  path: string;
  version: string;
  device: string;
  pageSize: number;
}

export interface Native {
  load(packageDir: string): LoadInfo;
  importArray(
    format: string,
    length: number,
    offset: number,
    nullCount: number,
    validity: ArrayBufferView | null,
    data: ArrayBufferView | null,
    offsets: ArrayBufferView | null,
  ): ArrayHandle;
  importRetained(h: ArrayHandle): boolean;
  exportArray(h: ArrayHandle): ExportedBuffers;
  length(h: ArrayHandle): number;
  nullCount(h: ArrayHandle): number;
  format(h: ArrayHandle): string;
  release(h: ArrayHandle): void;
  reduce(h: ArrayHandle, op: number): bigint | number | null;
  compareScalar(h: ArrayHandle, op: number, scalar: number | bigint | boolean): ArrayHandle;
  compareArray(a: ArrayHandle, op: number, b: ArrayHandle): ArrayHandle;
  arithScalar(h: ArrayHandle, op: number, scalar: number | bigint): ArrayHandle;
  cast(h: ArrayHandle, format: string): ArrayHandle;
  filter(a: ArrayHandle, mask: ArrayHandle): ArrayHandle;
  take(a: ArrayHandle, indices: ArrayHandle): ArrayHandle;
  slice(a: ArrayHandle, offset: number, length: number): ArrayHandle;
  argsort(a: ArrayHandle, descending: boolean): ArrayHandle;
  sort(a: ArrayHandle, descending: boolean): ArrayHandle;
  lexsort(columns: ArrayHandle[], descending: boolean[]): ArrayHandle;
  groupByKeys(columns: ArrayHandle[]): GroupByHandle;
  groupCount(gb: GroupByHandle): number;
  groupKeysResult(gb: GroupByHandle, i: number): ArrayHandle;
  groupAgg(gb: GroupByHandle, values: ArrayHandle | null, op: number, p1: number): ArrayHandle;
  planSourceCreate(name: string, columns: ArrayHandle[], names: string[]): PlanSourceHandle;
  planRun(json: string, sources: PlanSourceHandle[], optimize: boolean): PlanResultHandle;
  planExplain(json: string, sources: PlanSourceHandle[], optimize: boolean): string;
  planResultInfo(r: PlanResultHandle): { names: string[]; rows: number };
  planColumn(r: PlanResultHandle, i: number): ArrayHandle;
  lastError(): string;
  bufferAddress(view: ArrayBufferView | ArrayBuffer): bigint;
}

const packageDir = path.resolve(__dirname, '..');

function loadAddon(): Native {
  const candidates = [
    path.join(packageDir, 'build', 'Release', 'arrowmetal_native.node'),
    path.join(packageDir, 'build', 'Debug', 'arrowmetal_native.node'),
  ];
  const tried: string[] = [];
  for (const c of candidates) {
    try {
      // eslint-disable-next-line @typescript-eslint/no-var-requires
      return require(c) as Native;
    } catch (e) {
      tried.push(`${c}: ${(e as Error).message}`);
    }
  }
  throw new Error(
    'ArrowMetal: the native addon is not built. Run `npm run build:native` in the package ' +
      `directory. Tried:\n  ${tried.join('\n  ')}`,
  );
}

export const native: Native = loadAddon();

/** Path, version and device of the dylib this process is bound to. */
export const info: LoadInfo = native.load(packageDir);
