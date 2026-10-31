export const DEFAULT_MEMORY_SIMILARITY_THRESHOLD = 0.4

export function memorySimilarityThreshold(value: unknown): number {
  if (typeof value !== "number" || !Number.isFinite(value)) {
    return DEFAULT_MEMORY_SIMILARITY_THRESHOLD
  }
  return Math.min(2, Math.max(0.1, value))
}
