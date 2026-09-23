import { DailyData, MetricWithMetadata } from "../types";

/**
 * Merge per-key breakdown rows into the aggregated daily activity results.
 *
 * The aggregated endpoint returns no per-key rows because the key dimension
 * is unbounded; the per-key endpoint serves them for the keys the page
 * renders. Both responses share the same per-date breakdown shape, so the
 * merge only fills in what the aggregated response leaves empty: the api_keys
 * map and each family's nested api_key_breakdown.
 */

type KeyFamily = "models" | "model_groups" | "providers" | "mcp_servers" | "endpoints";

const KEY_FAMILIES: KeyFamily[] = ["models", "model_groups", "providers", "mcp_servers", "endpoints"];

const mergeFamily = (
  base: Record<string, MetricWithMetadata> | undefined,
  extra: Record<string, MetricWithMetadata> | undefined,
): Record<string, MetricWithMetadata> => {
  const merged: Record<string, MetricWithMetadata> = { ...(base ?? {}) };
  for (const [name, entry] of Object.entries(extra ?? {})) {
    const existing = merged[name];
    merged[name] = {
      metrics: existing?.metrics ?? entry.metrics,
      metadata: existing?.metadata ?? entry.metadata,
      api_key_breakdown: {
        ...(existing?.api_key_breakdown ?? {}),
        ...entry.api_key_breakdown,
      },
    };
  }
  return merged;
};

export const mergeApiKeyBreakdowns = (base: DailyData[], keys: DailyData[]): DailyData[] => {
  if (keys.length === 0) {
    return base;
  }
  const keysByDate = new Map(keys.map((day) => [day.date, day]));
  return base.map((day) => {
    const keysDay = keysByDate.get(day.date);
    if (!keysDay) {
      return day;
    }
    return {
      ...day,
      breakdown: {
        ...day.breakdown,
        api_keys: { ...day.breakdown.api_keys, ...keysDay.breakdown.api_keys },
        models: mergeFamily(day.breakdown.models, keysDay.breakdown.models),
        model_groups: mergeFamily(day.breakdown.model_groups, keysDay.breakdown.model_groups),
        providers: mergeFamily(day.breakdown.providers, keysDay.breakdown.providers),
        mcp_servers: mergeFamily(day.breakdown.mcp_servers, keysDay.breakdown.mcp_servers),
        endpoints: mergeFamily(day.breakdown.endpoints, keysDay.breakdown.endpoints),
      },
    };
  });
};
