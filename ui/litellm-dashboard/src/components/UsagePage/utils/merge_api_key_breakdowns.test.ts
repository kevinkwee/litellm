import { describe, expect, it } from "vitest";
import { DailyData, SpendMetrics } from "../types";
import { mergeApiKeyBreakdowns } from "./merge_api_key_breakdowns";

const metrics = (spend: number): SpendMetrics => ({
  spend,
  prompt_tokens: 10,
  completion_tokens: 5,
  total_tokens: 15,
  api_requests: 1,
  successful_requests: 1,
  failed_requests: 0,
  cache_read_input_tokens: 0,
  cache_creation_input_tokens: 0,
});

const aggregatedDay = (date: string): DailyData => ({
  date,
  metrics: metrics(100),
  breakdown: {
    models: {
      "gpt-4": { metrics: metrics(60), metadata: { region: "us" }, api_key_breakdown: {} },
    },
    model_groups: { "gpt-4": { metrics: metrics(60), metadata: {}, api_key_breakdown: {} } },
    mcp_servers: {},
    providers: { openai: { metrics: metrics(60), metadata: {}, api_key_breakdown: {} } },
    api_keys: {},
    entities: {},
    endpoints: { "/v1/chat/completions": { metrics: metrics(60), metadata: {}, api_key_breakdown: {} } },
  },
});

const keysDay = (date: string): DailyData => ({
  date,
  metrics: metrics(0),
  breakdown: {
    models: {
      "gpt-4": {
        metrics: metrics(0),
        metadata: {},
        api_key_breakdown: { "key-1": { metrics: metrics(60), metadata: { key_alias: "one", team_id: null } } },
      },
    },
    model_groups: {},
    mcp_servers: {
      "mcp:github": {
        metrics: metrics(0),
        metadata: {},
        api_key_breakdown: { "key-2": { metrics: metrics(5), metadata: { key_alias: "two", team_id: null } } },
      },
    },
    providers: {
      // A provider the rollups have never seen must not vanish in the merge
      bedrock: {
        metrics: metrics(0),
        metadata: {},
        api_key_breakdown: { "key-1": { metrics: metrics(7), metadata: { key_alias: "one", team_id: null } } },
      },
    },
    api_keys: { "key-1": { metrics: metrics(60), metadata: { key_alias: "one", team_id: null } } },
    entities: {},
    endpoints: {
      "/v1/chat/completions": {
        metrics: metrics(0),
        metadata: {},
        api_key_breakdown: { "key-1": { metrics: metrics(60), metadata: { key_alias: "one", team_id: null } } },
      },
    },
  },
});

describe("mergeApiKeyBreakdowns", () => {
  it("fills the api_keys map and nested per-family key breakdowns", () => {
    const merged = mergeApiKeyBreakdowns([aggregatedDay("2026-09-13")], [keysDay("2026-09-13")]);

    const day = merged[0];
    expect(Object.keys(day.breakdown.api_keys)).toEqual(["key-1"]);
    expect(day.breakdown.models["gpt-4"].api_key_breakdown["key-1"].metrics.spend).toBe(60);
    expect(day.breakdown.endpoints?.["/v1/chat/completions"].api_key_breakdown["key-1"].metrics.spend).toBe(60);
    expect(day.breakdown.mcp_servers["mcp:github"].api_key_breakdown["key-2"].metrics.spend).toBe(5);
    expect(day.breakdown.providers.bedrock.api_key_breakdown["key-1"].metrics.spend).toBe(7);
  });

  it("keeps the aggregated metrics and metadata untouched", () => {
    const merged = mergeApiKeyBreakdowns([aggregatedDay("2026-09-13")], [keysDay("2026-09-13")]);

    const day = merged[0];
    expect(day.metrics.spend).toBe(100);
    expect(day.breakdown.models["gpt-4"].metrics.spend).toBe(60);
    expect(day.breakdown.models["gpt-4"].metadata).toEqual({ region: "us" });
  });

  it("does not mutate the inputs and passes through days without key rows", () => {
    const base = [aggregatedDay("2026-09-13"), aggregatedDay("2026-09-14")];
    const merged = mergeApiKeyBreakdowns(base, [keysDay("2026-09-13")]);

    expect(merged[1]).toBe(base[1]);
    expect(base[0].breakdown.api_keys).toEqual({});
    expect(Object.keys(merged[0].breakdown.api_keys)).toEqual(["key-1"]);
  });

  it("returns the base as-is when there are no key rows", () => {
    const base = [aggregatedDay("2026-09-13")];
    expect(mergeApiKeyBreakdowns(base, [])).toBe(base);
  });
});
