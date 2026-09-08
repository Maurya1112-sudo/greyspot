// Typed client for the Greyspot API (services/api). Kept dependency-free
// (plain fetch) since this is a small prototype frontend, not a large app
// that would benefit from a heavier data-fetching library yet.

const API_BASE = import.meta.env.VITE_API_BASE ?? "http://127.0.0.1:8000";

export interface BoroughSummary {
  name: string;
  ons_code: string;
}

export interface ModelInfo {
  borough: string;
  model_version: string;
  seed: number;
  held_out_start: string;
  held_out_horizon_days: number;
  n_segments: number;
  this_run_acchr_at_20: number;
  reported_acchr_at_20: number | null;
  reported_acchr_source: string | null;
  conformal_confidence_level: number;
  generated_at: string;
  research_finding: string;
  interval_note: string;
  non_negotiable_boundary: string;
}

export type RankBy = "model" | "baseline";

export interface PriorityQueueRow {
  segment_id: string;
  name: string | null;
  priority_score: number;
  model_score: number;
  prior_year_count: number | null;
  highway: string | null;
  rank_by: RankBy;
}

export interface RoadEvidence {
  segment_id: string;
  borough: string;
  name: string | null;
  highway: string | null;
  observed_evidence: Record<string, number | null>;
  exposure: {
    aadf_all_motor_vehicles: number | null;
    aadf_pedal_cycles: number | null;
    has_aadf: boolean;
  };
  model_evidence: {
    model_version: string | null;
    predicted_crashes_next_14_days: number | null;
    conformal_interval_90pct: [number | null, number | null];
    as_of_window: string | null;
  };
  baseline_evidence: {
    method: string;
    score: number | null;
    research_note: string;
  };
  priority_score: {
    model_ranked: {
      score_0_100: number | null;
      components: Record<string, number | null>;
    };
    baseline_ranked: {
      score_0_100: number | null;
    };
    policy_profile: string;
    data_confidence_is_default: boolean;
  };
  limitations: string[];
}

async function getJSON<T>(path: string): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`);
  if (!res.ok) {
    const body = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(body.detail ?? `Request to ${path} failed (${res.status})`);
  }
  return res.json() as Promise<T>;
}

export const api = {
  boroughs: () => getJSON<BoroughSummary[]>("/boroughs"),
  modelInfo: (borough: string) => getJSON<ModelInfo>(`/boroughs/${encodeURIComponent(borough)}/model-info`),
  roadsGeoJSON: (borough: string) => getJSON<GeoJSON.FeatureCollection>(`/boroughs/${encodeURIComponent(borough)}/roads`),
  priorityQueue: (borough: string, limit = 25, rankBy: RankBy = "model") =>
    getJSON<PriorityQueueRow[]>(
      `/boroughs/${encodeURIComponent(borough)}/priority-queue?limit=${limit}&rank_by=${rankBy}`
    ),
  roadEvidence: (borough: string, segmentId: string) =>
    getJSON<RoadEvidence>(`/boroughs/${encodeURIComponent(borough)}/roads/${encodeURIComponent(segmentId)}`),
};
