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
  ons_code: string;
  model_version: string;
  test_year: number;
  n_segments: number;
  metrics: Record<string, number | null>;
  conformal: {
    target_confidence_level: number;
    empirical_coverage: number;
    mean_interval_width: number;
  };
  generated_at: string;
  non_negotiable_boundary: string;
}

export interface PriorityQueueRow {
  segment_id: string;
  priority_score: number;
  model_score: number;
  prior_year_count: number | null;
  highway: string | null;
}

export interface RoadEvidence {
  segment_id: string;
  borough: string;
  observed_evidence: Record<string, number | null>;
  exposure: {
    aadf_all_motor_vehicles: number | null;
    aadf_pedal_cycles: number | null;
    has_aadf: boolean;
  };
  model_evidence: {
    model_version: string;
    predicted_relative_risk: number | null;
    conformal_interval_width_90pct: number | null;
  };
  priority_score: {
    score_0_100: number | null;
    policy_profile: string;
    components: Record<string, number | null>;
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
  priorityQueue: (borough: string, limit = 25) =>
    getJSON<PriorityQueueRow[]>(`/boroughs/${encodeURIComponent(borough)}/priority-queue?limit=${limit}`),
  roadEvidence: (borough: string, segmentId: string) =>
    getJSON<RoadEvidence>(`/boroughs/${encodeURIComponent(borough)}/roads/${encodeURIComponent(segmentId)}`),
};
