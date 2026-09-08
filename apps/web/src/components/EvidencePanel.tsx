import { useEffect, useRef, useState } from "react";
import type { RoadEvidence } from "../api";

interface EvidencePanelProps {
  evidence: RoadEvidence | null;
  loading: boolean;
  error: string | null;
}

const COMPONENT_LABELS: Record<string, string> = {
  risk_signal: "Risk signal",
  severity: "Severity",
  vulnerable_users: "Vulnerable users",
  trend: "Recent trend",
  network_importance: "Network importance",
  data_confidence: "Data confidence",
};

// What each component actually measures, and its weight in the 0-100
// priority score (see src/greyspot/product/priority_score.py's
// DEFAULT_WEIGHTS - kept in sync with that module by hand, since it's a
// small, rarely-changed policy table, not worth importing into a
// frontend bundle). 2026-09-08 fix: the panel previously showed six bare
// 0-1 numbers with a bar and nothing else - a reported "I don't know
// what these numbers mean" usability bug.
const COMPONENT_INFO: Record<string, { weight: number; description: string }> = {
  risk_signal: {
    weight: 0.35,
    description: "The model's own predicted relative risk, scaled 0-1 against every other scored segment in this borough this window.",
  },
  severity: {
    weight: 0.25,
    description: "How severe prior-year casualties here were (fatal counts weighted most, slight least), scaled against every segment.",
  },
  vulnerable_users: {
    weight: 0.15,
    description: "Share of this segment's own prior-year casualties who were pedestrians or cyclists - not scaled against other segments.",
  },
  trend: {
    weight: 0.10,
    description: "Is the latest year's collision count above or below the recent 2-year average? A flat 0.50 means too little history to call a trend, not a neutral risk finding.",
  },
  network_importance: {
    weight: 0.10,
    description: "How connected this segment's endpoints are (junction degree) relative to other segments - a rough proxy for how much traffic passes through, not measured directly.",
  },
  data_confidence: {
    weight: 0.05,
    description: "How narrow the model's 90% uncertainty interval is here - narrower means more confident. Never used to imply the road is safe.",
  },
};

// A plain-language read of the CURRENT value, not just its definition -
// this is the part a bare number and a bar can't give a reader on its
// own ("what does 0.00 out of 1.00 actually mean for THIS road?").
function componentInterpretation(
  key: string,
  value: number | null,
  evidence: RoadEvidence,
): string {
  if (value === null) return "Not available for this segment.";
  if (key === "trend" && value === 0.5) {
    return "Not enough prior-year history on this segment to call a trend either way.";
  }
  if (key === "data_confidence" && evidence.priority_score.data_confidence_is_default) {
    return "No interval-based confidence data for this batch - shown as neutral, not computed.";
  }
  if (key === "vulnerable_users") {
    const priorCount = evidence.observed_evidence.prior_year_collision_count;
    if (!priorCount) return "No prior-year collisions recorded, so no vulnerable-user share to report.";
    return `${Math.round(value * 100)}% of this segment's prior-year casualties were pedestrians or cyclists.`;
  }
  // The five batch-normalised components (everything except
  // vulnerable_users) all share the same 0-1 "rank within this batch"
  // reading, so one banding covers them.
  if (value >= 0.8) return "Among the highest in this borough's batch.";
  if (value >= 0.5) return "Above the middle of this borough's batch.";
  if (value >= 0.2) return "Below the middle of this borough's batch.";
  return "Among the lowest in this borough's batch.";
}

function formatNumber(value: number | null | undefined, digits = 1): string {
  if (value === null || value === undefined) return "—";
  return value.toFixed(digits);
}

// Same RAG band as the map and priority queue (see App.css / DESIGN.md) -
// one colour language, never redefined per component.
function ragBand(score: number): { className: string; label: string } {
  if (score < 33) return { className: "govuk-tag--green", label: "Lower priority" };
  if (score < 66) return { className: "govuk-tag--orange", label: "Medium priority" };
  return { className: "govuk-tag--red", label: "Higher priority" };
}

// Score hero count-up + score-breakdown bar fill (Fieldscope redesign -
// see DESIGN.md's Motion section, "Selection handshake" / pattern 4).
// The score is the product's single most important number; snapping it
// to a new value on every selection discards the "how different is this
// from what I was just looking at" signal a count-up gives for free, and
// a static bar next to a live map/queue reads as stale by comparison.
function useCountUp(target: number | null, key: string): number {
  const [display, setDisplay] = useState(target ?? 0);
  const fromRef = useRef(target ?? 0);
  useEffect(() => {
    if (target === null) return;
    const from = fromRef.current;
    const start = performance.now();
    const duration = 420; // --duration-slow
    let raf = 0;
    const tick = (now: number) => {
      const t = Math.min(1, (now - start) / duration);
      const eased = 1 - Math.pow(1 - t, 3); // ease-out-cubic, no overshoot
      setDisplay(from + (target - from) * eased);
      if (t < 1) raf = requestAnimationFrame(tick);
      else fromRef.current = target;
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [target, key]);
  return display;
}

export function EvidencePanel({ evidence, loading, error }: EvidencePanelProps) {
  // Bars start at scale 0 and are flipped to their real value one frame
  // after mount/selection-change, so the fill genuinely animates in
  // rather than snapping to place on every road selected.
  const [barsFilled, setBarsFilled] = useState(false);
  useEffect(() => {
    setBarsFilled(false);
    const raf = requestAnimationFrame(() => setBarsFilled(true));
    return () => cancelAnimationFrame(raf);
  }, [evidence?.segment_id]);

  const modelScore = evidence?.priority_score.model_ranked.score_0_100 ?? null;
  const animatedScore = useCountUp(modelScore, evidence?.segment_id ?? "");

  if (loading) {
    return (
      <aside className="evidence-panel" aria-label="Evidence panel">
        <p className="muted">Loading evidence…</p>
      </aside>
    );
  }
  if (error) {
    return (
      <aside className="evidence-panel" aria-label="Evidence panel">
        <p className="error-text">{error}</p>
      </aside>
    );
  }
  if (!evidence) {
    return (
      <aside className="evidence-panel evidence-panel--empty" aria-label="Evidence panel">
        <p className="muted">Select a road on the map or from the priority queue to see its evidence.</p>
      </aside>
    );
  }

  const { observed_evidence, exposure, model_evidence, baseline_evidence, priority_score } = evidence;
  const [intervalLow, intervalHigh] = model_evidence.conformal_interval_90pct;

  return (
    <aside className="evidence-panel" aria-label="Evidence panel">
      <header className="panel-header">
        <h2>Why this road?</h2>
        {/* Real street name (OS Open Roads name_1) as the actual heading -
            2026-09-08 fix: this panel previously led with nothing but the
            raw segment UUID, a reported usability bug. A genuinely
            unnamed segment still falls back to its road class rather than
            a blank heading. */}
        <p className="panel-subtitle">{evidence.name ?? `Unnamed ${evidence.highway ?? "road"}`}</p>
        <p className="mono small evidence-segment-id">{evidence.segment_id}</p>
      </header>

      <div className="score-hero">
        <span className="score-hero-number">{modelScore !== null ? Math.round(animatedScore) : "—"}</span>
        <span className="score-hero-label">
          priority score (model-ranked)
          {modelScore !== null && (
            <>
              {" "}
              <span className={`govuk-tag ${ragBand(modelScore).className}`}>{ragBand(modelScore).label}</span>
            </>
          )}
          <br />
          <span className="mono small">{priority_score.policy_profile}</span>
        </span>
      </div>

      <section className="evidence-section evidence-section--baseline">
        <h3>Baseline comparison</h3>
        {/* This is the single most important disclosure in this panel:
            this project's own paper found the model does not reliably
            beat a ranking with no parameters and no training. Shown as a
            direct number-for-number comparison, not buried in prose. */}
        <dl className="evidence-grid">
          <div>
            <dt>Model score (0–100)</dt>
            <dd>{formatNumber(modelScore, 0)}</dd>
          </div>
          <div>
            <dt>Baseline score (0–100)</dt>
            <dd>{formatNumber(priority_score.baseline_ranked.score_0_100, 0)}</dd>
          </div>
        </dl>
        <p className="caveat">{baseline_evidence.research_note}</p>
      </section>

      <section className="evidence-section">
        <h3>Score breakdown (model-ranked)</h3>
        <p className="section-intro">
          The priority score is a weighted blend of six components, each scaled 0.00-1.00. Weights (how much each
          one counts toward the final 0-100 score) are shown in brackets.
        </p>
        <ul className="component-bars">
          {Object.entries(priority_score.model_ranked.components).map(([key, value]) => {
            const info = COMPONENT_INFO[key];
            return (
              <li key={key}>
                <div className="component-row">
                  <span className="component-label">
                    {COMPONENT_LABELS[key] ?? key}
                    {info && <span className="component-weight"> ({Math.round(info.weight * 100)}%)</span>}
                  </span>
                  <span className="component-value">{formatNumber(value, 2)}</span>
                </div>
                <span className="component-bar-track">
                  <span
                    className="component-bar-fill"
                    // Animates via transform (GPU-composited), not width -
                    // see App.css's .component-bar-fill comment. Starts at
                    // scale 0 and is flipped to the real fraction one frame
                    // after this segment's evidence mounts (barsFilled).
                    style={{
                      transform: `scaleX(${barsFilled ? Math.max(0, Math.min(1, value ?? 0)) : 0})`,
                    }}
                  />
                </span>
                {info && <p className="component-description">{info.description}</p>}
                <p className="component-interpretation">{componentInterpretation(key, value, evidence)}</p>
              </li>
            );
          })}
        </ul>
        {priority_score.data_confidence_is_default && (
          <p className="caveat">No conformal interval available for this row — confidence shown as neutral, not computed.</p>
        )}
      </section>

      <section className="evidence-section">
        <h3>Observed history (prior year)</h3>
        <dl className="evidence-grid">
          <div>
            <dt>Collisions</dt>
            <dd>{formatNumber(observed_evidence.prior_year_collision_count, 0)}</dd>
          </div>
          <div>
            <dt>Fatal casualties</dt>
            <dd>{formatNumber(observed_evidence.prior_year_fatal_casualties, 0)}</dd>
          </div>
          <div>
            <dt>Serious casualties</dt>
            <dd>{formatNumber(observed_evidence.prior_year_serious_casualties, 0)}</dd>
          </div>
          <div>
            <dt>Pedestrian casualties</dt>
            <dd>{formatNumber(observed_evidence.prior_year_pedestrian_casualties, 0)}</dd>
          </div>
          <div>
            <dt>Cyclist casualties</dt>
            <dd>{formatNumber(observed_evidence.prior_year_cyclist_casualties, 0)}</dd>
          </div>
        </dl>
      </section>

      <section className="evidence-section">
        <h3>Exposure</h3>
        {exposure.has_aadf ? (
          <dl className="evidence-grid">
            <div>
              <dt>Motor vehicles/day (AADF)</dt>
              <dd>{formatNumber(exposure.aadf_all_motor_vehicles, 0)}</dd>
            </div>
            <div>
              <dt>Pedal cycles/day (AADF)</dt>
              <dd>{formatNumber(exposure.aadf_pedal_cycles, 0)}</dd>
            </div>
          </dl>
        ) : (
          <p className="muted">No traffic-count point near this segment — not available, not zero.</p>
        )}
      </section>

      <section className="evidence-section">
        <h3>Model evidence</h3>
        <dl className="evidence-grid">
          <div>
            <dt>Predicted crashes, next 14 days</dt>
            <dd>{formatNumber(model_evidence.predicted_crashes_next_14_days, 3)}</dd>
          </div>
          <div>
            <dt>90% interval</dt>
            <dd>
              {intervalLow !== null && intervalHigh !== null
                ? `${formatNumber(intervalLow, 3)} – ${formatNumber(intervalHigh, 3)}`
                : "—"}
            </dd>
          </div>
          <div>
            <dt>Scored as of</dt>
            <dd>{model_evidence.as_of_window ?? "—"}</dd>
          </div>
          <div>
            <dt>Model version</dt>
            <dd className="mono small">{model_evidence.model_version ?? "—"}</dd>
          </div>
        </dl>
        <p className="section-intro">
          The prediction is an <em>expected count</em>, not a whole-road forecast - most 14-day windows on most
          segments see zero collisions, so a typical value is a small fraction (e.g. 0.015 means roughly 1 in 65
          such windows sees one). The interval is this project&rsquo;s 90% confidence range for that count.
        </p>
      </section>

      <section className="evidence-section evidence-section--limitations">
        <h3>Limitations</h3>
        {/* GOV.UK "warning text" component - the real pattern for exactly
            this kind of stated caveat, not a generic alert box. */}
        <div className="govuk-warning-text" role="note">
          <i className="govuk-warning-text__icon" aria-hidden="true">
            !
          </i>
          <div className="govuk-warning-text__text">
            <span className="govuk-visually-hidden">Warning</span>
            <ul>
              {evidence.limitations.map((line) => (
                <li key={line}>{line}</li>
              ))}
            </ul>
          </div>
        </div>
      </section>
    </aside>
  );
}
