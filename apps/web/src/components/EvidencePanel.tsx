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

  const animatedScore = useCountUp(evidence?.priority_score.score_0_100 ?? null, evidence?.segment_id ?? "");

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

  const { observed_evidence, exposure, model_evidence, priority_score } = evidence;

  return (
    <aside className="evidence-panel" aria-label="Evidence panel">
      <header className="panel-header">
        <h2>Why this road?</h2>
        <p className="panel-subtitle mono">{evidence.segment_id}</p>
      </header>

      <div className="score-hero">
        <span className="score-hero-number">
          {priority_score.score_0_100 !== null ? Math.round(animatedScore) : "—"}
        </span>
        <span className="score-hero-label">
          priority score
          {priority_score.score_0_100 !== null && priority_score.score_0_100 !== undefined && (
            <>
              {" "}
              <span className={`govuk-tag ${ragBand(priority_score.score_0_100).className}`}>
                {ragBand(priority_score.score_0_100).label}
              </span>
            </>
          )}
          <br />
          <span className="mono small">{priority_score.policy_profile}</span>
        </span>
      </div>

      <section className="evidence-section">
        <h3>Score breakdown</h3>
        <ul className="component-bars">
          {Object.entries(priority_score.components).map(([key, value]) => (
            <li key={key}>
              <span className="component-label">{COMPONENT_LABELS[key] ?? key}</span>
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
              <span className="component-value">{formatNumber(value, 2)}</span>
            </li>
          ))}
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
            <dt>Predicted relative risk</dt>
            <dd>{formatNumber(model_evidence.predicted_relative_risk, 2)}</dd>
          </div>
          <div>
            <dt>90% interval width</dt>
            <dd>{formatNumber(model_evidence.conformal_interval_width_90pct, 2)}</dd>
          </div>
          <div>
            <dt>Model version</dt>
            <dd className="mono small">{model_evidence.model_version}</dd>
          </div>
        </dl>
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
