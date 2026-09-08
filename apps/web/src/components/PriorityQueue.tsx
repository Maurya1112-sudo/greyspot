import { useEffect, useRef } from "react";
import type { PriorityQueueRow, RankBy } from "../api";

interface PriorityQueueProps {
  rows: PriorityQueueRow[];
  selectedSegmentId: string | null;
  onSelect: (segmentId: string) => void;
  loading: boolean;
  rankBy: RankBy;
  onChangeRankBy: (rankBy: RankBy) => void;
}

// RAG (Red-Amber-Green) band, matching the map's colour scale exactly -
// one colour language across the whole product (see App.css / DESIGN.md).
function ragBand(score: number): { className: string; label: string; color: string } {
  // --color-risk-mid is a text-contrast failure on white (2.78:1, needs
  // 4.5:1 - see App.css) - --color-risk-mid-text is the verified-safe
  // variant for exactly this use (colouring text, not a line or a tag's
  // light background).
  if (score < 33) return { className: "govuk-tag--green", label: "Lower", color: "var(--color-risk-low)" };
  if (score < 66) return { className: "govuk-tag--orange", label: "Medium", color: "var(--color-risk-mid-text)" };
  return { className: "govuk-tag--red", label: "Higher", color: "var(--color-risk-high)" };
}

export function PriorityQueue({
  rows,
  selectedSegmentId,
  onSelect,
  loading,
  rankBy,
  onChangeRankBy,
}: PriorityQueueProps) {
  const selectedRowRef = useRef<HTMLButtonElement>(null);

  // The reverse of MapView's "pan to the selected segment" fix: a road
  // selected directly on the map (rather than from this list) previously
  // highlighted here with zero visible effect if it fell outside the
  // currently-scrolled view of a 30-row list - found in the same
  // flow-verification pass. `block: "nearest"` avoids yanking the list to
  // the very top/centre when the row is already visible.
  useEffect(() => {
    selectedRowRef.current?.scrollIntoView({ block: "nearest", behavior: "smooth" });
  }, [selectedSegmentId]);

  return (
    <section className="priority-queue" aria-label="Priority queue">
      <header className="panel-header">
        <h2>Priority queue</h2>
        <p className="panel-subtitle">Ranked by transparent priority score, highest first</p>
        {/* This project's own replication study found that a
            parameter-free crash-count sort is not reliably beaten by the
            model (see the paper linked in the model-info audit trail).
            Hiding that behind only the model's ranking would misrepresent
            the research this product is built on, so the toggle is a
            first-class control, not a footnote. */}
        <div className="rank-toggle" role="radiogroup" aria-label="Ranking method">
          <button
            type="button"
            role="radio"
            aria-checked={rankBy === "model"}
            className={`rank-toggle-option${rankBy === "model" ? " rank-toggle-option--active" : ""}`}
            onClick={() => onChangeRankBy("model")}
          >
            Model (GAT+GRU+ZIP)
          </button>
          <button
            type="button"
            role="radio"
            aria-checked={rankBy === "baseline"}
            className={`rank-toggle-option${rankBy === "baseline" ? " rank-toggle-option--active" : ""}`}
            onClick={() => onChangeRankBy("baseline")}
            title="A parameter-free ranking by cumulative past crash count — no model, no training. This project's own research found it is not reliably beaten by the model above."
          >
            Baseline (crash-count sort)
          </button>
        </div>
      </header>
      {loading && <p className="muted">Loading…</p>}
      {!loading && rows.length === 0 && <p className="muted">No scored segments for this borough yet.</p>}
      <ol className="queue-list">
        {rows.map((row, index) => {
          const band = ragBand(row.priority_score);
          const isSelected = row.segment_id === selectedSegmentId;
          return (
            <li key={row.segment_id}>
              <button
                ref={isSelected ? selectedRowRef : undefined}
                type="button"
                className={`queue-row${isSelected ? " queue-row--selected" : ""}`}
                onClick={() => onSelect(row.segment_id)}
                aria-current={isSelected}
                // Panel-entrance stagger (Fieldscope redesign, see
                // DESIGN.md's Motion section) - each row's CSS animation
                // starts a little later than the one above it, so a fresh
                // borough's queue reads as newly-arrived data rather than
                // a static block. Capped at 24 rows' worth of delay so a
                // long list doesn't leave the bottom rows waiting almost
                // a second to appear.
                style={{ animationDelay: `${Math.min(index, 24) * 22}ms` }}
              >
                <span className="queue-rank">{index + 1}</span>
                <span className="queue-details">
                  {/* OS Open Roads carries a real street name for most
                      segments (name_1) - shown here as the primary line,
                      2026-09-08 fix for a reported bug where every row
                      showed nothing but a raw segment UUID. A genuinely
                      unnamed segment (service roads, tracks, some minor
                      residential stubs) falls back to its road-class
                      label instead of an empty row. */}
                  <span className="queue-road-name">{row.name ?? `Unnamed ${row.highway ?? "road"}`}</span>
                  <span className="queue-road-type">
                    {row.highway ?? "unclassified road"} · <span className="queue-segment-id">{row.segment_id}</span>
                  </span>
                </span>
                <span className="queue-score-group">
                  <span className={`govuk-tag ${band.className}`}>{band.label}</span>
                  <span
                    className="queue-score"
                    style={{ color: band.color }}
                    title="Priority score (0-100, policy-weighted composite - see Evidence panel)"
                  >
                    {row.priority_score.toFixed(0)}
                  </span>
                </span>
              </button>
            </li>
          );
        })}
      </ol>
    </section>
  );
}
