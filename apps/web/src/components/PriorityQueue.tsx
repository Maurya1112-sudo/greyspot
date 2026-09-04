import { useEffect, useRef } from "react";
import type { PriorityQueueRow } from "../api";

interface PriorityQueueProps {
  rows: PriorityQueueRow[];
  selectedSegmentId: string | null;
  onSelect: (segmentId: string) => void;
  loading: boolean;
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

export function PriorityQueue({ rows, selectedSegmentId, onSelect, loading }: PriorityQueueProps) {
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
                  <span className="queue-road-type">{row.highway ?? "unclassified road"}</span>
                  <span className="queue-segment-id">{row.segment_id}</span>
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
