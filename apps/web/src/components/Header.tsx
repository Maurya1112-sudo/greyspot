import type { BoroughSummary, ModelInfo } from "../api";

interface HeaderProps {
  boroughs: BoroughSummary[];
  selectedBorough: string;
  onSelectBorough: (name: string) => void;
  modelInfo: ModelInfo | null;
}

export function Header({ boroughs, selectedBorough, onSelectBorough, modelInfo }: HeaderProps) {
  return (
    <header className="app-header">
      <div className="app-header-left">
        <h1 className="app-logo">
          <span className="app-logo-mark">Greyspot</span>
          <span className="app-logo-suffix">Government Edition</span>
        </h1>
        {/* GOV.UK phase-banner convention: names this honestly as a
            prototype, not a live service - true of this project and a
            recognisable signal to a government-service audience. */}
        <span className="phase-tag">Beta</span>
        <select
          className="borough-select"
          value={selectedBorough}
          onChange={(e) => onSelectBorough(e.target.value)}
          aria-label="Select borough"
        >
          {boroughs.map((b) => (
            <option key={b.name} value={b.name}>
              {b.name}
            </option>
          ))}
        </select>
      </div>
      {modelInfo && (
        <div
          className="app-header-right"
          title="Audit trail — every score traceable to a model version and the exact historical window it was evaluated on"
        >
          <span className="audit-item">
            <span className="audit-label">Model</span>
            <span className="audit-value mono">{modelInfo.model_version}</span>
          </span>
          <span className="audit-item">
            <span className="audit-label">Evaluated as of</span>
            <span className="audit-value">{modelInfo.held_out_start}</span>
          </span>
          <span className="audit-item">
            <span className="audit-label">Segments</span>
            <span className="audit-value">{modelInfo.n_segments.toLocaleString()}</span>
          </span>
          {modelInfo.reported_acchr_at_20 !== null && (
            <span className="audit-item" title="AccHR@20 on this exact window, matching the published paper's figure">
              <span className="audit-label">AccHR@20</span>
              <span className="audit-value">{(modelInfo.reported_acchr_at_20 * 100).toFixed(1)}%</span>
            </span>
          )}
        </div>
      )}
    </header>
  );
}
