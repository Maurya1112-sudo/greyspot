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
        <div className="app-header-right" title="Audit trail — every number traceable to a model version and generation time">
          <span className="audit-item">
            <span className="audit-label">Model</span>
            <span className="audit-value">{modelInfo.model_version}</span>
          </span>
          <span className="audit-item">
            <span className="audit-label">Test year</span>
            <span className="audit-value">{modelInfo.test_year}</span>
          </span>
          <span className="audit-item">
            <span className="audit-label">Segments</span>
            <span className="audit-value">{modelInfo.n_segments.toLocaleString()}</span>
          </span>
          <span className="audit-item">
            <span className="audit-label">90% coverage</span>
            <span className="audit-value">{(modelInfo.conformal.empirical_coverage * 100).toFixed(1)}%</span>
          </span>
        </div>
      )}
    </header>
  );
}
