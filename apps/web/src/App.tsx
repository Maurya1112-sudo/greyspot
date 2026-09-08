import { useEffect, useRef, useState } from "react";
import { api, type BoroughSummary, type ModelInfo, type PriorityQueueRow, type RankBy, type RoadEvidence } from "./api";
import { Header } from "./components/Header";
import { MapView } from "./components/MapView";
import { PriorityQueue } from "./components/PriorityQueue";
import { EvidencePanel } from "./components/EvidencePanel";
import "./App.css";

export default function App() {
  const [boroughs, setBoroughs] = useState<BoroughSummary[]>([]);
  const [selectedBorough, setSelectedBorough] = useState<string>("Westminster");
  const [modelInfo, setModelInfo] = useState<ModelInfo | null>(null);
  const [roadsGeoJSON, setRoadsGeoJSON] = useState<GeoJSON.FeatureCollection | null>(null);
  const [queueRows, setQueueRows] = useState<PriorityQueueRow[]>([]);
  const [rankBy, setRankBy] = useState<RankBy>("model");
  const [selectedSegmentId, setSelectedSegmentId] = useState<string | null>(null);
  const [evidence, setEvidence] = useState<RoadEvidence | null>(null);
  const [loadingBorough, setLoadingBorough] = useState(true);
  const [loadingEvidence, setLoadingEvidence] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [evidenceError, setEvidenceError] = useState<string | null>(null);

  // Load the borough registry once.
  useEffect(() => {
    api.boroughs().then(setBoroughs).catch((e) => setError(String(e)));
  }, []);

  // Load everything for the selected borough.
  useEffect(() => {
    let cancelled = false;
    setLoadingBorough(true);
    setError(null);
    setSelectedSegmentId(null);
    setEvidence(null);

    Promise.all([
      api.modelInfo(selectedBorough),
      api.roadsGeoJSON(selectedBorough),
      api.priorityQueue(selectedBorough, 30, rankBy),
    ])
      .then(([info, roads, queue]) => {
        if (cancelled) return;
        setModelInfo(info);
        setRoadsGeoJSON(roads);
        setQueueRows(queue);
      })
      .catch((e) => !cancelled && setError(String(e)))
      .finally(() => !cancelled && setLoadingBorough(false));

    return () => {
      cancelled = true;
    };
    // rankBy intentionally excluded: switching it re-fetches only the
    // queue (see the effect below), not the whole borough - re-running
    // this effect too would flash the map/model-info loading state for a
    // change that doesn't affect either.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [selectedBorough]);

  // Re-fetch just the priority queue when the ranking method changes,
  // without touching the map or model-info (those are shared by both
  // rankings - only which segments sort to the top differs).
  useEffect(() => {
    let cancelled = false;
    api
      .priorityQueue(selectedBorough, 30, rankBy)
      .then((queue) => !cancelled && setQueueRows(queue))
      .catch((e) => !cancelled && setError(String(e)));
    return () => {
      cancelled = true;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [rankBy]);

  // Load evidence for the selected segment.
  //
  // The borough-switch effect above resets selectedSegmentId to null, but
  // that reset only *commits* on the next render - in the render where
  // selectedBorough has already changed, this effect still sees the old
  // (now wrong-borough) selectedSegmentId, since both effects fire off the
  // same render's state snapshot. Without the boroughChangedRef guard this
  // fires one real request for the previous borough's segment ID against
  // the new borough (harmless 404 today since IDs are effectively unique
  // per borough, but wrong in principle and would show mislabelled
  // evidence in the unlikely event of an ID collision across boroughs).
  const boroughChangedRef = useRef(selectedBorough);
  useEffect(() => {
    const boroughJustChanged = boroughChangedRef.current !== selectedBorough;
    boroughChangedRef.current = selectedBorough;
    if (boroughJustChanged || !selectedSegmentId) return;
    let cancelled = false;
    setLoadingEvidence(true);
    setEvidenceError(null);
    api
      .roadEvidence(selectedBorough, selectedSegmentId)
      .then((data) => !cancelled && setEvidence(data))
      .catch((e) => !cancelled && setEvidenceError(String(e)))
      .finally(() => !cancelled && setLoadingEvidence(false));
    return () => {
      cancelled = true;
    };
  }, [selectedBorough, selectedSegmentId]);

  return (
    <div className="app-shell">
      <Header
        boroughs={boroughs}
        selectedBorough={selectedBorough}
        onSelectBorough={setSelectedBorough}
        modelInfo={modelInfo}
      />
      {error && (
        <div className="banner banner--error">
          Could not reach the Greyspot API at the configured address. Is it running?
          <span className="mono small"> {error}</span>
        </div>
      )}
      <main className="app-main">
        <div className="map-column">
          <MapView
            roadsGeoJSON={roadsGeoJSON}
            selectedSegmentId={selectedSegmentId}
            onSelectSegment={setSelectedSegmentId}
            loading={loadingBorough}
          />
          <div className="disclaimer-bar">
            Research signal only — not a safety guarantee. Grey roads have no recent modelled estimate.
          </div>
        </div>
        <PriorityQueue
          rows={queueRows}
          selectedSegmentId={selectedSegmentId}
          onSelect={setSelectedSegmentId}
          loading={loadingBorough}
          rankBy={rankBy}
          onChangeRankBy={setRankBy}
        />
        <EvidencePanel evidence={evidence} loading={loadingEvidence} error={evidenceError} />
      </main>
    </div>
  );
}
