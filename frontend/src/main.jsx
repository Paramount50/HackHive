import React, { useEffect, useState } from "react";
import { createRoot } from "react-dom/client";
import "./styles.css";

const labels = {
  person_down: "Person down",
  crowd_gathering: "Crowd gathering",
  suspected_tailgating: "Suspected tailgating",
  aggression_fight: "Possible physical altercation",
  fire_smoke: "Possible fire or smoke",
  intrusion_unauthorized_access: "Possible restricted-area intrusion"
};
const maturity = {
  person_down: "Integrated",
  crowd_gathering: "Integrated",
  suspected_tailgating: "Integrated",
  aggression_fight: "Integrated",
  fire_smoke: "Experimental · no candidate generator",
  intrusion_unauthorized_access: "Experimental · no candidate generator"
};
const verificationLabels = {
  supported: "Supported",
  contradicted: "Contradicted",
  inconclusive: "Inconclusive",
  insufficient_evidence: "Insufficient evidence",
  category_conflict: "Category conflict",
  pending: "Pending",
  running: "Running",
  completed: "Completed",
  rejected: "Rejected",
  failed: "Failed",
  expired: "Expired",
  not_configured: "Not configured",
  not_run: "Not run"
};
const terminalReplayStates = new Set(["completed", "failed", "stopped"]);

const titleCase = (value = "") => value.replaceAll("_", " ").replace(/\b\w/g, (letter) => letter.toUpperCase());
const formatTime = (value) => value ? new Date(value).toLocaleString("en-IN", { timeZone: "Asia/Kolkata" }) : "Unknown";
const formatOffset = (value) => {
  if (!Number.isFinite(value)) return "Unknown";
  const total = Math.max(0, Math.floor(value));
  const hours = Math.floor(total / 3_600_000);
  const minutes = Math.floor((total % 3_600_000) / 60_000);
  const seconds = ((total % 60_000) / 1000).toFixed(3).padStart(6, "0");
  return `${hours ? `${String(hours).padStart(2, "0")}:` : ""}${String(minutes).padStart(2, "0")}:${seconds}`;
};
const eventEndOffset = (event) => Number.isFinite(event.end_pts_ms)
  ? event.end_pts_ms
  : event.start_pts_ms + Math.max(0, event.last_observed_utc_ms - event.start_utc_ms);
const hasVideoOffset = (event) => event.session_mode === "replay"
  && Number.isFinite(event.start_pts_ms);
const dateTimeToUtcMs = (value, timezone) => {
  if (!value) return null;
  const [datePart, timePart] = value.split("T");
  const [year, month, day] = datePart.split("-").map(Number);
  const [hour, minute] = timePart.split(":").map(Number);
  const offsetMinutes = timezone === "UTC" ? 0 : 330;
  return Date.UTC(year, month - 1, day, hour, minute) - offsetMinutes * 60_000;
};
const readableFilters = (filters, timezone) => [
  ...(filters?.categories ?? []).map((category) => labels[category] ?? category),
  ...(filters?.camera_ids ?? []),
  filters?.source_name,
  filters?.start_utc_ms != null ? `From ${formatTime(filters.start_utc_ms)} (${timezone})` : null,
  filters?.end_utc_ms != null ? `Before ${formatTime(filters.end_utc_ms)} (${timezone})` : null,
  filters?.start_pts_ms != null && filters?.end_pts_ms != null
    ? `Video offset ${formatOffset(filters.start_pts_ms)}–${formatOffset(filters.end_pts_ms)}` : null,
  ...(filters?.review_statuses ?? []).map((status) => `Human review: ${titleCase(status)}`),
  filters?.has_ready_media != null ? (filters.has_ready_media ? "Playable evidence" : "No playable evidence") : null
].filter(Boolean);
const displayVerification = (verification) => {
  const value = verificationLabels[verification?.verdict ?? verification?.status] ?? titleCase(verification?.status ?? "not_run");
  return verification?.verdict || ["pending", "running", "completed", "rejected", "failed", "expired"].includes(verification?.status)
    ? `VLM assessment · ${value}`
    : value;
};
const needsPriorityReview = (event) => [
  "contradicted", "inconclusive", "insufficient_evidence", "category_conflict"
].includes(event.verification?.verdict)
  || ["failed", "expired", "not_configured"].includes(event.verification?.status);
const evidenceAvailability = (event) => [event.has_keyframe && "Keyframe", event.has_clip && "Sampled evidence clip"].filter(Boolean).join(" · ") || "Unavailable";

async function request(path, options) {
  const response = await fetch(path, options);
  if (!response.ok) throw new Error(`${response.status}: ${await response.text()}`);
  return response.json();
}

function StatusBar({ status }) {
  const analysis = status?.analysis;
  const vlm = analysis?.vlm;
  const query = status?.query;
  return (
    <section className="status-bar panel">
      <div><span className="eyebrow">YOLO</span><strong className={analysis?.detector_ready ? "healthy" : "warning"}>{analysis?.detector_ready ? "Runtime ready" : "Unavailable"}</strong><span className="metadata">{analysis?.detector_weights ?? "No local weights"}</span></div>
      <div><span className="eyebrow">XCLIP</span><strong className={analysis?.action_model_ready ? "healthy" : "warning"}>{analysis?.action_model_ready ? "Runtime ready" : "Unavailable"}</strong><span className="metadata">Action ranking</span></div>
      <div><span className="eyebrow">SMOLVLM</span><strong className={vlm?.state === "ready" ? "healthy" : "warning"}>{titleCase(vlm?.state ?? "loading")}</strong><span className="metadata">{vlm?.variant ?? "—"} · {vlm?.pending ?? 0} queued · max {vlm?.capacity ?? 3}</span></div>
      <div><span className="eyebrow">SAMPLING</span><strong>{titleCase(vlm?.selection_strategy ?? "swift")}</strong><span className="metadata">Four-frame verification budget</span></div>
      <div><span className="eyebrow">CHAT</span><strong className={query?.state === "ready" ? "healthy" : "warning"}>{titleCase(query?.state ?? "loading")}</strong><span className="metadata">{query?.semantic_models_state === "ready" ? `Qwen + MiniLM · ${query?.device ?? "CPU"}` : "Structured search available"}</span></div>
    </section>
  );
}

function ReplayControl({ job, onJob, onError }) {
  const [sourceName, setSourceName] = useState("");
  const [sampleFps, setSampleFps] = useState(5);
  const [gateLineY, setGateLineY] = useState("");
  const [zone, setZone] = useState([0, 0, 100, 100]);
  const [uploadFile, setUploadFile] = useState(null);
  const [uploading, setUploading] = useState(false);
  const [recordings, setRecordings] = useState([]);
  const [uploadLimit, setUploadLimit] = useState(null);

  async function loadRecordings() {
    try {
      const result = await request("/api/uploads");
      setRecordings(result.recordings);
      setUploadLimit(result.max_upload_bytes);
    } catch (exception) { onError(exception.message); }
  }

  useEffect(() => { loadRecordings(); }, []);

  useEffect(() => {
    if (!job?.session_id || terminalReplayStates.has(job.status)) return undefined;
    const timer = window.setInterval(() => {
      request(`/api/replays/${job.session_id}`).then(onJob).catch((exception) => onError(exception.message));
    }, 500);
    return () => window.clearInterval(timer);
  }, [job?.session_id, job?.status]);

  async function start(event) {
    event.preventDefault();
    try {
      onJob(await request("/api/replays", {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({
          source_name: sourceName,
          sample_fps: Number(sampleFps),
          decode_strategy: "sequential",
          pace: true,
          ...(gateLineY === "" ? {} : { gate_line_y: Number(gateLineY) / 100 }),
          zone: zone.map((value) => Number(value) / 100)
        })
      }));
      onError("");
    } catch (exception) { onError(exception.message); }
  }

  async function stop() {
    try { onJob(await request(`/api/replays/${job.session_id}/stop`, { method: "POST" })); }
    catch (exception) { onError(exception.message); }
  }

  async function upload() {
    if (!uploadFile) return;
    setUploading(true);
    try {
      const body = new FormData();
      body.append("file", uploadFile);
      const result = await request("/api/uploads", { method: "POST", body });
      setSourceName(result.source_name);
      setUploadFile(null);
      await loadRecordings();
      onError("");
    } catch (exception) { onError(exception.message); }
    finally { setUploading(false); }
  }

  return (
    <section className="replay-control panel">
      <div className="upload-row">
        <div><span className="eyebrow">1 / UPLOAD</span><strong>Select a local video</strong><span className="metadata">MP4, MOV, AVI or MKV{uploadLimit ? ` · up to ${Math.floor(uploadLimit / 1024 / 1024)} MiB` : ""}</span></div>
        <input aria-label="Video file" type="file" accept=".mp4,.mov,.avi,.mkv,video/*" onChange={(event) => setUploadFile(event.target.files?.[0] ?? null)} />
        <button type="button" disabled={!uploadFile || uploading} onClick={upload}>{uploading ? "Validating…" : "Upload"}</button>
      </div>
      <form onSubmit={start}>
        <div><span className="eyebrow">2 / ANALYZE</span><strong>Configure the uploaded recording</strong></div>
        <label>Recording<select value={sourceName} onChange={(event) => setSourceName(event.target.value)} required><option value="">Choose a recording</option>{recordings.map((recording) => <option key={recording.recording_id} value={recording.stored_name}>{recording.original_name} · {(recording.duration_ms / 1000).toFixed(1)}s</option>)}</select></label>
        <label>Sample FPS<input type="number" min="0.5" max="10" step="0.5" value={sampleFps} onChange={(event) => setSampleFps(event.target.value)} /></label>
        <label>Gate line from top (%)<input type="number" min="0" max="100" placeholder="optional" value={gateLineY} onChange={(event) => setGateLineY(event.target.value)} /></label>
        {zone.map((value, index) => <label key={index}>Zone {["left", "top", "right", "bottom"][index]} (%)<input type="number" min="0" max="100" value={value} onChange={(event) => setZone((current) => current.map((item, position) => position === index ? event.target.value : item))} /></label>)}
        <p className="metadata replay-help">The zone is a rectangle in the image, ordered left, top, right, bottom. Each value is a percentage; 0, 0, 100, 100 means the whole frame. Crowd rules count people whose detected centers are inside it. The gate line is a separate crossing boundary.</p>
        <button type="submit" disabled={!sourceName || (job && !terminalReplayStates.has(job.status))}>Start analysis</button>
        {job && !terminalReplayStates.has(job.status) && <button type="button" className="secondary" onClick={stop}>Stop</button>}
      </form>
      {job && <div className="replay-progress">
        <div><span className={`job-state ${job.status}`}>{job.status}</span><strong>{job.frame_count} sampled frame(s) · {job.analysis}</strong><span className="metadata">{job.source_name}</span></div>
        {job.latest_frame && <img src={`${job.latest_frame.frame_url}?v=${job.frame_count}`} alt="Latest sampled replay frame" />}
        {job.analysis_message && <span className="metadata">{job.analysis_message}</span>}
        {job.error && <span className="error-text">{job.error}</span>}
      </div>}
    </section>
  );
}

function QueryPanel({ onSelect, onError }) {
  const [question, setQuestion] = useState("Show all tailgating events from camera 1 today");
  const [messages, setMessages] = useState([]);
  const [loading, setLoading] = useState(false);
  const suggestions = [
    "Show suspected tailgating today",
    "How many crowd gatherings were recorded?",
    "Did anyone appear to be lying on the ground?",
    "What unusual incidents happened on camera 1 this morning?"
  ];

  async function ask(event, promptedQuestion = question, offset = 0) {
    event?.preventDefault();
    if (!promptedQuestion.trim()) return;
    setLoading(true);
    try {
      const result = await request("/api/query", {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({ question: promptedQuestion, timezone: "Asia/Kolkata", limit: 20, offset })
      });
      setMessages((previous) => [...previous, { question: promptedQuestion, result }]);
      setQuestion("");
      onError("");
    } catch (exception) { onError(exception.message); }
    finally { setLoading(false); }
  }

  async function loadMore(index) {
    const current = messages[index];
    const offset = current.result.events.length;
    setLoading(true);
    try {
      const page = await request("/api/query", {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({
          question: current.question,
          timezone: current.result.timezone,
          limit: current.result.filters?.limit ?? 20,
          offset
        })
      });
      setMessages((previous) => previous.map((message, messageIndex) => {
        if (messageIndex !== index) return message;
        const events = [...message.result.events, ...page.events];
        return {
          ...message,
          result: {
            ...page,
            events,
            result_count: events.length,
            answer: `Showing ${events.length} of ${page.matching_total} matching recorded incidents.`
          }
        };
      }));
      onError("");
    } catch (exception) { onError(exception.message); }
    finally { setLoading(false); }
  }

  return (
    <section className="query-panel panel">
      <div><span className="eyebrow">ANALYST CHAT</span><h2>Ask about recorded incidents</h2><p className="metadata">Answers are grounded in stored events and cite the matching evidence.</p></div>
      <div className="suggestions">{suggestions.map((suggestion) => <button type="button" className="suggestion" key={suggestion} onClick={() => { setQuestion(suggestion); ask(null, suggestion); }}>{suggestion}</button>)}</div>
      <form onSubmit={ask}>
        <input aria-label="Incident question" placeholder="Ask: any tailgating incidents today?" value={question} onChange={(event) => setQuestion(event.target.value)} />
        <button disabled={loading}>{loading ? "Searching…" : "Send"}</button>
      </form>
      <div className="chat-history">
        {messages.map(({ question: asked, result }, index) => <div className="chat-turn" key={`${asked}-${index}`}>
          <div className="chat-question"><span className="chat-role">YOU</span>{asked}</div>
          <div className={result.needs_clarification ? "query-answer clarification" : "query-answer"}>
            <strong>{typeof result.answer === "string" ? result.answer : JSON.stringify(result.answer)}</strong>
            {result.retrieval_mode && <span className="metadata">
              {result.retrieval_mode === "structured" && result.retrieval?.degraded_reason?.startsWith("semantic_model_unavailable")
                ? "Semantic retrieval unavailable · results use structured filters"
                : `Retrieval: ${result.retrieval_mode}`}
              {result.matching_total != null && (result.retrieval_mode === "structured"
                ? ` · ${result.result_count} shown of ${result.matching_total} structured matches`
                : ` · ${result.result_count} semantic result(s) · ${result.matching_total} total structured candidates`)}
              {result.candidate_scan_truncated && ` · scanned first ${result.candidate_scan_count} candidates only`}
            </span>}
            {readableFilters(result.filters, result.timezone).length > 0 && <div className="applied-filters"><strong>Applied filters</strong>{readableFilters(result.filters, result.timezone).map((filter) => <span key={filter}>{filter}</span>)}</div>}
            {result.model?.id && <span className="metadata">Local model: {result.model.id}</span>}
            {result.retrieval?.degraded_reason && !result.retrieval.degraded_reason.startsWith("semantic_model_unavailable") && <span className="metadata">Fallback: {result.retrieval.degraded_reason}</span>}
            {result.needs_clarification && <span className="metadata">Try one of the supported prompts above.</span>}
            {result.events?.map((event) => <EvidenceResultCard key={event.event_id} event={event} onSelect={onSelect} />)}
            {result.filters?.operation === "list"
              && result.retrieval_mode === "structured"
              && !result.candidate_scan_truncated
              && result.events.length < result.matching_total
              && <button type="button" className="secondary load-more" disabled={loading} onClick={() => loadMore(index)}>Load more matching incidents</button>}
          </div>
        </div>)}
      </div>
    </section>
  );
}

function EvidenceSummary({ event }) {
  const endOffset = eventEndOffset(event);
  const startOffset = hasVideoOffset(event) ? event.start_pts_ms : null;
  const duration = Number.isFinite(startOffset) ? Math.max(0, endOffset - startOffset) : null;
  const recording = event.recording_original_name ?? event.source_name ?? "Recording source unavailable";
  const review = event.review_status === "unreviewed" ? "Human review required" : `Human review · ${titleCase(event.review_status)}`;
  return (
    <div className="evidence-summary">
      {event.thumbnail_media_id
        ? <img className="incident-thumbnail" src={`/api/media/${event.thumbnail_media_id}`} alt={`Keyframe for ${labels[event.category] ?? event.category}`} loading="lazy" />
        : <div className="incident-thumbnail unavailable">No keyframe available</div>}
      <div className="evidence-summary-body">
        <div className="event-heading">
          <strong>{labels[event.category] ?? event.category} · {event.camera_name ?? event.camera_id ?? "Camera unknown"}</strong>
          <span className={`verification ${event.verification?.verdict ?? event.verification?.status}`}>{displayVerification(event.verification)}</span>
        </div>
        <p>{event.summary}</p>
        <div className="source-interval"><strong>{recording}</strong><span>
          {Number.isFinite(startOffset)
            ? `Video offset ${formatOffset(startOffset)}–${formatOffset(endOffset)}${duration != null ? ` · ${(duration / 1000).toFixed(1)}s` : ""}`
            : "Video offset unavailable"}
        </span></div>
        <div className="evidence-badges"><span>{review}</span><span>Candidate · {event.candidate?.source ?? "Source unknown"}</span></div>
        {needsPriorityReview(event) && <span className="priority-review-badge">Priority manual review · {displayVerification(event.verification)}</span>}
        {event.session_mode === "replay" && <small className="metadata">Replay timestamp {formatTime(event.start_utc_ms)} · original capture time unknown</small>}
        {event.semantic_score != null && <small className="metadata">Semantic match · {Math.round(event.semantic_score * 100)}%</small>}
        <small className="metadata">Event {event.event_id}</small>
      </div>
    </div>
  );
}

function EvidenceResultCard({ event, onSelect }) {
  return (
    <article className="query-result-card">
      <EvidenceSummary event={event} />
      <div className="query-result-actions">
        <button type="button" onClick={() => onSelect(event)}>Open details</button>
        <button type="button" className="secondary" onClick={() => onSelect(event)}>Open playback & evidence</button>
      </div>
    </article>
  );
}

function EventCard({ event, selected, onSelect, onReview }) {
  return (
    <article className={`event-card ${selected ? "selected" : ""}`} onClick={() => onSelect(event)}>
      <div className="event-card-body">
      <EvidenceSummary event={event} />
      <div className="card-facts"><span><b>System decision</b>{titleCase(event.decision_state)}</span><span><b>Evidence availability</b>{evidenceAvailability(event)}</span></div>
      <span className={`maturity ${maturity[event.category]?.startsWith("Integrated") ? "integrated" : "experimental"}`}>{maturity[event.category]}</span>
      <div className="actions">
        <button onClick={(e) => { e.stopPropagation(); onReview(event, "confirm"); }}>Confirm</button>
        <button className="danger" onClick={(e) => { e.stopPropagation(); onReview(event, "reject"); }}>Reject</button>
      </div>
      </div>
    </article>
  );
}

function EvidenceDrawer({ event, onClose, onReview }) {
  if (!event) return <aside className="drawer panel empty">Select an incident to inspect its evidence.</aside>;
  const media = event.media ?? [];
  const verification = event.verification ?? { status: "not_run", verdict: null };
  const observations = verification.observation?.observations ?? [];
  const selection = verification.selection;
  const totalLatency = verification.timings_ms?.total;
  const selectedFrames = verification.selected_frames ?? [];
  const mediaTimestamp = (value) => hasVideoOffset(event) && Number.isFinite(value)
    ? formatOffset(value - event.session_started_utc_ms)
    : "Source timestamp unavailable";
  return (
    <aside className="drawer panel" id="evidence-drawer">
      <div className="drawer-header"><h2>Evidence</h2><button onClick={onClose}>Close</button></div>
      <div className="evidence-media">
        {media.filter((item) => item.status === "ready" && item.kind === "keyframe").map((item) =>
          <img key={item.media_id} src={`/api/media/${item.media_id}`} alt="Incident keyframe" />)}
        {media.filter((item) => item.status === "ready" && item.kind === "event_clip").map((item) =>
          <div key={item.media_id}>
            <span className="media-label">{item.kind === "event_clip" ? "Play incident clip" : "Sampled evidence clip"}</span>
            <video controls preload="metadata" src={`/api/media/${item.media_id}#t=${Math.max(0, event.start_utc_ms - item.coverage_start_utc_ms) / 1000}`}>Video playback is unavailable.</video>
            <small className="metadata">Available source coverage: {mediaTimestamp(item.coverage_start_utc_ms)}–{mediaTimestamp(item.coverage_end_utc_ms)}</small>
            {(item.coverage_start_utc_ms > item.requested_start_utc_ms || item.coverage_end_utc_ms < item.requested_end_utc_ms)
              && <small className="warning">Partial clip coverage · requested context is missing.</small>}
          </div>)}
        {!media.some((item) => item.status === "ready" && ["keyframe", "event_clip"].includes(item.kind)) && <div className="evidence-placeholder">No keyframe or playable event clip<br /><span>{media.length ? media.map((item) => `${item.kind}: ${item.status}`).join(" · ") : "No media record was created."}</span></div>}
      </div>
      <section className="evidence-section"><span className="eyebrow">INCIDENT</span><h3>{labels[event.category]}</h3><p>{event.summary}</p>
        <dl><dt>Candidate source</dt><dd>{event.candidate?.source ?? "Unknown"}</dd><dt>System decision</dt><dd>{titleCase(event.decision_state)}</dd><dt>Human review</dt><dd>{titleCase(event.review_status)}</dd><dt>Camera</dt><dd>{event.camera_name ?? event.camera_id}</dd><dt>Recording</dt><dd>{event.recording_original_name ?? event.source_name ?? "Source unavailable"}</dd><dt>Video offset</dt><dd>{hasVideoOffset(event) ? `${formatOffset(event.start_pts_ms)}–${formatOffset(eventEndOffset(event))}` : "Unavailable"}</dd><dt>Replay timestamp</dt><dd>{event.session_mode === "replay" ? `${formatTime(event.start_utc_ms)} · original capture timestamp unknown` : "Not applicable"}</dd><dt>Category maturity</dt><dd>{maturity[event.category]}</dd></dl>
      </section>
      <section className="evidence-section"><span className="eyebrow">SEMANTIC VERIFICATION</span><h3 className={`verification ${verification.verdict ?? verification.status}`}>{displayVerification(verification)}</h3>
        {observations.length ? <ul className="predicate-list">{observations.map((item) => <li key={item.predicate} className={item.supported === true ? "positive" : item.supported === false ? "negative" : "unknown"}><span>{item.supported === true ? "✓" : item.supported === false ? "✕" : "?"}</span><div><strong>{titleCase(item.predicate)}</strong><small>{item.supported === true ? (item.frame_indices ?? []).map((index) => selectedFrames[index] ? `Source ${formatOffset(selectedFrames[index].source_pts_ms)}` : `VLM input ${index}`).join(" · ") : item.supported === false ? "Not supported" : "Insufficient visual evidence"}</small></div></li>)}</ul> : <p className="metadata">{verification.error ?? "No visual predicate result is available."}</p>}
      </section>
      {selectedFrames.length > 0 && <section className="evidence-section"><span className="eyebrow">FRAMES ANALYZED BY THE VLM</span><div className="analyzed-frame-strip">{selectedFrames.map((frame, index) => <figure key={frame.media_id}><img src={`/api/media/${frame.media_id}`} alt={`VLM analyzed frame ${index + 1}`} /><figcaption>{formatOffset(frame.source_pts_ms)} source offset</figcaption></figure>)}</div></section>}
      {selection && <section className="evidence-section"><span className="eyebrow">SAMPLING</span><dl><dt>Actual strategy</dt><dd>{titleCase(selection.strategy)}</dd><dt>Candidate pool</dt><dd>{selection.candidate_frame_count} frames</dd><dt>Selected source timestamps</dt><dd>{(selection.selected_source_pts_ms ?? []).map(formatOffset).join(", ") || "Unavailable"}</dd><dt>Selected pool positions</dt><dd>{selection.selected_indices?.join(", ") ?? "Unavailable"} · internal details, not source frame numbers</dd>{selection.taylor_order && <><dt>Taylor order</dt><dd>{selection.taylor_order}</dd></>}<dt>Selection latency</dt><dd>{selection.latency_ms?.toFixed(1)} ms</dd></dl></section>}
      <section className="evidence-section"><span className="eyebrow">MODEL</span><dl><dt>Model</dt><dd>{verification.model_id ?? "Not run"}</dd><dt>Revision</dt><dd className="mono">{verification.model_revision ?? "—"}</dd><dt>Inference latency</dt><dd>{totalLatency != null ? `${totalLatency.toFixed(1)} ms` : "—"}</dd>{verification.error_code && <><dt>Error</dt><dd>{verification.error_code}</dd></>}</dl></section>
      <section className="evidence-section"><span className="eyebrow">HUMAN REVIEW</span><h3>{titleCase(event.review_status)}</h3><div className="actions"><button onClick={() => onReview(event, "confirm")}>Confirm</button><button className="danger" onClick={() => onReview(event, "reject")}>Reject</button></div></section>
      {(event.category === "suspected_tailgating" || event.category === "intrusion_unauthorized_access") && <section className="evidence-section"><span className="eyebrow">AUTHORIZATION</span><p>{titleCase(event.authorization_state)} · {event.access_evidence_source ?? "No policy/access evidence"}</p></section>}
      <section className="evidence-section"><span className="eyebrow">LIMITATIONS</span><ul>{event.limitations.map((item) => <li key={item}>{item}</li>)}</ul></section>
      <p className="metadata">Event ID: {event.event_id}</p>
    </aside>
  );
}

function App() {
  const [status, setStatus] = useState(null);
  const [events, setEvents] = useState([]);
  const [selected, setSelected] = useState(null);
  const [categories, setCategories] = useState([]);
  const [reviewStatus, setReviewStatus] = useState("");
  const [evidenceFilter, setEvidenceFilter] = useState("");
  const [sourceNameFilter, setSourceNameFilter] = useState("");
  const [startDateTime, setStartDateTime] = useState("");
  const [endDateTime, setEndDateTime] = useState("");
  const [offsetStartSeconds, setOffsetStartSeconds] = useState("");
  const [offsetEndSeconds, setOffsetEndSeconds] = useState("");
  const [message, setMessage] = useState("Loading recorded incidents…");
  const [error, setError] = useState("");
  const [replayJob, setReplayJob] = useState(null);
  const [eventTotal, setEventTotal] = useState(0);
  const [eventLimit, setEventLimit] = useState(20);

  function updateReplay(job) {
    setReplayJob(job);
    if (terminalReplayStates.has(job?.status)) loadEvents();
  }

  async function loadStatus() {
    try { setStatus(await request("/api/status")); }
    catch (exception) { setError(exception.message); }
  }

  async function loadEvents(append = false) {
    try {
      const filters = new URLSearchParams();
      categories.forEach((category) => filters.append("category", category));
      if (reviewStatus) filters.append("review_status", reviewStatus);
      if (evidenceFilter) filters.append("has_evidence", evidenceFilter);
      if (sourceNameFilter.trim()) filters.set("source_name", sourceNameFilter.trim());
      const timezone = status?.timezone ?? "Asia/Kolkata";
      if (startDateTime) filters.set("start_utc_ms", String(dateTimeToUtcMs(startDateTime, timezone)));
      if (endDateTime) filters.set("end_utc_ms", String(dateTimeToUtcMs(endDateTime, timezone)));
      if (offsetStartSeconds !== "") filters.set("start_pts_ms", String(Math.round(Number(offsetStartSeconds) * 1000)));
      if (offsetEndSeconds !== "") filters.set("end_pts_ms", String(Math.round(Number(offsetEndSeconds) * 1000)));
      const offset = append ? events.length : 0;
      filters.set("limit", String(eventLimit));
      filters.set("offset", String(offset));
      const query = filters.toString();
      const result = await request(`/api/events${query ? `?${query}` : ""}`);
      setEvents((previous) => append ? [...previous, ...result.events] : result.events);
      setEventTotal(result.count);
      setMessage(result.count ? `Showing ${Math.min(offset + result.returned, result.count)} of ${result.count} matching incidents.` : "No matching recorded incidents.");
      setError("");
    } catch (exception) { setError(exception.message); }
  }

  useEffect(() => {
    loadStatus();
    setEvents([]);
    loadEvents(false);
  }, [categories, reviewStatus, evidenceFilter, sourceNameFilter, startDateTime, endDateTime, offsetStartSeconds, offsetEndSeconds]);

  useEffect(() => {
    const pendingVerification = events.some((event) => ["pending", "running"].includes(event.verification?.status));
    if (!replayJob || (!pendingVerification && terminalReplayStates.has(replayJob.status))) return undefined;
    setMessage("Analysis is running; newly recorded incidents appear below.");
    const timer = window.setInterval(() => {
      loadEvents();
      loadStatus();
    }, 1000);
    return () => window.clearInterval(timer);
  }, [replayJob?.status, events.some((event) => ["pending", "running"].includes(event.verification?.status)), categories, reviewStatus, evidenceFilter]);

  async function review(event, action) {
    try {
      await request(`/api/events/${event.event_id}/reviews`, {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({ action, actor_label: "operator", note: "Reviewed in React dashboard", request_key: crypto.randomUUID() })
      });
      await loadEvents();
      setSelected(null);
    } catch (exception) { setError(exception.message); }
  }

  async function selectEvent(event) {
    try {
      setSelected(await request(`/api/events/${event.event_id}`));
      setError("");
      window.setTimeout(() => document.getElementById("evidence-drawer")?.scrollIntoView({ behavior: "smooth", block: "start" }), 0);
    } catch (exception) { setError(exception.message); }
  }

  return (
    <main>
      <header><div><span className="eyebrow">LOCAL VIDEO EVENT DESK</span><h1>HackHive</h1></div></header>
      <StatusBar status={status} />
      <div className="policy panel">
        <strong>{status?.analysis?.mode ?? "Checking local models"}</strong>
        <span>{status?.policy ?? "Loading policy"}</span>
        <span>Timezone: {status?.timezone ?? "Asia/Kolkata"}</span>
        <span className={status?.analysis?.degraded ? "warning" : "healthy"}>{status?.analysis?.degraded ? "Analysis unavailable · decode only" : "Local analysis available · operator review required"}</span>
      </div>
      <ReplayControl job={replayJob} onJob={updateReplay} onError={setError} />
      <QueryPanel onSelect={selectEvent} onError={setError} />
      {error && <div className="error panel">{error}</div>}
      <section className="workspace">
        <div className="feed panel">
          <div className="section-heading"><div><span className="eyebrow">INCIDENT TIMELINE</span><h2>Recorded incidents</h2></div>
            <div className="event-filters">
              <label>Categories<select multiple value={categories} onChange={(event) => setCategories(Array.from(event.target.selectedOptions, (option) => option.value))}>{Object.entries(labels).map(([value, label]) => <option key={value} value={value}>{label}</option>)}</select></label>
              <label>Review<select value={reviewStatus} onChange={(event) => setReviewStatus(event.target.value)}><option value="">Any review</option><option value="unreviewed">Unreviewed</option><option value="confirmed">Confirmed</option><option value="rejected">Rejected</option></select></label>
              <label>Evidence<select value={evidenceFilter} onChange={(event) => setEvidenceFilter(event.target.value)}><option value="">Any evidence</option><option value="true">Playable</option><option value="false">Unavailable</option></select></label>
              <label>Recording<input value={sourceNameFilter} placeholder="Filename" onChange={(event) => setSourceNameFilter(event.target.value)} /></label>
              <label>Observed from ({status?.timezone ?? "Asia/Kolkata"})<input type="datetime-local" value={startDateTime} onChange={(event) => setStartDateTime(event.target.value)} /></label>
              <label>Observed until ({status?.timezone ?? "Asia/Kolkata"})<input type="datetime-local" value={endDateTime} onChange={(event) => setEndDateTime(event.target.value)} /></label>
              <label>Video offset from (s)<input type="number" min="0" step="0.001" value={offsetStartSeconds} onChange={(event) => setOffsetStartSeconds(event.target.value)} /></label>
              <label>Video offset until (s)<input type="number" min="0" step="0.001" value={offsetEndSeconds} onChange={(event) => setOffsetEndSeconds(event.target.value)} /></label>
            </div>
          </div>
          <p className="metadata">{message}</p>
          {reviewStatus === "unreviewed" && <p className="review-priority-note">Priority order: VLM contradictions, inconclusive results, and verification failures first. VLM support is an assessment, not a human review.</p>}
          {events.map((event) => <EventCard key={event.event_id} event={event} selected={selected?.event_id === event.event_id} onSelect={selectEvent} onReview={review} />)}
          {events.length < eventTotal && <button className="secondary load-more" onClick={() => loadEvents(true)}>Load more incidents</button>}
          {!events.length && <div className="empty-state">No matching recorded incidents.</div>}
        </div>
        <EvidenceDrawer event={selected} onClose={() => setSelected(null)} onReview={review} />
      </section>
      <footer>{status?.analysis?.message ?? "Checking local analysis availability."} Candidate events remain unreviewed until a person checks them.</footer>
    </main>
  );
}

createRoot(document.getElementById("root")).render(<App />);
