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
const terminalReplayStates = new Set(["completed", "failed", "stopped"]);

async function request(path, options) {
  const response = await fetch(path, options);
  if (!response.ok) throw new Error(`${response.status}: ${await response.text()}`);
  return response.json();
}

function StatusBar({ status }) {
  const analysis = status?.analysis;
  return (
    <section className="status-bar panel">
      <div><span className="eyebrow">SOURCE</span><strong>Local video</strong><span className="badge replay">REPLAY</span></div>
      <div><span className="eyebrow">HEALTH</span><strong>{status?.source?.health ?? "loading"}</strong></div>
      <div><span className="eyebrow">ANALYSIS</span><strong className={analysis?.degraded ? "warning" : ""}>{analysis?.mode ?? "loading"}</strong></div>
      <div><span className="eyebrow">POLICY</span><strong>{analysis?.message ?? "Review required · local processing"}</strong></div>
    </section>
  );
}

function ReplayControl({ job, onJob, onError }) {
  const [sourceName, setSourceName] = useState("");
  const [sampleFps, setSampleFps] = useState(5);
  const [decodeStrategy, setDecodeStrategy] = useState("sequential");
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
          decode_strategy: decodeStrategy,
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
        <label>Decoder<select value={decodeStrategy} onChange={(event) => setDecodeStrategy(event.target.value)}><option value="sequential">Sequential baseline</option><option value="grab">Grab/retrieve trial</option></select></label>
        <label>Gate line from top (%)<input type="number" min="0" max="100" placeholder="optional" value={gateLineY} onChange={(event) => setGateLineY(event.target.value)} /></label>
        {zone.map((value, index) => <label key={index}>Zone {["left", "top", "right", "bottom"][index]} (%)<input type="number" min="0" max="100" value={value} onChange={(event) => setZone((current) => current.map((item, position) => position === index ? event.target.value : item))} /></label>)}
        <button type="submit" disabled={!sourceName || (job && !terminalReplayStates.has(job.status))}>Start analysis</button>
        {job && !terminalReplayStates.has(job.status) && <button type="button" className="secondary" onClick={stop}>Stop</button>}
      </form>
      {job && <div className="replay-progress">
        <div><span className={`job-state ${job.status}`}>{job.status}</span><strong>{job.frame_count} sampled frame(s) · {job.analysis}</strong><span className="metadata">{job.source_name} · {job.decode_strategy} decoder</span></div>
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

  async function ask(event, promptedQuestion = question) {
    event?.preventDefault();
    if (!promptedQuestion.trim()) return;
    setLoading(true);
    try {
      const result = await request("/api/query", {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({ question: promptedQuestion, timezone: "Asia/Kolkata" })
      });
      setMessages((previous) => [...previous, { question: promptedQuestion, result }]);
      setQuestion("");
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
            {result.retrieval_mode && <span className="metadata">Retrieval: {result.retrieval_mode} · {result.citations?.length ?? 0} recorded source(s)</span>}
            {result.model?.id && <span className="metadata">Local model: {result.model.id}</span>}
            {result.retrieval?.degraded_reason && <span className="metadata">Fallback: {result.retrieval.degraded_reason}</span>}
            {result.filters && <span className="metadata">Applied filters: {JSON.stringify(result.filters)}</span>}
            {result.needs_clarification && <span className="metadata">Try one of the supported prompts above.</span>}
            {result.events?.map((event) => <button className="query-result" key={event.event_id} onClick={() => onSelect(event)}>
              <span>{labels[event.category] ?? event.category}</span><small>{event.summary}</small>{event.semantic_score != null && <small>Semantic match · {Math.round(event.semantic_score * 100)}%</small>}<code>Citation · {event.event_id}</code>
            </button>)}
          </div>
        </div>)}
      </div>
    </section>
  );
}

function EventCard({ event, selected, onSelect, onReview }) {
  const score = Math.round(event.confidence_score * 4);
  return (
    <article className={`event-card ${selected ? "selected" : ""}`} onClick={() => onSelect(event)}>
      {event.thumbnail_media_id
        ? <img className="incident-thumbnail" src={`/api/media/${event.thumbnail_media_id}`} alt="Incident keyframe thumbnail" loading="lazy" />
        : <div className="incident-thumbnail unavailable">Evidence unavailable</div>}
      <div className="event-card-body">
      <div className="event-heading">
        <span className={`category ${event.category}`}>{labels[event.category] ?? event.category}</span>
        <span className="review-state">{event.review_status}</span>
      </div>
      <p>{event.summary}</p>
      <div className="metadata">System decision: {event.decision_state} · Evidence score {score}/4 · uncalibrated</div>
      <div className="metadata">{event.has_ready_media ? "Keyframe or clip available" : "No playable evidence available"}</div>
      <div className="actions">
        <button onClick={(e) => { e.stopPropagation(); onReview(event, "confirm"); }}>Confirm</button>
        <button className="danger" onClick={(e) => { e.stopPropagation(); onReview(event, "reject"); }}>Reject</button>
      </div>
      </div>
    </article>
  );
}

function EvidenceDrawer({ event, onClose }) {
  if (!event) return <aside className="drawer panel empty">Select an incident to inspect its evidence.</aside>;
  const media = event.media ?? [];
  return (
    <aside className="drawer panel">
      <div className="drawer-header"><h2>Evidence</h2><button onClick={onClose}>Close</button></div>
      <div className="evidence-media">
        {media.filter((item) => item.status === "ready" && item.kind === "keyframe").map((item) =>
          <img key={item.media_id} src={`/api/media/${item.media_id}`} alt="Incident keyframe" />)}
        {media.filter((item) => item.status === "ready" && item.kind !== "keyframe").map((item) =>
          <video key={item.media_id} controls preload="metadata" src={`/api/media/${item.media_id}`}>Video playback is unavailable.</video>)}
        {!media.some((item) => item.status === "ready") && <div className="evidence-placeholder">No playable evidence<br /><span>{media.length ? media.map((item) => `${item.kind}: ${item.status}`).join(" · ") : "No media record was created."}</span></div>}
      </div>
      <h3>{labels[event.category]}</h3>
      <p>{event.summary}</p>
      <dl>
        <dt>System prediction</dt><dd>{event.decision_state}</dd>
        <dt>Human review</dt><dd>{event.review_status}</dd>
        <dt>Authorization</dt><dd>{event.authorization_state} · {event.access_evidence_source ?? "no policy/access evidence"}</dd>
        <dt>Evidence score</dt><dd>{Math.round(event.confidence_score * 4)}/4 · uncalibrated</dd>
      </dl>
      <h3>Limitations</h3>
      <ul>{event.limitations.map((item) => <li key={item}>{item}</li>)}</ul>
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
  const [message, setMessage] = useState("Loading recorded incidents…");
  const [error, setError] = useState("");
  const [replayJob, setReplayJob] = useState(null);

  function updateReplay(job) {
    setReplayJob(job);
    if (terminalReplayStates.has(job?.status)) loadEvents();
  }

  async function loadStatus() {
    try { setStatus(await request("/api/status")); }
    catch (exception) { setError(exception.message); }
  }

  async function loadEvents() {
    try {
      const filters = new URLSearchParams();
      categories.forEach((category) => filters.append("category", category));
      if (reviewStatus) filters.append("review_status", reviewStatus);
      if (evidenceFilter) filters.append("has_evidence", evidenceFilter);
      const query = filters.toString();
      const result = await request(`/api/events${query ? `?${query}` : ""}`);
      setEvents(result.events);
      setMessage(result.events.length ? `${result.count} recorded incident(s)` : "No matching recorded incidents.");
      setError("");
    } catch (exception) { setError(exception.message); }
  }

  useEffect(() => {
    loadStatus();
    loadEvents();
  }, [categories, reviewStatus, evidenceFilter]);

  useEffect(() => {
    if (!replayJob || terminalReplayStates.has(replayJob.status)) return undefined;
    setMessage("Analysis is running; newly recorded incidents appear below.");
    const timer = window.setInterval(() => {
      loadEvents();
      loadStatus();
    }, 1000);
    return () => window.clearInterval(timer);
  }, [replayJob?.status, categories, reviewStatus, evidenceFilter]);

  async function runFixture(path = "/api/fixtures/run") {
    try {
      const result = await request(path, { method: "POST" });
      setMessage(result.message);
      await loadEvents();
    } catch (exception) { setError(exception.message); }
  }

  async function review(event, action) {
    try {
      await request(`/api/events/${event.event_id}/reviews`, {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({ action, actor_label: "operator", note: "Reviewed in React dashboard", request_key: `${event.event_id}-${action}` })
      });
      await loadEvents();
      setSelected(null);
    } catch (exception) { setError(exception.message); }
  }

  async function selectEvent(event) {
    try {
      setSelected(await request(`/api/events/${event.event_id}`));
      setError("");
    } catch (exception) { setError(exception.message); }
  }

  return (
    <main>
      <header><div><span className="eyebrow">LOCAL VIDEO EVENT DESK</span><h1>HackHive</h1></div><div className="header-actions"><button onClick={() => runFixture()}>Run analysis fixture</button><button onClick={() => runFixture("/api/fixtures/perception-run")}>Run perception fixture</button></div></header>
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
            </div>
          </div>
          <p className="metadata">{message}</p>
          {events.map((event) => <EventCard key={event.event_id} event={event} selected={selected?.event_id === event.event_id} onSelect={selectEvent} onReview={review} />)}
          {!events.length && <div className="empty-state">No matching recorded incidents.</div>}
        </div>
        <EvidenceDrawer event={selected} onClose={() => setSelected(null)} />
      </section>
      <footer>{status?.analysis?.message ?? "Checking local analysis availability."} Candidate events remain unreviewed until a person checks them.</footer>
    </main>
  );
}

createRoot(document.getElementById("root")).render(<App />);
