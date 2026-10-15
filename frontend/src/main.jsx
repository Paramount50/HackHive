import React, { useEffect, useState } from "react";
import { createRoot } from "react-dom/client";
import "./styles.css";

const labels = {
  person_down: "Person down",
  crowd_gathering: "Crowd gathering",
  suspected_tailgating: "Suspected tailgating"
};
const terminalReplayStates = new Set(["completed", "failed", "stopped"]);

async function request(path, options) {
  const response = await fetch(path, options);
  if (!response.ok) throw new Error(`${response.status}: ${await response.text()}`);
  return response.json();
}

function StatusBar({ status }) {
  return (
    <section className="status-bar panel">
      <div><span className="eyebrow">SOURCE</span><strong>Fixture Gate</strong><span className="badge replay">REPLAY</span></div>
      <div><span className="eyebrow">HEALTH</span><strong>{status?.source?.health ?? "loading"}</strong></div>
      <div><span className="eyebrow">ANALYSIS</span><strong className="warning">Degraded · fixture</strong></div>
      <div><span className="eyebrow">POLICY</span><strong>{status?.policy ?? "loading"}</strong></div>
    </section>
  );
}

function ReplayControl({ job, onJob, onError }) {
  const [sourceName, setSourceName] = useState("demo.mp4");
  const [sampleFps, setSampleFps] = useState(5);

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
        body: JSON.stringify({ source_name: sourceName, sample_fps: Number(sampleFps), pace: true })
      }));
      onError("");
    } catch (exception) { onError(exception.message); }
  }

  async function stop() {
    try { onJob(await request(`/api/replays/${job.session_id}/stop`, { method: "POST" })); }
    catch (exception) { onError(exception.message); }
  }

  return (
    <section className="replay-control panel">
      <form onSubmit={start}>
        <div><span className="eyebrow">REAL VIDEO REPLAY</span><strong>Decode a server-local recording</strong></div>
        <label>File in recordings/<input value={sourceName} onChange={(event) => setSourceName(event.target.value)} required /></label>
        <label>Sample FPS<input type="number" min="0.5" max="10" step="0.5" value={sampleFps} onChange={(event) => setSampleFps(event.target.value)} /></label>
        <button type="submit" disabled={job && !terminalReplayStates.has(job.status)}>Start replay</button>
        {job && !terminalReplayStates.has(job.status) && <button type="button" className="secondary" onClick={stop}>Stop</button>}
      </form>
      {job && <div className="replay-progress">
        <div><span className={`job-state ${job.status}`}>{job.status}</span><strong>{job.frame_count} sampled frame(s)</strong><span className="metadata">{job.source_name}</span></div>
        {job.latest_frame && <img src={`${job.latest_frame.frame_url}?v=${job.frame_count}`} alt="Latest sampled replay frame" />}
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
    "List person-down incidents from the last hour"
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
            {result.filters && <span className="metadata">Applied filters: {JSON.stringify(result.filters)}</span>}
            {result.needs_clarification && <span className="metadata">Try one of the supported prompts above.</span>}
            {result.events?.map((event) => <button className="query-result" key={event.event_id} onClick={() => onSelect(event)}>
              <span>{labels[event.category] ?? event.category}</span><small>{event.summary}</small><code>Citation · {event.event_id}</code>
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
      <div className="event-heading">
        <span className={`category ${event.category}`}>{labels[event.category] ?? event.category}</span>
        <span className="review-state">{event.review_status}</span>
      </div>
      <p>{event.summary}</p>
      <div className="metadata">System decision: {event.decision_state} · Evidence score {score}/4 · uncalibrated</div>
      <div className="actions">
        <button onClick={(e) => { e.stopPropagation(); onReview(event, "confirm"); }}>Confirm</button>
        <button className="danger" onClick={(e) => { e.stopPropagation(); onReview(event, "reject"); }}>Reject</button>
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
        <dt>Authorization</dt><dd>{event.authorization_state} · {event.access_evidence_source ?? "visual evidence only"}</dd>
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
  const [category, setCategory] = useState("");
  const [message, setMessage] = useState("Loading recorded incidents…");
  const [error, setError] = useState("");
  const [replayJob, setReplayJob] = useState(null);

  async function loadEvents() {
    try {
      const query = category ? `?category=${encodeURIComponent(category)}` : "";
      const result = await request(`/api/events${query}`);
      setEvents(result.events);
      setMessage(result.events.length ? `${result.count} recorded incident(s)` : "No matching recorded incidents.");
      setError("");
    } catch (exception) { setError(exception.message); }
  }

  useEffect(() => {
    request("/api/status").then(setStatus).catch((exception) => setError(exception.message));
    loadEvents();
  }, [category]);

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
      <div className="policy panel"><strong>Fixture mode</strong><span>Simulated access grants</span><span>Timezone: Asia/Kolkata</span><span className="warning">Not real inference</span></div>
      <ReplayControl job={replayJob} onJob={setReplayJob} onError={setError} />
      <QueryPanel onSelect={selectEvent} onError={setError} />
      {error && <div className="error panel">{error}</div>}
      <section className="workspace">
        <div className="feed panel">
          <div className="section-heading"><div><span className="eyebrow">INCIDENT TIMELINE</span><h2>Recorded incidents</h2></div>
            <select value={category} onChange={(e) => setCategory(e.target.value)}><option value="">All categories</option>{Object.entries(labels).map(([value, label]) => <option key={value} value={value}>{label}</option>)}</select>
          </div>
          <p className="metadata">{message}</p>
          {events.map((event) => <EventCard key={event.event_id} event={event} selected={selected?.event_id === event.event_id} onSelect={selectEvent} onReview={review} />)}
          {!events.length && <div className="empty-state">No matching recorded incidents.</div>}
        </div>
        <EvidenceDrawer event={selected} onClose={() => setSelected(null)} />
      </section>
      <footer>Coverage is limited to recorded fixture sessions. Real capture, model inference, and media clips are not connected yet.</footer>
    </main>
  );
}

createRoot(document.getElementById("root")).render(<App />);
