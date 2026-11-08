import { useMemo, useState, type ReactNode } from "react";
import {
  Bookmark,
  ChevronDown,
  ChevronRight,
  FileText,
  FlaskConical,
  History,
  Lightbulb,
  Moon,
  MoreHorizontal,
  Search,
  Send,
  Sun,
  TrendingUp,
} from "lucide-react";
import { research, type Citation, type ResearchResponse, type RetrievedChunk } from "./api/research";

type Page = "research" | "results" | "latest" | "trends" | "focus" | "paper";

const starterQuestions = [
  "How can AI explain medical reports to patients?",
  "What methods reduce the cost of routing queries between AI models?",
  "How do agents learn reusable task models from computer-use traces?",
];

const focusAreas = [
  "Responsible AI",
  "AI Agents",
  "AI Governance",
  "Cybersecurity",
  "Financial Services AI",
  "Healthcare / Life Sciences",
  "Model Efficiency",
  "Workforce Automation",
];

function App() {
  const [page, setPage] = useState<Page>("research");
  const [query, setQuery] = useState("");
  const [activeQuery, setActiveQuery] = useState("");
  const [focus, setFocus] = useState<string[]>([]);
  const [result, setResult] = useState<ResearchResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [dark, setDark] = useState(false);

  const submit = async () => {
    const trimmed = query.trim();
    if (!trimmed || loading) return;
    setError("");
    setLoading(true);
    setPage("results");
    setActiveQuery(trimmed);
    try {
      const nextResult = await research(trimmed, focus);
      setResult(nextResult);
    } catch (requestError) {
      setError(requestError instanceof Error ? requestError.message : "Research failed.");
      setResult(null);
    } finally {
      setLoading(false);
    }
  };

  const newResearch = () => {
    setQuery("");
    setActiveQuery("");
    setResult(null);
    setError("");
    setPage("research");
  };

  return (
    <div className={dark ? "app dark" : "app"}>
      <Sidebar page={page} onNavigate={setPage} onNewResearch={newResearch} />
      <main className="workspace">
        <header className="topbar">
          <div className="brand mobile-brand">
            <span className="brand-mark">KPMG</span>
            <span>Research Intelligence</span>
          </div>
          <div className="topbar-actions">
            <span className="model-pill">Model selector</span>
            <button className="icon-button" onClick={() => setDark((value) => !value)} aria-label="Toggle theme">
              {dark ? <Sun size={16} /> : <Moon size={16} />}
            </button>
            <button className="outline-button" onClick={newResearch}>New Research</button>
          </div>
        </header>
        {page === "research" && (
          <ResearchPage
            query={query}
            setQuery={setQuery}
            focus={focus}
            setFocus={setFocus}
            onSubmit={submit}
            onStarter={(value) => setQuery(value)}
            onNavigate={setPage}
          />
        )}
        {page === "results" && (
          <ResultsPage
            query={activeQuery}
            result={result}
            loading={loading}
            error={error}
            onBack={() => setPage("research")}
            onNewResearch={newResearch}
          />
        )}
        {page === "latest" && <PrototypePage title="Latest AI" icon={<FlaskConical />} copy="The latest AI research feed will connect here." />}
        {page === "trends" && <PrototypePage title="Trends" icon={<TrendingUp />} copy="Emerging AI research trends will connect here." />}
        {page === "focus" && <FocusPage />}
        {page === "paper" && <PrototypePage title="Paper Workspace" icon={<FileText />} copy="Paper metadata, PDF preview, and evidence tools will connect here." />}
      </main>
    </div>
  );
}

function Sidebar({ page, onNavigate, onNewResearch }: { page: Page; onNavigate: (page: Page) => void; onNewResearch: () => void }) {
  return (
    <aside className="sidebar">
      <div className="brand sidebar-brand">
        <span className="brand-mark">KPMG</span>
        <span><strong>Research Intelligence</strong><small>AI research workspace</small></span>
      </div>
      <button className="new-research" onClick={onNewResearch}><span>＋</span> New Research</button>
      <div className="side-label">Discover</div>
      <SideLink icon={<Search size={15} />} active={page === "research"} onClick={() => onNavigate("research")}>Research</SideLink>
      <SideLink icon={<FlaskConical size={15} />} active={page === "latest"} onClick={() => onNavigate("latest")}>Latest AI</SideLink>
      <SideLink icon={<TrendingUp size={15} />} active={page === "trends"} onClick={() => onNavigate("trends")}>Trends</SideLink>
      <SideLink icon={<Lightbulb size={15} />} active={page === "focus"} onClick={() => onNavigate("focus")}>KPMG Focus</SideLink>
      <div className="side-label">Recent</div>
      <div className="side-placeholder"><History size={15} />Recent research sessions will appear here.</div>
      <div className="side-label">Saved</div>
      <div className="side-placeholder"><Bookmark size={15} />Saved research will appear here.</div>
      <div className="sidebar-footer">Stage 1 frontend · prototype areas are clearly marked</div>
    </aside>
  );
}

function SideLink({ icon, active, onClick, children }: { icon: ReactNode; active: boolean; onClick: () => void; children: ReactNode }) {
  return <button className={active ? "side-link active" : "side-link"} onClick={onClick}>{icon}<span>{children}</span></button>;
}

function ResearchPage({ query, setQuery, focus, setFocus, onSubmit, onStarter, onNavigate }: { query: string; setQuery: (value: string) => void; focus: string[]; setFocus: (value: string[]) => void; onSubmit: () => void; onStarter: (value: string) => void; onNavigate: (page: Page) => void }) {
  const [showMoreFocus, setShowMoreFocus] = useState(false);
  const toggleFocus = (area: string) => setFocus(focus.includes(area) ? focus.filter((item) => item !== area) : [...focus, area]);
  const visibleFocusAreas = showMoreFocus ? focusAreas : focusAreas.slice(0, 4);
  return (
    <div className="page-content research-page">
      <div className="eyebrow">KPMG AI RESEARCH</div>
      <h1>What would you like to research?</h1>
      <p className="intro-copy">Explore AI research and translate evidence into business insight.</p>
      <section className="composer">
        <textarea value={query} onChange={(event) => setQuery(event.target.value)} onKeyDown={(event) => { if ((event.metaKey || event.ctrlKey) && event.key === "Enter") onSubmit(); }} placeholder="Ask about AI research, trends, risks, governance, agents, infrastructure, or business implications…" aria-label="Research question" />
        <div className="composer-focus">
          <div className="composer-focus-head">
            <span>Research focus</span>
            <button className="more-focus-button" onClick={() => setShowMoreFocus((value) => !value)} aria-expanded={showMoreFocus}>
              <MoreHorizontal size={15} /> More <ChevronDown className={showMoreFocus ? "rotated" : ""} size={13} />
            </button>
          </div>
          <div className="composer-chips">
            {visibleFocusAreas.map((area) => <button key={area} className={focus.includes(area) ? "composer-chip selected" : "composer-chip"} onClick={() => toggleFocus(area)}>{area}</button>)}
          </div>
          <div className="composer-prototype-note">Focus selections are prototype-only and do not affect retrieval yet.</div>
        </div>
        <div className="composer-bottom">
          <span className="evidence-footer"><span className="evidence-dot" /> Evidence-led research <span className="footer-divider" /> Source-backed business insight</span>
          <button className="send-button" onClick={onSubmit} disabled={!query.trim()} aria-label="Submit research question"><Send size={17} /></button>
        </div>
      </section>
      <section className="suggestions"><div className="section-heading">Suggested for you</div>{starterQuestions.map((question) => <button className="suggestion-row" key={question} onClick={() => onStarter(question)}><span className="suggestion-question"><Search size={14} /><span>{question}</span></span><ChevronRight size={15} /></button>)}</section>
      <section className="discovery"><div className="section-heading">Explore AI research</div><div className="discovery-grid"><EntryCard icon={<FlaskConical size={17} />} title="Latest AI" copy="Browse the newest research signals." onClick={() => onNavigate("latest")} /><EntryCard icon={<TrendingUp size={17} />} title="Trends" copy="Follow emerging areas across AI research." onClick={() => onNavigate("trends")} /><EntryCard icon={<Lightbulb size={17} />} title="KPMG Focus" copy="Explore business-relevant research themes." onClick={() => onNavigate("focus")} /></div></section>
    </div>
  );
}

function EntryCard({ icon, title, copy, onClick }: { icon: ReactNode; title: string; copy: string; onClick: () => void }) {
  return <button className="entry-card" onClick={onClick}><span className="entry-card-top"><span className="entry-icon">{icon}</span><ChevronRight className="entry-chevron" size={15} /></span><span className="entry-title">{title}</span><span className="entry-copy">{copy}</span></button>;
}

function ResultsPage({ query, result, loading, error, onBack, onNewResearch }: { query: string; result: ResearchResponse | null; loading: boolean; error: string; onBack: () => void; onNewResearch: () => void }) {
  return <div className="page-content results-page"><button className="back-link" onClick={onBack}>← Back to Research</button><div className="query-context"><span>Current research question</span><strong>{query}</strong></div>{loading && <LoadingState />}{error && <div className="error-banner"><strong>Research could not be completed.</strong><span>{error}</span><button onClick={onNewResearch}>Start a new research question</button></div>}{!loading && !error && result && <Briefing result={result} />}</div>;
}

function LoadingState() {
  return <section className="loading-card"><div className="loading-orb"><Search size={20} /></div><h2>Building your research briefing</h2><p>Searching AI research, evaluating evidence, and translating findings into business insight.</p><div className="progress-steps"><span className="complete">Question understood</span><span className="active-step">Searching research</span><span>Evaluating evidence</span><span>Preparing insight</span></div></section>;
}

function Briefing({ result }: { result: ResearchResponse }) {
  const citations = useMemo(() => new Map((result.citations ?? []).map((citation: Citation) => [citation.paper_id, citation.supporting_quote])), [result.citations]);
  const chunks = result.retrieved_chunks ?? [];
  return <><div className="section-heading result-heading">Executive Insight</div><section className="insight-card">{result.answer || "No answer was returned."}</section><div className="confidence-row"><strong>{result.confidence ? `${result.confidence[0].toUpperCase()}${result.confidence.slice(1)} confidence` : "Confidence unavailable"}</strong><span>{typeof result.top_score === "number" ? `Top relevance ${(result.top_score * 100).toFixed(1)}%` : ""}</span></div><div className="section-heading result-heading">Evidence / Sources</div><div className="evidence-list">{chunks.length ? chunks.map((chunk, index) => <EvidenceCard key={`${chunk.paper_id}-${index}`} chunk={chunk} quote={citations.get(chunk.paper_id)} />) : <div className="empty-card">No evidence was returned.</div>}</div><section className="more-card"><div className="section-heading">More to explore</div><p>Follow-up research controls will connect here in a later stage.</p></section></>;
}

function EvidenceCard({ chunk, quote }: { chunk: RetrievedChunk; quote?: string }) {
  return <article className="evidence-card"><div className="evidence-title"><span>{chunk.title || "Untitled paper"}</span>{typeof chunk.score === "number" && <b>{(chunk.score * 100).toFixed(1)}%</b>}</div>{chunk.section_title && <div className="evidence-meta">{chunk.section_title}</div>}{quote && <blockquote>{quote}</blockquote>}<details><summary>Show retrieved evidence</summary><p>{chunk.chunk_text || "No chunk preview returned."}</p></details></article>;
}

function PrototypePage({ title, icon, copy }: { title: string; icon: ReactNode; copy: string }) {
  return <div className="page-content prototype-page"><div className="prototype-icon">{icon}</div><div className="eyebrow">PROTOTYPE WORKSPACE</div><h1>{title}</h1><p className="intro-copy">{copy}</p><div className="empty-card"><strong>This page is not connected to live backend data yet.</strong><span>Stage 1 keeps the approved visual structure in place while the research flow is connected first.</span></div></div>;
}

function FocusPage() {
  return <div className="page-content prototype-page"><div className="eyebrow">KPMG RESEARCH TOPICS</div><h1>KPMG Focus</h1><p className="intro-copy">Business-oriented AI research themes reserved for future discovery and filtering.</p><div className="focus-grid">{focusAreas.map((area) => <div className="focus-tile" key={area}>{area}<span>Topic workspace</span></div>)}</div></div>;
}

export default App;
