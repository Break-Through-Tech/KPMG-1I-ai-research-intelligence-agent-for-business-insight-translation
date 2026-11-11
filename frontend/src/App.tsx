import { FormEvent, ReactNode, useEffect, useState } from "react";
import kpmgBlueLogo from "./assets/kpmg-logo-blue.svg";
import kpmgWhiteLogo from "./assets/kpmg-logo-white.svg";
import Markdown from "react-markdown";
import { runResearch, type ResearchResponse } from "./api/research";

type View = "home" | "results" | "latest" | "trends" | "focus" | "paper";
type Vote = "up" | "down" | null;

const Icon = ({ name, size = 18 }: { name: string; size?: number }) => {
  const paths: Record<string, ReactNode> = {
    search: (
      <>
        <circle cx="11" cy="11" r="7" />
        <path d="m20 20-4-4" />
      </>
    ),
    spark: (
      <>
        <path d="m12 3 1.3 4.3L18 9l-4.7 1.7L12 15l-1.3-4.3L6 9l4.7-1.7L12 3Z" />
        <path d="m19 16 .7 2.3L22 19l-2.3.7L19 22l-.7-2.3L16 19l2.3-.7L19 16Z" />
      </>
    ),
    clock: (
      <>
        <circle cx="12" cy="12" r="9" />
        <path d="M12 7v5l3 2" />
      </>
    ),
    trend: (
      <>
        <path d="M3 17 9 11l4 4 8-9" />
        <path d="M15 6h6v6" />
      </>
    ),
    target: (
      <>
        <circle cx="12" cy="12" r="9" />
        <circle cx="12" cy="12" r="4" />
        <path d="m15 9 6-6" />
      </>
    ),
    bookmark: <path d="M6 4.5A1.5 1.5 0 0 1 7.5 3h9A1.5 1.5 0 0 1 18 4.5V21l-6-4-6 4V4.5Z" />,
    info: (
      <>
        <circle cx="12" cy="12" r="9" />
        <path d="M12 11v6M12 7h.01" />
      </>
    ),
    plus: <path d="M12 5v14M5 12h14" />,
    sun: (
      <>
        <circle cx="12" cy="12" r="4" />
        <path d="M12 2v2M12 20v2M4.9 4.9l1.4 1.4M17.7 17.7l1.4 1.4M2 12h2M20 12h2M4.9 19.1l1.4-1.4M17.7 6.3l1.4-1.4" />
      </>
    ),
    moon: <path d="M20 15.5A8.5 8.5 0 0 1 8.5 4 8.5 8.5 0 1 0 20 15.5Z" />,
    arrow: (
      <>
        <path d="M5 12h14" />
        <path d="m15 8 4 4-4 4" />
      </>
    ),
    back: (
      <>
        <path d="M19 12H5" />
        <path d="m9 8-4 4 4 4" />
      </>
    ),
    chevron: <path d="m9 18 6-6-6-6" />,
    down: <path d="m6 9 6 6 6-6" />,
    file: (
      <>
        <path d="M7 3h7l4 4v14H7V3Z" />
        <path d="M14 3v5h5M10 12h5M10 16h5" />
      </>
    ),
    layers: (
      <>
        <path d="m12 3 9 5-9 5-9-5 9-5Z" />
        <path d="m3 12 9 5 9-5M3 16l9 5 9-5" />
      </>
    ),
    thumbsUp: (
      <path d="M7 10v10H3V10h4Zm0 9h9.3a2 2 0 0 0 2-1.6l1.2-6A2 2 0 0 0 17.5 9H14l.7-3.2A2.3 2.3 0 0 0 12.5 3L7 10" />
    ),
    thumbsDown: (
      <path d="M7 14V4H3v10h4Zm0-9h9.3a2 2 0 0 1 2 1.6l1.2 6a2 2 0 0 1-2 2.4H14l.7 3.2a2.3 2.3 0 0 1-2.2 2.8L7 14" />
    ),
    external: (
      <>
        <path d="M14 4h6v6M20 4l-9 9" />
        <path d="M18 13v6a1 1 0 0 1-1 1H5a1 1 0 0 1-1-1V7a1 1 0 0 1 1-1h6" />
      </>
    ),
    copy: (
      <>
        <rect x="8" y="8" width="11" height="12" rx="2" />
        <path d="M16 8V5a1 1 0 0 0-1-1H5a1 1 0 0 0-1 1v11a1 1 0 0 0 1 1h3" />
      </>
    ),
    mail: (
      <>
        <rect x="3" y="5" width="18" height="14" rx="2" />
        <path d="m3 7 9 6 9-6" />
      </>
    ),
    note: (
      <>
        <path d="M5 3h14v18H5zM8 8h8M8 12h8M8 16h5" />
      </>
    ),
    share: (
      <>
        <circle cx="18" cy="5" r="2" />
        <circle cx="6" cy="12" r="2" />
        <circle cx="18" cy="19" r="2" />
        <path d="m8 11 8-5M8 13l8 5" />
      </>
    ),
    highlight: (
      <>
        <path d="m8 16-3 3 4-1 10-10-3-3L6 15l2 1Z" />
        <path d="m14 7 3 3" />
      </>
    ),
    grid: (
      <>
        <rect x="4" y="4" width="6" height="6" />
        <rect x="14" y="4" width="6" height="6" />
        <rect x="4" y="14" width="6" height="6" />
        <rect x="14" y="14" width="6" height="6" />
      </>
    ),
    menu: (
      <>
        <path d="M4 7h16M4 12h16M4 17h16" />
      </>
    ),
    close: (
      <>
        <path d="m6 6 12 12M18 6 6 18" />
      </>
    ),
    check: <path d="m5 12 4 4L19 6" />,
  };
  return (
    <svg
      className="icon"
      width={size}
      height={size}
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.7"
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden="true"
    >
      {paths[name]}
    </svg>
  );
};

const papers = [
  {
    title: "Retrieval-Augmented Generation for Knowledge-Intensive Enterprise Tasks",
    authors: "M. Chen, R. Alvarez, S. Patel",
    date: "2025",
    source: "arXiv",
    category: "Enterprise RAG",
    relevance: 92,
    abstract:
      "A comparative evaluation of retrieval architectures for high-stakes enterprise knowledge work, with emphasis on attribution quality, latency, and operational cost.",
    quote:
      "Grounded retrieval reduced unsupported claims most consistently when evidence quality was evaluated before generation.",
  },
  {
    title: "Measuring the Value of AI Agents in Complex Business Workflows",
    authors: "L. Hoffman, A. Okafor, J. Reid",
    date: "2025",
    source: "Journal of Applied AI Systems",
    category: "AI Agents",
    relevance: 86,
    abstract:
      "A field-oriented framework for measuring agent performance across workflow completion, human intervention, error recovery, and business value.",
    quote:
      "Task completion alone overstated value; the strongest deployments combined bounded autonomy with explicit escalation paths.",
  },
  {
    title: "Operational Governance for Foundation Models: Evidence and Evaluation",
    authors: "E. Williams, N. Singh, T. Becker",
    date: "2024",
    source: "arXiv",
    category: "Responsible AI",
    relevance: 78,
    abstract:
      "An evidence-led review of model evaluation practices, governance controls, and post-deployment monitoring for foundation models.",
    quote:
      "Continuous evaluation linked to business context was more effective than static pre-release benchmarks alone.",
  },
];

type Paper = (typeof papers)[number];

function papersFromResearch(data: ResearchResponse | null): Paper[] {
  const chunks = data?.retrieved_chunks;
  if (!chunks?.length) return papers;

  return chunks.map((chunk) => {
    const citation = data?.citations?.find((item) => item.paper_id === chunk.paper_id);
    return {
      title: chunk.title || "Research source",
      authors: "Metadata unavailable",
      date: "—",
      source: chunk.paper_id || "Research source",
      category: chunk.section_title || "Research source",
      relevance: Math.round((chunk.score ?? 0) * 100),
      abstract: chunk.chunk_text || "",
      quote: citation?.supporting_quote || chunk.chunk_text || "",
    };
  });
}

const nav = [
  { view: "home" as View, label: "Research", icon: "search" },
  { view: "latest" as View, label: "Latest AI", icon: "clock" },
  { view: "trends" as View, label: "Trends", icon: "trend" },
  { view: "focus" as View, label: "KPMG Focus", icon: "target" },
];

function Brand({ compact = false }: { compact?: boolean }) {
  return (
    <div className={`brand ${compact ? "brand-compact" : ""}`}>
      <div className="brand-mark">
        <img className="logo-light" src={kpmgBlueLogo} alt="KPMG" />
        <img className="logo-dark" src={kpmgWhiteLogo} alt="KPMG" />
      </div>
      {!compact && (
        <div className="brand-product">
          <strong>Research</strong>
          <small>Intelligence</small>
        </div>
      )}
    </div>
  );
}

function Sidebar({
  view,
  setView,
  open,
  close,
  collapsed,
  toggleCollapsed,
}: {
  view: View;
  setView: (v: View) => void;
  open: boolean;
  close: () => void;
  collapsed: boolean;
  toggleCollapsed: () => void;
}) {
  const go = (v: View) => {
    setView(v);
    close();
  };
  return (
    <aside className={`sidebar ${open ? "sidebar-open" : ""} ${collapsed ? "collapsed" : ""}`}>
      <div className="sidebar-head">
        <Brand />
        <button className="icon-button mobile-close" onClick={close}>
          <Icon name="close" />
        </button>
      </div>
      <button
        className="sidebar-collapse"
        onClick={toggleCollapsed}
        aria-label={collapsed ? "Expand sidebar" : "Collapse sidebar"}
        title={collapsed ? "Expand sidebar" : "Collapse sidebar"}
      >
        <Icon name="chevron" size={14} />
      </button>
      <button
        className="new-research side-new"
        onClick={() => go("home")}
        data-tooltip="New Research"
        aria-label="New Research"
      >
        <Icon name="plus" size={16} /> <span className="nav-text">New Research</span>
      </button>
      <div className="nav-label">Discover</div>
      <nav>
        {nav.map((n) => (
          <button
            key={n.view}
            className={view === n.view || (n.view === "home" && view === "results") ? "active" : ""}
            onClick={() => go(n.view)}
            data-tooltip={n.label}
            aria-label={n.label}
          >
            <Icon name={n.icon} size={17} />
            <span className="nav-text">{n.label}</span>
          </button>
        ))}
      </nav>
      {/* Not connected to anything yet: no research history, saved items, system status or user accounts.
      <div className="nav-label nav-label-spaced">Your workspace</div>
      <nav>
        <button
          onClick={() => go("results")}
          data-tooltip="Recent Research"
          aria-label="Recent Research"
        >
          <Icon name="clock" size={17} />
          <span className="nav-text">Recent Research</span>
        </button>
        <button data-tooltip="Saved Research" aria-label="Saved Research">
          <Icon name="bookmark" size={17} />
          <span className="nav-text">Saved Research</span>
          <span className="nav-count">4</span>
        </button>
      </nav>
      <div className="sidebar-bottom">
        <button data-tooltip="System Info" aria-label="System Info">
          <Icon name="info" size={17} />
          <span className="nav-text">System Info</span>
          <span className="status-dot" />
        </button>
        <div className="user" data-tooltip="Alex Chen · KPMG Advisory">
          <div className="avatar">AC</div>
          <div className="user-details">
            <strong>Alex Chen</strong>
            <small>KPMG Advisory</small>
          </div>
        </div>
      </div>
      */}
    </aside>
  );
}

function Topbar({
  dark,
  toggleDark,
  openMenu,
}: {
  dark: boolean;
  toggleDark: () => void;
  openMenu: () => void;
}) {
  return (
    <header className="topbar">
      <button className="icon-button menu-button" onClick={openMenu}>
        <Icon name="menu" />
      </button>
      <div className="top-product">
        <span>Research Intelligence</span>
      </div>
      <div className="top-actions">
        <button className="model-select">
          <span className="model-dot" /> Research model <Icon name="down" size={14} />
        </button>
        <button
          className="icon-button"
          onClick={toggleDark}
          title={dark ? "Use light mode" : "Use dark mode"}
        >
          <Icon name={dark ? "sun" : "moon"} />
        </button>
        {/* No user accounts yet.
        <button className="profile-control" title="Account">
          <span>AC</span>
          <Icon name="down" size={13} />
        </button>
        */}
      </div>
    </header>
  );
}

const suggestions = [
  "How are enterprises reducing AI inference costs?",
  "What evidence exists for AI agents in business workflows?",
  "Which AI governance methods are gaining adoption?",
];

function Home({ research, openPaper }: { research: (q: string) => void; openPaper: () => void }) {
  const [query, setQuery] = useState("");
  const [selectedFocus, setSelectedFocus] = useState<string[]>(["Responsible AI"]);
  const [showMoreFocus, setShowMoreFocus] = useState(false);
  const [showSuggestedQuestions, setShowSuggestedQuestions] = useState(false);
  const [suggestedQuestionsHovered, setSuggestedQuestionsHovered] = useState(false);
  const primaryFocus = [
    "Responsible AI",
    "AI Agents",
    "AI Governance",
    "Cybersecurity",
    "Financial Services AI",
  ];
  const additionalFocus = [
    "Healthcare / Life Sciences AI",
    "Model Efficiency",
    "Workforce Automation",
  ];
  const toggleFocus = (focus: string) => {
    setSelectedFocus((current) =>
      current.includes(focus) ? current.filter((item) => item !== focus) : [...current, focus],
    );
  };
  const submit = (e: FormEvent) => {
    e.preventDefault();
    if (query.trim()) research(query);
  };
  return (
    <main className="page home-page">
      <section className="home-hero">
        <div className="eyebrow">
          <span />
          KPMG AI Research
        </div>
        <h1>What would you like to research?</h1>
        <p>Explore AI research and translate evidence into business insight.</p>
        <form className="composer" onSubmit={submit}>
          <div className="composer-input">
            <Icon name="search" size={21} />
            <textarea
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === "Enter" && !e.shiftKey) {
                  e.preventDefault();
                  if (query.trim()) research(query);
                }
              }}
              placeholder="Ask about AI research, trends, risks, governance, agents, infrastructure, or business implications…"
              rows={2}
            />
            <button className="send-button" disabled={!query.trim()} aria-label="Start research">
              <Icon name="arrow" size={19} />
            </button>
          </div>
          <div className="composer-focus">
            <span
              className="focus-label"
              style={{ fontSize: 10, fontWeight: 600, color: "var(--text)" }}
            >
              Research focus:
            </span>
            <div className="focus-chips">
              {primaryFocus.map((focus) => (
                <button
                  type="button"
                  className={selectedFocus.includes(focus) ? "selected" : ""}
                  aria-pressed={selectedFocus.includes(focus)}
                  onClick={() => toggleFocus(focus)}
                  style={{ fontSize: 9, padding: "5px 10px" }}
                  key={focus}
                >
                  {focus}
                </button>
              ))}
              <button
                type="button"
                className={`more-focus ${showMoreFocus ? "open" : ""}`}
                aria-expanded={showMoreFocus}
                onClick={() => setShowMoreFocus(!showMoreFocus)}
                style={{ fontSize: 9, padding: "5px 10px" }}
              >
                More <Icon name="down" size={12} />
              </button>
              {showMoreFocus &&
                additionalFocus.map((focus) => (
                  <button
                    type="button"
                    className={selectedFocus.includes(focus) ? "selected" : ""}
                    aria-pressed={selectedFocus.includes(focus)}
                    onClick={() => toggleFocus(focus)}
                    style={{ fontSize: 9, padding: "5px 10px" }}
                    key={focus}
                  >
                    {focus}
                  </button>
                ))}
            </div>
          </div>
          <div className="composer-meta">
            <button
              type="button"
              aria-expanded={showSuggestedQuestions}
              onClick={() => setShowSuggestedQuestions((current) => !current)}
              onMouseEnter={() => setSuggestedQuestionsHovered(true)}
              onMouseLeave={() => setSuggestedQuestionsHovered(false)}
              style={{
                display: "inline-flex",
                alignItems: "center",
                gap: 4,
                marginLeft: "auto",
                padding: 0,
                color: suggestedQuestionsHovered ? "var(--blue-dark)" : "var(--blue)",
                background: "none",
                border: 0,
                fontSize: 10,
                fontWeight: 600,
                transition: "color 150ms ease",
              }}
            >
              Suggested questions
              <span
                style={{
                  display: "inline-flex",
                  transform: showSuggestedQuestions ? "rotate(180deg)" : "none",
                  transition: "transform 160ms ease",
                }}
              >
                <Icon name="down" size={12} />
              </span>
            </button>
          </div>
          <div
            style={{
              maxHeight: showSuggestedQuestions ? 190 : 0,
              opacity: showSuggestedQuestions ? 1 : 0,
              overflow: "hidden",
              transition: "max-height 180ms ease, opacity 140ms ease",
            }}
          >
            <div className="suggestions">
              {suggestions.map((s, i) => (
                <button key={s} type="button" onClick={() => research(s)}>
                  <span className="suggest-icon">
                    <Icon name={["trend", "layers", "target"][i]} size={17} />
                  </span>
                  <span>{s}</span>
                  <Icon name="arrow" size={17} />
                </button>
              ))}
            </div>
          </div>
        </form>
      </section>
      <div className="home-content">
        <SectionTitle
          title="Trending research"
          subtitle="Research gaining attention across current AI themes."
        />
        <div className="recommended-papers">
          {papers.map((paper) => (
            <RecommendedPaper paper={paper} openPaper={openPaper} key={paper.title} />
          ))}
        </div>
      </div>
    </main>
  );
}

function SectionTitle({
  title,
  subtitle,
  action,
}: {
  title: string;
  subtitle?: string;
  action?: string;
}) {
  return (
    <div className="section-title">
      <div>
        <h2>{title}</h2>
        {subtitle && <p>{subtitle}</p>}
      </div>
      {action && (
        <button>
          {action}
          <Icon name="arrow" size={15} />
        </button>
      )}
    </div>
  );
}

function DemoNotice() {
  return (
    <div className="demo-notice-row">
      <p>Demo research data shown for prototype purposes.</p>
    </div>
  );
}

function PaperThumbnail({ index = 0 }: { index?: number }) {
  return (
    <div className={`paper-thumbnail paper-thumbnail-${index % 3}`} aria-hidden="true">
      <span className="thumbnail-masthead" />
      <strong />
      <strong />
      <i />
      <i />
      <i />
      <div>
        <span />
        <span />
        <span />
      </div>
      <small>{index + 1}</small>
    </div>
  );
}

function RecommendedPaper({
  paper,
  openPaper,
}: {
  paper: (typeof papers)[0];
  openPaper: () => void;
}) {
  const index = papers.indexOf(paper);
  return (
    <article className="recommended-paper">
      <PaperThumbnail index={index} />
      <div>
        <span className="paper-category">{paper.category}</span>
        <h3>{paper.title}</h3>
        <p className="recommended-meta">
          {paper.authors} · {paper.date}
        </p>
        <p>{paper.abstract}</p>
        <button onClick={openPaper}>
          View paper <Icon name="arrow" size={14} />
        </button>
      </div>
    </article>
  );
}

function Progress({ query }: { query: string }) {
  const [step, setStep] = useState(0);
  useEffect(() => {
    const id = setInterval(() => setStep((s) => Math.min(s + 1, 3)), 330);
    return () => clearInterval(id);
  }, []);
  const steps = [
    "Understanding question",
    "Searching AI research",
    "Evaluating evidence",
    "Building business insight",
  ];
  return (
    <main className="page progress-page">
      <div className="progress-card">
        <div className="progress-orbit">
          <Icon name="spark" size={24} />
        </div>
        <div className="eyebrow">Research in progress</div>
        <h1>Building an evidence-led answer</h1>
        <p className="progress-query">“{query}”</p>
        <div className="progress-steps">
          {steps.map((s, i) => (
            <div
              className={`progress-step ${i < step ? "done" : i === step ? "current" : ""}`}
              key={s}
            >
              <span>{i < step ? <Icon name="check" size={14} /> : i + 1}</span>
              <div>
                <strong>{s}</strong>
                {i === step && <small>Reviewing relevant research sources…</small>}
              </div>
            </div>
          ))}
        </div>
      </div>
    </main>
  );
}

function VoteButtons() {
  const [vote, setVote] = useState<Vote>(null);
  return (
    <div className="vote-buttons">
      <button
        className={vote === "up" ? "selected" : ""}
        onClick={() => setVote(vote === "up" ? null : "up")}
      >
        <Icon name="thumbsUp" size={15} /> Useful
      </button>
      <button
        className={vote === "down" ? "selected downvote" : ""}
        onClick={() => setVote(vote === "down" ? null : "down")}
      >
        <Icon name="thumbsDown" size={15} /> Not useful
      </button>
    </div>
  );
}

function PaperCard({
  paper,
  openPaper,
  compact = false,
}: {
  paper: Paper;
  openPaper: () => void;
  compact?: boolean;
}) {
  const [expanded, setExpanded] = useState(false);
  const [saved, setSaved] = useState(false);
  return (
    <article className={`paper-card ${compact ? "paper-compact" : ""}`}>
      <div className="paper-card-body">
        <PaperThumbnail index={papers.indexOf(paper) < 0 ? 1 : papers.indexOf(paper)} />
        <div className="paper-main">
          <div className="paper-top">
            <span className="category-badge">{paper.category}</span>
            <div className="relevance">
              <span>Relevance</span>
              <strong>{paper.relevance}%</strong>
            </div>
          </div>
          <h3 onClick={openPaper}>{paper.title}</h3>
          <p className="paper-meta">
            {paper.authors} <span /> {paper.date} <span /> {paper.source}
          </p>
          <p className="abstract">{paper.abstract}</p>
          {!compact && (
            <blockquote>
              <span>Supporting evidence</span>“{paper.quote}”
            </blockquote>
          )}
          {expanded && (
            <div className="expanded-details">
              <div>
                <strong>Evidence quality</strong>
                <span className="confidence-badge">
                  <i /> High confidence
                </span>
              </div>
              <p>
                Evidence quality reflects methodological fit, source recency, and direct relevance
                to the research question.
              </p>
            </div>
          )}
        </div>
      </div>
      <footer className="paper-footer">
        <div>
          {!compact && (
            <button className="text-button" onClick={() => setExpanded(!expanded)}>
              Advanced details <Icon name={expanded ? "down" : "chevron"} size={14} />
            </button>
          )}
          <VoteButtons />
        </div>
        <div>
          <button
            className={`icon-text-button ${saved ? "saved" : ""}`}
            onClick={() => setSaved(!saved)}
          >
            <Icon name="bookmark" size={15} />
            {saved ? "Saved" : "Save"}
          </button>
          <button className="secondary-button" onClick={openPaper}>
            {compact ? "Preview" : "View paper"}
            <Icon name="external" size={14} />
          </button>
        </div>
      </footer>
    </article>
  );
}

function Results({
  query,
  data,
  back,
  openPaper,
}: {
  query: string;
  data: ResearchResponse | null;
  back: () => void;
  openPaper: () => void;
}) {
  const resultPapers = papersFromResearch(data);
  const selectedSourceCount = data?.retrieved_chunks?.length ?? papers.length;
  const topRelevance = data?.top_score != null ? Math.round(data.top_score * 100) : 92;

  return (
    <main className="page results-page">
      <div className="page-header">
        <button className="back-link" onClick={back}>
          <Icon name="back" size={16} />
          Back to Research
        </button>
        <div className="question-label">Research question</div>
        <h1>{query}</h1>
        <div className="header-actions">
          <button>
            <Icon name="copy" size={15} />
            Copy
          </button>
          {/* Not connected to anything yet: no email or export.
          <button>
            <Icon name="mail" size={15} />
            Email finding
          </button>
          <button>
            <Icon name="share" size={15} />
            Export
          </button>
          */}
        </div>
      </div>
      <DemoNotice />
      <div className="results-layout">
        <div className="results-main">
          <section className="insight-card">
            <div className="insight-heading">
              <span>
                <Icon name="spark" size={18} />
              </span>
              <div>
                <small>Executive insight</small>
                <h2>{data?.answer ? "Executive insight" : "Research insight"}</h2>
              </div>
            </div>
            {data?.answer ? (
              <div className="insight-answer">
                <Markdown
                  components={{
                    a: ({ href, children }) => (
                      <a href={href} target="_blank" rel="noreferrer">
                        {children}
                      </a>
                    ),
                  }}
                >
                  {data.answer}
                </Markdown>
              </div>
            ) : (
              <p>No research answer was returned.</p>
            )}
            <div className="takeaways">
              <h3>Key takeaways</h3>
              <ul>
                <li>
                  Prioritize workflows where quality, cost, and intervention rates can be measured
                  together.
                </li>
                <li>
                  Ground AI outputs in governed enterprise knowledge and evaluate the evidence
                  chain.
                </li>
                <li>
                  Use bounded autonomy and escalation paths before expanding agent responsibility.
                </li>
              </ul>
            </div>
            <div className="confidence-row">
              <div>
                <span className="confidence-badge">
                  <i /> {data?.confidence}
                </span>
                <span>Based on {selectedSourceCount} selected sources</span>
              </div>
              <div>
                <span>Top relevance</span>
                <strong>{topRelevance}%</strong>
              </div>
            </div>
          </section>
          <div className="evidence-heading">
            <div>
              <h2>Evidence & sources</h2>
              <p>Research publications ranked by relevance to your question</p>
            </div>
            <button>
              Most relevant <Icon name="down" size={14} />
            </button>
          </div>
          <div className="papers">
            {resultPapers.map((p) => (
              <PaperCard key={p.title} paper={p} openPaper={openPaper} />
            ))}
          </div>
          <section className="more-explore">
            <SectionTitle title="More to explore" subtitle="Continue building your understanding" />
            <div>
              {[
                "Explore the business implications",
                "Compare evidence across industries",
                "Ask a follow-up question",
              ].map((x, i) => (
                <button key={x}>
                  <span>
                    <Icon name={["target", "layers", "spark"][i]} size={18} />
                  </span>
                  {x}
                  <Icon name="arrow" size={16} />
                </button>
              ))}
            </div>
          </section>
        </div>
        <aside className="results-aside">
          <div className="aside-card">
            <h3>Research summary</h3>
            <dl>
              <div>
                <dt>Sources reviewed</dt>
                <dd>18</dd>
              </div>
              <div>
                <dt>Evidence selected</dt>
                <dd>{selectedSourceCount}</dd>
              </div>
              <div>
                <dt>Year range</dt>
                <dd>2024–25</dd>
              </div>
            </dl>
          </div>
          <div className="aside-card">
            <h3>Research coverage</h3>
            {["Enterprise relevance", "Evidence quality", "Recency"].map((x, i) => (
              <div className="coverage" key={x}>
                <span>{x}</span>
                <div>
                  <i style={{ width: `${[88, 82, 94][i]}%` }} />
                </div>
              </div>
            ))}
          </div>
        </aside>
      </div>
    </main>
  );
}

function Latest({ openPaper }: { openPaper: () => void }) {
  const [filter, setFilter] = useState("All research");
  const filters = [
    "All research",
    "Agents",
    "Responsible AI",
    "LLMs",
    "Model Efficiency",
    "Cybersecurity",
    "Enterprise AI",
  ];
  return (
    <main className="page content-page">
      <PageIntro
        eyebrow="Research feed"
        title="Latest AI"
        text="Recent research selected for enterprise relevance."
      />
      <DemoNotice />
      <div className="filter-row">
        {filters.map((f) => (
          <button className={filter === f ? "active" : ""} onClick={() => setFilter(f)} key={f}>
            {f}
          </button>
        ))}
      </div>
      <div className="feed-layout">
        <div>
          <SectionTitle title="Recent research" subtitle="Curated papers and applied research" />
          <div className="papers">
            {papers
              .concat([
                {
                  ...papers[0],
                  title: "Efficient Language Model Serving Under Variable Enterprise Demand",
                  category: "Model Efficiency",
                  relevance: 83,
                },
              ])
              .map((p, i) => (
                <PaperCard key={i} paper={p} openPaper={openPaper} compact />
              ))}
          </div>
        </div>
        <aside className="feed-aside">
          <div className="aside-card">
            <h3>Research brief</h3>
            <strong>Enterprise agents move toward measurable outcomes</strong>
            <p>
              New evaluation approaches are connecting agent reliability with process-level business
              measures.
            </p>
            <button className="text-link">
              View topic brief <Icon name="arrow" size={14} />
            </button>
          </div>
          <div className="aside-card">
            <h3>Watching this week</h3>
            {["Agent evaluation", "Inference economics", "Model controls"].map((x, i) => (
              <div className="watch-item" key={x}>
                <span>0{i + 1}</span>
                <strong>{x}</strong>
                <Icon name="trend" size={15} />
              </div>
            ))}
          </div>
        </aside>
      </div>
    </main>
  );
}

function PageIntro({ eyebrow, title, text }: { eyebrow: string; title: string; text: string }) {
  return (
    <div className="content-header">
      <div className="eyebrow">
        <span />
        {eyebrow}
      </div>
      <h1>{title}</h1>
      <p>{text}</p>
    </div>
  );
}

const trends = [
  {
    title: "Agentic AI",
    volume: "1,284",
    change: "+38%",
    value: 94,
    text: "Research is shifting from capability demos to reliability, orchestration, and measurable workflow outcomes.",
  },
  {
    title: "AI Governance",
    volume: "968",
    change: "+24%",
    value: 78,
    text: "Operational controls and continuous model evaluation are becoming central to scaled adoption.",
  },
  {
    title: "Model Efficiency",
    volume: "746",
    change: "+19%",
    value: 69,
    text: "Inference optimization and smaller specialist models are gaining enterprise attention.",
  },
  {
    title: "Multimodal AI",
    volume: "691",
    change: "+17%",
    value: 62,
    text: "Document, audio, and visual understanding are converging in applied business workflows.",
  },
  {
    title: "Enterprise RAG",
    volume: "633",
    change: "+12%",
    value: 56,
    text: "Evidence quality and retrieval evaluation now matter more than architecture choice alone.",
  },
];

function Trends() {
  return (
    <main className="page content-page">
      <PageIntro
        eyebrow="AI intelligence"
        title="Research trends"
        text="Track emerging themes, evidence momentum, and potential business significance."
      />
      <DemoNotice />
      <div className="trend-summary">
        <div>
          <small>Research signals monitored</small>
          <strong>4,322</strong>
          <span>Research publications</span>
        </div>
        <div>
          <small>Fastest rising topic</small>
          <strong>Agentic AI</strong>
          <span className="positive">+38% momentum</span>
        </div>
        <div>
          <small>Business relevance</small>
          <strong>High</strong>
          <span>Across 4 of 5 themes</span>
        </div>
      </div>
      <SectionTitle
        title="Trending AI topics"
        subtitle="Momentum across selected research signals"
        action="Methodology"
      />
      <div className="trends-list">
        {trends.map((t, i) => (
          <article className="trend-card" key={t.title}>
            <span className="trend-rank">0{i + 1}</span>
            <div className="trend-info">
              <div>
                <h3>{t.title}</h3>
                <span className="momentum">
                  <Icon name="trend" size={14} />
                  {t.change}
                </span>
              </div>
              <p>{t.text}</p>
              <div className="momentum-bar">
                <i style={{ width: `${t.value}%` }} />
              </div>
            </div>
            <div className="trend-volume">
              <small>Research volume</small>
              <strong>{t.volume}</strong>
              <span>papers / signals</span>
            </div>
            <button className="icon-button">
              <Icon name="chevron" />
            </button>
          </article>
        ))}
      </div>
      <div className="interpretation">
        <span>
          <Icon name="spark" />
        </span>
        <div>
          <small>Business interpretation</small>
          <h3>Reliability is becoming the defining enterprise AI research theme</h3>
          <p>
            Across agents, governance, and RAG, research attention is moving from general capability
            toward measurable reliability, controls, and sustainable operating economics.
          </p>
        </div>
        <button>
          Explore evidence <Icon name="arrow" size={15} />
        </button>
      </div>
    </main>
  );
}

const focusAreas = [
  [
    "Responsible AI",
    "Governance approaches are moving toward continuous, context-specific evaluation.",
    "24 sources",
  ],
  [
    "Financial Services AI",
    "Explainability, model risk, and bounded automation remain high-priority themes.",
    "18 sources",
  ],
  [
    "Cybersecurity",
    "AI-enabled defense is accelerating alongside research into novel model attack surfaces.",
    "21 sources",
  ],
  [
    "Healthcare / Life Sciences AI",
    "Multimodal evidence and clinical validation are shaping responsible deployment.",
    "16 sources",
  ],
  [
    "Workforce Automation",
    "Research increasingly emphasizes task redesign over simple role replacement.",
    "14 sources",
  ],
  [
    "Enterprise AI",
    "Knowledge grounding and workflow integration lead current applied research.",
    "27 sources",
  ],
];

function Focus({ research }: { research: (q: string) => void }) {
  return (
    <main className="page content-page">
      <PageIntro
        eyebrow="KPMG perspective"
        title="KPMG focus"
        text="Research intelligence aligned with priority industries, services, and emerging client questions."
      />
      <DemoNotice />
      <div className="focus-feature">
        <div>
          <span className="category-badge">THIS MONTH</span>
          <h2>From AI experimentation to controlled enterprise value</h2>
          <p>
            Research signals suggest that leading organizations are concentrating investment on
            measurable workflows, governance by design, and scalable model economics.
          </p>
          <button
            onClick={() =>
              research("How are enterprises moving from AI experimentation to controlled value?")
            }
          >
            Research this theme <Icon name="arrow" size={15} />
          </button>
        </div>
        <div className="signal-visual">
          <span className="signal-label">Research signal</span>
          <strong>82</strong>
          <small>High relevance</small>
          <div>
            <i />
            <i />
            <i />
            <i />
            <i />
          </div>
        </div>
      </div>
      <SectionTitle
        title="Priority focus areas"
        subtitle="Research themes aligned to KPMG business priorities"
      />
      <div className="focus-grid">
        {focusAreas.map((a, i) => (
          <article key={a[0]}>
            <div className={`focus-icon focus-${i}`}>
              <Icon name={["target", "trend", "layers", "spark", "grid", "file"][i]} />
            </div>
            <div>
              <h3>{a[0]}</h3>
              <p>{a[1]}</p>
              <span>{a[2]}</span>
            </div>
            <button onClick={() => research(`What does current research say about ${a[0]}?`)}>
              <Icon name="chevron" size={18} />
            </button>
          </article>
        ))}
      </div>
    </main>
  );
}

function PaperWorkspace({ back }: { back: () => void }) {
  const [selected, setSelected] = useState(true);
  const [selectedPage, setSelectedPage] = useState(1);
  const [assistantText, setAssistantText] = useState("");
  const [messages, setMessages] = useState<string[]>([]);
  const [notice, setNotice] = useState("");
  const selectedQuote =
    "Our evaluation indicates that task completion alone can overstate the operational value of AI agents. The strongest deployments combined bounded autonomy with explicit escalation paths, allowing human experts to intervene when uncertainty or business risk exceeded predefined thresholds.";
  const action = (text: string) => {
    setNotice(text);
    setTimeout(() => setNotice(""), 1800);
  };
  const ask = (q: string) => {
    setMessages((m) => [...m, q]);
    setAssistantText("");
  };
  return (
    <main className="paper-workspace">
      {notice && (
        <div className="toast">
          <Icon name="check" size={16} />
          {notice}
        </div>
      )}
      <div className="workspace-toolbar">
        <button className="back-link" onClick={back}>
          <Icon name="back" size={16} />
          Back to evidence
        </button>
        <div className="workspace-actions">
          <button onClick={() => action("Citation copied")}>
            <Icon name="copy" size={15} />
            Copy citation
          </button>
          <button onClick={() => action("Finding ready to share")}>
            <Icon name="mail" size={15} />
            Email finding
          </button>
          <button onClick={() => action("Paper attached to draft")}>
            <Icon name="file" size={15} />
            Attach paper
          </button>
        </div>
      </div>
      <div className="workspace-grid">
        <section className="document-area">
          <aside className="thumbnails">
            <small>PAGES</small>
            {[1, 2, 3, 4].map((n) => (
              <button
                className={n === selectedPage ? "active" : ""}
                onClick={() => setSelectedPage(n)}
                aria-label={`View page ${n}`}
                key={n}
              >
                <span>
                  <i />
                  <i />
                  <i />
                  <i />
                </span>
                <small>{n}</small>
              </button>
            ))}
          </aside>
          <div className="document-scroll">
            <article className="paper-document">
              <div className="doc-journal">JOURNAL OF APPLIED AI SYSTEMS · VOLUME 12, ISSUE 4</div>
              <h1>{papers[1].title}</h1>
              <div className="doc-authors">{papers[1].authors}</div>
              <div className="doc-meta">
                <span>Research article · 2025 · Volume 12, Issue 4</span>
                <span className="demo-paper">Demo paper</span>
              </div>
              <h2>Abstract</h2>
              <p>
                {papers[1].abstract} This study examines operational conditions associated with
                reliable deployment and offers a framework for evidence-led evaluation.
              </p>
              <div className="doc-keywords">
                <strong>Keywords:</strong> AI agents, enterprise workflows, evaluation, human
                oversight
              </div>
              <h2>1. Introduction</h2>
              <p>
                Autonomous language-model agents have demonstrated the ability to plan and execute
                multi-step tasks. Yet enterprise value depends on more than isolated task
                completion. It requires consistent performance within operating constraints,
                transparent escalation, and meaningful measures of business impact.
              </p>
              <div
                className={`selected-passage ${selected ? "selected" : ""}`}
                onClick={() => setSelected(!selected)}
              >
                <p>{selectedQuote}</p>
                {selected && (
                  <div className="selection-tools">
                    <button
                      onClick={(e) => {
                        e.stopPropagation();
                        action("Passage highlighted");
                      }}
                    >
                      Highlight
                    </button>
                    <button
                      onClick={(e) => {
                        e.stopPropagation();
                        action("Note added");
                      }}
                    >
                      Note
                    </button>
                    <button
                      onClick={(e) => {
                        e.stopPropagation();
                        ask("Explain this selected passage in the context of enterprise AI.");
                      }}
                    >
                      Ask AI
                    </button>
                  </div>
                )}
              </div>
              <h2>2. Evaluation framework</h2>
              <p>
                We evaluate outcomes across four dimensions: completion quality, intervention
                frequency, recovery from error, and realized process value. These measures are
                intended to be considered together rather than as independent performance
                indicators.
              </p>
              <div className="doc-table">
                <div>
                  <strong>Evaluation dimension</strong>
                  <strong>Enterprise measure</strong>
                </div>
                <div>
                  <span>Completion quality</span>
                  <span>Outcome accepted without correction</span>
                </div>
                <div>
                  <span>Human intervention</span>
                  <span>Escalations per workflow</span>
                </div>
                <div>
                  <span>Process value</span>
                  <span>Time, quality, and risk impact</span>
                </div>
              </div>
            </article>
          </div>
        </section>
        <aside className="assistant-panel">
          <div className="assistant-head">
            <div>
              <div>
                <strong>Research Assistant</strong>
                <small>Answers grounded in this paper</small>
              </div>
            </div>
          </div>
          <div className="assistant-body">
            {selected && (
              <div className="selection-context">
                <div>Selected passage</div>
                <blockquote>“{selectedQuote}”</blockquote>
              </div>
            )}
            <div className="assistant-welcome">
              <h3>{selected ? "Ask about this passage" : "Explore this paper"}</h3>
              <p>
                {selected
                  ? "Explore the meaning, evidence, and business implications of the selected text."
                  : "Ask for an explanation, evidence summary, or enterprise interpretation."}
              </p>
            </div>
            {messages.map((m, i) => (
              <div className="research-exchange" key={i}>
                <div className="research-question">
                  <span>Question</span>
                  <p>{m}</p>
                </div>
                <div className="research-answer">
                  <span>Research Assistant</span>
                  <p>
                    This matters because bounded autonomy connects technical capability with
                    operational accountability. It lets organizations gain efficiency while
                    retaining human judgment for uncertain or high-risk decisions.
                  </p>
                  <small>Source: Evaluation framework, page 1</small>
                </div>
              </div>
            ))}
            <div className="assistant-section-label">Suggested questions</div>
            <div className="prompt-list">
              {[
                "Explain this in plain language",
                "Why does this matter for enterprise AI?",
                "Summarize the evidence",
              ].map((q) => (
                <button onClick={() => ask(q)} key={q}>
                  {q}
                  <Icon name="arrow" size={14} />
                </button>
              ))}
            </div>
          </div>
          <form
            className="assistant-composer"
            onSubmit={(e) => {
              e.preventDefault();
              if (assistantText.trim()) ask(assistantText);
            }}
          >
            <textarea
              rows={2}
              value={assistantText}
              onChange={(e) => setAssistantText(e.target.value)}
              placeholder="Ask about this paper…"
            />
            <button disabled={!assistantText.trim()}>
              <Icon name="arrow" size={17} />
            </button>
          </form>
        </aside>
      </div>
    </main>
  );
}

export default function App() {
  const [view, setView] = useState<View>("home");
  const [dark, setDark] = useState(false);
  const [menu, setMenu] = useState(false);
  const [sidebarCollapsed, setSidebarCollapsed] = useState(
    () => window.localStorage.getItem("research-sidebar-collapsed") === "true",
  );
  const [query, setQuery] = useState(suggestions[1]);
  const [progress, setProgress] = useState(false);
  const [researchData, setResearchData] = useState<ResearchResponse | null>(null);
  const [threadId] = useState(() => crypto.randomUUID());
  const [previousView, setPreviousView] = useState<View>("results");
  useEffect(() => {
    document.documentElement.classList.toggle("dark", dark);
  }, [dark]);
  useEffect(() => {
    window.localStorage.setItem("research-sidebar-collapsed", String(sidebarCollapsed));
  }, [sidebarCollapsed]);
const research = async (q: string) => {
  setQuery(q);
  setProgress(true);
  setResearchData(null);
  setView("results");

  try {
    const data = await runResearch(q, threadId);
    setResearchData(data);
  } catch (error) {
    console.error("Research request failed:", error);
  } finally {
    setProgress(false);
  }
};
  const openPaper = () => {
    setPreviousView(view);
    setView("paper");
  };
  return (
    <div
      className={`app-shell ${sidebarCollapsed ? "sidebar-collapsed" : ""} ${menu ? "nav-open" : ""}`}
    >
      <Sidebar
        view={view}
        setView={(v) => {
          setProgress(false);
          setView(v);
        }}
        open={menu}
        close={() => setMenu(false)}
        collapsed={sidebarCollapsed}
        toggleCollapsed={() => setSidebarCollapsed((current) => !current)}
      />
      {menu && <div className="sidebar-overlay" onClick={() => setMenu(false)} />}
      <div className="app-main">
        <Topbar dark={dark} toggleDark={() => setDark(!dark)} openMenu={() => setMenu(true)} />
        {view === "home" && <Home research={research} openPaper={openPaper} />}
        {view === "results" &&
          (progress ? (
            <Progress query={query} />
          ) : (
            <Results 
            query={query}
            data={researchData} 
            back={() => setView("home")} 
            openPaper={openPaper} />
          ))}
        {view === "latest" && <Latest openPaper={openPaper} />}
        {view === "trends" && <Trends />}
        {view === "focus" && <Focus research={research} />}
        {view === "paper" && <PaperWorkspace back={() => setView(previousView)} />}
      </div>
    </div>
  );
}
