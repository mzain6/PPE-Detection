"use client";
import { useState, useEffect, useCallback } from "react";
import { useSession } from "next-auth/react";
import { Download, Eye, CheckSquare, Film, AlertTriangle, Calendar, Video, ShieldAlert } from "lucide-react";
import DataTable from "@/components/ui/DataTable";
import Modal from "@/components/ui/Modal";
import { ViolationBadge } from "@/components/ui/Badge";
import { apiFetch, API_BASE } from "@/lib/api";
import { useAlertStore } from "@/store/alertStore";

const VIOLATION_TYPES = ["all", "NO_HELMET", "NO_VEST", "NO_BOTH", "FALL_DETECTED"];

function ConfidenceCell({ value }) {
  const pct = (value * 100).toFixed(1);
  const color = value > 0.8 ? "var(--color-success)" : value > 0.6 ? "var(--color-warning)" : "var(--color-danger)";
  return <span className="text-mono" style={{ color, fontWeight: 600 }}>{pct}%</span>;
}

const getMediaUrl = (path) => {
  if (!path) return null;
  if (path.startsWith("http")) return path;
  const clean = path.startsWith("/") ? path.slice(1) : path;
  return `${API_BASE}/${clean}`;
};

export default function ViolationsPage() {
  const { data: session } = useSession();
  const { addToast } = useAlertStore();
  const [data, setData] = useState({ items: [], total: 0, total_pages: 1 });
  const [loading, setLoading] = useState(true);
  const [cameras, setCameras] = useState([]);
  const [summary, setSummary] = useState(null);
  const [trendData, setTrendData] = useState([]);
  const [filters, setFilters] = useState({
    date_from: "",
    date_to: "",
    violation_type: "all",
    camera_id: "",
  });
  const [appliedFilters, setAppliedFilters] = useState(filters);
  const [page, setPage] = useState(1);
  const [pageSize, setPageSize] = useState(25);
  const [selectedViolation, setSelectedViolation] = useState(null);

  // Load initial options & daily stats
  useEffect(() => {
    if (!session) return;
    apiFetch("/api/cameras", {}, session).then(setCameras).catch(() => {});
    apiFetch("/api/analytics/summary", {}, session).then(setSummary).catch(() => {});
    apiFetch("/api/analytics/trend?days=7", {}, session).then(setTrendData).catch(() => {});
  }, [session]);

  const loadViolations = useCallback(async () => {
    if (!session) return;
    setLoading(true);
    try {
      const params = new URLSearchParams({ page, page_size: pageSize });
      if (appliedFilters.date_from) params.append("date_from", appliedFilters.date_from);
      if (appliedFilters.date_to)   params.append("date_to",   appliedFilters.date_to);
      if (appliedFilters.violation_type !== "all") params.append("violation_type", appliedFilters.violation_type);
      if (appliedFilters.camera_id) params.append("camera_id", appliedFilters.camera_id);

      const result = await apiFetch(`/api/violations?${params}`, {}, session);
      setData(result);
    } catch { addToast({ type: "error", message: "Failed to load violations" }); }
    finally { setLoading(false); }
  }, [session, page, pageSize, appliedFilters]);

  useEffect(() => { loadViolations(); }, [loadViolations]);

  async function markReviewed(id) {
    try {
      await apiFetch(`/api/violations/${id}/review`, { method: "PATCH" }, session);
      addToast({ type: "success", message: "Marked as reviewed" });
      loadViolations();
      // refresh summary
      apiFetch("/api/analytics/summary", {}, session).then(setSummary).catch(() => {});
      setSelectedViolation(null);
    } catch { addToast({ type: "error", message: "Failed to update" }); }
  }

  function exportCSV() {
    const params = new URLSearchParams();
    if (appliedFilters.date_from) params.append("date_from", appliedFilters.date_from);
    if (appliedFilters.date_to)   params.append("date_to",   appliedFilters.date_to);
    if (appliedFilters.violation_type !== "all") params.append("violation_type", appliedFilters.violation_type);
    if (appliedFilters.camera_id) params.append("camera_id", appliedFilters.camera_id);
    window.open(`${API_BASE}/api/violations/export/csv?${params}`, "_blank");
  }

  // Calculate 7-day total count
  const sevenDayTotal = trendData.reduce((acc, curr) => acc + (curr.count || 0), 0);

  const columns = [
    {
      key: "timestamp",
      label: "Time",
      render: (v) => (
        <span className="text-mono" style={{ fontSize: "0.8rem" }}>
          {new Date(v).toLocaleString("en-GB", { day: "2-digit", month: "short", hour: "2-digit", minute: "2-digit" })}
        </span>
      ),
    },
    { key: "camera_name", label: "Camera", render: (v) => v || "—" },
    {
      key: "violation_type",
      label: "Type",
      render: (v) => <ViolationBadge type={v} />,
    },
    {
      key: "person_name",
      label: "Person",
      render: (v, row) => (
        <span style={{ fontWeight: 600 }}>
          {v || `Track #${row.track_id || 1}`}
        </span>
      ),
    },
    {
      key: "confidence",
      label: "Confidence",
      render: (v) => <ConfidenceCell value={v} />,
    },
    {
      key: "evidence_video_path",
      label: "Evidence Clip",
      render: (v, row) => (
        v || row.video_link ? (
          <button
            className="btn btn-secondary btn-sm"
            style={{ color: "#38bdf8", borderColor: "#0284c733", display: "inline-flex", alignItems: "center", gap: 4 }}
            onClick={(e) => { e.stopPropagation(); setSelectedViolation(row); }}
          >
            <Film size={13} /> Play Clip
          </button>
        ) : (
          <span style={{ color: "var(--color-text-muted)", fontSize: "0.75rem" }}>Snapshot Only</span>
        )
      ),
    },
    {
      key: "is_reviewed",
      label: "Status",
      render: (v) => (
        <span className={`badge badge-${v ? "success" : "warning"}`}>{v ? "Reviewed" : "Pending"}</span>
      ),
    },
    {
      key: "id",
      label: "Actions",
      render: (_, row) => (
        <button
          className="btn btn-secondary btn-sm"
          onClick={() => setSelectedViolation(row)}
        >
          <Eye size={13} /> View
        </button>
      ),
    },
  ];

  return (
    <>
      <div className="page-header">
        <div>
          <h2 className="page-title">Violations & Evidence Clips</h2>
          <p className="page-subtitle">Daily incident logs with 4–5 second recorded evidence video clips</p>
        </div>
        <div style={{ display: "flex", gap: "0.5rem" }}>
          <button className="btn btn-secondary btn-sm" onClick={exportCSV}>
            <Download size={14} /> Export CSV
          </button>
        </div>
      </div>

      {/* Daily Analytics Summary Cards */}
      <div style={{
        display: "grid",
        gridTemplateColumns: "repeat(auto-fit, minmax(220px, 1fr))",
        gap: "1rem",
        marginBottom: "1.5rem"
      }}>
        <div className="card" style={{ padding: "1.25rem" }}>
          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "0.5rem" }}>
            <span style={{ fontSize: "0.8rem", color: "var(--color-text-muted)", fontWeight: 600 }}>TODAY'S VIOLATIONS</span>
            <ShieldAlert size={18} style={{ color: "#ef4444" }} />
          </div>
          <div style={{ fontSize: "1.75rem", fontWeight: 700, color: "var(--color-text)" }}>
            {summary ? summary.total_violations_today : "—"}
          </div>
          <div style={{ fontSize: "0.75rem", color: "var(--color-text-muted)", marginTop: "0.25rem" }}>
            Recorded on active cameras today
          </div>
        </div>

        <div className="card" style={{ padding: "1.25rem" }}>
          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "0.5rem" }}>
            <span style={{ fontSize: "0.8rem", color: "var(--color-text-muted)", fontWeight: 600 }}>7-DAY TOTAL</span>
            <Calendar size={18} style={{ color: "#38bdf8" }} />
          </div>
          <div style={{ fontSize: "1.75rem", fontWeight: 700, color: "var(--color-text)" }}>
            {sevenDayTotal || (summary ? summary.total_violations_today : 0)}
          </div>
          <div style={{ fontSize: "0.75rem", color: "var(--color-text-muted)", marginTop: "0.25rem" }}>
            Aggregated past 7 days incidents
          </div>
        </div>

        <div className="card" style={{ padding: "1.25rem" }}>
          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "0.5rem" }}>
            <span style={{ fontSize: "0.8rem", color: "var(--color-text-muted)", fontWeight: 600 }}>PENDING REVIEW</span>
            <AlertTriangle size={18} style={{ color: "#f59e0b" }} />
          </div>
          <div style={{ fontSize: "1.75rem", fontWeight: 700, color: "#f59e0b" }}>
            {summary ? summary.unreviewed_incidents : "—"}
          </div>
          <div style={{ fontSize: "0.75rem", color: "var(--color-text-muted)", marginTop: "0.25rem" }}>
            Awaiting supervisor verification
          </div>
        </div>

        <div className="card" style={{ padding: "1.25rem" }}>
          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "0.5rem" }}>
            <span style={{ fontSize: "0.8rem", color: "var(--color-text-muted)", fontWeight: 600 }}>ACTIVE CAMERAS</span>
            <Video size={18} style={{ color: "#10b981" }} />
          </div>
          <div style={{ fontSize: "1.75rem", fontWeight: 700, color: "var(--color-text)" }}>
            {summary ? `${summary.active_cameras} / ${summary.total_cameras}` : "—"}
          </div>
          <div style={{ fontSize: "0.75rem", color: "#10b981", marginTop: "0.25rem" }}>
            AI Streams Live
          </div>
        </div>
      </div>

      {/* Filter Bar */}
      <div className="filter-bar">
        <div className="form-group" style={{ flex: "1 1 140px" }}>
          <label className="input-label">From</label>
          <input type="date" className="input" value={filters.date_from}
            onChange={(e) => setFilters(f => ({ ...f, date_from: e.target.value }))} />
        </div>
        <div className="form-group" style={{ flex: "1 1 140px" }}>
          <label className="input-label">To</label>
          <input type="date" className="input" value={filters.date_to}
            onChange={(e) => setFilters(f => ({ ...f, date_to: e.target.value }))} />
        </div>
        <div className="form-group" style={{ flex: "1 1 160px" }}>
          <label className="input-label">Violation Type</label>
          <select className="input" value={filters.violation_type}
            onChange={(e) => setFilters(f => ({ ...f, violation_type: e.target.value }))}>
            {VIOLATION_TYPES.map(t => (
              <option key={t} value={t}>{t === "all" ? "All Types" : t.replace(/_/g, " ")}</option>
            ))}
          </select>
        </div>
        <div className="form-group" style={{ flex: "1 1 160px" }}>
          <label className="input-label">Camera</label>
          <select className="input" value={filters.camera_id}
            onChange={(e) => setFilters(f => ({ ...f, camera_id: e.target.value }))}>
            <option value="">All Cameras</option>
            {cameras.map(c => <option key={c.id} value={c.id}>{c.name}</option>)}
          </select>
        </div>
        <div style={{ display: "flex", gap: "0.5rem", alignItems: "flex-end" }}>
          <button className="btn btn-primary btn-sm" onClick={() => { setAppliedFilters(filters); setPage(1); }}>
            Apply
          </button>
          <button className="btn btn-ghost btn-sm" onClick={() => {
            const reset = { date_from: "", date_to: "", violation_type: "all", camera_id: "" };
            setFilters(reset);
            setAppliedFilters(reset);
            setPage(1);
          }}>
            Reset
          </button>
        </div>
      </div>

      <DataTable
        columns={columns}
        data={data.items}
        loading={loading}
        emptyMessage="No violations found for the selected filters"
        currentPage={page}
        totalPages={data.total_pages}
        total={data.total}
        pageSize={pageSize}
        onPageChange={setPage}
      />

      {/* Violation Details & Video Clip Player Modal */}
      <Modal
        isOpen={!!selectedViolation}
        onClose={() => setSelectedViolation(null)}
        title="Violation Evidence & Clip Player"
        size="lg"
      >
        {selectedViolation && (
          <>
            {/* 4-5s Evidence Video Clip Player or Snapshot Fallback */}
            {selectedViolation.evidence_video_path || selectedViolation.video_link ? (
              <div style={{ position: "relative", backgroundColor: "#000", borderRadius: "8px 8px 0 0", overflow: "hidden" }}>
                <video
                  src={getMediaUrl(selectedViolation.evidence_video_path || selectedViolation.video_link)}
                  controls
                  autoPlay
                  loop
                  playsInline
                  style={{ width: "100%", maxHeight: 350, display: "block" }}
                />
                <div style={{
                  position: "absolute",
                  top: 12,
                  left: 12,
                  background: "rgba(0,0,0,0.8)",
                  color: "#38bdf8",
                  padding: "4px 10px",
                  borderRadius: 6,
                  fontSize: "0.75rem",
                  fontWeight: 600,
                  display: "flex",
                  alignItems: "center",
                  gap: 6,
                  border: "1px solid rgba(56,189,248,0.3)"
                }}>
                  <Film size={14} /> 4-5s Video Recording Evidence
                </div>
              </div>
            ) : selectedViolation.screenshot_path ? (
              <img
                src={getMediaUrl(selectedViolation.screenshot_path)}
                alt="Violation screenshot"
                style={{ width: "100%", maxHeight: 300, objectFit: "cover", borderRadius: "8px 8px 0 0" }}
              />
            ) : (
              <div style={{ height: 160, display: "flex", alignItems: "center", justifyContent: "center", background: "#1e293b", color: "#94a3b8" }}>
                No video or screenshot recorded for this incident
              </div>
            )}

            <div className="modal-body" style={{ paddingTop: "1.25rem" }}>
              <div style={{
                display: "grid",
                gridTemplateColumns: "1fr 1fr",
                gap: "1rem",
                marginBottom: "1.25rem",
              }}>
                {[
                  ["Camera",     selectedViolation.camera_name || "—"],
                  ["Type",       <ViolationBadge type={selectedViolation.violation_type} />],
                  ["Time",       new Date(selectedViolation.timestamp).toLocaleString()],
                  ["Confidence", <ConfidenceCell value={selectedViolation.confidence} />],
                  ["Person",     selectedViolation.person_name || `Track #${selectedViolation.track_id || 1}`],
                  ["Status",     <span className={`badge badge-${selectedViolation.is_reviewed ? "success" : "warning"}`}>
                    {selectedViolation.is_reviewed ? "Reviewed" : "Pending"}
                  </span>],
                ].map(([label, value]) => (
                  <div key={label}>
                    <div style={{ fontSize: "0.7rem", color: "var(--color-text-muted)", marginBottom: 4 }}>
                      {label.toUpperCase()}
                    </div>
                    <div style={{ fontSize: "0.875rem", fontWeight: 500 }}>{value}</div>
                  </div>
                ))}
              </div>
            </div>
            {!selectedViolation.is_reviewed && (
              <div className="modal-footer">
                <button
                  className="btn btn-primary btn-sm"
                  onClick={() => markReviewed(selectedViolation.id)}
                >
                  <CheckSquare size={14} /> Mark as Reviewed
                </button>
              </div>
            )}
          </>
        )}
      </Modal>
    </>
  );
}
