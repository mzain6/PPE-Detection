"use client";
import { useState, useEffect } from "react";
import { useSession } from "next-auth/react";
import { UserPlus, Upload, Edit2, Clock } from "lucide-react";
import Modal from "@/components/ui/Modal";
import { apiFetch, API_BASE } from "@/lib/api";
import { useAlertStore } from "@/store/alertStore";

// ─── Employee Card ───────────────────────────────────
function EmployeeCard({ person, isAdmin, onEdit, onUploadFace }) {
  const initials = person.name.split(" ").map(n => n[0]).join("").slice(0, 2).toUpperCase();
  return (
    <div className="card" style={{ display: "flex", flexDirection: "column", gap: "1rem" }}>
      <div style={{ display: "flex", alignItems: "center", gap: "0.875rem" }}>
        {person.photo_path ? (
          <img
            src={`${API_BASE}/${person.photo_path}`}
            alt={person.name}
            style={{ width: 52, height: 52, borderRadius: "50%", objectFit: "cover", flexShrink: 0 }}
          />
        ) : (
          <div style={{
            width: 52, height: 52, borderRadius: "50%",
            background: "var(--color-primary-light)",
            display: "flex", alignItems: "center", justifyContent: "center",
            fontSize: "1rem", fontWeight: 700, color: "var(--color-primary)",
            flexShrink: 0,
          }}>
            {initials}
          </div>
        )}
        <div style={{ flex: 1, minWidth: 0 }}>
          <div style={{ fontWeight: 700, fontSize: "0.9rem" }}>{person.name}</div>
          <span className="badge badge-neutral" style={{ fontSize: "0.65rem" }}>
            #{person.employee_id}
          </span>
        </div>
        <span className={`badge badge-${person.is_active ? "success" : "neutral"}`}>
          {person.is_active ? "Active" : "Inactive"}
        </span>
      </div>

      <div style={{ fontSize: "0.78rem", color: "var(--color-text-secondary)" }}>
        {person.department && <div>🏢 {person.department}</div>}
        <div style={{ marginTop: 4, display: "flex", alignItems: "center", gap: 4 }}>
          <Clock size={12} />
          {person.last_seen
            ? `Last seen: ${new Date(person.last_seen).toLocaleDateString()}`
            : "Never seen"}
        </div>
      </div>

      {isAdmin && (
        <div style={{ display: "flex", gap: "0.5rem" }}>
          <button className="btn btn-secondary btn-sm" style={{ flex: 1 }} onClick={() => onEdit(person)}>
            <Edit2 size={12} /> Edit
          </button>
          <button className="btn btn-ghost btn-sm" style={{ flex: 1 }} onClick={() => onUploadFace(person)}>
            <Upload size={12} /> Photo
          </button>
        </div>
      )}
    </div>
  );
}

// ─── Main Page ────────────────────────────────────────
export default function PersonnelPage() {
  const { data: session } = useSession();
  const { addToast } = useAlertStore();
  const isAdmin = ["super_admin", "site_admin"].includes(session?.user?.role);

  const [tab, setTab] = useState("personnel");
  const [personnel, setPersonnel] = useState([]);
  const [accessLog, setAccessLog] = useState([]);
  const [loading, setLoading] = useState(true);

  const [showRegister, setShowRegister] = useState(false);
  const [editPerson, setEditPerson] = useState(null);
  const [uploadPerson, setUploadPerson] = useState(null);

  const [form, setForm] = useState({ name: "", employee_id: "", department: "", site_id: "" });

  useEffect(() => {
    if (!session) return;
    Promise.all([
      apiFetch("/api/personnel", {}, session),
      apiFetch("/api/access-log?page_size=50", {}, session),
    ]).then(([p, a]) => {
      setPersonnel(p || []);
      setAccessLog(a?.items || []);
    }).catch(() => {})
      .finally(() => setLoading(false));
  }, [session]);

  async function submitRegister(e) {
    e.preventDefault();
    try {
      const fd = new FormData(e.target);
      const photo = fd.get("photo");
      const body = new FormData();
      body.append("name",        form.name);
      body.append("employee_id", form.employee_id);
      body.append("department",  form.department);
      if (session?.user?.site_ids?.[0]) body.append("site_id", session.user.site_ids[0]);
      if (photo?.size) body.append("photo", photo);

      await fetch(`${API_BASE}/api/personnel`, {
        method: "POST",
        headers: { Authorization: `Bearer ${session?.access_token}` },
        body,
      });
      addToast({ type: "success", message: `${form.name} registered successfully` });
      setShowRegister(false);
      setForm({ name: "", employee_id: "", department: "", site_id: "" });
      const updated = await apiFetch("/api/personnel", {}, session);
      setPersonnel(updated || []);
    } catch { addToast({ type: "error", message: "Failed to register employee" }); }
  }

  return (
    <>
      <div className="page-header">
        <div>
          <h2 className="page-title">Personnel</h2>
          <p className="page-subtitle">Manage authorized employees and access records</p>
        </div>
        {isAdmin && (
          <button className="btn btn-primary btn-sm" onClick={() => setShowRegister(true)}>
            <UserPlus size={14} /> Register Employee
          </button>
        )}
      </div>

      <div className="tab-bar">
        {[
          { key: "personnel", label: "Registered Personnel" },
          { key: "log",       label: "Access Log"           },
        ].map(t => (
          <button
            key={t.key}
            className={`tab-item${tab === t.key ? " active" : ""}`}
            onClick={() => setTab(t.key)}
          >
            {t.label}
          </button>
        ))}
      </div>

      {tab === "personnel" && (
        loading ? (
          <div style={{ display: "grid", gridTemplateColumns: "repeat(3, 1fr)", gap: "1rem" }}>
            {Array.from({ length: 6 }).map((_, i) => (
              <div key={i} className="card skeleton" style={{ height: 160 }} />
            ))}
          </div>
        ) : personnel.length === 0 ? (
          <div className="empty-state card" style={{ padding: "4rem" }}>
            <span className="empty-state-title">No employees registered yet</span>
          </div>
        ) : (
          <div style={{ display: "grid", gridTemplateColumns: "repeat(3, 1fr)", gap: "1rem" }}>
            {personnel.map(p => (
              <EmployeeCard
                key={p.id}
                person={p}
                isAdmin={isAdmin}
                onEdit={setEditPerson}
                onUploadFace={setUploadPerson}
              />
            ))}
          </div>
        )
      )}

      {tab === "log" && (
        <div className="table-wrap">
          <table className="table">
            <thead>
              <tr>
                <th>Time</th><th>Person</th><th>Camera</th><th>Event</th><th>Confidence</th>
              </tr>
            </thead>
            <tbody>
              {accessLog.map(log => (
                <tr key={log.id}>
                  <td className="text-mono" style={{ fontSize: "0.8rem" }}>
                    {new Date(log.timestamp).toLocaleString()}
                  </td>
                  <td>{log.person_name || <span className="text-muted">Unknown</span>}</td>
                  <td>{log.camera_name || "—"}</td>
                  <td>
                    <span className={`badge badge-${log.event_type === "entry" ? "success" : log.event_type === "exit" ? "info" : "neutral"}`}>
                      {log.event_type}
                    </span>
                  </td>
                  <td className="text-mono" style={{ fontSize: "0.8rem" }}>
                    {log.confidence ? `${(log.confidence * 100).toFixed(1)}%` : "—"}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      {/* Register Modal */}
      <Modal isOpen={showRegister} onClose={() => setShowRegister(false)} title="Register Employee" size="md">
        <form onSubmit={submitRegister}>
          <div className="modal-body" style={{ display: "flex", flexDirection: "column", gap: "1rem" }}>
            <div className="form-group">
              <label className="input-label">Full Name</label>
              <input className="input" required value={form.name}
                onChange={e => setForm(f => ({ ...f, name: e.target.value }))} placeholder="John Doe" />
            </div>
            <div className="form-group">
              <label className="input-label">Employee ID</label>
              <input className="input" required value={form.employee_id}
                onChange={e => setForm(f => ({ ...f, employee_id: e.target.value }))} placeholder="EMP-001" />
            </div>
            <div className="form-group">
              <label className="input-label">Department</label>
              <input className="input" value={form.department}
                onChange={e => setForm(f => ({ ...f, department: e.target.value }))} placeholder="Engineering" />
            </div>
            <div className="form-group">
              <label className="input-label">Photo</label>
              <input type="file" name="photo" className="input" accept="image/*" />
            </div>
          </div>
          <div className="modal-footer">
            <button type="button" className="btn btn-ghost btn-sm" onClick={() => setShowRegister(false)}>Cancel</button>
            <button type="submit" className="btn btn-primary btn-sm">Register</button>
          </div>
        </form>
      </Modal>
    </>
  );
}
