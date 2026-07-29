"use client";
import { useState, useEffect } from "react";
import { useSession } from "next-auth/react";
import { Plus, Edit2, Trash2, CheckCircle, XCircle, Wifi, MapPin } from "lucide-react";
import Modal from "@/components/ui/Modal";
import { apiFetch, API_BASE } from "@/lib/api";
import { useAlertStore } from "@/store/alertStore";
import { useSiteStore } from "@/store/siteStore";

// ─── Camera Form Modal ───────────────────────────────
function CameraFormModal({ isOpen, onClose, camera, onSave, session, sites }) {
  const { addToast } = useAlertStore();

  const defaultSiteId = sites?.length > 0 ? sites[0].id : "";

  const [form, setForm] = useState({
    name: "", stream_url: "", type: "rtsp", fps_target: 15,
    confidence_threshold: 0.5, is_entrance: false, site_id: defaultSiteId,
  });
  const [testing, setTesting] = useState(false);
  const [testResult, setTestResult] = useState(null);
  const [saving, setSaving] = useState(false);

  useEffect(() => {
    if (camera) setForm({ ...camera });
    else setForm({ name: "", stream_url: "", type: "rtsp", fps_target: 15, confidence_threshold: 0.5, is_entrance: false, site_id: defaultSiteId });
    setTestResult(null);
  }, [camera, isOpen, defaultSiteId]);

  async function testStream() {
    setTesting(true);
    setTestResult(null);
    try {
      const r = await apiFetch("/api/cameras/test", {
        method: "POST",
        body: JSON.stringify({ stream_url: form.stream_url, type: form.type }),
      }, session);
      setTestResult(r);
    } catch { setTestResult({ success: false, message: "Test failed" }); }
    finally { setTesting(false); }
  }

  async function handleSave(e) {
    e.preventDefault();
    setSaving(true);
    try {
      const currentSiteId = form.site_id || defaultSiteId;
      if (!currentSiteId) {
        throw new Error("No site available. Please create a site first in the Sites tab.");
      }

      const payload = { ...form, fps_target: Number(form.fps_target), site_id: currentSiteId };
      if (camera) {
        await apiFetch(`/api/cameras/${camera.id}`, { method: "PUT", body: JSON.stringify(payload) }, session);
      } else {
        await apiFetch("/api/cameras", { method: "POST", body: JSON.stringify(payload) }, session);
      }
      addToast({ type: "success", message: `Camera ${camera ? "updated" : "added"} successfully` });
      onSave();
      onClose();
    } catch (err) {
      console.error("Save error:", err);
      addToast({ type: "error", message: `Save failed: ${err.message}` });
    } finally {
      setSaving(false);
    }
  }

  if (sites?.length === 0) {
    return (
      <Modal isOpen={isOpen} onClose={onClose} title="Add Camera" size="sm">
        <div className="modal-body" style={{ textAlign: "center", padding: "2rem 1rem" }}>
          <MapPin size={40} style={{ color: "var(--color-warning)", margin: "0 auto 1rem" }} />
          <p style={{ fontWeight: 600, marginBottom: "0.5rem" }}>No Site Found</p>
          <p style={{ color: "var(--color-text-secondary)", fontSize: "0.875rem" }}>
            Please go to the <strong>Sites</strong> tab and create a site before adding cameras.
          </p>
        </div>
        <div className="modal-footer">
          <button className="btn btn-primary btn-sm" onClick={onClose}>Go to Sites</button>
        </div>
      </Modal>
    );
  }

  return (
    <Modal isOpen={isOpen} onClose={onClose} title={camera ? "Edit Camera" : "Add Camera"} size="md">
      <form onSubmit={handleSave}>
        <div className="modal-body" style={{ display: "flex", flexDirection: "column", gap: "1rem" }}>
          {/* Site selector */}
          {sites?.length > 1 && (
            <div className="form-group">
              <label className="input-label">Site</label>
              <select className="input" value={form.site_id}
                onChange={e => setForm(f => ({ ...f, site_id: e.target.value }))}>
                {sites.map(s => <option key={s.id} value={s.id}>{s.name}</option>)}
              </select>
            </div>
          )}
          <div className="form-group">
            <label className="input-label">Camera Name</label>
            <input className="input" required value={form.name}
              onChange={e => setForm(f => ({ ...f, name: e.target.value }))} placeholder="Main Gate" />
          </div>
          <div className="form-group">
            <label className="input-label">Type</label>
            <select className="input" value={form.type}
              onChange={e => setForm(f => ({ ...f, type: e.target.value }))}>
              <option value="rtsp">RTSP (IP Camera)</option>
              <option value="webcam">Webcam (USB)</option>
            </select>
          </div>
          <div className="form-group">
            <label className="input-label">
              {form.type === "rtsp" ? "RTSP URL" : "Webcam Index"}
            </label>
            <div style={{ display: "flex", gap: "0.5rem" }}>
              <input className="input" required value={form.stream_url}
                onChange={e => setForm(f => ({ ...f, stream_url: e.target.value }))}
                placeholder={form.type === "rtsp" ? "rtsp://user:pass@192.168.1.50/stream" : "0"}
                style={{ flex: 1 }}
              />
              <button type="button" className="btn btn-secondary btn-sm" onClick={testStream} disabled={testing}>
                <Wifi size={13} /> {testing ? "Testing…" : "Test"}
              </button>
            </div>
            {testResult && (
              <div style={{
                marginTop: 6, fontSize: "0.78rem", fontWeight: 600,
                color: testResult.success ? "var(--color-success)" : "var(--color-danger)",
                display: "flex", alignItems: "center", gap: 4,
              }}>
                {testResult.success ? <CheckCircle size={13} /> : <XCircle size={13} />}
                {testResult.message}
              </div>
            )}
          </div>
          <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "0.75rem" }}>
            <div className="form-group">
              <label className="input-label">FPS Target</label>
              <input type="number" className="input" min={1} max={30} value={form.fps_target}
                onChange={e => setForm(f => ({ ...f, fps_target: +e.target.value }))} />
            </div>
            <div className="form-group">
              <label className="input-label">Confidence Threshold</label>
              <input type="number" className="input" min={0.3} max={0.95} step={0.05}
                value={form.confidence_threshold}
                onChange={e => setForm(f => ({ ...f, confidence_threshold: +e.target.value }))} />
            </div>
          </div>
          <label className="remember-label">
            <input type="checkbox" checked={form.is_entrance}
              onChange={e => setForm(f => ({ ...f, is_entrance: e.target.checked }))} />
            Entrance camera (enables face recognition)
          </label>
        </div>
        <div className="modal-footer">
          <button type="button" className="btn btn-ghost btn-sm" onClick={onClose}>Cancel</button>
          <button type="submit" className="btn btn-primary btn-sm" disabled={saving}>
            {saving ? "Saving…" : "Save Camera"}
          </button>
        </div>
      </form>
    </Modal>
  );
}

// ─── Create Site Modal ───────────────────────────────
function CreateSiteModal({ isOpen, onClose, onSave, session }) {
  const { addToast } = useAlertStore();
  const [form, setForm] = useState({ name: "", location: "" });
  const [saving, setSaving] = useState(false);

  useEffect(() => {
    if (isOpen) setForm({ name: "", location: "" });
  }, [isOpen]);

  async function handleSubmit(e) {
    e.preventDefault();
    setSaving(true);
    try {
      await apiFetch("/api/sites", {
        method: "POST",
        body: JSON.stringify({ name: form.name, location: form.location }),
      }, session);
      addToast({ type: "success", message: `Site "${form.name}" created successfully` });
      onSave();
      onClose();
    } catch (err) {
      addToast({ type: "error", message: `Failed to create site: ${err.message}` });
    } finally {
      setSaving(false);
    }
  }

  return (
    <Modal isOpen={isOpen} onClose={onClose} title="Create New Site" size="sm">
      <form onSubmit={handleSubmit}>
        <div className="modal-body" style={{ display: "flex", flexDirection: "column", gap: "1rem" }}>
          <div className="form-group">
            <label className="input-label">Site Name *</label>
            <input className="input" required value={form.name}
              onChange={e => setForm(f => ({ ...f, name: e.target.value }))}
              placeholder="e.g. Main Construction Site" />
          </div>
          <div className="form-group">
            <label className="input-label">Location</label>
            <input className="input" value={form.location}
              onChange={e => setForm(f => ({ ...f, location: e.target.value }))}
              placeholder="e.g. Lahore, Pakistan" />
          </div>
        </div>
        <div className="modal-footer">
          <button type="button" className="btn btn-ghost btn-sm" onClick={onClose}>Cancel</button>
          <button type="submit" className="btn btn-primary btn-sm" disabled={saving}>
            {saving ? "Creating…" : "Create Site"}
          </button>
        </div>
      </form>
    </Modal>
  );
}

// ─── Main Settings Page ──────────────────────────────
export default function SettingsPage() {
  const { data: session } = useSession();
  const { addToast } = useAlertStore();
  const [tab, setTab] = useState("cameras");
  const [cameras, setCameras] = useState([]);
  const [sites, setSites] = useState([]);
  const [loading, setLoading] = useState(true);
  const [showForm, setShowForm] = useState(false);
  const [showSiteForm, setShowSiteForm] = useState(false);
  const [editCam, setEditCam] = useState(null);
  const [deleteConfirm, setDeleteConfirm] = useState(null);
  const [deleteSiteConfirm, setDeleteSiteConfirm] = useState(null);

  async function loadData() {
    if (!session) return;
    try {
      const [c, s] = await Promise.all([
        apiFetch("/api/cameras", {}, session).catch(() => []),
        apiFetch("/api/sites", {}, session).catch(() => []),
      ]);
      setCameras(c || []);
      setSites(s || []);
    } catch {} finally { setLoading(false); }
  }

  useEffect(() => { loadData(); }, [session]);

  async function deleteCam(cam) {
    try {
      await apiFetch(`/api/cameras/${cam.id}`, { method: "DELETE" }, session);
      addToast({ type: "success", message: `${cam.name} removed` });
      loadData();
    } catch (err) {
      addToast({ type: "error", message: `Failed to delete: ${err.message}` });
    }
    setDeleteConfirm(null);
  }

  async function deleteSite(site) {
    try {
      await apiFetch(`/api/sites/${site.id}`, { method: "DELETE" }, session);
      addToast({ type: "success", message: `Site "${site.name}" deleted` });
      loadData();
    } catch (err) {
      addToast({ type: "error", message: `Failed to delete site: ${err.message}` });
    }
    setDeleteSiteConfirm(null);
  }

  async function toggleActive(cam) {
    await apiFetch(`/api/cameras/${cam.id}`, {
      method: "PUT",
      body: JSON.stringify({ is_active: !cam.is_active }),
    }, session);
    loadData();
  }

  return (
    <>
      <div className="page-header">
        <div>
          <h2 className="page-title">Settings</h2>
          <p className="page-subtitle">Manage cameras, sites, alerts, and detection configuration</p>
        </div>
      </div>

      <div className="tab-bar">
        {[
          { key: "cameras",    label: "Cameras"              },
          { key: "sites",      label: `Sites (${sites.length})`},
          { key: "alerts",     label: "Alert Config"         },
          { key: "thresholds", label: "Detection Thresholds" },
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

      {/* ── Cameras Tab ───────────────────────── */}
      {tab === "cameras" && (
        <>
          {sites.length === 0 && !loading && (
            <div className="card" style={{
              background: "rgba(251,191,36,0.08)", border: "1px solid rgba(251,191,36,0.3)",
              marginBottom: "1rem", padding: "1rem 1.25rem",
              display: "flex", alignItems: "center", gap: "0.75rem",
            }}>
              <MapPin size={18} style={{ color: "var(--color-warning)", flexShrink: 0 }} />
              <span style={{ color: "var(--color-warning)", fontSize: "0.875rem", fontWeight: 500 }}>
                No site exists yet. Please{" "}
                <button
                  onClick={() => { setTab("sites"); setShowSiteForm(true); }}
                  style={{ textDecoration: "underline", background: "none", border: "none",
                    color: "var(--color-warning)", cursor: "pointer", fontWeight: 700 }}
                >
                  create a site first
                </button>{" "}
                before adding cameras.
              </span>
            </div>
          )}
          <div style={{ display: "flex", justifyContent: "flex-end", marginBottom: "1rem" }}>
            <button className="btn btn-primary btn-sm" onClick={() => { setEditCam(null); setShowForm(true); }}>
              <Plus size={14} /> Add Camera
            </button>
          </div>
          <div className="table-wrap">
            <table className="table">
              <thead>
                <tr>
                  <th>Name</th><th>Type</th><th>Stream URL</th>
                  <th>FPS</th><th>Threshold</th><th>Status</th><th>Active</th><th>Actions</th>
                </tr>
              </thead>
              <tbody>
                {loading ? (
                  <tr><td colSpan={8}><div className="empty-state">Loading…</div></td></tr>
                ) : cameras.length === 0 ? (
                  <tr><td colSpan={8}><div className="empty-state">No cameras yet. Add your first camera above.</div></td></tr>
                ) : cameras.map(cam => (
                  <tr key={cam.id}>
                    <td style={{ fontWeight: 600 }}>{cam.name}</td>
                    <td><span className="badge badge-neutral">{cam.type?.toUpperCase()}</span></td>
                    <td style={{ maxWidth: 200, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
                      <span className="text-mono" style={{ fontSize: "0.75rem" }}>{cam.stream_url}</span>
                    </td>
                    <td>{cam.fps_target}</td>
                    <td className="text-mono">{(cam.confidence_threshold * 100).toFixed(0)}%</td>
                    <td>
                      <span className={`badge badge-${cam.last_seen ? "success" : "neutral"}`}>
                        {cam.last_seen ? "Online" : "Unknown"}
                      </span>
                    </td>
                    <td>
                      <label className="toggle">
                        <input type="checkbox" checked={cam.is_active}
                          onChange={() => toggleActive(cam)} />
                        <div className="toggle-track" />
                      </label>
                    </td>
                    <td>
                      <div style={{ display: "flex", gap: "0.375rem" }}>
                        <button className="btn btn-ghost btn-sm" onClick={() => { setEditCam(cam); setShowForm(true); }}>
                          <Edit2 size={13} />
                        </button>
                        <button className="btn btn-danger btn-sm" onClick={() => setDeleteConfirm(cam)}>
                          <Trash2 size={13} />
                        </button>
                      </div>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </>
      )}

      {/* ── Sites Tab ─────────────────────────── */}
      {tab === "sites" && (
        <>
          <div style={{ display: "flex", justifyContent: "flex-end", marginBottom: "1rem" }}>
            <button className="btn btn-primary btn-sm" onClick={() => setShowSiteForm(true)}>
              <Plus size={14} /> Create Site
            </button>
          </div>
          <div className="table-wrap">
            <table className="table">
              <thead>
                <tr><th>Site Name</th><th>Location</th><th>Created</th><th>Actions</th></tr>
              </thead>
              <tbody>
                {loading ? (
                  <tr><td colSpan={4}><div className="empty-state">Loading…</div></td></tr>
                ) : sites.length === 0 ? (
                  <tr><td colSpan={4}>
                    <div className="empty-state" style={{ padding: "2.5rem" }}>
                      <MapPin size={32} style={{ opacity: 0.3, margin: "0 auto 0.75rem", display: "block" }} />
                      No sites yet. Create your first site to start adding cameras.
                    </div>
                  </td></tr>
                ) : sites.map(site => (
                  <tr key={site.id}>
                    <td style={{ fontWeight: 600 }}>
                      <div style={{ display: "flex", alignItems: "center", gap: "0.5rem" }}>
                        <MapPin size={14} style={{ color: "var(--color-primary)", flexShrink: 0 }} />
                        {site.name}
                      </div>
                    </td>
                    <td style={{ color: "var(--color-text-secondary)" }}>{site.location || "—"}</td>
                    <td style={{ fontSize: "0.8rem", color: "var(--color-text-secondary)" }}>
                      {new Date(site.created_at).toLocaleDateString()}
                    </td>
                    <td>
                      <button className="btn btn-danger btn-sm" onClick={() => setDeleteSiteConfirm(site)}>
                        <Trash2 size={13} />
                      </button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </>
      )}

      {/* ── Alert Config Tab ──────────────────── */}
      {tab === "alerts" && (
        <div className="card" style={{ maxWidth: 600 }}>
          <p style={{ color: "var(--color-text-secondary)", fontSize: "0.875rem" }}>
            Alert configuration will be available here. Connect your email SMTP settings to enable notifications.
          </p>
        </div>
      )}

      {/* ── Detection Thresholds Tab ──────────── */}
      {tab === "thresholds" && (
        <div className="table-wrap">
          <table className="table">
            <thead>
              <tr><th>Camera</th><th>Threshold</th><th>Value</th></tr>
            </thead>
            <tbody>
              {cameras.map(cam => (
                <tr key={cam.id}>
                  <td style={{ fontWeight: 600 }}>{cam.name}</td>
                  <td style={{ width: "50%" }}>
                    <input
                      type="range" className="slider"
                      min={0.30} max={0.95} step={0.05}
                      defaultValue={cam.confidence_threshold}
                      onChange={async (e) => {
                        await apiFetch(`/api/cameras/${cam.id}`, {
                          method: "PUT",
                          body: JSON.stringify({ confidence_threshold: +e.target.value }),
                        }, session);
                      }}
                    />
                  </td>
                  <td className="text-mono">
                    {(cam.confidence_threshold * 100).toFixed(0)}%
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      {/* Camera Form Modal */}
      <CameraFormModal
        isOpen={showForm}
        onClose={() => setShowForm(false)}
        camera={editCam}
        onSave={loadData}
        session={session}
        sites={sites}
      />

      {/* Create Site Modal */}
      <CreateSiteModal
        isOpen={showSiteForm}
        onClose={() => setShowSiteForm(false)}
        onSave={loadData}
        session={session}
      />

      {/* Delete Camera Confirmation */}
      <Modal isOpen={!!deleteConfirm} onClose={() => setDeleteConfirm(null)} title="Delete Camera" size="sm">
        <div className="modal-body">
          <p>Are you sure you want to remove <strong>{deleteConfirm?.name}</strong>? This cannot be undone.</p>
        </div>
        <div className="modal-footer">
          <button className="btn btn-ghost btn-sm" onClick={() => setDeleteConfirm(null)}>Cancel</button>
          <button className="btn btn-danger btn-sm" onClick={() => deleteCam(deleteConfirm)}>
            <Trash2 size={13} /> Delete
          </button>
        </div>
      </Modal>

      {/* Delete Site Confirmation */}
      <Modal isOpen={!!deleteSiteConfirm} onClose={() => setDeleteSiteConfirm(null)} title="Delete Site" size="sm">
        <div className="modal-body">
          <p>Delete site <strong>{deleteSiteConfirm?.name}</strong>? All cameras assigned to this site will also be removed.</p>
        </div>
        <div className="modal-footer">
          <button className="btn btn-ghost btn-sm" onClick={() => setDeleteSiteConfirm(null)}>Cancel</button>
          <button className="btn btn-danger btn-sm" onClick={() => deleteSite(deleteSiteConfirm)}>
            <Trash2 size={13} /> Delete Site
          </button>
        </div>
      </Modal>
    </>
  );
}
