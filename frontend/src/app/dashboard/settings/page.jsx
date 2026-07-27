"use client";
import { useState, useEffect } from "react";
import { useSession } from "next-auth/react";
import { Plus, Edit2, Trash2, CheckCircle, XCircle, Wifi } from "lucide-react";
import Modal from "@/components/ui/Modal";
import { apiFetch, API_BASE } from "@/lib/api";
import { useAlertStore } from "@/store/alertStore";
import { useSiteStore } from "@/store/siteStore";

// ─── Camera Form Modal ───────────────────────────────
function CameraFormModal({ isOpen, onClose, camera, onSave, session }) {
  const { addToast } = useAlertStore();
  const { sites, activeSite } = useSiteStore();
  
  const defaultSiteId = activeSite?.id || (sites?.length > 0 ? sites[0].id : "");

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
      let currentSiteId = form.site_id;
      if (!currentSiteId) {
        // Fallback: if store didn't load site for some reason, fetch it now
        const fetchedSites = await apiFetch("/api/sites", {}, session);
        if (fetchedSites && fetchedSites.length > 0) {
          currentSiteId = fetchedSites[0].id;
        } else {
          throw new Error("No site available to assign camera. Please create a site first.");
        }
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

  return (
    <Modal isOpen={isOpen} onClose={onClose} title={camera ? "Edit Camera" : "Add Camera"} size="md">
      <form onSubmit={handleSave}>
        <div className="modal-body" style={{ display: "flex", flexDirection: "column", gap: "1rem" }}>
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

// ─── Main Settings Page ──────────────────────────────
export default function SettingsPage() {
  const { data: session } = useSession();
  const { addToast } = useAlertStore();
  const [tab, setTab] = useState("cameras");
  const [cameras, setCameras] = useState([]);
  const [loading, setLoading] = useState(true);
  const [showForm, setShowForm] = useState(false);
  const [editCam, setEditCam] = useState(null);
  const [deleteConfirm, setDeleteConfirm] = useState(null);

  async function loadCameras() {
    if (!session) return;
    try {
      const c = await apiFetch("/api/cameras", {}, session);
      setCameras(c || []);
    } catch {} finally { setLoading(false); }
  }

  useEffect(() => { loadCameras(); }, [session]);

  async function deleteCam(cam) {
    try {
      await apiFetch(`/api/cameras/${cam.id}`, { method: "DELETE" }, session);
      addToast({ type: "success", message: `${cam.name} removed` });
      loadCameras();
    } catch (err) { 
      console.error("Delete error:", err);
      addToast({ type: "error", message: `Failed to delete: ${err.message}` }); 
    }
    setDeleteConfirm(null);
  }

  async function toggleActive(cam) {
    await apiFetch(`/api/cameras/${cam.id}`, {
      method: "PUT",
      body: JSON.stringify({ is_active: !cam.is_active }),
    }, session);
    loadCameras();
  }

  return (
    <>
      <div className="page-header">
        <div>
          <h2 className="page-title">Settings</h2>
          <p className="page-subtitle">Manage cameras, alerts, and detection configuration</p>
        </div>
      </div>

      <div className="tab-bar">
        {[
          { key: "cameras",    label: "Cameras"               },
          { key: "alerts",     label: "Alert Config"          },
          { key: "thresholds", label: "Detection Thresholds"  },
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
        onSave={loadCameras}
        session={session}
      />

      {/* Delete Confirmation Modal */}
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
    </>
  );
}
