"use client";
import { useState, useEffect, useRef } from "react";
import { useSession } from "next-auth/react";
import { Maximize2, Wifi, WifiOff } from "lucide-react";
import Modal from "@/components/ui/Modal";
import { ViolationBadge } from "@/components/ui/Badge";
import { apiFetch, API_BASE } from "@/lib/api";

const PAGE_SIZE = 6;

function CameraTile({ cam, onExpand }) {
  const [offline, setOffline] = useState(false);
  const [visible, setVisible] = useState(false);
  const ref = useRef(null);

  // Intersection Observer — only load stream when visible
  useEffect(() => {
    const obs = new IntersectionObserver(
      ([entry]) => setVisible(entry.isIntersecting),
      { threshold: 0.1 }
    );
    if (ref.current) obs.observe(ref.current);
    return () => obs.disconnect();
  }, []);

  const streamSrc = `${API_BASE}/api/cameras/${cam.id}/stream`;

  return (
    <div className="camera-tile" ref={ref} onClick={() => onExpand(cam)}>
      {offline ? (
        <div className="camera-offline">
          <WifiOff size={28} />
          <span>Offline</span>
        </div>
      ) : (
        visible && (
          <img
            src={streamSrc}
            alt={cam.name}
            onError={() => setOffline(true)}
          />
        )
      )}

      {!visible && !offline && (
        <div className="camera-offline" style={{ color: "var(--color-text-muted)" }}>
          <span style={{ fontSize: "0.75rem" }}>Loading…</span>
        </div>
      )}

      <div className="camera-tile-overlay">
        <Maximize2 size={24} color="white" style={{ opacity: 0.9 }} />
      </div>

      {/* Top labels */}
      <div className="camera-tile-label-top">
        <span className="camera-chip">{cam.name}</span>
        <span
          className="camera-chip"
          style={{ background: offline ? "rgba(239,68,68,0.8)" : "rgba(16,185,129,0.8)" }}
        >
          {offline ? "Offline" : "Online"}
        </span>
      </div>

      {/* Bottom labels */}
      <div className="camera-tile-label-bottom">
        <span className="camera-chip" style={{ fontSize: "0.65rem" }}>
          {cam.fps_target} FPS
        </span>
        {cam.site_name && (
          <span className="camera-chip" style={{ fontSize: "0.65rem" }}>
            {cam.site_name}
          </span>
        )}
      </div>
    </div>
  );
}

function CameraModal({ cam, onClose, session }) {
  const [recentViolations, setRecentViolations] = useState([]);

  useEffect(() => {
    if (!cam || !session) return;
    apiFetch(`/api/violations?camera_id=${cam.id}&page_size=5`, {}, session)
      .then((d) => setRecentViolations(d?.items || []))
      .catch(() => {});
  }, [cam, session]);

  if (!cam) return null;

  return (
    <Modal isOpen={!!cam} onClose={onClose} title={cam.name} size="xl">
      <div className="modal-body" style={{ padding: 0 }}>
        <img
          src={`${API_BASE}/api/cameras/${cam.id}/stream`}
          alt={cam.name}
          style={{ width: "100%", maxHeight: 480, objectFit: "cover", display: "block" }}
          onError={(e) => { e.target.style.display = "none"; }}
        />
        <div style={{ padding: "1.25rem 1.5rem" }}>
          <div style={{
            display: "grid", gridTemplateColumns: "1fr 1fr 1fr",
            gap: "0.75rem", marginBottom: "1.25rem"
          }}>
            <div>
              <div style={{ fontSize: "0.7rem", color: "var(--color-text-muted)", marginBottom: 2 }}>STATUS</div>
              <span className={`badge badge-${cam.is_active ? "success" : "danger"}`}>
                {cam.is_active ? "Active" : "Inactive"}
              </span>
            </div>
            <div>
              <div style={{ fontSize: "0.7rem", color: "var(--color-text-muted)", marginBottom: 2 }}>TYPE</div>
              <span className="badge badge-neutral">{cam.type?.toUpperCase()}</span>
            </div>
            <div>
              <div style={{ fontSize: "0.7rem", color: "var(--color-text-muted)", marginBottom: 2 }}>FPS TARGET</div>
              <span style={{ fontSize: "0.875rem", fontWeight: 600 }}>{cam.fps_target}</span>
            </div>
          </div>

          {recentViolations.length > 0 && (
            <>
              <h4 style={{ fontSize: "0.8rem", fontWeight: 700, marginBottom: "0.625rem", color: "var(--color-text-secondary)" }}>
                RECENT VIOLATIONS FROM THIS CAMERA
              </h4>
              <div style={{ display: "flex", flexDirection: "column", gap: "0.5rem" }}>
                {recentViolations.map((v) => (
                  <div key={v.id} style={{
                    display: "flex", alignItems: "center", gap: "0.75rem",
                    padding: "0.625rem", background: "var(--color-bg-surface)",
                    borderRadius: "var(--radius-md)", fontSize: "0.8rem",
                  }}>
                    <ViolationBadge type={v.violation_type} />
                    <span style={{ color: "var(--color-text-muted)" }}>
                      {new Date(v.timestamp).toLocaleString()}
                    </span>
                    <span style={{ marginLeft: "auto", fontFamily: "var(--font-mono)", fontSize: "0.75rem" }}>
                      {(v.confidence * 100).toFixed(1)}%
                    </span>
                  </div>
                ))}
              </div>
            </>
          )}
        </div>
      </div>
    </Modal>
  );
}

export default function CamerasPage() {
  const { data: session } = useSession();
  const [cameras, setCameras] = useState([]);
  const [loading, setLoading] = useState(true);
  const [page, setPage] = useState(1);
  const [selectedCam, setSelectedCam] = useState(null);

  useEffect(() => {
    if (!session) return;
    apiFetch("/api/cameras", {}, session)
      .then(setCameras)
      .catch(() => {})
      .finally(() => setLoading(false));
  }, [session]);

  const totalPages = Math.ceil(cameras.length / PAGE_SIZE);
  const pageCams   = cameras.slice((page - 1) * PAGE_SIZE, page * PAGE_SIZE);

  return (
    <>
      <div className="page-header">
        <div>
          <h2 className="page-title">Live Cameras</h2>
          <p className="page-subtitle">
            Showing {pageCams.length} of {cameras.length} cameras
          </p>
        </div>
      </div>

      {loading ? (
        <div className="camera-grid">
          {Array.from({ length: 6 }).map((_, i) => (
            <div key={i} className="camera-tile">
              <div className="skeleton" style={{ width: "100%", height: "100%" }} />
            </div>
          ))}
        </div>
      ) : cameras.length === 0 ? (
        <div className="empty-state card" style={{ padding: "4rem" }}>
          <Wifi size={48} className="empty-state-icon" />
          <span className="empty-state-title">No cameras configured</span>
          <a href="/dashboard/settings" style={{ fontSize: "0.8rem", color: "var(--color-primary)" }}>
            Add cameras in Settings →
          </a>
        </div>
      ) : (
        <>
          <div className="camera-grid">
            {pageCams.map((cam) => (
              <CameraTile key={cam.id} cam={cam} onExpand={setSelectedCam} />
            ))}
          </div>

          {totalPages > 1 && (
            <div className="pagination" style={{ marginTop: "1rem", border: "none", paddingTop: "1rem" }}>
              <span style={{ fontSize: "0.8rem", color: "var(--color-text-secondary)" }}>
                Showing {(page - 1) * PAGE_SIZE + 1}–{Math.min(page * PAGE_SIZE, cameras.length)} of {cameras.length} cameras
              </span>
              <div className="pagination-controls">
                <button className="page-btn" onClick={() => setPage(p => p - 1)} disabled={page === 1}>‹</button>
                {Array.from({ length: totalPages }).map((_, i) => (
                  <button
                    key={i}
                    className="page-btn"
                    style={page === i + 1 ? { background: "var(--color-primary)", color: "white", borderColor: "var(--color-primary)" } : {}}
                    onClick={() => setPage(i + 1)}
                  >{i + 1}</button>
                ))}
                <button className="page-btn" onClick={() => setPage(p => p + 1)} disabled={page === totalPages}>›</button>
              </div>
            </div>
          )}
        </>
      )}

      <CameraModal
        cam={selectedCam}
        onClose={() => setSelectedCam(null)}
        session={session}
      />
    </>
  );
}
