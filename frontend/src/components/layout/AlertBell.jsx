"use client";
import { useState, useRef, useEffect } from "react";
import { Bell, ShieldAlert, X } from "lucide-react";
import { useAlertStore } from "@/store/alertStore";
import { API_BASE } from "@/lib/api";

function timeAgo(ts) {
  const diff = (Date.now() - new Date(ts)) / 1000;
  if (diff < 60) return `${Math.round(diff)}s ago`;
  if (diff < 3600) return `${Math.round(diff / 60)}m ago`;
  return `${Math.round(diff / 3600)}h ago`;
}

export default function AlertBell() {
  const [open, setOpen] = useState(false);
  const ref = useRef(null);
  const { unreadCount, recentAlerts, clearUnread } = useAlertStore();

  // Close on outside click
  useEffect(() => {
    function handler(e) {
      if (ref.current && !ref.current.contains(e.target)) setOpen(false);
    }
    document.addEventListener("mousedown", handler);
    return () => document.removeEventListener("mousedown", handler);
  }, []);

  function handleOpen() {
    setOpen((v) => !v);
    if (!open) clearUnread();
  }

  const typeColor = {
    no_helmet: "var(--color-warning)",
    no_vest:   "var(--color-danger)",
    no_gloves: "var(--color-warning)",
    no_both:   "var(--color-danger)",
  };

  return (
    <div style={{ position: "relative" }} ref={ref}>
      <button className="bell-btn" onClick={handleOpen} aria-label="Alerts">
        <Bell size={18} />
        {unreadCount > 0 && (
          <span className="bell-badge">{unreadCount > 99 ? "99+" : unreadCount}</span>
        )}
      </button>

      {open && (
        <div className="bell-dropdown">
          <div className="bell-dropdown-header">
            <span className="bell-dropdown-title">Recent Violations</span>
            <button className="bell-dropdown-clear" onClick={clearUnread}>
              Clear all
            </button>
          </div>

          {recentAlerts.length === 0 ? (
            <div style={{ padding: "2rem 1rem", textAlign: "center", color: "var(--color-text-muted)", fontSize: "0.8rem" }}>
              No new violations
            </div>
          ) : (
            recentAlerts.map((alert, i) => (
              <div key={alert.id || i} className="bell-alert-item">
                <ShieldAlert
                  size={16}
                  style={{ color: typeColor[alert.violation_type] || "var(--color-text-muted)", flexShrink: 0, marginTop: 2 }}
                />
                <div style={{ flex: 1, minWidth: 0 }}>
                  <div style={{ fontSize: "0.8rem", fontWeight: 600, color: "var(--color-text-primary)" }}>
                    {alert.violation_type?.replace(/_/g, " ").toUpperCase()}
                  </div>
                  <div style={{ fontSize: "0.72rem", color: "var(--color-text-muted)" }}>
                    {alert.camera_name || "Unknown camera"} · {timeAgo(alert.timestamp)}
                  </div>
                </div>
                {alert.screenshot_path && (
                  <img
                    src={`${API_BASE}/${alert.screenshot_path}`}
                    alt="Violation"
                    style={{ width: 44, height: 32, objectFit: "cover", borderRadius: 4, flexShrink: 0 }}
                  />
                )}
              </div>
            ))
          )}

          <div style={{ padding: "0.625rem 1rem", borderTop: "1px solid var(--color-border)" }}>
            <a
              href="/dashboard/violations"
              style={{ fontSize: "0.78rem", color: "var(--color-primary)" }}
              onClick={() => setOpen(false)}
            >
              View all violations →
            </a>
          </div>
        </div>
      )}
    </div>
  );
}
