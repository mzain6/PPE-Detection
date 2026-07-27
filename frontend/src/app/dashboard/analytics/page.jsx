"use client";
import { useState, useEffect } from "react";
import { useSession } from "next-auth/react";
import {
  PieChart, Pie, Cell, BarChart, Bar, LineChart, Line,
  XAxis, YAxis, Tooltip, Legend, ResponsiveContainer
} from "recharts";
import { apiFetch } from "@/lib/api";

const COLORS = [
  "var(--color-danger)",
  "var(--color-warning)",
  "var(--color-info)",
  "var(--color-primary)",
];

const CHART_STYLE = {
  background: "var(--color-bg-surface)",
  border: "1px solid var(--color-border)",
  borderRadius: "var(--radius-lg)",
  padding: "1.25rem",
};

const TOOLTIP_STYLE = {
  contentStyle: {
    background: "var(--color-bg-elevated)",
    border: "1px solid var(--color-border)",
    borderRadius: 8,
    fontSize: 12,
    color: "var(--color-text-primary)",
  },
};

const QUICK_RANGES = [
  { label: "Today",       days: 1  },
  { label: "Last 7 Days", days: 7  },
  { label: "Last 30 Days",days: 30 },
];

// ─── Peak Hours Heatmap ─────────────────────────────
function PeakHoursChart({ data }) {
  if (!data?.length) return (
    <div className="empty-state" style={{ padding: "3rem 0" }}>No data</div>
  );

  const DAY_LABELS = ["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"];
  const maxCount = Math.max(...data.map(d => d.count), 1);

  const grid = {};
  data.forEach(({ day, hour, count }) => {
    grid[`${day}-${hour}`] = count;
  });

  return (
    <div style={{ overflowX: "auto" }}>
      <div style={{ minWidth: 420 }}>
        <div style={{
          display: "grid",
          gridTemplateColumns: "32px repeat(24, 1fr)",
          gap: 2,
          fontSize: "0.6rem",
          color: "var(--color-text-muted)",
        }}>
          {/* Hour labels */}
          <div />
          {Array.from({ length: 24 }, (_, h) => (
            <div key={h} style={{ textAlign: "center", marginBottom: 4 }}>
              {h % 6 === 0 ? `${h}h` : ""}
            </div>
          ))}

          {/* Day rows */}
          {DAY_LABELS.map((day, dayIdx) => (
            <>
              <div key={`label-${dayIdx}`} style={{ display: "flex", alignItems: "center", fontWeight: 600 }}>
                {day}
              </div>
              {Array.from({ length: 24 }, (_, hour) => {
                const count = grid[`${dayIdx}-${hour}`] || 0;
                const intensity = count / maxCount;
                const alpha = 0.1 + intensity * 0.85;
                return (
                  <div
                    key={`${dayIdx}-${hour}`}
                    title={`${day} ${hour}:00 — ${count} violations`}
                    style={{
                      aspectRatio: "1",
                      borderRadius: 2,
                      background: count === 0
                        ? "var(--color-bg-elevated)"
                        : `rgba(239, 68, 68, ${alpha})`,
                      transition: "opacity 0.15s",
                      cursor: "default",
                    }}
                  />
                );
              })}
            </>
          ))}
        </div>
      </div>
    </div>
  );
}

export default function AnalyticsPage() {
  const { data: session } = useSession();
  const [days, setDays] = useState(7);
  const [byType, setByType]       = useState([]);
  const [byCamera, setByCamera]   = useState([]);
  const [trend, setTrend]         = useState([]);
  const [peakHours, setPeakHours] = useState([]);
  const [loading, setLoading]     = useState(true);

  useEffect(() => {
    if (!session) return;
    setLoading(true);
    Promise.all([
      apiFetch("/api/analytics/violations-by-type", {}, session),
      apiFetch("/api/analytics/violations-by-camera", {}, session),
      apiFetch(`/api/analytics/trend?days=${days}`, {}, session),
      apiFetch(`/api/analytics/peak-hours?days=${days}`, {}, session),
    ]).then(([t, c, tr, ph]) => {
      setByType(t || []);
      setByCamera(c || []);
      setTrend(tr || []);
      setPeakHours(ph || []);
    }).catch(() => {})
      .finally(() => setLoading(false));
  }, [session, days]);

  return (
    <>
      <div className="page-header">
        <div>
          <h2 className="page-title">Analytics</h2>
          <p className="page-subtitle">Safety trend analysis and reporting</p>
        </div>
        {/* Quick Range */}
        <div style={{ display: "flex", gap: "0.5rem" }}>
          {QUICK_RANGES.map(r => (
            <button
              key={r.days}
              className={`btn btn-sm ${days === r.days ? "btn-primary" : "btn-secondary"}`}
              onClick={() => setDays(r.days)}
            >
              {r.label}
            </button>
          ))}
        </div>
      </div>

      <div className="chart-grid-2x2">
        {/* Violations by Type */}
        <div style={CHART_STYLE}>
          <h3 style={{ fontSize: "0.875rem", fontWeight: 700, marginBottom: "1rem" }}>Violations by Type</h3>
          {byType.length === 0 ? (
            <div className="empty-state" style={{ padding: "3rem 0" }}>No data</div>
          ) : (
            <ResponsiveContainer width="100%" height={220}>
              <PieChart>
                <Pie data={byType} dataKey="count" nameKey="type" cx="50%" cy="50%"
                  innerRadius={55} outerRadius={85} paddingAngle={3}>
                  {byType.map((_, i) => (
                    <Cell key={i} fill={COLORS[i % COLORS.length]} />
                  ))}
                </Pie>
                <Tooltip {...TOOLTIP_STYLE} />
                <Legend wrapperStyle={{ fontSize: "0.72rem", color: "var(--color-text-secondary)" }} />
              </PieChart>
            </ResponsiveContainer>
          )}
        </div>

        {/* Violations by Camera */}
        <div style={CHART_STYLE}>
          <h3 style={{ fontSize: "0.875rem", fontWeight: 700, marginBottom: "1rem" }}>Violations by Camera</h3>
          {byCamera.length === 0 ? (
            <div className="empty-state" style={{ padding: "3rem 0" }}>No data</div>
          ) : (
            <ResponsiveContainer width="100%" height={220}>
              <BarChart data={byCamera} layout="vertical">
                <XAxis type="number" tick={{ fontSize: 10, fill: "var(--color-text-muted)" }} axisLine={false} tickLine={false} />
                <YAxis type="category" dataKey="camera_name" width={90}
                  tick={{ fontSize: 10, fill: "var(--color-text-muted)" }} axisLine={false} tickLine={false} />
                <Tooltip {...TOOLTIP_STYLE} />
                <Bar dataKey="count" fill="var(--color-primary)" radius={[0, 4, 4, 0]} />
              </BarChart>
            </ResponsiveContainer>
          )}
        </div>

        {/* Violation Trend */}
        <div style={CHART_STYLE}>
          <h3 style={{ fontSize: "0.875rem", fontWeight: 700, marginBottom: "1rem" }}>Violation Trend</h3>
          {trend.length === 0 ? (
            <div className="empty-state" style={{ padding: "3rem 0" }}>No data</div>
          ) : (
            <ResponsiveContainer width="100%" height={220}>
              <LineChart data={trend}>
                <XAxis dataKey="date" tick={{ fontSize: 10, fill: "var(--color-text-muted)" }} axisLine={false} tickLine={false} />
                <YAxis tick={{ fontSize: 10, fill: "var(--color-text-muted)" }} axisLine={false} tickLine={false} />
                <Tooltip {...TOOLTIP_STYLE} />
                <Line
                  type="monotone" dataKey="count"
                  stroke="var(--color-primary)" strokeWidth={2}
                  dot={{ r: 3, fill: "var(--color-primary)" }}
                  activeDot={{ r: 5 }}
                />
              </LineChart>
            </ResponsiveContainer>
          )}
        </div>

        {/* Peak Hours Heatmap */}
        <div style={CHART_STYLE}>
          <h3 style={{ fontSize: "0.875rem", fontWeight: 700, marginBottom: "1rem" }}>Peak Violation Hours</h3>
          <PeakHoursChart data={peakHours} />
        </div>
      </div>
    </>
  );
}
