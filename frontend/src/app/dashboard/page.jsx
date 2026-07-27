"use client";
import { useEffect, useState } from "react";
import { useSession } from "next-auth/react";
import {
  ShieldAlert, Camera, ClipboardList, Activity
} from "lucide-react";
import {
  BarChart, Bar, PieChart, Pie, Cell,
  XAxis, YAxis, Tooltip, ResponsiveContainer, Legend
} from "recharts";
import StatCard from "@/components/ui/StatCard";
import { ViolationBadge } from "@/components/ui/Badge";
import { apiFetch, API_BASE } from "@/lib/api";
import Link from "next/link";

function timeAgo(ts) {
  const diff = (Date.now() - new Date(ts)) / 1000;
  if (diff < 60) return `${Math.round(diff)}s ago`;
  if (diff < 3600) return `${Math.round(diff / 60)}m ago`;
  return new Date(ts).toLocaleTimeString();
}

const PIE_COLORS = [
  "var(--color-danger)",
  "var(--color-warning)",
  "var(--color-info)",
  "var(--color-primary)",
];

const CustomTooltip = ({ active, payload, label }) => {
  if (!active || !payload?.length) return null;
  return (
    <div style={{
      background: "var(--color-bg-elevated)",
      border: "1px solid var(--color-border)",
      borderRadius: "var(--radius-md)",
      padding: "0.5rem 0.75rem",
      fontSize: "0.8rem",
    }}>
      <div style={{ color: "var(--color-text-secondary)", marginBottom: 2 }}>{label}</div>
      <div style={{ fontWeight: 700, color: "var(--color-text-primary)" }}>
        {payload[0].value} violations
      </div>
    </div>
  );
};

export default function OverviewPage() {
  const { data: session } = useSession();
  const [summary, setSummary] = useState(null);
  const [hourData, setHourData] = useState([]);
  const [typeData, setTypeData] = useState([]);
  const [recentAlerts, setRecentAlerts] = useState([]);
  const [loading, setLoading] = useState(true);

  async function loadData() {
    if (!session) return;
    try {
      const [sum, byType, violations] = await Promise.all([
        apiFetch("/api/analytics/summary", {}, session),
        apiFetch("/api/analytics/violations-by-type", {}, session),
        apiFetch("/api/violations?page=1&page_size=5", {}, session),
      ]);
      setSummary(sum);
      setTypeData(byType);
      setRecentAlerts(violations?.items || []);

      // Build 24h bar chart data from trend
      const trend = await apiFetch("/api/analytics/trend?days=1", {}, session);
      setHourData(trend || []);
    } catch (err) {
      console.error(err);
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => { loadData(); }, [session]);
  useEffect(() => {
    const interval = setInterval(loadData, 30000);
    return () => clearInterval(interval);
  }, [session]);

  return (
    <>
      {/* KPI Row */}
      <div className="kpi-grid">
        <StatCard
          title="Total Violations Today"
          value={summary?.total_violations_today ?? null}
          icon={ShieldAlert}
          variant="danger"
          danger={summary?.total_violations_today > 10}
        />
        <StatCard
          title="Active Cameras"
          value={summary ? `${summary.active_cameras} / ${summary.total_cameras}` : null}
          icon={Camera}
          variant="success"
        />
        <StatCard
          title="Unreviewed Incidents"
          value={summary?.unreviewed_incidents ?? null}
          icon={ClipboardList}
          variant="warning"
        />
        <StatCard
          title="System Health"
          value={summary
            ? summary.system_healthy
              ? "All Online"
              : `${summary.offline_cameras} Offline`
            : null}
          icon={Activity}
          variant="success"
          danger={!summary?.system_healthy}
        />
      </div>

      {/* Charts Row */}
      <div className="charts-row" style={{ marginBottom: "1.5rem" }}>
        {/* Bar Chart */}
        <div className="card violations-chart-card">
          <div style={{ marginBottom: "1rem" }}>
            <h3 style={{ fontSize: "0.9rem", fontWeight: 700 }}>Violations — Last 7 Days</h3>
          </div>
          <ResponsiveContainer width="100%" height={220}>
            <BarChart data={hourData} barCategoryGap="30%">
              <XAxis
                dataKey="date"
                tick={{ fontSize: 10, fill: "var(--color-text-muted)" }}
                axisLine={false} tickLine={false}
              />
              <YAxis
                tick={{ fontSize: 10, fill: "var(--color-text-muted)" }}
                axisLine={false} tickLine={false}
              />
              <Tooltip content={<CustomTooltip />} cursor={{ fill: "rgba(255,255,255,0.04)" }} />
              <Bar dataKey="count" fill="var(--color-primary)" radius={[4, 4, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>

        {/* Pie Chart */}
        <div className="card">
          <div style={{ marginBottom: "1rem" }}>
            <h3 style={{ fontSize: "0.9rem", fontWeight: 700 }}>Violations by Type</h3>
          </div>
          {typeData.length === 0 ? (
            <div className="empty-state" style={{ padding: "2rem 0" }}>No data</div>
          ) : (
            <ResponsiveContainer width="100%" height={220}>
              <PieChart>
                <Pie
                  data={typeData}
                  dataKey="count"
                  nameKey="type"
                  cx="50%"
                  cy="50%"
                  innerRadius={55}
                  outerRadius={85}
                  paddingAngle={3}
                >
                  {typeData.map((_, i) => (
                    <Cell key={i} fill={PIE_COLORS[i % PIE_COLORS.length]} />
                  ))}
                </Pie>
                <Tooltip
                  contentStyle={{
                    background: "var(--color-bg-elevated)",
                    border: "1px solid var(--color-border)",
                    borderRadius: 8,
                    fontSize: 12,
                  }}
                />
                <Legend
                  wrapperStyle={{ fontSize: "0.72rem", color: "var(--color-text-secondary)" }}
                />
              </PieChart>
            </ResponsiveContainer>
          )}
        </div>
      </div>

      {/* Recent Alerts Feed */}
      <div className="card">
        <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: "1rem" }}>
          <h3 style={{ fontSize: "0.9rem", fontWeight: 700 }}>Recent Violations</h3>
          <Link href="/dashboard/violations" style={{ fontSize: "0.78rem", color: "var(--color-primary)" }}>
            View all violations →
          </Link>
        </div>

        {recentAlerts.length === 0 ? (
          <div className="empty-state">
            <ShieldAlert size={36} className="empty-state-icon" />
            <span className="empty-state-title">No violations recorded yet</span>
          </div>
        ) : (
          <div style={{ display: "flex", flexDirection: "column", gap: "0.625rem" }}>
            {recentAlerts.map((alert) => (
              <div
                key={alert.id}
                style={{
                  display: "flex",
                  alignItems: "center",
                  gap: "0.875rem",
                  padding: "0.75rem",
                  background: "var(--color-bg-elevated)",
                  borderRadius: "var(--radius-md)",
                  border: "1px solid var(--color-border-subtle)",
                }}
              >
                <ViolationBadge type={alert.violation_type} />
                <div style={{ flex: 1, minWidth: 0 }}>
                  <div style={{ fontSize: "0.8rem", fontWeight: 600, color: "var(--color-text-primary)" }}>
                    {alert.camera_name || "Unknown camera"}
                  </div>
                  <div style={{ fontSize: "0.72rem", color: "var(--color-text-muted)" }}>
                    {timeAgo(alert.timestamp)}
                    {alert.person_name ? ` · ${alert.person_name}` : ""}
                  </div>
                </div>
                {alert.screenshot_path && (
                  <img
                    src={`${API_BASE}/${alert.screenshot_path}`}
                    alt="Violation"
                    style={{
                      width: 64, height: 44,
                      objectFit: "cover",
                      borderRadius: "var(--radius-sm)",
                      flexShrink: 0,
                    }}
                  />
                )}
              </div>
            ))}
          </div>
        )}
      </div>
    </>
  );
}
