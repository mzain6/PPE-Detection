export default function StatCard({ title, value, icon: Icon, delta, variant = "primary", danger = false }) {
  const iconColors = {
    primary: { bg: "rgba(255, 255, 255, 0.04)",  color: "#9ca3af"  },
    success: { bg: "rgba(255, 255, 255, 0.04)",  color: "#9ca3af"  },
    warning: { bg: "rgba(255, 255, 255, 0.04)",  color: "#9ca3af"  },
    danger:  { bg: "rgba(255, 255, 255, 0.04)",  color: "#9ca3af"  },
    info:    { bg: "rgba(255, 255, 255, 0.04)",  color: "#9ca3af"  },
  };

  const activeVariant = danger ? "danger" : variant;
  const { bg, color } = iconColors[activeVariant] || iconColors.primary;

  const isAllOnline = value === "All Online";

  return (
    <div className="stat-card">
      {Icon && (
        <div className="stat-card-icon-wrap" style={{ background: bg }}>
          <Icon size={22} style={{ color }} strokeWidth={1.5} />
        </div>
      )}
      <div className="stat-card-body">
        <div
          className={`stat-card-value${danger ? " danger" : ""}`}
          style={danger ? { color: "var(--color-danger)" } : undefined}
        >
          {isAllOnline && <span className="pulsing-dot"></span>}
          {value ?? (
            <span className="skeleton" style={{ display: "inline-block", width: 60, height: 32 }} />
          )}
        </div>
        <div className="stat-card-title">{title}</div>
        {delta && (
          <div
            className="stat-card-delta"
            style={{ color: delta.startsWith("+") ? "var(--color-success)" : "var(--color-danger)" }}
          >
            {delta}
          </div>
        )}
      </div>
    </div>
  );
}
