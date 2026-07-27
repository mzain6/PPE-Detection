"use client";
import { useEffect } from "react";
import { AlertTriangle, RotateCcw, Home } from "lucide-react";

export default function ErrorBoundary({ error, reset }) {
  useEffect(() => {
    // Log the error to an error reporting service
    console.error("Dashboard error captured:", error);
  }, [error]);

  return (
    <div style={{
      display: "flex",
      flexDirection: "column",
      alignItems: "center",
      justifyContent: "center",
      minHeight: "70vh",
      padding: "2rem",
      textAlign: "center"
    }}>
      <div style={{
        background: "var(--color-danger-light)",
        border: "1px solid var(--color-danger)",
        borderRadius: "var(--radius-full)",
        width: "64px",
        height: "64px",
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
        marginBottom: "1.5rem",
        boxShadow: "0 0 15px rgba(239, 68, 68, 0.2)"
      }}>
        <AlertTriangle size={32} color="var(--color-danger)" />
      </div>

      <h2 style={{
        fontSize: "1.5rem",
        fontWeight: "600",
        marginBottom: "0.75rem",
        color: "var(--color-text-primary)"
      }}>
        Something went wrong
      </h2>

      <p style={{
        color: "var(--color-text-secondary)",
        maxWidth: "400px",
        marginBottom: "2rem",
        fontSize: "0.95rem",
        lineHeight: "1.5"
      }}>
        {error?.message || "An unexpected error occurred while rendering the dashboard. Please try reloading or check back in a moment."}
      </p>

      <div style={{
        display: "flex",
        gap: "1rem",
        alignItems: "center"
      }}>
        <button
          onClick={() => reset()}
          className="btn btn-primary"
          style={{
            display: "flex",
            alignItems: "center",
            gap: "0.5rem",
            padding: "0.625rem 1.25rem",
            fontSize: "0.9rem"
          }}
        >
          <RotateCcw size={16} />
          Try Again
        </button>

        <a
          href="/dashboard"
          className="btn"
          style={{
            display: "flex",
            alignItems: "center",
            gap: "0.5rem",
            padding: "0.625rem 1.25rem",
            fontSize: "0.9rem",
            border: "1px solid var(--color-border)",
            background: "var(--color-bg-surface)",
            color: "var(--color-text-primary)",
            borderRadius: "var(--radius-md)",
            transition: "all var(--transition-fast)"
          }}
          onMouseEnter={(e) => {
            e.currentTarget.style.background = "var(--color-bg-hover)";
            e.currentTarget.style.borderColor = "var(--color-text-secondary)";
          }}
          onMouseLeave={(e) => {
            e.currentTarget.style.background = "var(--color-bg-surface)";
            e.currentTarget.style.borderColor = "var(--color-border)";
          }}
        >
          <Home size={16} />
          Go to Overview
        </a>
      </div>
    </div>
  );
}
