"use client";
import { useEffect, useRef, useState } from "react";
import { useSession } from "next-auth/react";
import { usePathname } from "next/navigation";
import Sidebar from "@/components/layout/Sidebar";
import AlertBell from "@/components/layout/AlertBell";
import ToastContainer from "@/components/ui/Toast";
import { useSiteStore } from "@/store/siteStore";
import { useAlertStore } from "@/store/alertStore";
import { apiFetch } from "@/lib/api";
import { Menu } from "lucide-react";

const PAGE_TITLES = {
  "/dashboard":            "Overview",
  "/dashboard/cameras":    "Live Cameras",
  "/dashboard/violations": "Violations",
  "/dashboard/personnel":  "Personnel",
  "/dashboard/analytics":  "Analytics",
  "/dashboard/settings":   "Settings",
  "/dashboard/users":      "User Management",
};

export default function DashboardLayout({ children }) {
  const { data: session } = useSession();
  const pathname = usePathname();
  const { activeSite, sites, setActiveSite, setSites } = useSiteStore();
  const { setAlerts, addToast } = useAlertStore();
  const lastPollRef = useRef(new Date().toISOString());
  const pageTitle = PAGE_TITLES[pathname] || "Dashboard";
  const [mobileOpen, setMobileOpen] = useState(false);

  // Load sites on mount
  useEffect(() => {
    if (!session) return;
    apiFetch("/api/sites", {}, session)
      .then((data) => setSites(data))
      .catch(() => {});
  }, [session]);

  // Global violation polling every 10s
  useEffect(() => {
    if (!session) return;
    const poll = async () => {
      try {
        const data = await apiFetch(
          `/api/violations?since=${lastPollRef.current}&limit=10`,
          {},
          session
        );
        if (data?.items?.length > 0) {
          setAlerts(data.items);
          addToast({
            type: "warning",
            message: `${data.items.length} new violation${data.items.length > 1 ? "s" : ""} detected`,
          });
        }
        lastPollRef.current = new Date().toISOString();
      } catch {}
    };

    const interval = setInterval(poll, 10000);
    return () => clearInterval(interval);
  }, [session]);

  return (
    <div className="dashboard-layout">
      <Sidebar
        activeSite={activeSite}
        sites={sites}
        onSiteChange={setActiveSite}
        mobileOpen={mobileOpen}
        onClose={() => setMobileOpen(false)}
      />

      {mobileOpen && (
        <div className="mobile-sidebar-overlay" onClick={() => setMobileOpen(false)} />
      )}

      <div className="main-content">
        {/* Topbar */}
        <header className="topbar">
          <div className="topbar-left">
            <button className="mobile-menu-toggle" onClick={() => setMobileOpen(true)} aria-label="Open menu">
              <Menu size={20} />
            </button>
            <h1 className="topbar-title">{pageTitle}</h1>
          </div>
          <div className="topbar-actions">
            <AlertBell />
          </div>
        </header>

        {/* Page content */}
        <main className="page-content">
          {children}
        </main>
      </div>

      <ToastContainer />
    </div>
  );
}
