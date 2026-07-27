"use client";
import Link from "next/link";
import { usePathname } from "next/navigation";
import { useSession, signOut } from "next-auth/react";
import {
  LayoutDashboard, Video, ShieldAlert, Users,
  BarChart2, Settings, UserCog, LogOut, X
} from "lucide-react";
import Logo from "@/components/ui/Logo";

const NAV_ITEMS = [
  { href: "/dashboard",            label: "Overview",        icon: LayoutDashboard },
  { href: "/dashboard/cameras",    label: "Live Cameras",    icon: Video           },
  { href: "/dashboard/violations", label: "Violations",      icon: ShieldAlert     },
  { href: "/dashboard/personnel",  label: "Personnel",       icon: Users           },
  { href: "/dashboard/analytics",  label: "Analytics",       icon: BarChart2       },
  { href: "/dashboard/settings",   label: "Settings",        icon: Settings        },
];

const ADMIN_NAV = [
  { href: "/dashboard/users", label: "User Management", icon: UserCog },
];

export default function Sidebar({ activeSite, sites, onSiteChange, mobileOpen, onClose }) {
  const pathname = usePathname();
  const { data: session } = useSession();
  const isAdmin = ["super_admin", "site_admin"].includes(session?.user?.role);
  const initials = session?.user?.name
    ?.split(" ")
    .map((n) => n[0])
    .join("")
    .slice(0, 2)
    .toUpperCase() || "?";

  const allNav = isAdmin ? [...NAV_ITEMS, ...ADMIN_NAV] : NAV_ITEMS;

  return (
    <aside className={`sidebar${mobileOpen ? " mobile-open" : ""}`}>
      {/* Mobile Close Button */}
      <button className="sidebar-close-btn" onClick={onClose} aria-label="Close sidebar">
        <X size={18} />
      </button>
      {/* Logo */}
      <div className="sidebar-logo">
        <div className="sidebar-logo-icon">
          <Logo size={18} strokeWidth={2} />
        </div>
        <div>
          <div className="sidebar-logo-text">SafeSite AI</div>
          <div className="sidebar-logo-sub">Safety Monitor</div>
        </div>
      </div>

      {/* Site Selector */}
      {(isAdmin || (sites && sites.length > 1)) && (
        <select
          className="site-selector"
          value={activeSite?.id || ""}
          onChange={(e) => {
            const site = sites.find((s) => s.id === e.target.value);
            onSiteChange?.(site || null);
          }}
        >
          <option value="">All Sites</option>
          {sites?.map((s) => (
            <option key={s.id} value={s.id}>{s.name}</option>
          ))}
        </select>
      )}

      {/* Navigation */}
      <nav className="sidebar-nav">
        <span className="nav-section-label">Navigation</span>
        {allNav.map(({ href, label, icon: Icon }) => {
          const isActive = href === "/dashboard"
            ? pathname === "/dashboard"
            : pathname.startsWith(href);
          return (
            <Link
              key={href}
              href={href}
              className={`nav-item${isActive ? " active" : ""}`}
              onClick={() => onClose?.()}
            >
              <Icon size={17} className="nav-item-icon" />
              {label}
            </Link>
          );
        })}
      </nav>

      {/* User Profile */}
      <div className="sidebar-footer">
        <div className="user-profile">
          <div className="user-avatar">{initials}</div>
          <div className="user-info">
            <div className="user-name">{session?.user?.name || "User"}</div>
            <div className="user-role">{session?.user?.role?.replace("_", " ")}</div>
          </div>
        </div>
        <button
          className="signout-btn"
          onClick={() => signOut({ callbackUrl: "/login" })}
        >
          <LogOut size={14} />
          Sign Out
        </button>
      </div>
    </aside>
  );
}
