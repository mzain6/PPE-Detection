"use client";
import { useState, useEffect } from "react";
import { useSession } from "next-auth/react";
import { redirect } from "next/navigation";
import { UserPlus, Edit2 } from "lucide-react";
import Modal from "@/components/ui/Modal";
import { RoleBadge } from "@/components/ui/Badge";
import { apiFetch } from "@/lib/api";
import { useAlertStore } from "@/store/alertStore";

const ROLES = ["super_admin", "site_admin", "safety_officer", "viewer"];

export default function UsersPage() {
  const { data: session, status } = useSession();
  const { addToast } = useAlertStore();
  const isAdmin = ["super_admin", "site_admin"].includes(session?.user?.role);

  const [users, setUsers] = useState([]);
  const [loading, setLoading] = useState(true);
  const [showInvite, setShowInvite] = useState(false);
  const [editUser, setEditUser] = useState(null);
  const [form, setForm] = useState({ name: "", email: "", role: "viewer" });

  // Redirect non-admins
  useEffect(() => {
    if (status === "authenticated" && !isAdmin) {
      redirect("/dashboard");
    }
  }, [status, isAdmin]);

  async function loadUsers() {
    if (!session) return;
    try {
      const data = await apiFetch("/api/users", {}, session);
      setUsers(data || []);
    } catch {} finally { setLoading(false); }
  }

  useEffect(() => { loadUsers(); }, [session]);

  async function handleInvite(e) {
    e.preventDefault();
    try {
      await apiFetch("/api/users", { method: "POST", body: JSON.stringify(form) }, session);
      addToast({ type: "success", message: `Invite sent to ${form.email}` });
      setShowInvite(false);
      setForm({ name: "", email: "", role: "viewer" });
      loadUsers();
    } catch { addToast({ type: "error", message: "Failed to send invite" }); }
  }

  async function handleEditSave(e) {
    e.preventDefault();
    try {
      await apiFetch(`/api/users/${editUser.id}`, {
        method: "PUT",
        body: JSON.stringify({ role: editUser.role, is_active: editUser.is_active }),
      }, session);
      addToast({ type: "success", message: "User updated" });
      setEditUser(null);
      loadUsers();
    } catch { addToast({ type: "error", message: "Failed to update user" }); }
  }

  if (!isAdmin) return null;

  return (
    <>
      <div className="page-header">
        <div>
          <h2 className="page-title">User Management</h2>
          <p className="page-subtitle">Manage team members, roles, and site access</p>
        </div>
        <button className="btn btn-primary btn-sm" onClick={() => setShowInvite(true)}>
          <UserPlus size={14} /> Invite User
        </button>
      </div>

      <div className="table-wrap">
        <table className="table">
          <thead>
            <tr>
              <th>User</th><th>Email</th><th>Role</th>
              <th>Last Login</th><th>Status</th><th>Actions</th>
            </tr>
          </thead>
          <tbody>
            {loading ? (
              <tr><td colSpan={6}><div className="empty-state">Loading users…</div></td></tr>
            ) : users.map(user => (
              <tr key={user.id}>
                <td>
                  <div style={{ display: "flex", alignItems: "center", gap: "0.625rem" }}>
                    <div style={{
                      width: 32, height: 32, borderRadius: "50%",
                      background: "var(--color-primary-light)",
                      display: "flex", alignItems: "center", justifyContent: "center",
                      fontSize: "0.7rem", fontWeight: 700, color: "var(--color-primary)", flexShrink: 0,
                    }}>
                      {user.name.split(" ").map(n => n[0]).join("").slice(0, 2).toUpperCase()}
                    </div>
                    <span style={{ fontWeight: 600, fontSize: "0.875rem" }}>{user.name}</span>
                  </div>
                </td>
                <td style={{ color: "var(--color-text-secondary)", fontSize: "0.8rem" }}>{user.email}</td>
                <td><RoleBadge role={user.role} /></td>
                <td style={{ color: "var(--color-text-muted)", fontSize: "0.8rem" }}>
                  {user.last_login ? new Date(user.last_login).toLocaleDateString() : "Never"}
                </td>
                <td>
                  <span className={`badge badge-${user.is_active ? "success" : "danger"}`}>
                    {user.is_active ? "Active" : "Inactive"}
                  </span>
                </td>
                <td>
                  <button className="btn btn-ghost btn-sm"
                    onClick={() => setEditUser({ ...user })}>
                    <Edit2 size={13} />
                  </button>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {/* Invite Modal */}
      <Modal isOpen={showInvite} onClose={() => setShowInvite(false)} title="Invite User" size="sm">
        <form onSubmit={handleInvite}>
          <div className="modal-body" style={{ display: "flex", flexDirection: "column", gap: "1rem" }}>
            <div className="form-group">
              <label className="input-label">Full Name</label>
              <input className="input" required value={form.name}
                onChange={e => setForm(f => ({ ...f, name: e.target.value }))} placeholder="Jane Smith" />
            </div>
            <div className="form-group">
              <label className="input-label">Email</label>
              <input type="email" className="input" required value={form.email}
                onChange={e => setForm(f => ({ ...f, email: e.target.value }))} placeholder="jane@company.com" />
            </div>
            <div className="form-group">
              <label className="input-label">Role</label>
              <select className="input" value={form.role}
                onChange={e => setForm(f => ({ ...f, role: e.target.value }))}>
                {ROLES.map(r => <option key={r} value={r}>{r.replace(/_/g, " ")}</option>)}
              </select>
            </div>
          </div>
          <div className="modal-footer">
            <button type="button" className="btn btn-ghost btn-sm" onClick={() => setShowInvite(false)}>Cancel</button>
            <button type="submit" className="btn btn-primary btn-sm">Send Invite</button>
          </div>
        </form>
      </Modal>

      {/* Edit User Modal */}
      <Modal isOpen={!!editUser} onClose={() => setEditUser(null)} title="Edit User" size="sm">
        {editUser && (
          <form onSubmit={handleEditSave}>
            <div className="modal-body" style={{ display: "flex", flexDirection: "column", gap: "1rem" }}>
              <div className="form-group">
                <label className="input-label">Role</label>
                <select className="input" value={editUser.role}
                  onChange={e => setEditUser(u => ({ ...u, role: e.target.value }))}>
                  {ROLES.map(r => <option key={r} value={r}>{r.replace(/_/g, " ")}</option>)}
                </select>
              </div>
              <label className="remember-label">
                <input type="checkbox" checked={editUser.is_active}
                  onChange={e => setEditUser(u => ({ ...u, is_active: e.target.checked }))} />
                Account is active
              </label>
            </div>
            <div className="modal-footer">
              <button type="button" className="btn btn-ghost btn-sm" onClick={() => setEditUser(null)}>Cancel</button>
              <button type="submit" className="btn btn-primary btn-sm">Save Changes</button>
            </div>
          </form>
        )}
      </Modal>
    </>
  );
}
