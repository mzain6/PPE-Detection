"use client";
import { CheckCircle, AlertTriangle, XCircle, Info, X } from "lucide-react";
import { useAlertStore } from "@/store/alertStore";

const ICONS = {
  success: CheckCircle,
  warning: AlertTriangle,
  error:   XCircle,
  info:    Info,
};

export default function ToastContainer() {
  const { toasts, dismissToast } = useAlertStore();

  return (
    <div className="toast-container">
      {toasts.map((toast) => {
        const Icon = ICONS[toast.type] || Info;
        return (
          <div key={toast.id} className={`toast toast-${toast.type}`}>
            <Icon size={17} className="toast-icon" />
            <span className="toast-msg">{toast.message}</span>
            <button
              className="toast-close"
              onClick={() => dismissToast(toast.id)}
              aria-label="Dismiss"
            >
              <X size={14} />
            </button>
          </div>
        );
      })}
    </div>
  );
}
