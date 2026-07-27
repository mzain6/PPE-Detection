import { create } from "zustand";

let toastId = 0;

export const useAlertStore = create((set, get) => ({
  toasts: [],
  unreadCount: 0,
  recentAlerts: [],

  addToast: ({ type = "info", message, duration = 5000 }) => {
    const id = ++toastId;
    set((state) => ({
      toasts: [...state.toasts.slice(-2), { id, type, message }],
    }));
    if (duration > 0) {
      setTimeout(() => get().dismissToast(id), duration);
    }
  },

  dismissToast: (id) => {
    set((state) => ({
      toasts: state.toasts.filter((t) => t.id !== id),
    }));
  },

  setAlerts: (alerts) => {
    set((state) => ({
      recentAlerts: alerts.slice(0, 5),
      unreadCount: state.unreadCount + alerts.length,
    }));
  },

  clearUnread: () => set({ unreadCount: 0 }),
}));
