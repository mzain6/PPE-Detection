import { create } from "zustand";

export const useSiteStore = create((set) => ({
  activeSite: null, // { id, name } or null = all sites
  sites: [],

  setActiveSite: (site) => set({ activeSite: site }),
  setSites: (sites) => set({ sites }),
}));
