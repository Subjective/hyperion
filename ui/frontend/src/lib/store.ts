import { create } from "zustand";

interface AppState {
  selectedTrialId: string | null;
  highlightedPath: string[];
  selectionSource: "hierarchy" | "graph" | null;
  eventFilter: string;
  sidebarOpen: boolean;

  setSelectedTrial: (id: string | null, source?: "hierarchy" | "graph") => void;
  setHighlightedPath: (path: string[]) => void;
  setEventFilter: (filter: string) => void;
  toggleSidebar: () => void;
}

export const useStore = create<AppState>((set) => ({
  selectedTrialId: null,
  highlightedPath: [],
  selectionSource: null,
  eventFilter: "",
  sidebarOpen: true,

  setSelectedTrial: (id, source = "hierarchy") =>
    set({
      selectedTrialId: id,
      selectionSource: source,
    }),
  setHighlightedPath: (path) => set({ highlightedPath: path }),
  setEventFilter: (filter) => set({ eventFilter: filter }),
  toggleSidebar: () => set((state) => ({ sidebarOpen: !state.sidebarOpen })),
}));
