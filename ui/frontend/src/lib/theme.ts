export const theme = {
  colors: {
    status: {
      running: "#3b82f6",
      completed: "#22c55e",
      failed: "#ef4444",
      pending: "#6b7280",
      killed: "#f59e0b",
    },
    depth: [
      "#8b5cf6", // violet
      "#6366f1", // indigo
      "#3b82f6", // blue
      "#06b6d4", // cyan
      "#10b981", // emerald
      "#84cc16", // lime
      "#eab308", // yellow
      "#f97316", // orange
      "#ef4444", // red
      "#ec4899", // pink
    ],
  },
  animation: {
    spring: {
      type: "spring",
      stiffness: 260,
      damping: 20,
    },
    smooth: {
      type: "tween",
      duration: 0.3,
    },
  },
};
