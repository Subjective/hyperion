export const API_BASE = (import.meta as any).env?.VITE_API_BASE || "http://localhost:8000";

async function get<T>(path: string): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`);
  if (!res.ok) throw new Error(`${res.status} ${res.statusText}`);
  return await res.json();
}

export const api = {
  experiments: () =>
    get<Array<{ id: string; name: string; status: string; created_at: string }>>(
      "/api/experiments",
    ),
  experiment: (id: string) =>
    get<{ id: string; name: string; status: string; created_at: string }>(`/api/experiments/${id}`),
  trials: (id: string) => get<Array<any>>(`/api/experiments/${id}/trials?limit=200`),
  events: (id: string, limit = 50) =>
    get<Array<any>>(`/api/events?experiment_id=${id}&limit=${limit}`),
  lineage: (id: string) => get<{ nodes: any[]; edges: any[] }>(`/api/experiments/${id}/lineage`),
};

export function openEventsSocket(params: { experimentId?: string; sinceTs?: string }) {
  const q: string[] = [];
  if (params.experimentId) q.push(`experiment_id=${encodeURIComponent(params.experimentId)}`);
  if (params.sinceTs) q.push(`since_ts=${encodeURIComponent(params.sinceTs)}`);
  const url = `${API_BASE.replace("http", "ws")}/ws/events${q.length ? `?${q.join("&")}` : ""}`;
  return new WebSocket(url);
}
