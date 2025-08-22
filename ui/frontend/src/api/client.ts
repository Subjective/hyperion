import { QueryClient } from "@tanstack/react-query";

export const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      staleTime: 1000 * 60 * 2, // 2 minutes
      gcTime: 1000 * 60 * 5, // 5 minutes
      retry: 1,
      refetchOnWindowFocus: false,
    },
  },
});

export const API_BASE = (import.meta as any).env?.VITE_API_BASE || "http://localhost:8000";

export async function fetcher<T>(path: string): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`);
  if (!res.ok) {
    const error = new Error(`${res.status} ${res.statusText}`);
    (error as any).status = res.status;
    throw error;
  }
  return await res.json();
}

export type Experiment = {
  id: string;
  name: string;
  status: string;
  created_at: number;
  config?: Record<string, any>;
  tags?: Record<string, any>;
};

export type Trial = {
  id: string;
  experiment_id: string;
  status: string;
  score?: number | null;
  params: Record<string, any>;
  metrics_last?: Record<string, any> | null;
  depth: number;
  parent_trial_id?: string | null;
  branch_id?: string | null;
  mutation_op?: string | null;
  tags?: Record<string, any> | null;
};

export type Event = {
  id: string;
  type: string;
  ts: number;
  aggregate_id?: string | null;
  correlation_id?: string | null;
  causation_id?: string | null;
  data: Record<string, any>;
};

export type LineageNode = {
  id: string;
  depth: number;
  status: string;
  score?: number | null;
  params: Record<string, any>;
  tags?: Record<string, any>;
  branchId?: string | null;
  mutationOp?: string | null;
  rationale?: string | null;
  actorType?: string | null;
  actorId?: string | null;
  startedAt?: number | null;
};

export type LineageEdge = {
  id: string;
  source: string;
  target: string;
};

export type Lineage = {
  nodes: LineageNode[];
  edges: LineageEdge[];
};
