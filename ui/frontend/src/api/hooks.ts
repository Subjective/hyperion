import { useQuery, useQueryClient } from "@tanstack/react-query";
import { useEffect, useRef } from "react";
import { fetcher, API_BASE, type Experiment, type Trial, type Event, type Lineage } from "./client";

export function useExperiments() {
  return useQuery({
    queryKey: ["experiments"],
    queryFn: () => fetcher<Experiment[]>("/api/experiments"),
    refetchOnMount: true,
    staleTime: 30000, // 30 seconds
  });
}

export function useExperiment(id: string) {
  return useQuery({
    queryKey: ["experiment", id],
    queryFn: () => fetcher<Experiment>(`/api/experiments/${id}`),
    enabled: !!id,
  });
}

export function useTrials(experimentId: string) {
  return useQuery({
    queryKey: ["trials", experimentId],
    queryFn: () => fetcher<Trial[]>(`/api/experiments/${experimentId}/trials?limit=500`),
    enabled: !!experimentId,
    // No polling - WebSocket will invalidate when trials change
  });
}

export function useEvents(experimentId: string, limit?: number) {
  return useQuery({
    queryKey: ["events", experimentId, limit ?? "all"],
    // If limit is provided and > 0, pass it; otherwise request full history
    queryFn: () =>
      fetcher<Event[]>(
        `/api/events?experiment_id=${experimentId}${limit && limit > 0 ? `&limit=${limit}` : ""}`,
      ),
    enabled: !!experimentId,
  });
}

export function useLineage(experimentId: string) {
  return useQuery({
    queryKey: ["lineage", experimentId],
    queryFn: () => fetcher<Lineage>(`/api/experiments/${experimentId}/lineage`),
    enabled: !!experimentId,
  });
}

export function useEventStream(experimentId: string | undefined, onEvent: (event: Event) => void) {
  const wsRef = useRef<WebSocket | null>(null);
  const onEventRef = useRef(onEvent);
  const queryClient = useQueryClient();
  const invalidateTimerRef = useRef<number | null>(null);
  const pendingInvalidateRef = useRef<{ trials: boolean; lineage: boolean }>({
    trials: false,
    lineage: false,
  });

  // Update the ref when onEvent changes
  useEffect(() => {
    onEventRef.current = onEvent;
  }, [onEvent]);

  useEffect(() => {
    // Don't create new connection if one is already open or connecting
    if (
      wsRef.current &&
      (wsRef.current.readyState === WebSocket.CONNECTING ||
        wsRef.current.readyState === WebSocket.OPEN)
    ) {
      return;
    }

    const qs = new URLSearchParams();
    if (experimentId) {
      qs.set("experiment_id", experimentId);
    }
    const ws = new WebSocket(`${API_BASE.replace("http", "ws")}/ws/events?${qs.toString()}`);

    ws.onmessage = (e) => {
      const event = JSON.parse(e.data) as Event;
      onEventRef.current(event); // Use ref instead of direct callback

      // Only invalidate trial/lineage queries when we have a specific experiment
      if (experimentId) {
        if (
          ["TRIAL_STARTED", "TRIAL_COMPLETED", "TRIAL_FAILED", "TRIAL_KILLED"].includes(event.type)
        ) {
          pendingInvalidateRef.current.trials = true;
          pendingInvalidateRef.current.lineage = true;
          if (invalidateTimerRef.current == null) {
            // Debounce invalidations to avoid request floods
            invalidateTimerRef.current = window.setTimeout(() => {
              const pending = pendingInvalidateRef.current;
              invalidateTimerRef.current = null;
              pendingInvalidateRef.current = { trials: false, lineage: false };
              if (pending.trials) {
                queryClient.invalidateQueries({ queryKey: ["trials", experimentId] });
              }
              if (pending.lineage) {
                queryClient.invalidateQueries({ queryKey: ["lineage", experimentId] });
              }
            }, 400);
          }
        }
      }

      // Invalidate experiments list when any experiment status changes
      if (["EXPERIMENT_STARTED"].includes(event.type)) {
        queryClient.invalidateQueries({ queryKey: ["experiments"] });
      }

      // Keep experiment status + trials/lineage in sync; close WS when finished
      if (
        ["EXPERIMENT_COMPLETED", "EXPERIMENT_FAILED", "EXPERIMENT_STOPPED"].includes(event.type)
      ) {
        if (experimentId) {
          queryClient.invalidateQueries({ queryKey: ["experiment", experimentId] });
          queryClient.invalidateQueries({ queryKey: ["trials", experimentId] });
          queryClient.invalidateQueries({ queryKey: ["lineage", experimentId] });
        }
        queryClient.invalidateQueries({ queryKey: ["experiments"] });
        // Only close WS if watching a specific experiment that finished
        if (experimentId && wsRef.current) wsRef.current.close();
      }
    };

    ws.onerror = () => ws.close();
    wsRef.current = ws;

    return () => {
      if (wsRef.current) {
        wsRef.current.close();
        wsRef.current = null;
      }
      if (invalidateTimerRef.current != null) {
        window.clearTimeout(invalidateTimerRef.current);
        invalidateTimerRef.current = null;
      }
    };
  }, [experimentId, queryClient]);

  return wsRef.current;
}
