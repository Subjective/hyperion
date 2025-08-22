import { clsx, type ClassValue } from "clsx";
import { twMerge } from "tailwind-merge";
import type { LineageNode, LineageEdge, Trial } from "@/api/client";

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}

export function calculateTrialPath(
  nodeId: string,
  nodes: LineageNode[] | Trial[],
  edges: LineageEdge[] | Array<{ source: string; target: string }>,
): string[] {
  const path: string[] = [nodeId];
  let current = nodes.find((n) => n.id === nodeId);

  while (current) {
    const parentEdge = edges.find((e) => e.target === current!.id);
    if (parentEdge) {
      path.push(parentEdge.source);
      current = nodes.find((n) => n.id === parentEdge.source);
    } else {
      break;
    }
  }

  return path;
}
