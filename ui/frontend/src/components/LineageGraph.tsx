import { useState, useMemo, useCallback, useEffect, useRef } from "react";
import {
  ReactFlow,
  ReactFlowProvider,
  MiniMap,
  Background,
  useNodesState,
  useEdgesState,
  useReactFlow,
  Handle,
  Position,
  BackgroundVariant,
  getBezierPath,
} from "@xyflow/react";
import type { Node, Edge, NodeProps, EdgeProps } from "@xyflow/react";
import "@xyflow/react/dist/style.css";
import "./LineageGraph.css";
import { motion } from "framer-motion";
import dagre from "dagre";
import type { LineageNode, LineageEdge } from "@/api/client";
import { theme } from "@/lib/theme";
import { useStore } from "@/lib/store";
import { Card } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import {
  GitBranch,
  Target,
  Brain,
  RotateCw,
  Clock,
  Maximize,
  ZoomIn,
  ZoomOut,
  AlignLeft,
  ChartScatter,
  WandSparkles,
} from "lucide-react";
import { cn, calculateTrialPath } from "@/lib/utils";

// Layout and card sizing constants (single source of truth)
const CARD = {
  width: 256, // Tailwind w-64 (16rem)
  baseHeight: 160, // Tailwind min-h-40 (10rem)
};

const LAYOUT = {
  nodesep: 50, // vertical gap between nodes in same column
  ranksep: 350, // horizontal gap between depth columns for columned layouts
  xsep: 350, // horizontal gap between items for staggered layout
  depthXJitter: 12, // small per-depth x offset in staggered to avoid any vertical align illusions
};

interface LineageGraphProps {
  nodes: LineageNode[];
  edges: LineageEdge[];
  className?: string;
}

// Custom edge with animation
function AnimatedEdge({
  id,
  sourceX,
  sourceY,
  targetX,
  targetY,
  sourcePosition,
  targetPosition,
  style = {},
  markerEnd,
  data,
}: EdgeProps) {
  const [path] = getBezierPath({
    sourceX,
    sourceY,
    sourcePosition,
    targetX,
    targetY,
    targetPosition,
  });

  // Get animation tracking ref and source depth from data
  const animatedEdgesRef = (data as any)?.animatedEdgesRef;
  const sourceDepth = (data as any)?.sourceDepth || 0;
  const animationGeneration = (data as any)?.animationGeneration || 0;

  // Check if this edge has already animated
  const hasAnimated = animatedEdgesRef?.current?.has(id) || false;

  // Mark as animated after rendering
  useEffect(() => {
    if (animatedEdgesRef?.current && !hasAnimated) {
      animatedEdgesRef.current.set(id, true);
    }
  }, [id, animatedEdgesRef, hasAnimated]);

  return (
    <>
      <motion.path
        key={`${id}-${animationGeneration}`}
        id={id}
        style={style}
        className="react-flow__edge-path"
        d={path}
        markerEnd={markerEnd}
        initial={!hasAnimated ? { pathLength: 0 } : { pathLength: 1 }}
        animate={{ pathLength: 1 }}
        transition={!hasAnimated ? { duration: 0.5, delay: sourceDepth * 0.15 } : { duration: 0 }}
      />
    </>
  );
}

// Custom node component with flippable card
function FlippableCardNode({ data }: NodeProps) {
  const [isFlipped, setIsFlipped] = useState(false);
  const { selectedTrialId, setSelectedTrial, highlightedPath } = useStore();
  const nodeData = data as LineageNode;
  const depthColor = theme.colors.depth[nodeData.depth % theme.colors.depth.length];
  const statusColor = theme.colors.status[nodeData.status as keyof typeof theme.colors.status];

  const isSelected = selectedTrialId === nodeData.id;
  const isHighlighted = highlightedPath.includes(nodeData.id);

  const handleSelect = useCallback(
    (e: React.MouseEvent) => {
      e.stopPropagation();
      setSelectedTrial(nodeData.id);
    },
    [nodeData.id, setSelectedTrial],
  );

  const handleDoubleClick = useCallback(
    (e: React.MouseEvent) => {
      e.stopPropagation();
      setIsFlipped(!isFlipped);
    },
    [isFlipped],
  );

  return (
    <>
      <Handle type="target" position={Position.Left} style={{ visibility: "hidden" }} />
      <motion.div
        key={`${nodeData.id}-${(data as any)?.animationGeneration ?? 0}`}
        initial={{ opacity: 0, scale: 0.8, x: -20 }}
        animate={{ opacity: 1, scale: 1, x: 0 }}
        transition={{
          type: "spring",
          delay: nodeData.depth * 0.1,
          duration: 0.5,
          stiffness: 100,
          damping: 15,
        }}
      >
        <motion.div
          className="preserve-3d relative w-64 cursor-pointer"
          onClick={handleSelect}
          onDoubleClick={handleDoubleClick}
          whileHover={{ scale: 1.04 }}
          whileTap={{ scale: 0.96 }}
          style={{ transformStyle: "preserve-3d" }}
          animate={{ rotateY: isFlipped ? 180 : 0 }}
          transition={{ duration: 0.6, type: "spring" }}
        >
          {/* Front of card */}
          <div
            className={cn(
              "backface-hidden relative min-h-40 rounded-xl border-2 bg-white/95 p-4 shadow-lg backdrop-blur-sm dark:bg-gray-900/95",
              isSelected && "border-violet-500 shadow-xl shadow-violet-500/20",
              !isSelected && isHighlighted && "border-indigo-400",
              !isSelected && !isHighlighted && "border-gray-200 dark:border-gray-700",
            )}
          >
            <div className="flex flex-col">
              <div className="mb-2 flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <div className="h-2 w-2 rounded-full" style={{ backgroundColor: depthColor }} />
                  <span className="font-mono text-xs text-gray-500">depth {nodeData.depth}</span>
                </div>
                <Badge className="text-xs text-white" style={{ backgroundColor: statusColor }}>
                  {nodeData.status}
                </Badge>
              </div>

              <div className="mb-2">
                <p className="font-mono text-xs text-gray-600 dark:text-gray-400">
                  {nodeData.id.slice(0, 16)}...
                </p>
              </div>

              <div className="mb-3 flex h-8 items-center gap-2">
                {nodeData.score !== null && nodeData.score !== undefined ? (
                  <>
                    <Target className="h-4 w-4 text-violet-500" />
                    <span className="text-2xl font-bold text-violet-600">
                      {nodeData.score.toFixed(4)}
                    </span>
                  </>
                ) : (
                  <div className="h-4 w-16 rounded bg-gray-200 dark:bg-gray-800 animate-pulse" />
                )}
              </div>

              {nodeData.rationale && (
                <div className="mt-2">
                  <p className="text-xs italic text-gray-600 dark:text-gray-400">
                    "{nodeData.rationale}"
                  </p>
                </div>
              )}

              <div className="absolute bottom-2 right-2">
                <div className="flex items-center gap-1 rounded-md bg-gray-100/80 px-2 py-1 text-xs text-gray-500 dark:bg-gray-800/80 dark:text-gray-400">
                  <RotateCw className="h-3 w-3" />
                </div>
              </div>
            </div>

            {nodeData.status === "RUNNING" && (
              <div className="absolute -right-2 -top-2">
                <span className="relative flex h-4 w-4">
                  <span className="absolute inline-flex h-full w-full animate-ping rounded-full bg-blue-400 opacity-75" />
                  <span className="relative inline-flex h-4 w-4 rounded-full bg-blue-500" />
                </span>
              </div>
            )}
          </div>

          {/* Back of card */}
          <div
            className={cn(
              "backface-hidden absolute inset-0 top-0 rounded-xl border-2 bg-gradient-to-br from-violet-50 to-indigo-50 p-4 shadow-lg dark:from-violet-900/20 dark:to-indigo-900/20",
              isSelected && "border-violet-500",
              !isSelected && "border-gray-200 dark:border-gray-700",
            )}
            style={{ transform: "rotateY(180deg)" }}
          >
            <div className="flex flex-col min-h-32">
              <h4 className="mb-2 font-semibold text-sm">Parameters</h4>
              <div className="flex-1 overflow-y-auto">
                <div className="space-y-1">
                  {Object.entries(nodeData.params).map(([key, value]) => (
                    <div key={key} className="flex justify-between text-xs">
                      <span className="text-gray-600 dark:text-gray-400">{key}:</span>
                      <span className="font-mono font-medium">
                        {typeof value === "number" ? value.toFixed(4) : String(value)}
                      </span>
                    </div>
                  ))}
                </div>
              </div>

              {nodeData.actorType && (
                <div className="mt-2 border-t pt-2">
                  <div className="flex items-center gap-2 text-xs">
                    <Brain className="h-3 w-3 text-purple-500 flex-shrink-0" />
                    <span className="text-gray-600">
                      {nodeData.actorType} • {nodeData.actorId}
                    </span>
                  </div>
                </div>
              )}
            </div>
          </div>
        </motion.div>
      </motion.div>
      <Handle type="source" position={Position.Right} style={{ visibility: "hidden" }} />
    </>
  );
}

// Calculate dynamic node dimensions based on content
function calculateNodeDimensions(node: LineageNode) {
  const baseHeight = CARD.baseHeight;
  const width = CARD.width;

  let additionalHeight = 0;

  // Add height for rationale text (estimate ~20px per line, ~40 chars per line)
  if (node.rationale) {
    const estimatedLines = Math.ceil(node.rationale.length / 40);
    additionalHeight += estimatedLines * 20;
  }

  // Add height for parameters (24px per param, max 100px)
  const paramsCount = Object.keys(node.params || {}).length;
  additionalHeight += Math.min(paramsCount * 24, 100);

  // Add some padding
  const totalHeight = baseHeight + additionalHeight + 20;

  return { width, height: totalHeight };
}

// Build a stable per-depth comparator
function makeComparator(sortMode: "score" | "time") {
  return (a: LineageNode, b: LineageNode) => {
    if (sortMode === "score") {
      const as = a.score ?? -Infinity;
      const bs = b.score ?? -Infinity;
      if (bs !== as) return bs - as; // desc score
      const at = a.startedAt ?? Number.POSITIVE_INFINITY;
      const bt = b.startedAt ?? Number.POSITIVE_INFINITY;
      if (at !== bt) return at - bt; // asc time
      return a.id.localeCompare(b.id);
    } else {
      const at = a.startedAt ?? Number.POSITIVE_INFINITY;
      const bt = b.startedAt ?? Number.POSITIVE_INFINITY;
      if (at !== bt) return at - bt; // asc time
      // stable tie-breaker only to avoid reorder on score/status updates
      return a.id.localeCompare(b.id);
    }
  };
}

// Group nodes by depth and sort inside each depth
function groupNodesByDepth(
  nodes: LineageNode[],
  compare: (a: LineageNode, b: LineageNode) => number,
) {
  const byDepth = new Map<number, LineageNode[]>();
  for (const n of nodes) {
    const d = n.depth || 0;
    const arr = byDepth.get(d) || [];
    arr.push(n);
    byDepth.set(d, arr);
  }
  for (const arr of byDepth.values()) arr.sort(compare);
  const depths = Array.from(byDepth.keys()).sort((a, b) => a - b);
  const flattened: LineageNode[] = [];
  for (const d of depths) flattened.push(...(byDepth.get(d) || []));
  return { nodesByDepth: byDepth, depths, flattened };
}

// Consistent edge builder
function buildEdges(edges: LineageEdge[], nodes: LineageNode[]): Edge[] {
  return edges.map((edge) => {
    const sourceNode = nodes.find((n) => n.id === edge.source);
    const sourceDepth = sourceNode?.depth || 0;
    return {
      id: edge.id,
      source: edge.source,
      target: edge.target,
      type: "animatedEdge",
      data: { sourceDepth },
      animated: false,
      style: { strokeWidth: 2, stroke: "#e5e7eb" },
    };
  });
}

// Dagre layout (balanced look) with order-preserving vertical remap
function layoutDagre(
  nodes: LineageNode[],
  edges: LineageEdge[],
  nodesByDepth: Map<number, LineageNode[]>,
  depths: number[],
  flattened: LineageNode[],
): Node<LineageNode>[] {
  const g = new dagre.graphlib.Graph();
  g.setGraph({
    rankdir: "LR",
    nodesep: LAYOUT.nodesep,
    ranksep: LAYOUT.ranksep,
    ranker: "network-simplex",
  });
  g.setDefaultEdgeLabel(() => ({}));
  for (const n of flattened) g.setNode(n.id, calculateNodeDimensions(n));
  for (const e of edges) g.setEdge(e.source, e.target);
  dagre.layout(g);

  const infoById = new Map<string, { x: number; y: number; width: number; height: number }>();
  for (const n of nodes) {
    const dn = g.node(n.id);
    infoById.set(n.id, { x: dn.x, y: dn.y, width: dn.width, height: dn.height });
  }
  const newCenterYById = new Map<string, number>();
  for (const d of depths) {
    const atDepth = nodesByDepth.get(d)!; // already in comparator order
    const yCenters = atDepth.map((n) => infoById.get(n.id)!.y).sort((a, b) => a - b);
    atDepth.forEach((n, i) => newCenterYById.set(n.id, yCenters[i]));
  }
  return nodes.map((n) => {
    const info = infoById.get(n.id)!;
    const cy = newCenterYById.get(n.id) ?? info.y;
    return {
      id: n.id,
      type: "flippableCard",
      data: n,
      position: { x: info.x - info.width / 2, y: cy - info.height / 2 },
    };
  });
}

// Stacked top-left layout (strict order)
function layoutStacked(
  nodesByDepth: Map<number, LineageNode[]>,
  depths: number[],
): Node<LineageNode>[] {
  const dims = new Map<string, { width: number; height: number }>();
  for (const arr of nodesByDepth.values())
    for (const n of arr) dims.set(n.id, calculateNodeDimensions(n));
  const currentYByDepth = new Map<number, number>();
  const baseTopPadding = 0;
  const out: Node<LineageNode>[] = [];
  for (const d of depths) {
    currentYByDepth.set(d, baseTopPadding);
    const x = d * (CARD.width + LAYOUT.ranksep);
    const arr = nodesByDepth.get(d)!;
    for (const n of arr) {
      const { height } = dims.get(n.id)!;
      const y = currentYByDepth.get(d)!;
      out.push({ id: n.id, type: "flippableCard", data: n, position: { x, y } });
      currentYByDepth.set(d, y + height + LAYOUT.nodesep);
    }
  }
  return out;
}

// Staggered (timeline) layout: global left->right order by sort; rows by depth
function layoutStaggered(
  nodes: LineageNode[],
  edges: LineageEdge[],
  compare: (a: LineageNode, b: LineageNode) => number,
  sortMode?: "score" | "time",
): Node<LineageNode>[] {
  const dims = new Map<string, { width: number; height: number }>();
  nodes.forEach((n) => dims.set(n.id, calculateNodeDimensions(n)));

  // Global ordering across all nodes (ignores depth), stable
  // For 'time': asc time (via compare). For 'score': we want higher score further right, so ASC by score.
  const ordered = [...nodes];
  if (sortMode === "score") {
    ordered.sort((a, b) => {
      const as = a.score ?? -Infinity;
      const bs = b.score ?? -Infinity;
      if (as !== bs) return as - bs; // asc score so higher is further right
      const at = a.startedAt ?? Number.POSITIVE_INFINITY;
      const bt = b.startedAt ?? Number.POSITIVE_INFINITY;
      if (at !== bt) return at - bt;
      return a.id.localeCompare(b.id);
    });
  } else {
    ordered.sort(compare);
  }

  // Compute per-depth row top using max height per depth
  const depths = Array.from(new Set(nodes.map((n) => n.depth || 0))).sort((a, b) => a - b);
  const maxHeightByDepth = new Map<number, number>();
  depths.forEach((d) => {
    let mh = 0;
    for (const n of nodes) if ((n.depth || 0) === d) mh = Math.max(mh, dims.get(n.id)!.height);
    maxHeightByDepth.set(d, mh || CARD.baseHeight);
  });
  const rowTopByDepth = new Map<number, number>();
  let accY = 0;
  depths.forEach((d, idx) => {
    if (idx === 0) accY = 0;
    else accY = accY + (maxHeightByDepth.get(depths[idx - 1]) || CARD.baseHeight) + LAYOUT.nodesep;
    rowTopByDepth.set(d, accY);
  });

  // Place nodes left->right by global order, apply small depth-based x jitter to avoid perfect vertical alignment
  const out: Node<LineageNode>[] = [];
  ordered.forEach((n, i) => {
    const d = n.depth || 0;
    const x = i * LAYOUT.xsep + d * LAYOUT.depthXJitter;
    const y = rowTopByDepth.get(d) || 0;
    out.push({ id: n.id, type: "flippableCard", data: n, position: { x, y } });
  });
  return out;
}

// Raw dagre layout (classic, no order preservation)
function layoutDagreRaw(nodes: LineageNode[], edges: LineageEdge[]): Node<LineageNode>[] {
  const g = new dagre.graphlib.Graph();
  g.setGraph({
    rankdir: "LR",
    nodesep: LAYOUT.nodesep,
    ranksep: LAYOUT.ranksep,
    ranker: "network-simplex",
  });
  g.setDefaultEdgeLabel(() => ({}));
  for (const n of nodes) g.setNode(n.id, calculateNodeDimensions(n));
  for (const e of edges) g.setEdge(e.source, e.target);
  dagre.layout(g);
  return nodes.map((n) => {
    const dn = g.node(n.id);
    return {
      id: n.id,
      type: "flippableCard",
      data: n,
      position: { x: dn.x - dn.width / 2, y: dn.y - dn.height / 2 },
    };
  });
}

// Unified layout entry point (current engines: dagre, topleft, staggered, auto)
function getLayoutedElements(
  nodes: LineageNode[],
  edges: LineageEdge[],
  sortMode: "score" | "time",
  layoutMode: "dagre" | "topleft" | "staggered" | "auto",
) {
  const compare = makeComparator(sortMode);
  const { nodesByDepth, depths, flattened } = groupNodesByDepth(nodes, compare);
  let layoutedNodes: Node<LineageNode>[];
  switch (layoutMode) {
    case "dagre":
      layoutedNodes = layoutDagre(nodes, edges, nodesByDepth, depths, flattened);
      break;
    case "staggered":
      layoutedNodes = layoutStaggered(nodes, edges, compare, sortMode);
      break;
    case "auto":
      layoutedNodes = layoutDagreRaw(nodes, edges);
      break;
    case "topleft":
    default:
      layoutedNodes = layoutStacked(nodesByDepth, depths);
  }
  const layoutedEdges = buildEdges(edges, nodes);
  return { nodes: layoutedNodes, edges: layoutedEdges };
}

const nodeTypes = {
  flippableCard: FlippableCardNode,
};

const edgeTypes = {
  animatedEdge: AnimatedEdge,
};

function LineageGraphInner({
  nodes: lineageNodes,
  edges: lineageEdges,
  className,
}: LineageGraphProps) {
  const [sortMode, setSortMode] = useState<"score" | "time">("time");
  const [layoutMode, setLayoutMode] = useState<"dagre" | "topleft" | "staggered" | "auto">(
    "topleft",
  );
  const { selectedTrialId, highlightedPath, setHighlightedPath } = useStore();
  const { fitView, zoomIn, zoomOut } = useReactFlow();

  // Track which edges have animated to prevent re-animation
  const animatedEdgesRef = useRef(new Map<string, boolean>());

  // Anchor handling to reduce jank: keep visual top-left fixed
  const anchorNodePosRef = useRef<{ x: number; y: number } | null>(null);

  // Bump this to re-animate nodes/edges when layout or sort changes
  const [animationGeneration, setAnimationGeneration] = useState(0);

  // Track previous node positions and per-node move counters to reanimate moved nodes smoothly
  const prevPositionsRef = useRef(new Map<string, { x: number; y: number }>());
  const moveCountersRef = useRef(new Map<string, number>());

  // Select the actual visual top-left node from positioned nodes
  const pickVisualTopLeft = (positioned: Node<LineageNode>[]) => {
    if (positioned.length === 0) return null;
    return positioned.reduce(
      (best, n) => {
        if (!best) return n;
        if (n.position.x < best.position.x) return n;
        if (n.position.x > best.position.x) return best;
        // same column → pick the one higher up (smaller y)
        return n.position.y < best.position.y ? n : best;
      },
      null as Node<LineageNode> | null,
    );
  };

  // Re-animate nodes on sort/layout changes
  useEffect(() => {
    setAnimationGeneration((g) => g + 1);
    // Also reset edge animation flags so paths can reanimate
    animatedEdgesRef.current.clear();
  }, [sortMode, layoutMode]);

  // Layout nodes and edges with animation tracking ref
  const { nodes: layoutedNodes, edges: layoutedEdges } = useMemo(() => {
    const result = getLayoutedElements(lineageNodes, lineageEdges, sortMode, layoutMode);

    // Keep the visual top-left node anchored across both layouts
    if (result.nodes.length > 0) {
      const topLeft = pickVisualTopLeft(result.nodes);
      if (topLeft) {
        if (!anchorNodePosRef.current) {
          // Adopt current top-left position as the fixed anchor position
          anchorNodePosRef.current = { x: topLeft.position.x, y: topLeft.position.y };
        } else {
          const dx = anchorNodePosRef.current.x - topLeft.position.x;
          const dy = anchorNodePosRef.current.y - topLeft.position.y;
          if (dx !== 0 || dy !== 0) {
            result.nodes = result.nodes.map((n) => ({
              ...n,
              position: { x: n.position.x + dx, y: n.position.y + dy },
            }));
          }
        }
      }
    }
    // Detect nodes that moved (post-anchor) to trigger re-animation for just those nodes
    const movedIds = new Set<string>();
    for (const n of result.nodes) {
      const prev = prevPositionsRef.current.get(n.id);
      if (prev) {
        const dx = Math.abs(prev.x - n.position.x);
        const dy = Math.abs(prev.y - n.position.y);
        if (dx > 0.5 || dy > 0.5) movedIds.add(n.id);
      }
    }
    for (const id of movedIds) {
      const cur = moveCountersRef.current.get(id) ?? 0;
      moveCountersRef.current.set(id, cur + 1);
    }
    // Update snapshot after computing moves
    const snap = new Map<string, { x: number; y: number }>();
    for (const n of result.nodes) snap.set(n.id, { x: n.position.x, y: n.position.y });
    prevPositionsRef.current = snap;

    // Add animation tracking ref and animation generation to edges
    result.edges = result.edges.map((edge) => ({
      ...edge,
      data: {
        ...edge.data,
        animatedEdgesRef,
        animationGeneration,
      },
    }));
    // Attach animation generation so nodes remount on sort/layout toggles
    // and also bump when individual nodes moved due to layout updates
    result.nodes = result.nodes.map((n) => ({
      ...n,
      data: {
        ...(n.data as any),
        animationGeneration: animationGeneration + (moveCountersRef.current.get(n.id) ?? 0),
      },
    }));

    return result;
  }, [lineageNodes, lineageEdges, sortMode, layoutMode, animationGeneration]);

  const [nodes, setNodes, onNodesChange] = useNodesState(layoutedNodes);
  const [edges, setEdges, onEdgesChange] = useEdgesState(layoutedEdges);

  // Update nodes and edges when layout or highlighting changes
  useEffect(() => {
    setNodes(layoutedNodes);

    // Apply styles based on highlighting
    const styledEdges = layoutedEdges.map((edge) => {
      // Check if this edge connects consecutive nodes in the path
      const sourceIndex = highlightedPath.indexOf(edge.source);
      const targetIndex = highlightedPath.indexOf(edge.target);

      // Edge is highlighted if it connects adjacent nodes in the path
      // Path goes from selected (index 0) to root (last index)
      // So edge from parent to child means sourceIndex = targetIndex + 1
      const isHighlighted =
        sourceIndex !== -1 && targetIndex !== -1 && sourceIndex === targetIndex + 1;

      return {
        ...edge,
        data: edge.data, // Keep exact same data object reference
        animated: false,
        style: {
          strokeWidth: isHighlighted ? 3 : 2,
          stroke: isHighlighted ? "#6366f1" : "#e5e7eb",
        },
      };
    });

    setEdges(styledEdges);
  }, [layoutedNodes, layoutedEdges, highlightedPath, setNodes, setEdges]);

  // Calculate path from selected node to root using shared utility
  const calculatePath = useCallback(
    (nodeId: string): string[] => {
      return calculateTrialPath(nodeId, lineageNodes, lineageEdges);
    },
    [lineageNodes, lineageEdges],
  );

  // Update highlighted path when selection changes
  useEffect(() => {
    if (selectedTrialId) {
      const path = calculatePath(selectedTrialId);
      setHighlightedPath(path);
    }
  }, [selectedTrialId, calculatePath, setHighlightedPath]);

  if (lineageNodes.length === 0) {
    return (
      <Card className={cn("flex items-center justify-center", className)}>
        <div className="text-center">
          <GitBranch className="mx-auto mb-2 h-12 w-12 text-gray-400" />
          <p className="text-gray-500">No lineage data yet</p>
        </div>
      </Card>
    );
  }

  return (
    <Card className={cn("relative overflow-hidden", className)}>
      <div className="absolute left-4 top-4 z-10 flex items-center gap-4">
        <div className="flex gap-2">
          <Button variant="outline" size="icon" onClick={() => zoomIn()}>
            <ZoomIn className="h-4 w-4" />
          </Button>
          <Button variant="outline" size="icon" onClick={() => zoomOut()}>
            <ZoomOut className="h-4 w-4" />
          </Button>
          <Button variant="outline" size="icon" onClick={() => fitView({ duration: 800 })}>
            <Maximize className="h-4 w-4" />
          </Button>
        </div>

        <div className="flex items-center gap-1 rounded-lg border bg-white/80 px-1 py-1 backdrop-blur-sm dark:bg-gray-900/80">
          <Button
            variant={layoutMode === "auto" ? "ghost" : sortMode === "time" ? "default" : "ghost"}
            size="icon"
            onClick={() => layoutMode !== "auto" && setSortMode("time")}
            className="h-7 w-7"
            title={layoutMode === "auto" ? "Auto layout ignores sorting" : "Sort by time"}
            disabled={layoutMode === "auto"}
          >
            <Clock className="h-4 w-4" />
          </Button>
          <Button
            variant={layoutMode === "auto" ? "ghost" : sortMode === "score" ? "default" : "ghost"}
            size="icon"
            onClick={() => layoutMode !== "auto" && setSortMode("score")}
            className="h-7 w-7"
            title={layoutMode === "auto" ? "Auto layout ignores sorting" : "Sort by score"}
            disabled={layoutMode === "auto"}
          >
            <Target className="h-4 w-4" />
          </Button>
        </div>

        <div className="flex items-center gap-1 rounded-lg border bg-white/80 px-1 py-1 backdrop-blur-sm dark:bg-gray-900/80">
          <Button
            variant={layoutMode === "topleft" ? "default" : "ghost"}
            size="icon"
            onClick={() => setLayoutMode("topleft")}
            className="h-7 w-7"
            title="Stacked layout"
          >
            <AlignLeft className="h-4 w-4" />
          </Button>
          <Button
            variant={layoutMode === "dagre" ? "default" : "ghost"}
            size="icon"
            onClick={() => setLayoutMode("dagre")}
            className="h-7 w-7"
            title="Balanced layout"
          >
            <GitBranch className="h-4 w-4" />
          </Button>
          <Button
            variant={layoutMode === "staggered" ? "default" : "ghost"}
            size="icon"
            onClick={() => setLayoutMode("staggered")}
            className="h-7 w-7"
            title="Staggered layout"
          >
            <ChartScatter className="h-4 w-4" />
          </Button>
          <Button
            variant={layoutMode === "auto" ? "default" : "ghost"}
            size="icon"
            onClick={() => setLayoutMode("auto")}
            className="h-7 w-7"
            title="Auto layout"
          >
            <WandSparkles className="h-4 w-4" />
          </Button>
        </div>
      </div>

      <div className="absolute right-4 top-4 z-10">
        <Badge variant="secondary">
          {lineageNodes.length} nodes • {lineageEdges.length} edges
        </Badge>
      </div>

      <div className="h-full w-full">
        <ReactFlow
          nodes={nodes}
          edges={edges}
          onNodesChange={onNodesChange}
          onEdgesChange={onEdgesChange}
          nodeTypes={nodeTypes}
          edgeTypes={edgeTypes}
          nodesDraggable={false}
          nodesConnectable={false}
          nodesFocusable={true}
          elementsSelectable={true}
          zoomOnScroll={true}
          zoomOnDoubleClick={false}
          panOnDrag={true}
          panOnScroll={false}
          selectNodesOnDrag={false}
          fitView
          fitViewOptions={{
            padding: 0.2,
            minZoom: 0.5,
            maxZoom: 1.5,
          }}
          minZoom={0.3}
          maxZoom={2}
          defaultViewport={{ x: 0, y: 0, zoom: 1 }}
          proOptions={{ hideAttribution: true }}
        >
          <MiniMap
            nodeStrokeWidth={3}
            zoomable
            pannable
            position="bottom-right"
            className="!bg-gray-50 dark:!bg-gray-900"
          />
          <Background variant={BackgroundVariant.Dots} gap={12} size={1} />
        </ReactFlow>
      </div>
    </Card>
  );
}

// Export wrapped component with ReactFlowProvider
export default function LineageGraph(props: LineageGraphProps) {
  return (
    <ReactFlowProvider>
      <LineageGraphInner {...props} />
    </ReactFlowProvider>
  );
}
