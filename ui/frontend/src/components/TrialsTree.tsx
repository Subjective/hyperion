import { useState, useMemo, useEffect } from "react";
import { motion, AnimatePresence } from "framer-motion";
import {
  ChevronRight,
  ChevronDown,
  GitBranch,
  Target,
  AlertCircle,
  CheckCircle,
  XCircle,
  Clock,
  Zap,
} from "lucide-react";
import type { Trial } from "@/api/client";
import { theme } from "@/lib/theme";
import { cn, calculateTrialPath } from "@/lib/utils";
import { Badge } from "@/components/ui/badge";
import { Card } from "@/components/ui/card";
import { useStore } from "@/lib/store";

interface TrialNode extends Trial {
  children: TrialNode[];
}

interface TrialsTreeProps {
  trials: Trial[];
  onSelectTrial?: (trial: Trial) => void;
}

const statusIcons = {
  RUNNING: Clock,
  COMPLETED: CheckCircle,
  FAILED: XCircle,
  PENDING: AlertCircle,
  KILLED: Zap,
};

function buildTree(trials: Trial[]): TrialNode[] {
  const nodeMap = new Map<string, TrialNode>();
  const roots: TrialNode[] = [];

  // Create nodes
  trials.forEach((trial) => {
    nodeMap.set(trial.id, { ...trial, children: [] });
  });

  // Build tree structure
  trials.forEach((trial) => {
    const node = nodeMap.get(trial.id)!;
    if (trial.parent_trial_id && nodeMap.has(trial.parent_trial_id)) {
      nodeMap.get(trial.parent_trial_id)!.children.push(node);
    } else {
      roots.push(node);
    }
  });

  // Sort children by score (best first)
  const sortNodes = (nodes: TrialNode[]) => {
    nodes.sort((a, b) => (b.score ?? -Infinity) - (a.score ?? -Infinity));
    nodes.forEach((node) => sortNodes(node.children));
  };
  sortNodes(roots);

  return roots;
}

function TrialNodeComponent({
  node,
  depth = 0,
  onSelect,
}: {
  node: TrialNode;
  depth?: number;
  onSelect?: (trial: Trial) => void;
}) {
  const [expanded, setExpanded] = useState(depth < 2);
  const { selectedTrialId, setSelectedTrial, highlightedPath } = useStore();
  const hasChildren = node.children.length > 0;
  const isSelected = selectedTrialId === node.id;
  const isHighlighted = highlightedPath.includes(node.id);
  const StatusIcon = statusIcons[node.status as keyof typeof statusIcons] || AlertCircle;

  const depthColor = theme.colors.depth[depth % theme.colors.depth.length];

  return (
    <div className="select-none">
      <motion.div
        initial={{ opacity: 0, x: -20 }}
        animate={{ opacity: 1, x: 0 }}
        transition={{ delay: depth * 0.05 }}
        className={cn(
          "group relative mb-2 cursor-pointer rounded-xl border-2 bg-white/90 p-3 backdrop-blur-sm transition-all hover:shadow-lg dark:bg-gray-900/90",
          isSelected && "border-violet-500 shadow-lg shadow-violet-500/20",
          !isSelected && isHighlighted && "border-indigo-400",
          !isSelected && !isHighlighted && "border-gray-200 dark:border-gray-700",
        )}
        onClick={() => {
          setSelectedTrial(node.id);
          onSelect?.(node);
        }}
      >
        <div className="flex items-start gap-3">
          {hasChildren && (
            <button
              onClick={(e) => {
                e.stopPropagation();
                setExpanded(!expanded);
              }}
              className="mt-1 rounded-md p-1 hover:bg-gray-100 dark:hover:bg-gray-800"
            >
              {expanded ? (
                <ChevronDown className="h-4 w-4" />
              ) : (
                <ChevronRight className="h-4 w-4" />
              )}
            </button>
          )}

          <div className="flex-1">
            <div className="mb-2 flex items-center gap-2">
              <div className="h-2 w-2 rounded-full" style={{ backgroundColor: depthColor }} />
              <span className="font-mono text-xs text-gray-500">{node.id.slice(0, 12)}...</span>
              <Badge
                variant="outline"
                className={cn(
                  "text-xs",
                  node.status === "COMPLETED" && "border-green-500 text-green-600",
                  node.status === "RUNNING" && "border-blue-500 text-blue-600",
                  node.status === "FAILED" && "border-red-500 text-red-600",
                )}
              >
                <StatusIcon className="mr-1 h-3 w-3" />
                {node.status}
              </Badge>
              {node.mutation_op && (
                <Badge variant="secondary" className="text-xs">
                  <GitBranch className="mr-1 h-3 w-3" />
                  {node.mutation_op}
                </Badge>
              )}
            </div>

            <div className="flex items-center justify-between">
              <div className="flex items-center gap-4">
                <div className="text-sm">
                  <span className="text-gray-500">Depth:</span>{" "}
                  <span className="font-semibold">{node.depth}</span>
                </div>
                {node.score !== null && node.score !== undefined && (
                  <div className="flex items-center gap-1">
                    <Target className="h-4 w-4 text-violet-500" />
                    <span className="text-lg font-bold text-violet-600">
                      {node.score.toFixed(4)}
                    </span>
                  </div>
                )}
              </div>
              {hasChildren && (
                <span className="text-xs text-gray-500">{node.children.length} children</span>
              )}
            </div>

            {Object.keys(node.params).length > 0 && (
              <div className="mt-2 grid grid-cols-2 gap-2 text-xs">
                {Object.entries(node.params)
                  .slice(0, 4)
                  .map(([key, value]) => (
                    <div key={key} className="rounded bg-gray-50 px-2 py-1 dark:bg-gray-800">
                      <span className="text-gray-500">{key}:</span>{" "}
                      <span className="font-medium">
                        {typeof value === "number" ? value.toFixed(3) : String(value)}
                      </span>
                    </div>
                  ))}
                {Object.keys(node.params).length > 4 && (
                  <div className="text-gray-400">+{Object.keys(node.params).length - 4} more</div>
                )}
              </div>
            )}
          </div>
        </div>

        {node.status === "RUNNING" && (
          <div className="absolute -right-1 -top-1">
            <span className="relative flex h-3 w-3">
              <span className="absolute inline-flex h-full w-full animate-ping rounded-full bg-blue-400 opacity-75" />
              <span className="relative inline-flex h-3 w-3 rounded-full bg-blue-500" />
            </span>
          </div>
        )}
      </motion.div>

      <AnimatePresence>
        {expanded && hasChildren && (
          <motion.div
            initial={{ opacity: 0, height: 0 }}
            animate={{ opacity: 1, height: "auto" }}
            exit={{ opacity: 0, height: 0 }}
            className="ml-8 border-l-2 border-gray-200 pl-4 dark:border-gray-700"
          >
            {node.children.map((child) => (
              <TrialNodeComponent
                key={child.id}
                node={child}
                depth={depth + 1}
                onSelect={onSelect}
              />
            ))}
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}

export default function TrialsTree({ trials, onSelectTrial }: TrialsTreeProps) {
  const tree = useMemo(() => buildTree(trials), [trials]);
  const { selectedTrialId, setHighlightedPath } = useStore();

  // Build edges from parent-child relationships
  const edges = useMemo(() => {
    const edgeList: Array<{ source: string; target: string }> = [];
    trials.forEach((trial) => {
      if (trial.parent_trial_id) {
        edgeList.push({
          source: trial.parent_trial_id,
          target: trial.id,
        });
      }
    });
    return edgeList;
  }, [trials]);

  // Calculate and set highlighted path when selection changes
  useEffect(() => {
    if (selectedTrialId && trials.length > 0) {
      const path = calculateTrialPath(selectedTrialId, trials, edges);
      setHighlightedPath(path);
    }
  }, [selectedTrialId, trials, edges, setHighlightedPath]);

  if (trials.length === 0) {
    return (
      <Card className="flex h-64 items-center justify-center">
        <div className="text-center">
          <AlertCircle className="mx-auto mb-2 h-12 w-12 text-gray-400" />
          <p className="text-gray-500">No trials yet</p>
        </div>
      </Card>
    );
  }

  return (
    <div className="space-y-2">
      {tree.map((node) => (
        <TrialNodeComponent key={node.id} node={node} onSelect={onSelectTrial} />
      ))}
    </div>
  );
}
