import { useState, useEffect, useRef } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { format } from "date-fns";
import {
  Activity,
  Play,
  CheckCircle,
  XCircle,
  Zap,
  Brain,
  Target,
  GitBranch,
  Filter,
  ChevronDown,
  ChevronRight,
  Sparkles,
  FlaskConical,
  Beaker,
  Layers,
  ArrowUpDown,
} from "lucide-react";
import type { Event } from "@/api/client";
import { Card } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { ScrollArea } from "@/components/ui/scroll-area";
import { cn } from "@/lib/utils";

interface EventTimelineProps {
  events: Event[];
  className?: string;
}

const eventConfig = {
  EXPERIMENT_STARTED: {
    icon: Sparkles,
    color: "text-violet-500",
    bg: "bg-violet-50 dark:bg-violet-900/20",
  },
  TRIAL_STARTED: {
    icon: Play,
    color: "text-blue-500",
    bg: "bg-blue-50 dark:bg-blue-900/20",
  },
  TRIAL_COMPLETED: {
    icon: CheckCircle,
    color: "text-green-500",
    bg: "bg-green-50 dark:bg-green-900/20",
  },
  TRIAL_FAILED: {
    icon: XCircle,
    color: "text-red-500",
    bg: "bg-red-50 dark:bg-red-900/20",
  },
  TRIAL_KILLED: {
    icon: Zap,
    color: "text-orange-500",
    bg: "bg-orange-50 dark:bg-orange-900/20",
  },
  TRIAL_PROGRESS: {
    icon: Activity,
    color: "text-indigo-500",
    bg: "bg-indigo-50 dark:bg-indigo-900/20",
  },
  DECISION_RECORDED: {
    icon: Brain,
    color: "text-purple-500",
    bg: "bg-purple-50 dark:bg-purple-900/20",
  },
  CAPACITY_AVAILABLE: {
    icon: Target,
    color: "text-gray-500",
    bg: "bg-gray-50 dark:bg-gray-900/20",
  },
  START_TRIAL: {
    icon: GitBranch,
    color: "text-cyan-500",
    bg: "bg-cyan-50 dark:bg-cyan-900/20",
  },
};

interface GroupedEvent {
  groupId: string;
  events: Event[];
  expanded: boolean;
  label: string;
  icon: any;
}

function EventItem({ event, isNew }: { event: Event; isNew?: boolean }) {
  const [showDetails, setShowDetails] = useState(false);
  const config = eventConfig[event.type as keyof typeof eventConfig] || {
    icon: Activity,
    color: "text-gray-500",
    bg: "bg-gray-50 dark:bg-gray-900/20",
  };
  const Icon = config.icon;

  return (
    <motion.div
      initial={isNew ? { opacity: 0, x: -20, scale: 0.95 } : false}
      animate={{ opacity: 1, x: 0, scale: 1 }}
      className={cn(
        "group relative rounded-lg border bg-white/80 p-3 backdrop-blur-sm transition-all hover:shadow-md dark:bg-gray-900/80",
        isNew && "ring-2 ring-violet-500 ring-offset-2",
      )}
    >
      <div className="flex items-start gap-3">
        <div className={cn("rounded-lg p-2", config.bg)}>
          <Icon className={cn("h-4 w-4", config.color)} />
        </div>

        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-2 mb-1">
            <span className="font-semibold text-sm">{event.type}</span>
            <span className="text-xs text-gray-500">
              {format(new Date(event.ts), "HH:mm:ss.SSS")}
            </span>
          </div>

          {event.data && Object.keys(event.data).length > 0 && (
            <button
              onClick={() => setShowDetails(!showDetails)}
              className="flex items-center gap-1 text-xs text-gray-600 hover:text-gray-900 dark:text-gray-400 dark:hover:text-gray-200"
            >
              {showDetails ? (
                <ChevronDown className="h-3 w-3" />
              ) : (
                <ChevronRight className="h-3 w-3" />
              )}
              {Object.keys(event.data).length} properties
            </button>
          )}

          <AnimatePresence>
            {showDetails && (
              <motion.div
                initial={{ opacity: 0, height: 0 }}
                animate={{ opacity: 1, height: "auto" }}
                exit={{ opacity: 0, height: 0 }}
                className="mt-2"
              >
                <pre className="rounded bg-gray-50 p-2 text-xs overflow-x-auto dark:bg-gray-800">
                  {JSON.stringify(event.data, null, 2)}
                </pre>
              </motion.div>
            )}
          </AnimatePresence>
        </div>
      </div>
    </motion.div>
  );
}

export default function EventTimeline({ events, className }: EventTimelineProps) {
  const [filter, setFilter] = useState<string>("");
  const [groupByTrial, setGroupByTrial] = useState(false);
  const [reverseOrder, setReverseOrder] = useState(false);
  const [expandedGroups, setExpandedGroups] = useState<Set<string>>(new Set());
  const scrollRef = useRef<HTMLDivElement>(null);
  const [newEventIds, setNewEventIds] = useState<Set<string>>(new Set());

  // Track new events
  useEffect(() => {
    const latestEvents = events.slice(-5).map((e) => e.id);
    setNewEventIds(new Set(latestEvents));

    const timer = setTimeout(() => {
      setNewEventIds(new Set());
    }, 3000);

    return () => clearTimeout(timer);
  }, [events.length]);

  // Determine group label and icon based on event types
  const getGroupInfo = (groupId: string, events: Event[]) => {
    if (groupId.startsWith("independent_")) {
      return { label: "Independent Events", icon: Sparkles };
    }

    const eventTypes = new Set(events.map((e) => e.type));

    // Check for trial events
    if (
      eventTypes.has("TRIAL_STARTED") ||
      eventTypes.has("TRIAL_COMPLETED") ||
      eventTypes.has("TRIAL_FAILED") ||
      eventTypes.has("TRIAL_KILLED")
    ) {
      return { label: "Trial Sequence", icon: FlaskConical };
    }

    // Check for decision events
    if (eventTypes.has("DECISION_RECORDED")) {
      return { label: "Decision Flow", icon: Brain };
    }

    // Check for experiment events
    if (eventTypes.has("EXPERIMENT_STARTED") || eventTypes.has("EXPERIMENT_COMPLETED")) {
      return { label: "Experiment", icon: Beaker };
    }

    // Default for other correlated events
    return { label: "Event Sequence", icon: Layers };
  };

  // Group events by correlation
  const groupedEvents = (): GroupedEvent[] => {
    const groups = new Map<string, Event[]>();

    sortedFilteredEvents.forEach((event) => {
      // Use correlation_id for grouping, or treat as independent
      const groupId = event.correlation_id || `independent_${event.id}`;

      if (!groups.has(groupId)) {
        groups.set(groupId, []);
      }
      groups.get(groupId)!.push(event);
    });

    return Array.from(groups.entries()).map(([groupId, events]) => {
      const { label, icon } = getGroupInfo(groupId, events);
      return {
        groupId,
        events: reverseOrder
          ? [...events].reverse() // Exact reversal preserves order within same timestamps
          : events,
        expanded: expandedGroups.has(groupId),
        label,
        icon,
      };
    });
  };

  const filteredEvents = filter
    ? events.filter(
        (e) =>
          e.type.toLowerCase().includes(filter.toLowerCase()) ||
          JSON.stringify(e.data).toLowerCase().includes(filter.toLowerCase()),
      )
    : events;

  // Apply sorting based on reverseOrder
  const sortedFilteredEvents = reverseOrder
    ? [...filteredEvents].reverse() // Exact reversal preserves order within same timestamps
    : filteredEvents; // Keep original order

  const toggleGroup = (groupId: string) => {
    setExpandedGroups((prev) => {
      const next = new Set(prev);
      if (next.has(groupId)) {
        next.delete(groupId);
      } else {
        next.add(groupId);
      }
      return next;
    });
  };

  return (
    <Card className={cn("flex flex-col overflow-hidden", className)}>
      <div className="border-b p-4 flex-shrink-0">
        <div className="flex items-center justify-between mb-3">
          <h3 className="text-lg font-semibold">Event Timeline</h3>
          <Badge variant="secondary">{events.length} events</Badge>
        </div>

        <div className="flex gap-2">
          <div className="relative flex-1">
            <Filter className="absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-gray-400" />
            <input
              type="text"
              placeholder="Filter events..."
              value={filter}
              onChange={(e) => setFilter(e.target.value)}
              className="w-full rounded-lg border bg-white pl-10 pr-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-violet-500 dark:bg-gray-900"
            />
          </div>
          <Button
            variant={groupByTrial ? "default" : "outline"}
            size="sm"
            onClick={() => setGroupByTrial(!groupByTrial)}
          >
            <GitBranch className="mr-1 h-4 w-4" />
            Group
          </Button>
          <Button
            variant="outline"
            size="sm"
            onClick={() => setReverseOrder(!reverseOrder)}
            title={reverseOrder ? "Showing oldest first" : "Showing newest first"}
          >
            <ArrowUpDown className="h-4 w-4" />
          </Button>
        </div>
      </div>

      <ScrollArea className="flex-1 min-h-0" ref={scrollRef}>
        <div className="space-y-2 p-4" key={groupByTrial ? "grouped" : "ungrouped"}>
          {groupByTrial
            ? groupedEvents().map((group) => {
                const Icon = group.icon;
                return (
                  <motion.div
                    key={group.groupId}
                    initial={{ opacity: 0, y: 10 }}
                    animate={{ opacity: 1, y: 0 }}
                    className="group relative overflow-hidden rounded-xl border bg-white/80 backdrop-blur-sm transition-all hover:shadow-lg dark:border-gray-700 dark:bg-gray-900/80"
                  >
                    <button
                      onClick={() => toggleGroup(group.groupId)}
                      className="flex w-full items-center gap-3 p-4 text-left transition-colors hover:bg-gray-50/50 dark:hover:bg-gray-800/50"
                    >
                      <div className="rounded-lg bg-gradient-to-br from-violet-100 to-indigo-100 p-2 dark:from-violet-900/30 dark:to-indigo-900/30">
                        <Icon className="h-4 w-4 text-violet-600 dark:text-violet-400" />
                      </div>

                      <div className="flex-1">
                        <div className="flex items-center gap-2">
                          <span className="font-semibold text-sm">{group.label}</span>
                          {!group.groupId.startsWith("independent_") && (
                            <span className="font-mono text-xs text-gray-500">
                              {group.groupId.slice(0, 8)}...
                            </span>
                          )}
                        </div>
                        <div className="mt-1 flex items-center gap-2 text-xs text-gray-600 dark:text-gray-400">
                          <span>{group.events.length} events</span>
                          <span>•</span>
                          <span>{format(new Date(group.events[0].ts), "HH:mm:ss")}</span>
                        </div>
                      </div>

                      <div className="flex items-center gap-2">
                        <Badge variant="secondary" className="text-xs">
                          {group.events.length}
                        </Badge>
                        {group.expanded ? (
                          <ChevronDown className="h-4 w-4 text-gray-400 transition-transform" />
                        ) : (
                          <ChevronRight className="h-4 w-4 text-gray-400 transition-transform group-hover:translate-x-0.5" />
                        )}
                      </div>
                    </button>

                    <AnimatePresence>
                      {group.expanded && (
                        <motion.div
                          initial={{ opacity: 0, height: 0 }}
                          animate={{ opacity: 1, height: "auto" }}
                          exit={{ opacity: 0, height: 0 }}
                          className="border-t border-gray-100 bg-gray-50/30 p-4 dark:border-gray-800 dark:bg-gray-900/30"
                        >
                          <div className="space-y-2">
                            {group.events.map((event) => (
                              <EventItem
                                key={event.id}
                                event={event}
                                isNew={newEventIds.has(event.id)}
                              />
                            ))}
                          </div>
                        </motion.div>
                      )}
                    </AnimatePresence>
                  </motion.div>
                );
              })
            : sortedFilteredEvents.map((event) => (
                <EventItem key={event.id} event={event} isNew={newEventIds.has(event.id)} />
              ))}
        </div>
      </ScrollArea>
    </Card>
  );
}
