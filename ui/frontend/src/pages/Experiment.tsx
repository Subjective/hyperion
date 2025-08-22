import { useState, useCallback } from "react";
import { useParams, Link } from "react-router-dom";
import { motion } from "framer-motion";
import { useExperiment, useTrials, useEvents, useLineage, useEventStream } from "@/api/hooks";
import type { Event } from "@/api/client";
import TrialsTree from "@/components/TrialsTree";
import EventTimeline from "@/components/EventTimeline";
import LineageGraph from "@/components/LineageGraph";
import { Card } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import {
  ArrowLeft,
  Activity,
  GitBranch,
  Calendar,
  Target,
  Brain,
  Network,
  FlaskConical,
  LineChart,
  Search,
} from "lucide-react";
import { format } from "date-fns";
import { theme } from "@/lib/theme";
import { toast, Toaster } from "sonner";

export default function Experiment() {
  const { id } = useParams();
  const experimentId = id as string;
  const [realtimeEvents, setRealtimeEvents] = useState<Event[]>([]);

  const { data: experiment, isLoading: expLoading } = useExperiment(experimentId);
  const { data: trials = [], isLoading: trialsLoading } = useTrials(experimentId);
  // Load full history for this experiment (no limit)
  const { data: events = [], isLoading: eventsLoading } = useEvents(experimentId);
  const { data: lineage } = useLineage(experimentId);

  // Merge REST and WS events by id to avoid duplicates
  const allEvents: Event[] = (() => {
    const map = new Map<string, Event>();
    for (const e of events) map.set(e.id, e);
    for (const e of realtimeEvents) map.set(e.id, e);
    const arr = Array.from(map.values());
    arr.sort((a, b) => new Date(a.ts).getTime() - new Date(b.ts).getTime());
    return arr;
  })();

  // Memoize the event handler to prevent WebSocket reconnections
  const handleEvent = useCallback((event: Event) => {
    setRealtimeEvents((prev) => [...prev.slice(-100), event]);

    // Show toast for important events
    if (event.type === "TRIAL_COMPLETED" && event.data?.score) {
      toast.success(`Trial completed with score: ${event.data.score.toFixed(4)}`);
    } else if (event.type === "TRIAL_FAILED") {
      toast.error("Trial failed");
    }
  }, []);

  // Connect WebSocket for this specific experiment
  useEventStream(experimentId, handleEvent);

  // Calculate statistics
  const stats = {
    totalTrials: trials.length,
    completedTrials: trials.filter((t) => t.status === "COMPLETED").length,
    runningTrials: trials.filter((t) => t.status === "RUNNING").length,
    bestScore: Math.max(...trials.filter((t) => t.score).map((t) => t.score!), 0),
    maxDepth: Math.max(...trials.map((t) => t.depth), 0),
  };

  if (expLoading || trialsLoading || eventsLoading) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-gray-50 to-gray-100 dark:from-gray-900 dark:to-gray-950">
        <div className="container mx-auto px-6 py-12">
          <div className="space-y-6">
            <div className="h-32 animate-pulse rounded-2xl bg-gray-200 dark:bg-gray-800" />
            <div className="grid gap-6 lg:grid-cols-3">
              <div className="h-96 animate-pulse rounded-2xl bg-gray-200 dark:bg-gray-800" />
              <div className="h-96 animate-pulse rounded-2xl bg-gray-200 dark:bg-gray-800 lg:col-span-2" />
            </div>
          </div>
        </div>
      </div>
    );
  }

  if (!experiment) {
    return (
      <div className="flex min-h-screen items-center justify-center">
        <Card className="max-w-md p-8 text-center">
          <div className="mb-4 flex justify-center">
            <div className="rounded-full bg-gray-100 p-4 dark:bg-gray-800">
              <Search className="h-12 w-12 text-gray-500 dark:text-gray-400" />
            </div>
          </div>
          <h2 className="mb-2 text-2xl font-bold">Experiment Not Found</h2>
          <p className="mb-4 text-gray-600 dark:text-gray-400">
            The experiment you're looking for doesn't exist.
          </p>
          <Link to="/experiments">
            <Button>Back to Experiments</Button>
          </Link>
        </Card>
      </div>
    );
  }

  const statusColor = theme.colors.status[experiment.status as keyof typeof theme.colors.status];

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-50 via-white to-gray-100 dark:from-gray-900 dark:via-gray-950 dark:to-black">
      <Toaster richColors position="top-right" />

      <div className="container mx-auto px-6 py-8">
        {/* Header */}
        <motion.div
          initial={{ opacity: 0, y: -20 }}
          animate={{ opacity: 1, y: 0 }}
          className="mb-8"
        >
          <Link
            to="/experiments"
            className="mb-4 inline-flex items-center gap-2 text-gray-600 hover:text-gray-900 dark:text-gray-400 dark:hover:text-gray-200"
          >
            <ArrowLeft className="h-4 w-4" />
            Back to Experiments
          </Link>

          <div className="flex items-start justify-between">
            <div>
              <div className="flex items-center gap-3 mb-2">
                <h1 className="text-3xl font-bold">{experiment.name}</h1>
                <Badge className="text-white" style={{ backgroundColor: statusColor }}>
                  <Activity className="mr-1 h-3 w-3" />
                  {experiment.status}
                </Badge>
              </div>
              <div className="flex items-center gap-4 text-sm text-gray-600 dark:text-gray-400">
                <span className="font-mono">{experiment.id}</span>
                <span className="flex items-center gap-1">
                  <Calendar className="h-4 w-4" />
                  {format(new Date(experiment.created_at), "PPp")}
                </span>
              </div>
            </div>
          </div>
        </motion.div>

        {/* Stats Cards */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.1 }}
          className="mb-8 grid gap-4 sm:grid-cols-2 lg:grid-cols-5"
        >
          <Card className="border-0 bg-gradient-to-br from-violet-50 to-violet-100 p-4 dark:from-violet-900/20 dark:to-violet-800/20">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm text-violet-600 dark:text-violet-400">Total Trials</p>
                <p className="text-2xl font-bold">{stats.totalTrials}</p>
              </div>
              <Brain className="h-8 w-8 text-violet-500 opacity-50" />
            </div>
          </Card>

          <Card className="border-0 bg-gradient-to-br from-green-50 to-green-100 p-4 dark:from-green-900/20 dark:to-green-800/20">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm text-green-600 dark:text-green-400">Completed</p>
                <p className="text-2xl font-bold">{stats.completedTrials}</p>
              </div>
              <Activity className="h-8 w-8 text-green-500 opacity-50" />
            </div>
          </Card>

          <Card className="border-0 bg-gradient-to-br from-blue-50 to-blue-100 p-4 dark:from-blue-900/20 dark:to-blue-800/20">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm text-blue-600 dark:text-blue-400">Running</p>
                <p className="text-2xl font-bold">{stats.runningTrials}</p>
              </div>
              <Activity className="h-8 w-8 text-blue-500 opacity-50" />
            </div>
          </Card>

          <Card className="border-0 bg-gradient-to-br from-purple-50 to-purple-100 p-4 dark:from-purple-900/20 dark:to-purple-800/20">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm text-purple-600 dark:text-purple-400">Best Score</p>
                <p className="text-2xl font-bold">{stats.bestScore.toFixed(4)}</p>
              </div>
              <Target className="h-8 w-8 text-purple-500 opacity-50" />
            </div>
          </Card>

          <Card className="border-0 bg-gradient-to-br from-indigo-50 to-indigo-100 p-4 dark:from-indigo-900/20 dark:to-indigo-800/20">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm text-indigo-600 dark:text-indigo-400">Max Depth</p>
                <p className="text-2xl font-bold">{stats.maxDepth}</p>
              </div>
              <GitBranch className="h-8 w-8 text-indigo-500 opacity-50" />
            </div>
          </Card>
        </motion.div>

        {/* Main Content */}
        <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: 0.2 }}>
          <Tabs defaultValue="trials" className="space-y-6">
            <TabsList className="grid w-full grid-cols-3 lg:w-[480px]">
              <TabsTrigger value="trials" className="flex items-center gap-2">
                <FlaskConical className="h-4 w-4" />
                Trials
              </TabsTrigger>
              <TabsTrigger value="lineage" className="flex items-center gap-2">
                <Network className="h-4 w-4" />
                Lineage
              </TabsTrigger>
              <TabsTrigger value="metrics" className="flex items-center gap-2">
                <LineChart className="h-4 w-4" />
                Metrics
              </TabsTrigger>
            </TabsList>

            <TabsContent value="trials" className="space-y-6">
              <div className="grid gap-6 lg:grid-cols-3">
                <div className="lg:col-span-2">
                  <div className="flex items-center gap-2 mb-4">
                    <FlaskConical className="h-5 w-5 text-violet-500" />
                    <h3 className="text-lg font-semibold">Trial Hierarchy</h3>
                  </div>
                  <TrialsTree trials={trials} />
                </div>
                <div>
                  <div className="flex items-center gap-2 mb-4">
                    <Activity className="h-5 w-5 text-indigo-500" />
                    <h3 className="text-lg font-semibold">Event Stream</h3>
                  </div>
                  <EventTimeline events={allEvents} className="h-[650px]" />
                </div>
              </div>
            </TabsContent>

            <TabsContent value="lineage" className="space-y-6">
              <div className="flex items-center gap-2 mb-4">
                <Network className="h-5 w-5 text-violet-500" />
                <h3 className="text-lg font-semibold">Lineage Graph</h3>
                <Badge variant="secondary" className="ml-auto">
                  Double-click cards to flip
                </Badge>
              </div>
              {lineage && (
                <LineageGraph nodes={lineage.nodes} edges={lineage.edges} className="h-[750px]" />
              )}
            </TabsContent>

            <TabsContent value="metrics" className="space-y-6">
              <Card className="p-6">
                <div className="flex items-center gap-2 mb-4">
                  <LineChart className="h-5 w-5 text-violet-500" />
                  <h3 className="text-lg font-semibold">Performance Metrics</h3>
                </div>
                <div className="text-center py-12 text-gray-500">
                  Metrics visualization coming soon...
                </div>
              </Card>
            </TabsContent>
          </Tabs>
        </motion.div>
      </div>
    </div>
  );
}
