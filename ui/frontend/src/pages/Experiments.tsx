import { Link } from "react-router-dom";
import { motion } from "framer-motion";
import { useExperiments, useEventStream } from "@/api/hooks";
import { format, formatDistanceToNow } from "date-fns";
import { useCallback } from "react";
import {
  Activity,
  Clock,
  Beaker,
  ChevronRight,
  Sparkles,
  TrendingUp,
  Zap,
  Brain,
  AlertTriangle,
} from "lucide-react";
import { Card } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";

const statusConfig = {
  RUNNING: { color: "bg-blue-500", pulse: true, icon: Activity },
  COMPLETED: { color: "bg-green-500", pulse: false, icon: Sparkles },
  FAILED: { color: "bg-red-500", pulse: false, icon: Activity },
  PENDING: { color: "bg-gray-500", pulse: false, icon: Clock },
};

export default function Experiments() {
  const { data: experiments, isLoading, error, refetch } = useExperiments();

  // Listen to all events across all experiments to update the list in real-time
  const handleEvent = useCallback(() => {
    // Events trigger automatic query invalidation in useEventStream
  }, []);

  useEventStream(undefined, handleEvent);

  if (isLoading) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-gray-50 to-gray-100 dark:from-gray-900 dark:to-gray-950">
        <div className="container mx-auto px-6 py-12">
          <div className="grid gap-6 md:grid-cols-2 lg:grid-cols-3">
            {[...Array(6)].map((_, i) => (
              <div
                key={i}
                className="h-64 animate-pulse rounded-2xl bg-gray-200 dark:bg-gray-800"
              />
            ))}
          </div>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="flex min-h-screen items-center justify-center">
        <Card className="max-w-md p-8 text-center">
          <div className="mb-4 flex justify-center">
            <div className="rounded-full bg-red-100 p-4 dark:bg-red-900/20">
              <AlertTriangle className="h-12 w-12 text-red-600 dark:text-red-400" />
            </div>
          </div>
          <h2 className="mb-2 text-2xl font-bold">Connection Error</h2>
          <p className="text-gray-600 dark:text-gray-400">Unable to connect to Hyperion server</p>
        </Card>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-50 via-white to-gray-100 dark:from-gray-900 dark:via-gray-950 dark:to-black">
      <div className="container mx-auto px-6 py-12">
        <motion.div
          initial={{ opacity: 0, y: -20 }}
          animate={{ opacity: 1, y: 0 }}
          className="mb-12"
        >
          <div className="flex items-center gap-3 mb-2">
            <div className="p-3 rounded-xl bg-gradient-to-br from-violet-500 to-indigo-600 text-white">
              <Brain className="h-8 w-8" />
            </div>
            <div>
              <h1 className="text-4xl font-bold bg-gradient-to-r from-violet-600 to-indigo-600 bg-clip-text text-transparent">
                Hyperion Dashboard
              </h1>
              <p className="text-gray-600 dark:text-gray-400">
                Hyperparameter optimization experiments
              </p>
            </div>
          </div>
        </motion.div>

        {experiments?.length === 0 ? (
          <motion.div
            initial={{ opacity: 0, scale: 0.95 }}
            animate={{ opacity: 1, scale: 1 }}
            className="flex flex-col items-center justify-center py-24"
          >
            <div className="mb-6 rounded-full bg-gray-100 p-8 dark:bg-gray-800">
              <Beaker className="h-16 w-16 text-gray-400" />
            </div>
            <h2 className="mb-2 text-2xl font-semibold">No experiments yet</h2>
            <p className="text-gray-600 dark:text-gray-400">
              Run your first experiment to see it here
            </p>
          </motion.div>
        ) : (
          <div className="grid gap-6 md:grid-cols-2 lg:grid-cols-3">
            {experiments?.map((exp, index) => (
              <motion.div
                key={exp.id}
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: index * 0.05 }}
              >
                <Link to={`/experiments/${exp.id}`}>
                  <Card className="group relative overflow-hidden border-0 bg-white/80 backdrop-blur-sm transition-all hover:scale-[1.02] hover:shadow-2xl dark:bg-gray-900/80">
                    <div className="absolute inset-0 bg-gradient-to-br from-violet-500/5 to-indigo-600/5 opacity-0 transition-opacity group-hover:opacity-100" />

                    <div className="p-6">
                      <div className="mb-4 flex items-start justify-between">
                        <div className="flex-1">
                          <h3 className="text-xl font-semibold text-gray-900 dark:text-white">
                            {exp.name}
                          </h3>
                          <p className="mt-1 text-sm text-gray-500 dark:text-gray-400">
                            {exp.id.slice(0, 12)}...
                          </p>
                        </div>
                        <div className="relative">
                          {statusConfig[exp.status as keyof typeof statusConfig]?.pulse && (
                            <span className="absolute inset-0 animate-ping rounded-full bg-blue-400 opacity-75" />
                          )}
                          <Badge
                            className={`${
                              statusConfig[exp.status as keyof typeof statusConfig]?.color
                            } text-white border-0`}
                          >
                            {exp.status}
                          </Badge>
                        </div>
                      </div>

                      <div className="mb-4 space-y-2">
                        <div className="flex items-center gap-2 text-sm text-gray-600 dark:text-gray-400">
                          <Clock className="h-4 w-4" />
                          <span>{formatDistanceToNow(new Date(exp.created_at))} ago</span>
                        </div>
                        <div className="flex items-center gap-2 text-sm text-gray-600 dark:text-gray-400">
                          <Activity className="h-4 w-4" />
                          <span>Started {format(new Date(exp.created_at), "PPp")}</span>
                        </div>
                      </div>

                      <div className="flex items-center justify-between">
                        <div className="flex gap-2">
                          <div className="flex h-8 w-8 items-center justify-center rounded-lg bg-violet-100 dark:bg-violet-900/30">
                            <TrendingUp className="h-4 w-4 text-violet-600 dark:text-violet-400" />
                          </div>
                          <div className="flex h-8 w-8 items-center justify-center rounded-lg bg-indigo-100 dark:bg-indigo-900/30">
                            <Zap className="h-4 w-4 text-indigo-600 dark:text-indigo-400" />
                          </div>
                        </div>
                        <ChevronRight className="h-5 w-5 text-gray-400 transition-transform group-hover:translate-x-1" />
                      </div>
                    </div>

                    <div className="h-1 bg-gradient-to-r from-violet-500 to-indigo-600 opacity-0 transition-opacity group-hover:opacity-100" />
                  </Card>
                </Link>
              </motion.div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
