import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import {
  abortV11Task,
  applyV11HistoryResult,
  getBestObservation,
  getV11FinalReport,
  getV11BestState,
  getV11Observations,
  getV11SweepPoints,
  getV11Task,
  getV11TaskEvents,
  pauseV11Task,
  recoverV11Task,
  resumeV11Task,
} from "../api/client";

export const v11Keys = {
  task: (id: string) => ["v11", "task", id] as const,
  best: (id: string) => ["v11", "best", id] as const,
  sweep: (id: string, query: string) => ["v11", "sweep", id, query] as const,
  recovery: (id: string) => ["v11", "recovery", id] as const,
  observations: (id: string) => ["v11", "task", id, "observations"] as const,
  events: (id: string) => ["v11", "task", id, "events"] as const,
  report: (id: string) => ["v11", "task", id, "final-report"] as const,
};

export function useBestObservation(campaignId?: string) {
  return useQuery({
    queryKey: v11Keys.best(campaignId ?? "none"),
    queryFn: () => getBestObservation(campaignId!),
    enabled: Boolean(campaignId),
  });
}

function useTaskAction<T>(fn: (id: string) => Promise<T>, taskId: string) {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: () => fn(taskId),
    onSuccess: async () => {
      await Promise.all([
        queryClient.invalidateQueries({ queryKey: ["campaigns"] }),
        queryClient.invalidateQueries({ queryKey: ["campaign", taskId] }),
        queryClient.invalidateQueries({ queryKey: v11Keys.task(taskId) }),
        queryClient.invalidateQueries({ queryKey: v11Keys.observations(taskId) }),
        queryClient.invalidateQueries({ queryKey: v11Keys.events(taskId) }),
        queryClient.invalidateQueries({ queryKey: ["v11", "tasks", "active"] }),
        queryClient.invalidateQueries({ queryKey: ["v11", "sweep", taskId] }),
      ]);
    },
  });
}

export const usePauseCampaign = (id: string) => useTaskAction(pauseV11Task, id);
export const useResumeCampaign = (id: string) => useTaskAction(resumeV11Task, id);
export const useAbortCampaign = (id: string) => useTaskAction(
  (taskId) => abortV11Task(taskId, "operator requested abort from dashboard"),
  id,
);

export function useRecovery(taskId: string) {
  return useQuery({
    queryKey: v11Keys.recovery(taskId),
    queryFn: () => getV11Task(taskId),
  });
}

export function useV11Task(taskId: string) {
  return useQuery({
    queryKey: v11Keys.task(taskId),
    queryFn: () => getV11Task(taskId),
    refetchInterval: 2000,
  });
}

export function useV11BestState(taskId: string) {
  return useQuery({
    queryKey: v11Keys.best(taskId),
    queryFn: () => getV11BestState(taskId),
    refetchInterval: 2000,
  });
}

export function useV11Observations(taskId: string) {
  return useQuery({
    queryKey: v11Keys.observations(taskId),
    queryFn: () => getV11Observations(taskId),
    refetchInterval: 2000,
  });
}

export function useV11TaskEvents(taskId: string) {
  return useQuery({
    queryKey: v11Keys.events(taskId),
    queryFn: () => getV11TaskEvents(taskId),
    refetchInterval: 2000,
  });
}

export function useV11FinalReport(taskId: string) {
  return useQuery({
    queryKey: v11Keys.report(taskId),
    queryFn: () => getV11FinalReport(taskId),
  });
}

export const useContinueRecoveredCampaign = (id: string) => useTaskAction(
  (taskId) => recoverV11Task(taskId, "RESUME"),
  id,
);
export const useEndRecoveredCampaign = (id: string) => useTaskAction(
  (taskId) => recoverV11Task(taskId, "ABORT_RECOVERED"),
  id,
);

export function useV11SweepPoints(taskId: string, page: number, pageSize = 25) {
  return useQuery({
    queryKey: v11Keys.sweep(taskId, `${page}:${pageSize}`),
    queryFn: () => getV11SweepPoints(taskId, (page - 1) * pageSize, pageSize),
    refetchInterval: 2000,
  });
}

export function useApplyHistory(taskId: string) {
  return useMutation({
    mutationFn: async () => {
      const best = await getV11BestState(taskId);
      if (!best) throw new Error("No persisted complete best state is available");
      return applyV11HistoryResult({
        source_task_id: taskId,
        observation_id: best.observation_id,
      });
    },
  });
}
