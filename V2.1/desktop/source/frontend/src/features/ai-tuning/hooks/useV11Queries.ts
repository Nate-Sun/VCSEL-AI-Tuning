import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import {
  abortCampaign,
  applyHistoryResult,
  continueRecoveredCampaign,
  endRecoveredCampaign,
  getBestObservation,
  getHistoryApplyStatus,
  getRecovery,
  getSweepPage,
  pauseCampaign,
  restoreBaseline,
  resumeCampaign,
} from "../api/client";

export const v11Keys = {
  task: (id: string) => ["v11", "task", id] as const,
  best: (id: string) => ["v11", "best", id] as const,
  sweep: (id: string, query: string) => ["v11", "sweep", id, query] as const,
  recovery: (id: string) => ["v11", "recovery", id] as const,
};

export function useBestObservation(campaignId?: string) {
  return useQuery({
    queryKey: v11Keys.best(campaignId ?? "none"),
    queryFn: () => getBestObservation(campaignId!),
    enabled: Boolean(campaignId),
  });
}

function useCampaignAction(fn: (id: string) => Promise<unknown>, campaignId: string) {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: () => fn(campaignId),
    onSuccess: async () => {
      await Promise.all([
        queryClient.invalidateQueries({ queryKey: ["campaigns"] }),
        queryClient.invalidateQueries({ queryKey: ["campaign", campaignId] }),
        queryClient.invalidateQueries({ queryKey: v11Keys.best(campaignId) }),
        queryClient.invalidateQueries({ queryKey: ["v11", "sweep", campaignId] }),
      ]);
    },
  });
}

export const usePauseCampaign = (id: string) => useCampaignAction(pauseCampaign, id);
export const useResumeCampaign = (id: string) => useCampaignAction(resumeCampaign, id);
export const useAbortCampaign = (id: string) => useCampaignAction(abortCampaign, id);
export function useRestoreBaseline(id: string) {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: () => restoreBaseline(id),
    onSuccess: async () => {
      await Promise.all([
        queryClient.invalidateQueries({ queryKey: v11Keys.recovery(id) }),
        queryClient.invalidateQueries({ queryKey: ["campaigns"] }),
      ]);
    },
  });
}

function useRecoveryAction(fn: (id: string) => Promise<unknown>, id: string) {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: () => fn(id),
    onSuccess: async () => {
      await Promise.all([
        queryClient.invalidateQueries({ queryKey: v11Keys.recovery(id) }),
        queryClient.invalidateQueries({ queryKey: ["campaigns"] }),
        queryClient.invalidateQueries({ queryKey: ["campaign", id] }),
      ]);
    },
  });
}

export const useContinueRecoveredCampaign = (id: string) => useRecoveryAction(continueRecoveredCampaign, id);
export const useEndRecoveredCampaign = (id: string) => useRecoveryAction(endRecoveredCampaign, id);

export function useRecovery(campaignId: string) {
  return useQuery({ queryKey: v11Keys.recovery(campaignId), queryFn: () => getRecovery(campaignId) });
}

export function useSweepPage(campaignId: string, query: string) {
  return useQuery({ queryKey: v11Keys.sweep(campaignId, query), queryFn: () => getSweepPage(campaignId, query), refetchInterval: 2000 });
}

export function useApplyHistory(campaignId: string) {
  return useMutation({ mutationFn: () => applyHistoryResult(campaignId) });
}

export function useHistoryApplyStatus(operationId?: string, enabled = false) {
  return useQuery({
    queryKey: ["v11", "history-apply", operationId ?? "none"],
    queryFn: () => getHistoryApplyStatus(operationId!),
    enabled: Boolean(operationId) && enabled,
    refetchInterval: (query) => query.state.data?.status === "RUNNING" ? 1000 : false,
  });
}
