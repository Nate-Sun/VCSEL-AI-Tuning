import { useQuery } from "@tanstack/react-query";
import { getCampaigns, getCapabilities, getHealth } from "../api/client";

export const RUNTIME_POLL_INTERVAL_MS = 5000;

export function useRuntimeQueries() {
  const mockScenario = import.meta.env.VITE_ENABLE_MOCKS === "true" && typeof window !== "undefined"
    ? new URLSearchParams(window.location.search).get("scenario")
    : null;
  const health = useQuery({
    queryKey: ["runtime", "health"],
    queryFn: getHealth,
    refetchInterval: RUNTIME_POLL_INTERVAL_MS,
  });
  const capabilities = useQuery({
    queryKey: ["runtime", "capabilities"],
    queryFn: getCapabilities,
    refetchInterval: RUNTIME_POLL_INTERVAL_MS,
  });
  const campaigns = useQuery({
    queryKey: ["campaigns", mockScenario],
    queryFn: () => getCampaigns(50, 0, mockScenario),
    refetchInterval: RUNTIME_POLL_INTERVAL_MS,
  });

  return { health, capabilities, campaigns };
}
