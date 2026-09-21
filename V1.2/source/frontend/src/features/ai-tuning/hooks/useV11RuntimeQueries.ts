import { useQuery } from "@tanstack/react-query";
import {
  getV11ActiveTask,
  getV11RuntimeCapabilities,
} from "../api/client";
import { RUNTIME_POLL_INTERVAL_MS } from "./useRuntimeQueries";

export function useV11RuntimeQueries() {
  const capabilities = useQuery({
    queryKey: ["v11", "runtime", "capabilities"],
    queryFn: getV11RuntimeCapabilities,
    refetchInterval: RUNTIME_POLL_INTERVAL_MS,
  });
  const activeTask = useQuery({
    queryKey: ["v11", "tasks", "active"],
    queryFn: getV11ActiveTask,
    refetchInterval: RUNTIME_POLL_INTERVAL_MS,
  });

  return { capabilities, activeTask };
}
