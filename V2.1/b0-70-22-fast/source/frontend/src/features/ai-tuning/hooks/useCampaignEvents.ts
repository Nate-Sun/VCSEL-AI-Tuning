import { useQueryClient } from "@tanstack/react-query";
import { useEffect, useState } from "react";
import { runCampaignEventStream, type StreamStatus } from "../store/eventStream";
import type { CampaignEvent } from "../schemas/contracts";

export function useCampaignEvents(campaignId: string) {
  const queryClient = useQueryClient();
  const [status, setStatus] = useState<StreamStatus>("connecting");
  const [events, setEvents] = useState<CampaignEvent[]>([]);

  useEffect(() => {
    const controller = new AbortController();
    void runCampaignEventStream({
      campaignId,
      signal: controller.signal,
      onEvent: (event) => setEvents((current) => [...current, event]),
      onStatus: setStatus,
      onGap: () => {
        void queryClient.invalidateQueries({ queryKey: ["campaign", campaignId] });
        void queryClient.invalidateQueries({
          queryKey: ["campaign", campaignId, "observations"],
        });
      },
    });
    return () => controller.abort();
  }, [campaignId, queryClient]);

  return { status, events };
}
