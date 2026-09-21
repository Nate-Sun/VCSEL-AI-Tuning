import {
  campaignEventSchema,
  type CampaignEvent,
} from "../schemas/contracts";
import { eventStreamUrl } from "../api/client";

const TERMINAL_STATES = new Set([
  "COMPLETED",
  "FAILED",
  "RECOVERY_REQUIRED",
]);

export type StreamStatus =
  | "connecting"
  | "connected"
  | "reconnecting"
  | "stale"
  | "closed";

export interface EventCursor {
  lastSequence: number;
  events: CampaignEvent[];
  gap: boolean;
}

export function retryDelayMs(attempt: number): number {
  return Math.min(1000 * 2 ** Math.max(0, attempt), 10_000);
}

export function parseSseFrames(raw: string): CampaignEvent[] {
  const bySequence = new Map<number, CampaignEvent>();
  for (const frame of raw.split(/\r?\n\r?\n/)) {
    const dataLines = frame
      .split(/\r?\n/)
      .filter((line) => line.startsWith("data:"))
      .map((line) => line.slice(5).trimStart());
    if (dataLines.length === 0) continue;
    try {
      const parsed = campaignEventSchema.safeParse(JSON.parse(dataLines.join("\n")));
      if (parsed.success) bySequence.set(parsed.data.sequence, parsed.data);
    } catch {
      // Invalid frames are intentionally isolated from current UI state.
    }
  }
  return [...bySequence.values()].sort((left, right) => left.sequence - right.sequence);
}

export function applyEvent(
  cursor: EventCursor,
  event: CampaignEvent,
): EventCursor {
  if (event.sequence <= cursor.lastSequence) return cursor;
  const expected = cursor.lastSequence + 1;
  if (event.sequence !== expected) {
    return { ...cursor, gap: true };
  }
  return {
    lastSequence: event.sequence,
    events: [...cursor.events, event],
    gap: false,
  };
}

interface RunEventStreamOptions {
  campaignId: string;
  initialSequence?: number;
  fetcher?: typeof fetch;
  wait?: (delayMs: number) => Promise<void>;
  signal?: AbortSignal;
  onEvent?: (event: CampaignEvent) => void;
  onGap?: (expectedSequence: number) => void;
  onStatus?: (status: StreamStatus) => void;
}

interface RunEventStreamResult {
  cursor: EventCursor;
  terminal: boolean;
}

async function framesFromResponse(
  response: Response,
  receive: (events: CampaignEvent[]) => boolean,
): Promise<void> {
  if (response.body === null) {
    receive(parseSseFrames(await response.text()));
    return;
  }
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  while (true) {
    const { done, value } = await reader.read();
    buffer += decoder.decode(value, { stream: !done });
    const frames = buffer.split(/\r?\n\r?\n/);
    buffer = frames.pop() ?? "";
    if (frames.length > 0 && receive(parseSseFrames(`${frames.join("\n\n")}\n\n`))) {
      await reader.cancel();
      return;
    }
    if (done) {
      if (buffer.trim() && receive(parseSseFrames(buffer))) return;
      return;
    }
  }
}

export async function runCampaignEventStream(
  options: RunEventStreamOptions,
): Promise<RunEventStreamResult> {
  const fetcher = options.fetcher ?? fetch;
  const wait = options.wait ?? ((delay) => new Promise((resolve) => setTimeout(resolve, delay)));
  let cursor: EventCursor = {
    lastSequence: options.initialSequence ?? 0,
    events: [],
    gap: false,
  };
  let attempt = 0;
  options.onStatus?.("connecting");

  while (!options.signal?.aborted) {
    try {
      const headers = new Headers({ Accept: "text/event-stream" });
      if (cursor.lastSequence > 0) {
        headers.set("Last-Event-ID", String(cursor.lastSequence));
      }
      const request: RequestInit = {
        method: "GET",
        headers,
      };
      if (options.signal) request.signal = options.signal;
      const response = await fetcher(eventStreamUrl(options.campaignId), request);
      if (!response.ok) throw new Error(`SSE HTTP ${response.status}`);
      options.onStatus?.("connected");
      attempt = 0;
      let terminal = false;
      await framesFromResponse(response, (events) => {
        for (const event of events) {
          const next = applyEvent(cursor, event);
          if (next === cursor) continue;
          if (next.gap) {
            cursor = next;
            options.onStatus?.("stale");
            options.onGap?.(cursor.lastSequence + 1);
            return true;
          }
          cursor = next;
          options.onEvent?.(event);
          if (event.to_state !== null && TERMINAL_STATES.has(event.to_state)) {
            terminal = true;
            return true;
          }
        }
        return false;
      });
      if (cursor.gap) return { cursor, terminal: false };
      if (terminal) {
        options.onStatus?.("closed");
        return { cursor, terminal: true };
      }
      options.onStatus?.("reconnecting");
      await wait(retryDelayMs(attempt));
      attempt += 1;
    } catch {
      if (options.signal?.aborted) break;
      options.onStatus?.("reconnecting");
      await wait(retryDelayMs(attempt));
      attempt += 1;
    }
  }
  options.onStatus?.("closed");
  return { cursor, terminal: false };
}
