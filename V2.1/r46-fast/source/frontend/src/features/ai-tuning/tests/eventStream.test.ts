import {
  applyEvent,
  parseSseFrames,
  retryDelayMs,
  runCampaignEventStream,
  type EventCursor,
} from "../store/eventStream";
import type { CampaignEvent } from "../schemas/contracts";

function event(
  sequence: number,
  toState: CampaignEvent["to_state"] = "I2C_COARSE",
): CampaignEvent {
  return {
    campaign_id: "campaign-1",
    created_at: "2026-07-24T03:00:00Z",
    event_type: "OBSERVATION_RECORDED",
    from_state: "I2C_SCREENING",
    payload: {},
    sequence,
    to_state: toState,
  };
}

function sseBody(events: ReturnType<typeof event>[]) {
  return events
    .map(
      (item) =>
        `id: ${item.sequence}\nevent: ${item.event_type}\ndata: ${JSON.stringify(item)}\n\n`,
    )
    .join("");
}

describe("resumable campaign event stream", () => {
  it("sorts frames and ignores duplicate sequences", () => {
    const parsed = parseSseFrames(sseBody([event(2), event(1), event(2)]));
    expect(parsed.map((item) => item.sequence)).toEqual([1, 2]);
  });

  it("rejects invalid payloads before they reach state", () => {
    const body = `id: 1\nevent: BAD\ndata: ${JSON.stringify({ ...event(1), sequence: "1" })}\n\n`;
    expect(parseSseFrames(body)).toEqual([]);
  });

  it("detects a sequence gap and keeps the prior cursor", () => {
    const cursor: EventCursor = { lastSequence: 3, events: [], gap: false };
    const next = applyEvent(cursor, event(5));
    expect(next.gap).toBe(true);
    expect(next.lastSequence).toBe(3);
    expect(next.events).toEqual([]);
  });

  it("ignores an already applied sequence", () => {
    const cursor: EventCursor = {
      lastSequence: 2,
      events: [event(2)],
      gap: false,
    };
    expect(applyEvent(cursor, event(2))).toBe(cursor);
  });

  it("uses capped exponential backoff", () => {
    expect([0, 1, 2, 3, 4, 8].map(retryDelayMs)).toEqual([
      1000, 2000, 4000, 8000, 10000, 10000,
    ]);
  });

  it("reconnects with Last-Event-ID and closes on terminal state", async () => {
    const headers: Array<string | null> = [];
    let attempt = 0;
    const fetcher: typeof fetch = (_input, init) => {
      headers.push(new Headers(init?.headers).get("Last-Event-ID"));
      attempt += 1;
      if (attempt === 1) return Promise.reject(new TypeError("offline"));
      return Promise.resolve(
        new Response(sseBody([event(4, "COMPLETED")]), {
          status: 200,
          headers: { "Content-Type": "text/event-stream" },
        }),
      );
    };
    const statuses: string[] = [];
    const delays: number[] = [];

    const result = await runCampaignEventStream({
      campaignId: "campaign-1",
      initialSequence: 3,
      fetcher,
      wait: (delay) => {
        delays.push(delay);
        return Promise.resolve();
      },
      onStatus: (status) => statuses.push(status),
    });

    expect(headers).toEqual(["3", "3"]);
    expect(delays).toEqual([1000]);
    expect(statuses).toContain("reconnecting");
    expect(result.terminal).toBe(true);
    expect(result.cursor.lastSequence).toBe(4);
  });

  it("requests a REST resync when a gap appears", async () => {
    const gaps: number[] = [];
    const result = await runCampaignEventStream({
      campaignId: "campaign-1",
      initialSequence: 1,
      fetcher: () =>
        Promise.resolve(
          new Response(sseBody([event(3)]), {
            headers: { "Content-Type": "text/event-stream" },
          }),
        ),
      wait: () => Promise.resolve(),
      onGap: (expected) => gaps.push(expected),
    });

    expect(gaps).toEqual([2]);
    expect(result.cursor.gap).toBe(true);
  });
});
