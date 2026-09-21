import type { StreamStatus } from "../store/eventStream";
import { StatusBadge } from "./StatusBadge";

interface DisplayEvent {
  sequence: number;
  event_type: string;
  created_at: string;
}

export function EventLog({ events, status }: { events: DisplayEvent[]; status: StreamStatus }) {
  const tone = status === "connected" ? "healthy" : status === "stale" ? "fault" : status === "reconnecting" ? "warning" : "neutral";
  return (
    <section className="panel" aria-labelledby="events-title">
      <div className="panel-heading-row"><div><div className="panel-kicker">ORDERED SERVER EVENTS</div><h2 id="events-title">Event log / 事件流</h2></div><StatusBadge label={status.toUpperCase()} tone={tone} /></div>
      <ol className="event-log">
        {events.length === 0 ? <li className="empty-state">Waiting for validated events / 等待已校验事件</li> : events.map((event) => <li key={event.sequence}><span>#{event.sequence}</span><strong>{event.event_type}</strong><time dateTime={event.created_at}>{new Date(event.created_at).toLocaleTimeString()}</time></li>)}
      </ol>
    </section>
  );
}
