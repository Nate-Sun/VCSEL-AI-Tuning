import type { RuntimeHealth } from "../schemas/contracts";
import { StatusBadge } from "./StatusBadge";

export function HardwareStatus({ health }: { health: RuntimeHealth }) {
  const instruments = [
    ["DCA", health.hardware?.dca],
    ["I2C", health.hardware?.i2c],
    ["BERT", health.hardware?.bert],
  ] as const;
  return (
    <section className="hardware-strip" aria-label="Hardware snapshot status">
      {instruments.map(([name, status]) => (
        <article className="instrument" key={name}>
          <div className="instrument-name">{name}</div>
          <StatusBadge
            label={status === undefined ? "UNAVAILABLE" : status.ok ? "HEALTHY" : "FAULT"}
            tone={status === undefined ? "neutral" : status.ok ? "healthy" : "fault"}
          />
          <small>
            {status === undefined
              ? "No server snapshot / 无服务端快照"
              : status.message ?? "Server-reported / 服务端状态"}
          </small>
        </article>
      ))}
    </section>
  );
}
