import type { Capability } from "../schemas/contracts";
import { StatusBadge } from "./StatusBadge";

const capabilityNames = [
  "hierarchical_ready",
  "fir_proposer_ready",
  "exhaustive_ready",
  "tdecq_bayesian_ready",
] as const;

export function CapabilityPanel({ capability }: { capability: Capability }) {
  return (
    <section className="panel" aria-labelledby="capability-title">
      <div className="panel-kicker">SERVER CAPABILITIES</div>
      <h2 id="capability-title">能力矩阵 / Capability matrix</h2>
      <div className="capability-list">
        {capabilityNames.map((name) => {
          const value = capability[name];
          return (
            <div className="capability-row" key={name}>
              <code>{name}</code>
              <StatusBadge
                label={value === undefined ? "UNAVAILABLE" : value ? "READY" : "NOT READY"}
                tone={value === undefined ? "neutral" : value ? "healthy" : "warning"}
              />
            </div>
          );
        })}
      </div>
      <div className="authorization-line">
        Authorization <strong>{capability.authorization_modes.join(", ") || "UNAVAILABLE"}</strong>
      </div>
    </section>
  );
}
