import type { Capability, V11RuntimeCapabilities } from "../schemas/contracts";
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

export function StagedLiveAuthority({ capability }: { capability: V11RuntimeCapabilities }) {
  if (capability.live_ai_profile !== "PILOT_3_THEN_FAST24") return null;
  return (
    <div className="staged-live-authority" aria-label="Native V1.2 staged LIVE authority">
      <strong>PILOT_3 → FAST24</strong>
      <span>27 observations · PILOT 69, 71, 73 · FAST ibias_dac 60..80 / res_gain 0..10</span>
      <span>6 coarse · 16 fine · 1 first FIR · 1 iterative FIR</span>
      <span>Formula FIR only · Main, NLC and pattern remain frozen · L1 ≤ 170</span>
      <span>Dataset required · Final hardware state: B0</span>
      <span>DCA timeout {capability.measurement_timeout_s} s · cleanup {capability.cleanup_grace_s} s</span>
      <small>原生 V1.2 分阶段真实调优权限；PILOT 门禁通过前不会开始 FAST24。</small>
    </div>
  );
}
