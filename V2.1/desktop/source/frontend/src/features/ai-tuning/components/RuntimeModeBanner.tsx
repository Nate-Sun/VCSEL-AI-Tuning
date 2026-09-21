import { Link } from "react-router-dom";
import type { Campaign, Capability } from "../schemas/contracts";
import { deriveRuntimeGate } from "../hooks/useRuntimeGate";

export function RuntimeModeBanner({ capability, activeCampaign }: { capability?: Capability | undefined; activeCampaign?: Campaign | null | undefined }) {
  if (!capability) {
    return <div className="mode-banner mode-loading">Runtime / 运行模式 · Waiting for API…</div>;
  }
  const gate = deriveRuntimeGate(capability, activeCampaign ?? null);
  if (gate.recoveryOnly && activeCampaign) {
    return (
      <div className="mode-banner mode-recovery" role="alert">
        <strong>RECOVERY_REQUIRED</strong>
        <span>全局写入已锁定 / All mutations are locked until baseline recovery is verified.</span>
        <Link to={`/recovery/${activeCampaign.campaign_id}`}>Open Recovery / 打开恢复</Link>
      </div>
    );
  }
  if (capability.runtime_mode === "FAKE") {
    return <div className="mode-banner mode-fake"><strong>FAKE · SIMULATION</strong><span>所有控制仅由 MSW 模拟，不连接真实硬件 / Simulated responses only; no physical hardware.</span></div>;
  }
  if (capability.runtime_mode === "READ_ONLY") {
    return <div className="mode-banner mode-readonly"><strong>READ_ONLY</strong><span>可查看数据，所有写操作均被禁用 / Monitoring enabled; mutations disabled.</span></div>;
  }
  return <div className="mode-banner mode-live" role="alert"><strong>LIVE · REAL HARDWARE</strong><span>操作会通过 FastAPI 作用于真实设备 / Authorized operations affect physical instruments.</span></div>;
}
