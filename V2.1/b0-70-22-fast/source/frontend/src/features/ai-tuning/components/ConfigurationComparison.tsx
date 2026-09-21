import type { AppliedSettings } from "../schemas/contracts";
import { B0, FIR_TAP_LABELS } from "../fixtures/baseline";

export function ConfigurationComparison({
  finalSettings,
}: {
  finalSettings: AppliedSettings | undefined;
}) {
  return (
    <section className="panel" aria-labelledby="comparison-title">
      <div className="panel-kicker">B0 → FINAL</div><h2 id="comparison-title">Configuration comparison / 配置对比</h2>
      <div className="comparison-grid">
        {FIR_TAP_LABELS.map((label, index) => <div key={label}><span>{label}</span><b>{B0.fir.taps[index]}</b><span aria-hidden="true">→</span><strong>{finalSettings?.readback_fir?.taps[index] ?? "—"}</strong></div>)}
      </div>
    </section>
  );
}
