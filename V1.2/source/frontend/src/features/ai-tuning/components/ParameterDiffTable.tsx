import type { AppliedSettings } from "../schemas/contracts";
import { FIR_TAP_LABELS } from "../fixtures/baseline";

export function ParameterDiffTable({ settings }: { settings: AppliedSettings }) {
  const i2cNames = [...new Set([
    ...Object.keys(settings.requested_i2c),
    ...Object.keys(settings.readback_i2c),
  ])];
  const mismatch = i2cNames.some(
    (name) => settings.requested_i2c[name] !== settings.readback_i2c[name],
  ) || settings.requested_fir?.taps.some(
    (value, index) => value !== settings.readback_fir?.taps[index],
  );
  return (
    <section className="panel" aria-labelledby="diff-title">
      <div className="panel-heading-row">
        <div><div className="panel-kicker">SERVER READBACK</div><h2 id="diff-title">参数差异 / Parameter diff</h2></div>
        <span className={mismatch ? "diff-warning" : "diff-match"}>
          {mismatch ? "◆ Requested ≠ Readback" : "● Requested = Readback"}
        </span>
      </div>
      <div className="table-scroll">
        <table>
          <thead><tr><th>Parameter</th><th>Requested</th><th>Readback</th><th>Status</th></tr></thead>
          <tbody>
            {i2cNames.map((name) => {
              const requested = settings.requested_i2c[name];
              const readback = settings.readback_i2c[name];
              return <tr key={name}><th><code>{name}</code></th><td>{requested ?? "—"}</td><td>{readback ?? "—"}</td><td>{requested === readback ? "MATCH" : "MISMATCH"}</td></tr>;
            })}
            {settings.requested_fir?.taps.map((requested, index) => {
              const readback = settings.readback_fir?.taps[index];
              return <tr key={FIR_TAP_LABELS[index]}><th>FIR · {FIR_TAP_LABELS[index]}</th><td>{requested}</td><td>{readback ?? "—"}</td><td>{requested === readback ? "MATCH" : "MISMATCH"}</td></tr>;
            })}
          </tbody>
        </table>
      </div>
    </section>
  );
}
