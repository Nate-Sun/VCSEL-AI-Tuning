import { B0, FIR_TAP_LABELS } from "../fixtures/baseline";

export function BaselineCard() {
  return (
    <section className="panel" aria-labelledby="b0-title">
      <div className="panel-heading-row">
        <div>
          <div className="panel-kicker">COMMISSIONED BASELINE</div>
          <h2 id="b0-title">B0 / 固定基线</h2>
        </div>
        <span className="pattern-tag">Pattern · {B0.pattern}</span>
      </div>
      <div className="fir-grid" aria-label="B0 FIR labeled values">
        {FIR_TAP_LABELS.map((label, index) => (
          <div className={label === "Main" ? "fir-cell fir-main" : "fir-cell"} key={label}>
            <span>{label}</span>
            <strong>{B0.fir.taps[index]}</strong>
          </div>
        ))}
      </div>
      <div className="nlc-line">
        <span>Upper NLC <b>{B0.fir.upper_nlc}</b></span>
        <span>Lower NLC <b>{B0.fir.lower_nlc}</b></span>
      </div>
      <div className="i2c-grid">
        {Object.entries(B0.i2c).map(([name, value]) => (
          <span key={name}><code>{name}</code><b>{value}</b></span>
        ))}
      </div>
    </section>
  );
}
