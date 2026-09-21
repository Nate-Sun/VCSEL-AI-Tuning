export function ReadOnlyBanner() {
  return (
    <div className="readonly-banner" role="status">
      <span className="readonly-lock" aria-hidden="true">◇</span>
      <strong>READ-ONLY / 只读</strong>
      <span>Monitoring surface only · 仅用于监控，不执行硬件写入</span>
    </div>
  );
}
