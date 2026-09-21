interface DcaDriftPanelProps {
  driftDetected: boolean;
  original: Record<string, string>;
  current: Record<string, string>;
  b0RestoreStatus: string;
  preflightAllowed: boolean;
}

export function DcaDriftPanel({ driftDetected, original, current, b0RestoreStatus, preflightAllowed }: DcaDriftPanelProps) {
  const keys = [...new Set([...Object.keys(original), ...Object.keys(current)])];
  return <article className={driftDetected ? "panel drift-alert" : "panel"}><div className="panel-kicker">DCA CONFIGURATION DRIFT</div><h2>{driftDetected ? "Task paused / 配置漂移已暂停任务" : "Reference alignment / 配置一致"}</h2><div className="table-scroll"><table><thead><tr><th>Field</th><th>Original</th><th>Current</th></tr></thead><tbody>{keys.map((key) => <tr key={key}><td>{key}</td><td>{original[key] ?? "—"}</td><td className={original[key] === current[key] ? "" : "diff-warning"}>{current[key] ?? "—"}</td></tr>)}</tbody></table></div><div className="status-line"><span>B0 restore</span><b>{b0RestoreStatus}</b></div><div className="status-line"><span>Resume preflight</span><b>{preflightAllowed ? "ALLOWED" : "BLOCKED"}</b></div>{!preflightAllowed ? <p className="notice-warning">Resume remains disabled until FastAPI reruns Preflight and explicitly allows continuation.</p> : null}</article>;
}
