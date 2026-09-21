type Tone = "healthy" | "warning" | "fault" | "neutral" | "info";

export function StatusBadge({
  label,
  tone = "neutral",
}: {
  label: string;
  tone?: Tone;
}) {
  const icons: Record<Tone, string> = {
    healthy: "●",
    warning: "◆",
    fault: "▲",
    neutral: "○",
    info: "◇",
  };
  return (
    <span className={`status-badge status-${tone}`}>
      <span aria-hidden="true">{icons[tone]}</span> {label}
    </span>
  );
}
