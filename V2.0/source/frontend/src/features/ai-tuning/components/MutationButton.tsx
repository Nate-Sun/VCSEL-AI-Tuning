import type { ButtonHTMLAttributes, ReactNode } from "react";

interface MutationButtonProps extends ButtonHTMLAttributes<HTMLButtonElement> {
  pending?: boolean;
  disabledReason?: string | null | undefined;
  pendingLabel?: ReactNode;
}

export function MutationButton({
  pending = false,
  disabledReason,
  pendingLabel = "Working… / 正在处理…",
  children,
  className = "primary-button",
  disabled,
  ...props
}: MutationButtonProps) {
  const isDisabled = disabled || pending || Boolean(disabledReason);
  return (
    <div className="mutation-control">
      <button {...props} className={className} type={props.type ?? "button"} disabled={isDisabled} aria-busy={pending}>
        {pending ? pendingLabel : children}
      </button>
      {disabledReason ? <small className="disabled-reason" role="status">{disabledReason}</small> : null}
    </div>
  );
}
