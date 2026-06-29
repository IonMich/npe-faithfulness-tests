import { Check, ChevronDown, RefreshCw, X } from "lucide-react";
import { useEffect, useRef, useState, type ButtonHTMLAttributes, type ReactNode } from "react";

import { cn } from "../lib/utils";

type SelectOption<T extends string> = {
  value: T;
  label: string;
  icon?: ReactNode;
  refreshable?: boolean;
};

export function Button({
  children,
  className,
  variant = "default",
  ...props
}: ButtonHTMLAttributes<HTMLButtonElement> & {
  variant?: "default" | "outline" | "ghost";
}) {
  return (
    <button className={cn("btn", `btn-${variant}`, className)} {...props}>
      {children}
    </button>
  );
}

export function Card({
  children,
  className
}: {
  children: ReactNode;
  className?: string;
}) {
  return <section className={cn("card", className)}>{children}</section>;
}

export function CardHeader({
  title,
  meta,
  children
}: {
  title: string;
  meta?: ReactNode;
  children?: ReactNode;
}) {
  return (
    <div className="card-header">
      <div className="card-title-row">
        <h2>{title}</h2>
        {meta}
      </div>
      {children}
    </div>
  );
}

export function Badge({
  children,
  tone = "default"
}: {
  children: ReactNode;
  tone?: "default" | "muted" | "warn" | "ok";
}) {
  return <span className={cn("badge", `badge-${tone}`)}>{children}</span>;
}

export function Field({
  label,
  children,
  className
}: {
  label: string;
  children: ReactNode;
  className?: string;
}) {
  return (
    <label className={cn("field", className)}>
      <span>{label}</span>
      {children}
    </label>
  );
}

export function SelectField({
  value,
  onChange,
  children,
  ariaLabel
}: {
  value: string;
  onChange: (value: string) => void;
  children: ReactNode;
  ariaLabel: string;
}) {
  return (
    <select
      aria-label={ariaLabel}
      className="select"
      value={value}
      onChange={(event) => onChange(event.target.value)}
    >
      {children}
    </select>
  );
}

export function NumberField({
  value,
  onChange,
  min,
  max,
  step,
  ariaLabel
}: {
  value: number;
  onChange: (value: number) => void;
  min: number;
  max: number;
  step: number;
  ariaLabel: string;
}) {
  const [draft, setDraft] = useState(String(value));
  const skipNextBlurCommit = useRef(false);

  useEffect(() => {
    setDraft(String(value));
  }, [value]);

  function commitDraft() {
    const parsed = Number.parseInt(draft, 10);
    if (!Number.isFinite(parsed)) {
      setDraft(String(value));
      return;
    }
    const clamped = Math.min(Math.max(parsed, min), max);
    setDraft(String(clamped));
    if (clamped !== value) onChange(clamped);
  }

  return (
    <input
      aria-label={ariaLabel}
      className="input"
      max={max}
      min={min}
      step={step}
      type="number"
      value={draft}
      onBlur={() => {
        if (skipNextBlurCommit.current) {
          skipNextBlurCommit.current = false;
          return;
        }
        commitDraft();
      }}
      onChange={(event) => {
        setDraft(event.target.value);
      }}
      onKeyDown={(event) => {
        if (event.key === "Enter") {
          event.preventDefault();
          event.currentTarget.blur();
        } else if (event.key === "Escape") {
          skipNextBlurCommit.current = true;
          setDraft(String(value));
          event.currentTarget.blur();
        }
      }}
    />
  );
}

export function Tabs<T extends string>({
  value,
  onChange,
  options
}: {
  value: T;
  onChange: (value: T) => void;
  options: SelectOption<T>[];
}) {
  return (
    <div className="tabs" role="tablist">
      {options.map((option) => (
        <button
          aria-selected={value === option.value}
          className={cn("tab", value === option.value && "tab-active")}
          key={option.value}
          onClick={() => onChange(option.value)}
          type="button"
        >
          {option.label}
        </button>
      ))}
    </div>
  );
}

export function MultiSelect<T extends string>({
  value,
  onChange,
  onRefresh,
  refreshDisabled = false,
  options,
  placeholder
}: {
  value: T[];
  onChange: (value: T[]) => void;
  onRefresh?: (value: T) => void;
  refreshDisabled?: boolean;
  options: SelectOption<T>[];
  placeholder: string;
}) {
  const [open, setOpen] = useState(false);
  const rootRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    function close(event: MouseEvent) {
      if (rootRef.current && !rootRef.current.contains(event.target as Node)) {
        setOpen(false);
      }
    }
    document.addEventListener("mousedown", close);
    return () => document.removeEventListener("mousedown", close);
  }, []);

  const selectedOptions = options.filter((option) => value.includes(option.value));
  const selectedLabels = selectedOptions.map((option) => option.label);

  function toggle(item: T) {
    if (value.includes(item)) {
      onChange(value.filter((current) => current !== item));
    } else {
      onChange([...value, item]);
    }
  }

  return (
    <div className="multi-select" ref={rootRef}>
      <button
        aria-expanded={open}
        className="multi-trigger"
        onClick={() => setOpen((current) => !current)}
        type="button"
      >
        {selectedOptions.length ? (
          <span className="multi-trigger-icons" aria-hidden="true">
            {selectedOptions.map((option) =>
              option.icon ? (
                <span className="multi-option-icon" key={option.value}>
                  {option.icon}
                </span>
              ) : null
            )}
          </span>
        ) : null}
        <span className={cn("multi-trigger-text", selectedLabels.length === 0 && "muted")}>
          {selectedLabels.length ? selectedLabels.join(", ") : placeholder}
        </span>
        <ChevronDown size={15} />
      </button>
      {open ? (
        <div className="multi-menu">
          {options.map((option) => {
            const selected = value.includes(option.value);
            return (
              <div className="multi-item-row" key={option.value}>
                <button
                  className="multi-item"
                  onClick={() => toggle(option.value)}
                  type="button"
                >
                  <span className={cn("check-box", selected && "check-box-on")}>
                    {selected ? <Check size={13} /> : null}
                  </span>
                  {option.icon ? (
                    <span className="multi-option-icon" aria-hidden="true">
                      {option.icon}
                    </span>
                  ) : null}
                  <span className="multi-item-label">{option.label}</span>
                </button>
                {onRefresh && option.refreshable ? (
                  <button
                    aria-label={`Redraw ${option.label}`}
                    className="multi-refresh"
                    disabled={refreshDisabled}
                    onClick={() => onRefresh(option.value)}
                    title={`Redraw ${option.label} for the current signal`}
                    type="button"
                  >
                    <RefreshCw size={13} />
                  </button>
                ) : null}
              </div>
            );
          })}
          {value.length ? (
            <button className="multi-clear" onClick={() => onChange([])} type="button">
              <X size={13} />
              Clear selections
            </button>
          ) : null}
        </div>
      ) : null}
    </div>
  );
}
