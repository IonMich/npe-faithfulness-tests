import { clsx, type ClassValue } from "clsx";
import { twMerge } from "tailwind-merge";

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}

export function formatValue(value: unknown) {
  if (value === null || value === undefined) return "n/a";
  if (typeof value === "number") return value.toPrecision(4);
  return String(value);
}

export function compactPath(value: unknown) {
  if (value === null || value === undefined) return "n/a";
  const parts = String(value).split("/").filter(Boolean);
  if (parts.length <= 4) return String(value);
  const tail = parts.slice(-2).join("/");
  return tail.length <= 44 ? `.../${tail}` : `.../${parts[parts.length - 1]}`;
}
