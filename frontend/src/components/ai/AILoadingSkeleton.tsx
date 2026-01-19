import { Sparkles } from "lucide-react";

interface AILoadingSkeletonProps {
  variant?: "panel" | "card" | "inline";
}

export const AILoadingSkeleton = ({ variant = "panel" }: AILoadingSkeletonProps) => {
  if (variant === "inline") {
    return (
      <div className="flex items-center gap-2 text-muted-foreground">
        <Sparkles className="w-4 h-4 animate-pulse text-primary" />
        <span className="text-sm">Analyzing...</span>
        <div className="flex gap-1">
          <span className="w-1.5 h-1.5 rounded-full bg-primary animate-bounce" style={{ animationDelay: "0ms" }} />
          <span className="w-1.5 h-1.5 rounded-full bg-primary animate-bounce" style={{ animationDelay: "150ms" }} />
          <span className="w-1.5 h-1.5 rounded-full bg-primary animate-bounce" style={{ animationDelay: "300ms" }} />
        </div>
      </div>
    );
  }

  if (variant === "card") {
    return (
      <div className="p-4 rounded-xl bg-secondary/50 animate-pulse">
        <div className="flex items-start gap-3">
          <div className="w-8 h-8 rounded-lg bg-primary/20" />
          <div className="flex-1 space-y-2">
            <div className="h-4 w-3/4 rounded bg-muted-foreground/20" />
            <div className="h-3 w-full rounded bg-muted-foreground/10" />
            <div className="h-3 w-2/3 rounded bg-muted-foreground/10" />
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-4">
      {/* Header skeleton */}
      <div className="flex items-center gap-3 animate-pulse">
        <div className="w-12 h-12 rounded-xl bg-primary/20 flex items-center justify-center">
          <Sparkles className="w-6 h-6 text-primary/50" />
        </div>
        <div className="space-y-2">
          <div className="h-5 w-32 rounded bg-muted-foreground/20" />
          <div className="h-3 w-48 rounded bg-muted-foreground/10" />
        </div>
      </div>

      {/* Score skeleton */}
      <div className="flex items-center gap-6 p-4 rounded-xl bg-secondary/50 animate-pulse">
        <div className="w-20 h-20 rounded-full border-4 border-primary/20" />
        <div className="flex-1 space-y-2">
          <div className="h-4 w-24 rounded bg-muted-foreground/20" />
          <div className="grid grid-cols-4 gap-2">
            {[1, 2, 3, 4].map((i) => (
              <div key={i} className="h-2 rounded bg-muted-foreground/10" />
            ))}
          </div>
        </div>
      </div>

      {/* Feedback cards skeleton */}
      <div className="space-y-3">
        {[1, 2, 3].map((i) => (
          <div key={i} className="p-4 rounded-xl bg-secondary/50 animate-pulse" style={{ animationDelay: `${i * 100}ms` }}>
            <div className="flex items-start gap-3">
              <div className="w-8 h-8 rounded-lg bg-primary/20" />
              <div className="flex-1 space-y-2">
                <div className="h-4 w-3/4 rounded bg-muted-foreground/20" />
                <div className="h-3 w-full rounded bg-muted-foreground/10" />
                <div className="h-3 w-2/3 rounded bg-muted-foreground/10" />
              </div>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
};
