import { useState } from "react";
import {
  FileText, User, Flame, GitBranch, XCircle, TrendingDown,
  AlertTriangle, ArrowRight, Copy, Check, ChevronDown, ChevronUp,
  Brain, Clock, Heart, DollarSign, Users, Zap, BookOpen, Lock
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Phase1DetailedResult, PainMoment, RootCause, RiskFlag } from "@/lib/mockAIData";
import { AIQualityScore } from "./AIQualityScore";
import { cn } from "@/lib/utils";

interface AIPhase1DetailedAnalysisProps {
  result: Phase1DetailedResult;
}

export const AIPhase1DetailedAnalysis = ({ result }: AIPhase1DetailedAnalysisProps) => {
  const [copiedId, setCopiedId] = useState<string | null>(null);
  const [expandedPainMoments, setExpandedPainMoments] = useState<string[]>([result.painMoments[0]?.id]);

  const copyToClipboard = (text: string, id: string) => {
    navigator.clipboard.writeText(text);
    setCopiedId(id);
    setTimeout(() => setCopiedId(null), 2000);
  };

  const togglePainMoment = (id: string) => {
    setExpandedPainMoments(prev =>
      prev.includes(id) ? prev.filter(p => p !== id) : [...prev, id]
    );
  };

  const getCategoryIcon = (category: RootCause['category']) => {
    switch (category) {
      case 'Knowledge': return <BookOpen className="w-4 h-4" />;
      case 'Process': return <GitBranch className="w-4 h-4" />;
      case 'Access': return <Lock className="w-4 h-4" />;
      case 'Psychology': return <Brain className="w-4 h-4" />;
    }
  };

  const getCategoryColor = (category: RootCause['category']) => {
    switch (category) {
      case 'Knowledge': return "bg-blue-500/10 text-blue-600 dark:text-blue-400 border-blue-500/20";
      case 'Process': return "bg-purple-500/10 text-purple-600 dark:text-purple-400 border-purple-500/20";
      case 'Access': return "bg-amber-500/10 text-amber-600 dark:text-amber-400 border-amber-500/20";
      case 'Psychology': return "bg-pink-500/10 text-pink-600 dark:text-pink-400 border-pink-500/20";
    }
  };

  const formatPersonaText = () => {
    const p = result.personaClarification.primaryUser;
    return `Primary User (MVP):
- Role: ${p.role}
- Context: ${p.context}
- Goal: ${p.goal}
- Urgency: ${p.urgency}
- Failure Impact: ${p.failureImpact}`;
  };

  const formatPainMomentText = (pm: PainMoment) => {
    return `Moment: ${pm.moment}
Trigger: ${pm.trigger}
Current Behavior: ${pm.currentBehavior}
Why It Hurts: ${pm.whyItHurts}`;
  };

  const formatRootCausesText = () => {
    return `Root Causes:\n${result.rootCauses.map((rc, i) =>
      `${i + 1}. ${rc.category} Gap — ${rc.description}`
    ).join('\n')}`;
  };

  const formatExistingSolutionsText = () => {
    return `Existing Solutions:\n${result.existingSolutions.map(s =>
      `- ${s.name}: ${s.gap}`
    ).join('\n')}`;
  };

  const formatImpactText = () => {
    const i = result.problemImpact;
    return `Impact:
- Time: ${i.time}
- Quality: ${i.quality}
- Emotional: ${i.emotional}
- Opportunity: ${i.opportunity}`;
  };

  const formatRiskFlagsText = () => {
    return `Key Risks:\n${result.riskFlags.map(rf => `- ${rf.description}`).join('\n')}`;
  };

  return (
    <div className="space-y-8 animate-fade-in">
      {/* Overall Quality Score */}
      <AIQualityScore
        score={result.overallScore}
        breakdown={result.breakdown}
        animated={true}
      />

      {/* Section 1: Normalized Problem Summary */}
      <section className="space-y-4">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="w-8 h-8 rounded-lg bg-primary/10 flex items-center justify-center text-sm font-bold text-primary">1</div>
            <div>
              <h3 className="font-display font-bold text-lg">Normalized Problem Summary</h3>
              <p className="text-xs text-muted-foreground">{result.normalizedProblem.purpose}</p>
            </div>
          </div>
          <Button
            variant="ghost"
            size="sm"
            className="gap-1.5 text-xs"
            onClick={() => copyToClipboard(result.normalizedProblem.rewrittenProblem, "problem-summary")}
          >
            {copiedId === "problem-summary" ? <Check className="w-3.5 h-3.5" /> : <Copy className="w-3.5 h-3.5" />}
            Copy
          </Button>
        </div>

        <div className="bg-secondary/50 border border-border/50 rounded-xl p-5">
          <div className="border-l-4 border-primary pl-4">
            <p className="text-sm leading-relaxed">{result.normalizedProblem.rewrittenProblem}</p>
          </div>
        </div>
      </section>

      {/* Section 2: User Persona Clarification */}
      <section className="space-y-4">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="w-8 h-8 rounded-lg bg-primary/10 flex items-center justify-center text-sm font-bold text-primary">2</div>
            <div>
              <h3 className="font-display font-bold text-lg">User Persona Clarification</h3>
              <p className="text-xs text-muted-foreground">Kill ambiguous users early. Forced choice.</p>
            </div>
          </div>
          <Button
            variant="ghost"
            size="sm"
            className="gap-1.5 text-xs"
            onClick={() => copyToClipboard(formatPersonaText(), "persona")}
          >
            {copiedId === "persona" ? <Check className="w-3.5 h-3.5" /> : <Copy className="w-3.5 h-3.5" />}
            Copy
          </Button>
        </div>

        <div className="bg-secondary/50 border border-border/50 rounded-xl p-5 space-y-4">
          <div className="flex items-center gap-2 mb-3">
            <User className="w-5 h-5 text-primary" />
            <span className="font-semibold">Primary User (MVP)</span>
            <Badge variant="outline" className="text-xs bg-primary/10 text-primary border-primary/20">Focus</Badge>
          </div>

          <div className="font-mono text-sm space-y-1 bg-background/50 rounded-lg p-4 border border-border/30">
            <p><span className="text-muted-foreground">- Role:</span> {result.personaClarification.primaryUser.role}</p>
            <p><span className="text-muted-foreground">- Context:</span> {result.personaClarification.primaryUser.context}</p>
            <p><span className="text-muted-foreground">- Goal:</span> {result.personaClarification.primaryUser.goal}</p>
            <p><span className="text-muted-foreground">- Urgency:</span> {result.personaClarification.primaryUser.urgency}</p>
            <p><span className="text-muted-foreground">- Failure Impact:</span> {result.personaClarification.primaryUser.failureImpact}</p>
          </div>

          {result.personaClarification.secondaryPersonas.length > 0 && (
            <div className="pt-3 border-t border-border/30">
              <p className="text-xs text-muted-foreground mb-2 font-medium">Secondary Personas (Excluded from MVP):</p>
              <div className="space-y-1.5">
                {result.personaClarification.secondaryPersonas.map((sp, i) => (
                  <div key={i} className="flex items-start gap-2 text-xs">
                    <Users className="w-3.5 h-3.5 text-muted-foreground mt-0.5" />
                    <span><span className="font-medium">{sp.role}</span> — <span className="text-muted-foreground">{sp.reason}</span></span>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      </section>

      {/* Section 3: Concrete Pain Moments */}
      <section className="space-y-4">
        <div className="flex items-center gap-3">
          <div className="w-8 h-8 rounded-lg bg-destructive/10 flex items-center justify-center text-sm font-bold text-destructive">3</div>
          <div>
            <h3 className="font-display font-bold text-lg">Concrete Pain Moments</h3>
            <p className="text-xs text-muted-foreground">Replace abstract pain with lived moments. Each written as a scene.</p>
          </div>
        </div>

        <div className="space-y-3">
          {result.painMoments.map((pm) => (
            <div
              key={pm.id}
              className="rounded-xl border border-border/50 bg-secondary/30 overflow-hidden"
            >
              <div
                className="w-full p-4 flex items-center justify-between text-left hover:bg-secondary/50 transition-colors cursor-pointer"
                onClick={() => togglePainMoment(pm.id)}
              >
                <div className="flex items-center gap-3">
                  <Flame className="w-5 h-5 text-destructive" />
                  <span className="font-semibold">{pm.moment}</span>
                </div>
                <div className="flex items-center gap-2">
                  <Button
                    variant="ghost"
                    size="sm"
                    className="gap-1.5 text-xs h-7"
                    onClick={(e) => {
                      e.stopPropagation();
                      copyToClipboard(formatPainMomentText(pm), pm.id);
                    }}
                  >
                    {copiedId === pm.id ? <Check className="w-3 h-3" /> : <Copy className="w-3 h-3" />}
                  </Button>
                  {expandedPainMoments.includes(pm.id) ? (
                    <ChevronUp className="w-5 h-5 text-muted-foreground" />
                  ) : (
                    <ChevronDown className="w-5 h-5 text-muted-foreground" />
                  )}
                </div>
              </div>

              {expandedPainMoments.includes(pm.id) && (
                <div className="px-4 pb-4">
                  <div className="font-mono text-sm space-y-2 bg-background/50 rounded-lg p-4 border border-border/30">
                    <p><span className="text-amber-500 font-semibold">Trigger:</span> {pm.trigger}</p>
                    <p><span className="text-blue-500 font-semibold">Current Behavior:</span> {pm.currentBehavior}</p>
                    <p><span className="text-destructive font-semibold">Why It Hurts:</span> {pm.whyItHurts}</p>
                  </div>
                </div>
              )}
            </div>
          ))}
        </div>
      </section>

      {/* Section 4: Root Cause Analysis */}
      <section className="space-y-4">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="w-8 h-8 rounded-lg bg-purple-500/10 flex items-center justify-center text-sm font-bold text-purple-500">4</div>
            <div>
              <h3 className="font-display font-bold text-lg">Root Cause Analysis</h3>
              <p className="text-xs text-muted-foreground">Prevent shallow solutions. Why this problem exists.</p>
            </div>
          </div>
          <Button
            variant="ghost"
            size="sm"
            className="gap-1.5 text-xs"
            onClick={() => copyToClipboard(formatRootCausesText(), "root-causes")}
          >
            {copiedId === "root-causes" ? <Check className="w-3.5 h-3.5" /> : <Copy className="w-3.5 h-3.5" />}
            Copy
          </Button>
        </div>

        <div className="bg-secondary/50 border border-border/50 rounded-xl p-5">
          <div className="font-mono text-sm space-y-3 bg-background/50 rounded-lg p-4 border border-border/30">
            <p className="font-semibold text-foreground mb-3">Root Causes:</p>
            {result.rootCauses.map((rc, i) => (
              <div key={rc.id} className="flex items-start gap-3">
                <span className="text-muted-foreground">{i + 1}.</span>
                <div className="flex items-start gap-2 flex-1">
                  <Badge variant="outline" className={cn("text-xs shrink-0 gap-1", getCategoryColor(rc.category))}>
                    {getCategoryIcon(rc.category)}
                    {rc.category} Gap
                  </Badge>
                  <span className="text-muted-foreground">—</span>
                  <span>{rc.description}</span>
                </div>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* Section 5: Existing Solutions & Why They Fail */}
      <section className="space-y-4">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="w-8 h-8 rounded-lg bg-amber-500/10 flex items-center justify-center text-sm font-bold text-amber-500">5</div>
            <div>
              <h3 className="font-display font-bold text-lg">Existing Solutions & Why They Fail</h3>
              <p className="text-xs text-muted-foreground">Competitive realism, not feature envy. Explicit gaps.</p>
            </div>
          </div>
          <Button
            variant="ghost"
            size="sm"
            className="gap-1.5 text-xs"
            onClick={() => copyToClipboard(formatExistingSolutionsText(), "solutions")}
          >
            {copiedId === "solutions" ? <Check className="w-3.5 h-3.5" /> : <Copy className="w-3.5 h-3.5" />}
            Copy
          </Button>
        </div>

        <div className="bg-secondary/50 border border-border/50 rounded-xl p-5">
          <div className="font-mono text-sm space-y-2 bg-background/50 rounded-lg p-4 border border-border/30">
            <p className="font-semibold text-foreground mb-3">Existing Solutions:</p>
            {result.existingSolutions.map((s, i) => (
              <div key={i} className="flex items-start gap-2">
                <XCircle className="w-4 h-4 text-destructive mt-0.5 shrink-0" />
                <span><span className="font-semibold">{s.name}:</span> <span className="text-muted-foreground">{s.gap}</span></span>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* Section 6: Problem Impact & Stakes */}
      <section className="space-y-4">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="w-8 h-8 rounded-lg bg-red-500/10 flex items-center justify-center text-sm font-bold text-red-500">6</div>
            <div>
              <h3 className="font-display font-bold text-lg">Problem Impact & Stakes</h3>
              <p className="text-xs text-muted-foreground">Force seriousness. Quantified or bounded estimates.</p>
            </div>
          </div>
          <Button
            variant="ghost"
            size="sm"
            className="gap-1.5 text-xs"
            onClick={() => copyToClipboard(formatImpactText(), "impact")}
          >
            {copiedId === "impact" ? <Check className="w-3.5 h-3.5" /> : <Copy className="w-3.5 h-3.5" />}
            Copy
          </Button>
        </div>

        <div className="bg-secondary/50 border border-border/50 rounded-xl p-5">
          <div className="font-mono text-sm space-y-2 bg-background/50 rounded-lg p-4 border border-border/30">
            <p className="font-semibold text-foreground mb-3">Impact:</p>
            <div className="space-y-2">
              <div className="flex items-start gap-3">
                <Clock className="w-4 h-4 text-blue-500 mt-0.5" />
                <span><span className="font-semibold">Time:</span> {result.problemImpact.time}</span>
              </div>
              <div className="flex items-start gap-3">
                <TrendingDown className="w-4 h-4 text-amber-500 mt-0.5" />
                <span><span className="font-semibold">Quality:</span> {result.problemImpact.quality}</span>
              </div>
              <div className="flex items-start gap-3">
                <Heart className="w-4 h-4 text-pink-500 mt-0.5" />
                <span><span className="font-semibold">Emotional:</span> {result.problemImpact.emotional}</span>
              </div>
              <div className="flex items-start gap-3">
                <DollarSign className="w-4 h-4 text-emerald-500 mt-0.5" />
                <span><span className="font-semibold">Opportunity:</span> {result.problemImpact.opportunity}</span>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* Section 7: Readiness & Risk Flags */}
      <section className="space-y-4">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="w-8 h-8 rounded-lg bg-orange-500/10 flex items-center justify-center text-sm font-bold text-orange-500">7</div>
            <div>
              <h3 className="font-display font-bold text-lg">Readiness & Risk Flags</h3>
              <p className="text-xs text-muted-foreground">BA-style honesty. What is unclear, what must be validated.</p>
            </div>
          </div>
          <Button
            variant="ghost"
            size="sm"
            className="gap-1.5 text-xs"
            onClick={() => copyToClipboard(formatRiskFlagsText(), "risks")}
          >
            {copiedId === "risks" ? <Check className="w-3.5 h-3.5" /> : <Copy className="w-3.5 h-3.5" />}
            Copy
          </Button>
        </div>

        <div className="bg-secondary/50 border border-border/50 rounded-xl p-5">
          <div className="font-mono text-sm space-y-2 bg-background/50 rounded-lg p-4 border border-border/30">
            <p className="font-semibold text-foreground mb-3">Key Risks:</p>
            {result.riskFlags.map((rf) => (
              <div key={rf.id} className="flex items-start gap-2">
                <AlertTriangle className={cn(
                  "w-4 h-4 mt-0.5 shrink-0",
                  rf.type === 'unclear' ? "text-amber-500" : "text-orange-500"
                )} />
                <span>{rf.description}</span>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* Section 8: Transition Questions to Phase 2 */}
      <section className="space-y-4">
        <div className="flex items-center gap-3">
          <div className="w-8 h-8 rounded-lg bg-emerald-500/10 flex items-center justify-center text-sm font-bold text-emerald-500">8</div>
          <div>
            <h3 className="font-display font-bold text-lg">Transition Questions to Phase 2</h3>
            <p className="text-xs text-muted-foreground">Smooth handoff to Requirements Analysis.</p>
          </div>
        </div>

        <div className="bg-secondary/50 border border-border/50 rounded-xl p-5">
          <ul className="space-y-3">
            {result.transitionQuestions.map((q, i) => (
              <li key={i} className="flex items-start gap-3 text-sm">
                <ArrowRight className="w-4 h-4 text-emerald-500 mt-0.5 shrink-0" />
                <span>{q}</span>
              </li>
            ))}
          </ul>
        </div>
      </section>
    </div>
  );
};
