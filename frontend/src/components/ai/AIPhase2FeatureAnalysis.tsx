import { useState, Fragment } from "react";
import {
  Copy, Check, ChevronDown, ChevronUp, ListChecks, Users, Target,
  Clock, Code, UserCog, Calendar, AlertTriangle, Sparkles, FileText,
  ArrowRight, Zap, Shield, Database, Globe, Gauge, CheckCircle2
} from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";

export interface NormalizedFeature {
  id: string;
  original: string;
  normalized: string;
  category: 'core' | 'enhancement' | 'nice-to-have';
  duplicateOf?: string;
}

export interface RequirementRow {
  id: string;
  requirement: string;
  type: 'functional' | 'non-functional';
  category: string;
  moscow: 'Must' | 'Should' | 'Could' | 'Won\'t';
  rationale: string;
  complexity: 'Low' | 'Medium' | 'High';
}

export interface Phase2FeatureResult {
  normalizedFeatures: NormalizedFeature[];
  requirements: RequirementRow[];
  mvpScope: string[];
  warnings: string[];
}

interface AIPhase2FeatureAnalysisProps {
  result: Phase2FeatureResult;
}

const categoryIcons: Record<string, React.ReactNode> = {
  core: <Zap className="w-4 h-4" />,
  enhancement: <Sparkles className="w-4 h-4" />,
  'nice-to-have': <Globe className="w-4 h-4" />,
};

const categoryColors: Record<string, string> = {
  core: 'bg-primary/20 text-primary border-primary/30',
  enhancement: 'bg-purple-500/20 text-purple-400 border-purple-500/30',
  'nice-to-have': 'bg-muted text-muted-foreground border-border',
};

const moscowColors: Record<string, string> = {
  'Must': 'bg-destructive/20 text-destructive border-destructive/30',
  'Should': 'bg-orange-500/20 text-orange-400 border-orange-500/30',
  'Could': 'bg-primary/20 text-primary border-primary/30',
  'Won\'t': 'bg-muted text-muted-foreground border-border',
};

const complexityColors: Record<string, string> = {
  'Low': 'text-green-400',
  'Medium': 'text-yellow-400',
  'High': 'text-destructive',
};

const reqCategoryIcons: Record<string, React.ReactNode> = {
  'Authentication': <Shield className="w-4 h-4" />,
  'Data': <Database className="w-4 h-4" />,
  'Performance': <Gauge className="w-4 h-4" />,
  'Security': <Shield className="w-4 h-4" />,
  'User Interface': <Globe className="w-4 h-4" />,
  'AI/ML': <Sparkles className="w-4 h-4" />,
  'Integration': <Zap className="w-4 h-4" />,
};

export const AIPhase2FeatureAnalysis = ({ result }: AIPhase2FeatureAnalysisProps) => {
  const [copiedId, setCopiedId] = useState<string | null>(null);
  const [expandedReq, setExpandedReq] = useState<string | null>(null);

  const copyToClipboard = (text: string, id: string) => {
    navigator.clipboard.writeText(text);
    setCopiedId(id);
    setTimeout(() => setCopiedId(null), 2000);
  };

  const functionalReqs = result.requirements.filter(r => r.type === 'functional');
  const nonFunctionalReqs = result.requirements.filter(r => r.type === 'non-functional');

  return (
    <div className="space-y-8 animate-fade-in">
      {/* Section 1: Normalized Feature List */}
      <div className="space-y-4">
        <div className="flex items-center gap-3">
          <div className="w-8 h-8 rounded-lg bg-primary/20 flex items-center justify-center">
            <ListChecks className="w-4 h-4 text-primary" />
          </div>
          <div>
            <h3 className="font-display font-bold text-lg">Normalized Feature List</h3>
            <p className="text-sm text-muted-foreground">Deduplicated, renamed in user-outcome language</p>
          </div>
        </div>

        <div className="rounded-xl border border-border overflow-hidden">
          <table className="w-full">
            <thead className="bg-muted/50">
              <tr>
                <th className="text-left text-xs font-medium text-muted-foreground uppercase tracking-wider px-4 py-3">Original</th>
                <th className="text-left text-xs font-medium text-muted-foreground uppercase tracking-wider px-4 py-3">Normalized (User-Outcome)</th>
                <th className="text-left text-xs font-medium text-muted-foreground uppercase tracking-wider px-4 py-3">Category</th>
                <th className="text-center text-xs font-medium text-muted-foreground uppercase tracking-wider px-4 py-3">Actions</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-border">
              {result.normalizedFeatures.map((feature) => (
                <tr key={feature.id} className={`${feature.duplicateOf ? 'bg-destructive/5' : 'hover:bg-muted/30'} transition-colors`}>
                  <td className="px-4 py-3">
                    <span className={`text-sm ${feature.duplicateOf ? 'line-through text-muted-foreground' : ''}`}>
                      {feature.original}
                    </span>
                    {feature.duplicateOf && (
                      <Badge variant="outline" className="ml-2 text-xs bg-destructive/10 text-destructive border-destructive/30">
                        Duplicate
                      </Badge>
                    )}
                  </td>
                  <td className="px-4 py-3">
                    <span className="text-sm font-medium">{feature.normalized}</span>
                  </td>
                  <td className="px-4 py-3">
                    <Badge variant="outline" className={`${categoryColors[feature.category]} flex items-center gap-1.5 w-fit`}>
                      {categoryIcons[feature.category]}
                      {feature.category}
                    </Badge>
                  </td>
                  <td className="px-4 py-3 text-center">
                    <Button
                      variant="ghost"
                      size="sm"
                      onClick={() => copyToClipboard(feature.normalized, feature.id)}
                      className="h-8 w-8 p-0"
                    >
                      {copiedId === feature.id ? (
                        <Check className="w-4 h-4 text-green-400" />
                      ) : (
                        <Copy className="w-4 h-4 text-muted-foreground" />
                      )}
                    </Button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {/* Section 2: Functional Requirements Table */}
      <div className="space-y-4">
        <div className="flex items-center gap-3">
          <div className="w-8 h-8 rounded-lg bg-green-500/20 flex items-center justify-center">
            <CheckCircle2 className="w-4 h-4 text-green-400" />
          </div>
          <div>
            <h3 className="font-display font-bold text-lg">Functional Requirements</h3>
            <p className="text-sm text-muted-foreground">What the system must do</p>
          </div>
        </div>

        <div className="rounded-xl border border-border overflow-hidden">
          <table className="w-full">
            <thead className="bg-muted/50">
              <tr>
                <th className="text-left text-xs font-medium text-muted-foreground uppercase tracking-wider px-4 py-3">Requirement</th>
                <th className="text-left text-xs font-medium text-muted-foreground uppercase tracking-wider px-4 py-3">Category</th>
                <th className="text-center text-xs font-medium text-muted-foreground uppercase tracking-wider px-4 py-3">MoSCoW</th>
                <th className="text-center text-xs font-medium text-muted-foreground uppercase tracking-wider px-4 py-3">Complexity</th>
                <th className="text-center text-xs font-medium text-muted-foreground uppercase tracking-wider px-4 py-3 w-10"></th>
              </tr>
            </thead>
            <tbody className="divide-y divide-border">
              {functionalReqs.map((req) => (
                <Fragment key={req.id}>
                  <tr className="hover:bg-muted/30 transition-colors cursor-pointer" onClick={() => setExpandedReq(expandedReq === req.id ? null : req.id)}>
                    <td className="px-4 py-3">
                      <span className="text-sm font-medium">{req.requirement}</span>
                    </td>
                    <td className="px-4 py-3">
                      <Badge variant="outline" className="flex items-center gap-1.5 w-fit">
                        {reqCategoryIcons[req.category] || <FileText className="w-4 h-4" />}
                        {req.category}
                      </Badge>
                    </td>
                    <td className="px-4 py-3 text-center">
                      <Badge variant="outline" className={moscowColors[req.moscow]}>
                        {req.moscow}
                      </Badge>
                    </td>
                    <td className="px-4 py-3 text-center">
                      <span className={`text-sm font-medium ${complexityColors[req.complexity]}`}>
                        {req.complexity}
                      </span>
                    </td>
                    <td className="px-4 py-3 text-center">
                      {expandedReq === req.id ? (
                        <ChevronUp className="w-4 h-4 text-muted-foreground" />
                      ) : (
                        <ChevronDown className="w-4 h-4 text-muted-foreground" />
                      )}
                    </td>
                  </tr>
                  {expandedReq === req.id && (
                    <tr className="bg-muted/20">
                      <td colSpan={5} className="px-4 py-3">
                        <div className="flex items-start gap-2 text-sm text-muted-foreground">
                          <AlertTriangle className="w-4 h-4 mt-0.5 text-primary" />
                          <span><strong>Rationale:</strong> {req.rationale}</span>
                        </div>
                      </td>
                    </tr>
                  )}
                </Fragment>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {/* Section 3: Non-Functional Requirements Table */}
      <div className="space-y-4">
        <div className="flex items-center gap-3">
          <div className="w-8 h-8 rounded-lg bg-purple-500/20 flex items-center justify-center">
            <Gauge className="w-4 h-4 text-purple-400" />
          </div>
          <div>
            <h3 className="font-display font-bold text-lg">Non-Functional Requirements</h3>
            <p className="text-sm text-muted-foreground">How the system should perform</p>
          </div>
        </div>

        <div className="rounded-xl border border-border overflow-hidden">
          <table className="w-full">
            <thead className="bg-muted/50">
              <tr>
                <th className="text-left text-xs font-medium text-muted-foreground uppercase tracking-wider px-4 py-3">Requirement</th>
                <th className="text-left text-xs font-medium text-muted-foreground uppercase tracking-wider px-4 py-3">Category</th>
                <th className="text-center text-xs font-medium text-muted-foreground uppercase tracking-wider px-4 py-3">MoSCoW</th>
                <th className="text-center text-xs font-medium text-muted-foreground uppercase tracking-wider px-4 py-3">Complexity</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-border">
              {nonFunctionalReqs.map((req) => (
                <tr key={req.id} className="hover:bg-muted/30 transition-colors">
                  <td className="px-4 py-3">
                    <span className="text-sm font-medium">{req.requirement}</span>
                  </td>
                  <td className="px-4 py-3">
                    <Badge variant="outline" className="flex items-center gap-1.5 w-fit">
                      {reqCategoryIcons[req.category] || <FileText className="w-4 h-4" />}
                      {req.category}
                    </Badge>
                  </td>
                  <td className="px-4 py-3 text-center">
                    <Badge variant="outline" className={moscowColors[req.moscow]}>
                      {req.moscow}
                    </Badge>
                  </td>
                  <td className="px-4 py-3 text-center">
                    <span className={`text-sm font-medium ${complexityColors[req.complexity]}`}>
                      {req.complexity}
                    </span>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {/* Section 4: MVP Scope Summary */}
      <div className="space-y-4">
        <div className="flex items-center gap-3">
          <div className="w-8 h-8 rounded-lg bg-orange-500/20 flex items-center justify-center">
            <Target className="w-4 h-4 text-orange-400" />
          </div>
          <div>
            <h3 className="font-display font-bold text-lg">MVP Scope Summary</h3>
            <p className="text-sm text-muted-foreground">Minimum viable features for launch</p>
          </div>
        </div>

        <div className="p-4 rounded-xl bg-gradient-to-br from-primary/10 via-background to-purple-500/10 border border-primary/20">
          <ul className="space-y-2">
            {result.mvpScope.map((item, index) => (
              <li key={index} className="flex items-start gap-3">
                <CheckCircle2 className="w-5 h-5 text-primary mt-0.5 shrink-0" />
                <span className="text-sm">{item}</span>
              </li>
            ))}
          </ul>
        </div>
      </div>

      {/* Section 5: Warnings */}
      {result.warnings.length > 0 && (
        <div className="space-y-4">
          <div className="flex items-center gap-3">
            <div className="w-8 h-8 rounded-lg bg-destructive/20 flex items-center justify-center">
              <AlertTriangle className="w-4 h-4 text-destructive" />
            </div>
            <div>
              <h3 className="font-display font-bold text-lg">Scope Warnings</h3>
              <p className="text-sm text-muted-foreground">Potential risks and concerns</p>
            </div>
          </div>

          <div className="space-y-2">
            {result.warnings.map((warning, index) => (
              <div key={index} className="p-3 rounded-lg bg-destructive/10 border border-destructive/20 flex items-start gap-3">
                <AlertTriangle className="w-4 h-4 text-destructive mt-0.5 shrink-0" />
                <span className="text-sm">{warning}</span>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
};
