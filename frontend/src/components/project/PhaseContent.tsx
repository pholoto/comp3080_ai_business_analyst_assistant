import { AIFeedbackPanel } from "@/components/ai/AIFeedbackPanel";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Switch } from "@/components/ui/switch";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Textarea } from "@/components/ui/textarea";
import { useToast } from "@/hooks/use-toast";
import {
  analyzePhase1,
  analyzePhase2Features,
  analyzePhase3Market,
  APIError,
  generatePhase2UserJourney,
  researchCompetitorsApi,
  generatePhase7Documentation
} from "@/lib/api";
import {
  transformPhase1Response,
  transformPhase2FeatureResponse,
  transformPhase2JourneyResponse,
  transformPhase3Response,
} from "@/lib/apiTransformers";
import { AIAnalysisResult, mockAIResponses } from "@/lib/mockAIData";
import {
  AlertCircle,
  AlertTriangle,
  ArrowRight,
  BarChart3,
  Building2,
  CheckCircle2,
  ChevronLeft,
  Code,
  Cpu,
  Database,
  Download,
  ExternalLink,
  Eye,
  FileCheck,
  FileText,
  Globe,
  GripVertical,
  Layers,
  Layout,
  Lightbulb,
  ListChecks,
  Loader2,
  Monitor,
  Plus,
  Rocket,
  Search,
  Server,
  Sparkles,
  Tag,
  Target,
  TrendingUp,
  Users,
  X,
  Zap
} from "lucide-react";
import { useEffect, useState } from "react";

interface PhaseContentProps {
  phaseId: number;
  onComplete: () => void;
  isCompleted: boolean;
  projectId?: string;
}

// Storage key helper
const getStorageKey = (projectId: string | undefined, key: string) =>
  projectId ? `aiba-project-${projectId}-${key}` : null;

// Load from localStorage
const loadFromStorage = <T,>(projectId: string | undefined, key: string, defaultValue: T): T => {
  const storageKey = getStorageKey(projectId, key);
  if (!storageKey) return defaultValue;
  try {
    const stored = localStorage.getItem(storageKey);
    return stored ? JSON.parse(stored) : defaultValue;
  } catch {
    return defaultValue;
  }
};

// Save to localStorage
const saveToStorage = <T,>(projectId: string | undefined, key: string, value: T) => {
  const storageKey = getStorageKey(projectId, key);
  if (!storageKey) return;
  try {
    localStorage.setItem(storageKey, JSON.stringify(value));
  } catch {
    console.error("Failed to save to localStorage");
  }
};

const PhaseContent = ({ phaseId, onComplete, isCompleted, projectId }: PhaseContentProps) => {
  // All hooks must be called at the top level unconditionally
  const { toast } = useToast();

  const CompleteButton = ({ label = "Complete Phase & Continue" }: { label?: string }) => (
    <Button
      onClick={onComplete}
      className="aiba-button-primary gap-2 py-3"
      disabled={isCompleted}
    >
      <CheckCircle2 className="w-5 h-5" />
      {label}
    </Button>
  );

  // Phase 1: Problem Definition
  const [problemData, setProblemData] = useState(() => loadFromStorage(projectId, "problemData", {
    description: "",
    targetUsers: "",
    whyItMatters: "",
    currentSolutions: "",
    hasExistingSolution: false
  }));
  const [painPoints, setPainPoints] = useState<string[]>(() => loadFromStorage(projectId, "painPoints", []));
  const [newPainPoint, setNewPainPoint] = useState("");

  // AI feedback state for each phase
  const [phase1Result, setPhase1Result] = useState<AIAnalysisResult | null>(() => loadFromStorage(projectId, "phase1Result", null));
  const [isAnalyzing, setIsAnalyzing] = useState(false);

  // Phase 2: Requirements Analysis
  const [functionalReqs, setFunctionalReqs] = useState<string[]>(() => loadFromStorage(projectId, "functionalReqs", []));
  const [nonFunctionalReqs, setNonFunctionalReqs] = useState<string[]>(() => loadFromStorage(projectId, "nonFunctionalReqs", []));
  const [newReq, setNewReq] = useState("");
  // Phase 2 Results Persistence
  const [featureResult, setFeatureResult] = useState<AIAnalysisResult | null>(() => loadFromStorage(projectId, "featureResult", null));
  const [journeyResult, setJourneyResult] = useState<AIAnalysisResult | null>(() => loadFromStorage(projectId, "journeyResult", null));

  // Phase 3: Market Analysis
  // Phase 3 uses a separate result state
  const [phase3Result, setPhase3Result] = useState<AIAnalysisResult | null>(() => loadFromStorage(projectId, "phase3Result", null));
  const [competitors, setCompetitors] = useState<{ name: string; strengths: string; weaknesses: string }[]>(() =>
    loadFromStorage(projectId, "competitors", [])
  );
  const [uspIdeas, setUspIdeas] = useState<string[]>(() => loadFromStorage(projectId, "uspIdeas", []));

  // Phase 4: Solution Design
  const [phase4Tab, setPhase4Tab] = useState<'tech-stack' | 'wireframe'>('tech-stack');

  // Tech Stack Analyzer state
  const [techStackInputs, setTechStackInputs] = useState(() => loadFromStorage(projectId, "techStackInputs", {
    frontend: "",
    backend: "",
    database: "",
    aiml: ""
  }));
  const [techStackResult, setTechStackResult] = useState<any>(() => loadFromStorage(projectId, "techStackResult", null));
  const [isTechStackAnalyzing, setIsTechStackAnalyzing] = useState(false);

  // Wireframe Sandbox state
  const [screensList, setScreensList] = useState(() => loadFromStorage(projectId, "screensList", ""));
  const [wireframeResult, setWireframeResult] = useState<any>(() => loadFromStorage(projectId, "wireframeResult", null));
  const [isWireframeAnalyzing, setIsWireframeAnalyzing] = useState(false);

  // Phase 5: Prototype Development
  const [phase5Tab, setPhase5Tab] = useState<'task-breakdown' | 'sprint-planning'>('task-breakdown');

  // Task Breakdown state
  const [taskBreakdownInputs, setTaskBreakdownInputs] = useState(() => loadFromStorage(projectId, "taskBreakdownInputs", {
    projectScope: "",
    complexity: "",

    priorityFeatures: ""
  }));
  const [taskBreakdownResult, setTaskBreakdownResult] = useState<any>(() => loadFromStorage(projectId, "taskBreakdownResult", null));
  const [isTaskBreakdownAnalyzing, setIsTaskBreakdownAnalyzing] = useState(false);

  // Sprint Planning state
  const [sprintInputs, setSprintInputs] = useState(() => loadFromStorage(projectId, "sprintInputs", {
    totalWeeks: "",
    teamSize: "",
    workHoursPerDay: "",
    selectedMilestones: [] as string[]
  }));
  const [sprintResult, setSprintResult] = useState<any>(() => loadFromStorage(projectId, "sprintResult", null));
  const [isSprintPlanning, setIsSprintPlanning] = useState(false);

  // Phase 7: Documentation
  const [docType, setDocType] = useState(() => loadFromStorage(projectId, "docType", ""));

  // Save to localStorage whenever state changes
  useEffect(() => { saveToStorage(projectId, "problemData", problemData); }, [projectId, problemData]);
  useEffect(() => { saveToStorage(projectId, "painPoints", painPoints); }, [projectId, painPoints]);
  useEffect(() => { saveToStorage(projectId, "functionalReqs", functionalReqs); }, [projectId, functionalReqs]);
  useEffect(() => { saveToStorage(projectId, "nonFunctionalReqs", nonFunctionalReqs); }, [projectId, nonFunctionalReqs]);
  useEffect(() => { saveToStorage(projectId, "competitors", competitors); }, [projectId, competitors]);
  useEffect(() => { saveToStorage(projectId, "uspIdeas", uspIdeas); }, [projectId, uspIdeas]);
  useEffect(() => { saveToStorage(projectId, "techStackInputs", techStackInputs); }, [projectId, techStackInputs]);
  useEffect(() => { saveToStorage(projectId, "screensList", screensList); }, [projectId, screensList]);
  useEffect(() => { saveToStorage(projectId, "taskBreakdownInputs", taskBreakdownInputs); }, [projectId, taskBreakdownInputs]);
  useEffect(() => { saveToStorage(projectId, "sprintInputs", sprintInputs); }, [projectId, sprintInputs]);
  useEffect(() => { saveToStorage(projectId, "docType", docType); }, [projectId, docType]);

  // Save AI Results
  useEffect(() => { saveToStorage(projectId, "phase1Result", phase1Result); }, [projectId, phase1Result]);
  useEffect(() => { saveToStorage(projectId, "phase3Result", phase3Result); }, [projectId, phase3Result]);
  useEffect(() => { saveToStorage(projectId, "featureResult", featureResult); }, [projectId, featureResult]);
  useEffect(() => { saveToStorage(projectId, "journeyResult", journeyResult); }, [projectId, journeyResult]);
  useEffect(() => { saveToStorage(projectId, "techStackResult", techStackResult); }, [projectId, techStackResult]);
  useEffect(() => { saveToStorage(projectId, "wireframeResult", wireframeResult); }, [projectId, wireframeResult]);
  useEffect(() => { saveToStorage(projectId, "taskBreakdownResult", taskBreakdownResult); }, [projectId, taskBreakdownResult]);
  useEffect(() => { saveToStorage(projectId, "sprintResult", sprintResult); }, [projectId, sprintResult]);

  const addPainPoint = () => {
    if (newPainPoint.trim()) {
      setPainPoints([...painPoints, newPainPoint.trim()]);
      setNewPainPoint("");
    }
  };

  const removePainPoint = (index: number) => {
    setPainPoints(painPoints.filter((_, i) => i !== index));
  };

  const analyzeWithAI = async (phase: number) => {
    setIsAnalyzing(true);
    // Remove global reset of aiResult, handle per phase
    if (phase === 1) setPhase1Result(null);
    if (phase === 3) setPhase3Result(null);

    // Use projectId as userId for API calls, fallback to a default
    const userId = projectId || 'default-user';

    try {
      if (phase === 1) {
        // Phase 1: Problem Definition Validation
        if (problemData.description.length < 10) {
          throw new Error("Problem description must be at least 10 characters long.");
        }
        if (problemData.targetUsers.length < 5) {
          throw new Error("Target users description must be at least 5 characters long.");
        }
        if (problemData.whyItMatters.length < 10) {
          throw new Error("Why it matters description must be at least 10 characters long.");
        }

        // Phase 1: Problem Definition
        const response = await analyzePhase1(userId, {
          problem_description: problemData.description,
          target_users: problemData.targetUsers,
          why_it_matters: problemData.whyItMatters,
          pain_points: painPoints.length > 0 ? painPoints : undefined,
          has_existing_solutions: problemData.hasExistingSolution,
          current_solutions: problemData.hasExistingSolution ? problemData.currentSolutions : null,
        });
        const transformedResult = transformPhase1Response(response);
        setPhase1Result(transformedResult);
      } else if (phase === 3) {
        // Phase 3: Market Analysis
        const inputCompetitors = knownCompetitors.trim()
          ? knownCompetitors.split(/[,\n]/).map(c => c.trim()).filter(c => c)
          : [];

        const researchedNames = researchedCompetitors.map(c => c.name);
        const combinedCompetitors = Array.from(new Set([...inputCompetitors, ...researchedNames]));

        const competitorList = combinedCompetitors.length > 0 ? combinedCompetitors : undefined;

        // Map geographic scope to API format
        const geoScopeMap: Record<string, 'local' | 'regional' | 'national' | 'international' | 'global'> = {
          'vietnam': 'local',
          'sea': 'regional',
          'apac': 'international',
          'global': 'global',
        };

        const response = await analyzePhase3Market(userId, {
          geographic_scope: geographicScope ? geoScopeMap[geographicScope] || null : null,
          industry_context: industryContext || null,
          competitors: competitorList,
        });
        const transformedResult = transformPhase3Response(response);
        setPhase3Result(transformedResult);
      } else {
        // For other phases, fall back to mock data for now
        if (phase === 1) setPhase1Result(mockAIResponses[1]);
        if (phase === 3) setPhase3Result(mockAIResponses[3] || mockAIResponses[1]);
      }
    } catch (error) {
      console.error(`Phase ${phase} analysis failed:`, error);
      const errorMessage = error instanceof APIError
        ? error.message
        : 'Analysis failed. Please try again.';
      toast({
        title: "Analysis Failed",
        description: errorMessage,
        variant: "destructive",
      });
      // Fall back to mock data on error
      if (phase === 1) setPhase1Result(mockAIResponses[phase] || mockAIResponses[1]);
      if (phase === 3) setPhase3Result(mockAIResponses[phase] || mockAIResponses[1]);
    } finally {
      setIsAnalyzing(false);
    }
  };

  const renderPhase1 = () => (
    <div className="space-y-8">
      <div className="flex items-start justify-between gap-4 mb-8">
        <div>
          <h2 className="text-3xl font-display font-bold mb-3">Problem Definition Canvas</h2>
          <p className="text-muted-foreground max-w-2xl">
            Let's clearly define the problem you're solving. Fill out this guided form, and our AI will help you refine your problem statement, identify personas, and cluster pain points.
          </p>
        </div>
        <CompleteButton />
      </div>

      <div className="glass-card rounded-xl p-6 space-y-6">
        {/* Problem Description */}
        <div className="space-y-3">
          <Label className="text-lg font-display font-semibold flex items-center gap-2">
            <Target className="w-5 h-5 text-primary" />
            Problem Description *
          </Label>
          <p className="text-sm text-muted-foreground">
            Describe the problem in your own words. What challenge or pain point are you addressing?
          </p>
          <Textarea
            placeholder="Example: Students struggle to manage their engineering project workflow and often miss important milestones..."
            value={problemData.description}
            onChange={(e) => setProblemData({ ...problemData, description: e.target.value })}
            className="min-h-[120px]"
          />
        </div>

        {/* Target Users */}
        <div className="space-y-3">
          <Label className="text-lg font-display font-semibold flex items-center gap-2">
            <Users className="w-5 h-5 text-primary" />
            Target Users *
          </Label>
          <p className="text-sm text-muted-foreground">
            Who experiences this problem? Be specific about the user group.
          </p>
          <Input
            placeholder="Example: Engineering students working on capstone projects"
            value={problemData.targetUsers}
            onChange={(e) => setProblemData({ ...problemData, targetUsers: e.target.value })}
          />
        </div>

        {/* Why It Matters */}
        <div className="space-y-3">
          <Label className="text-lg font-display font-semibold flex items-center gap-2">
            <Zap className="w-5 h-5 text-primary" />
            Why It Matters *
          </Label>
          <p className="text-sm text-muted-foreground">
            What's the impact? Why should this problem be solved?
          </p>
          <Textarea
            placeholder="Example: Poor project management leads to lower grades, missed deadlines, and increased stress..."
            value={problemData.whyItMatters}
            onChange={(e) => setProblemData({ ...problemData, whyItMatters: e.target.value })}
            className="min-h-[100px]"
          />
        </div>

        {/* Pain Points Tags */}
        <div className="space-y-3">
          <Label className="text-lg font-display font-semibold flex items-center gap-2">
            <Tag className="w-5 h-5 text-primary" />
            Pain Points
          </Label>
          <p className="text-sm text-muted-foreground">
            Add specific pain points your users experience. Press Enter or click Add.
          </p>
          <div className="flex gap-2">
            <Input
              placeholder="e.g., Difficulty tracking progress"
              value={newPainPoint}
              onChange={(e) => setNewPainPoint(e.target.value)}
              onKeyDown={(e) => e.key === "Enter" && addPainPoint()}
            />
            <Button onClick={addPainPoint} variant="secondary">
              <Plus className="w-4 h-4" />
            </Button>
          </div>
          <div className="flex flex-wrap gap-2 mt-3">
            {painPoints.map((point, index) => (
              <Badge key={index} variant="secondary" className="px-3 py-1.5 gap-2">
                {point}
                <button onClick={() => removePainPoint(index)} className="hover:text-destructive">
                  <X className="w-3 h-3" />
                </button>
              </Badge>
            ))}
          </div>
        </div>

        {/* Existing Solutions Toggle */}
        <div className="flex items-center justify-between p-4 rounded-xl bg-secondary">
          <div className="space-y-1">
            <Label className="font-display font-semibold">Existing Solutions?</Label>
            <p className="text-sm text-muted-foreground">Are there current solutions users are using?</p>
          </div>
          <Switch
            checked={problemData.hasExistingSolution}
            onCheckedChange={(checked) => setProblemData({ ...problemData, hasExistingSolution: checked })}
          />
        </div>

        {problemData.hasExistingSolution && (
          <div className="space-y-3 animate-fade-in">
            <Label className="font-display font-semibold">Current Solutions Used</Label>
            <Textarea
              placeholder="What solutions are users currently using? What are their limitations?"
              value={problemData.currentSolutions}
              onChange={(e) => setProblemData({ ...problemData, currentSolutions: e.target.value })}
            />
          </div>
        )}
      </div>

      {/* AI Analysis Section */}
      <AIFeedbackPanel
        isLoading={isAnalyzing}
        result={phase1Result}
        onAnalyze={() => analyzeWithAI(1)}
        title="AI Problem Analysis"
        description="Critical BA-style analysis of your problem definition"
        disabled={!problemData.description}
        phaseId={1}
      />
    </div>
  );

  // Phase 2 State
  const [phase2Tab, setPhase2Tab] = useState<'feature-analyzer' | 'user-journey'>('feature-analyzer');
  const [featureInputs, setFeatureInputs] = useState(() => loadFromStorage(projectId, "featureInputs", {
    featureList: "",
    primaryPersona: "High school students preparing university applications independently",
    mvpGoal: "",
    constraints: "",
    deadline: "",
    skillLevel: ""
  }));
  const [journeyFeature, setJourneyFeature] = useState(() => loadFromStorage(projectId, "journeyFeature", ""));
  // These are now defined above with persistence
  // const [featureResult, setFeatureResult] = useState<AIAnalysisResult | null>(null);
  // const [journeyResult, setJourneyResult] = useState<AIAnalysisResult | null>(null);
  const [isFeatureAnalyzing, setIsFeatureAnalyzing] = useState(false);
  const [isJourneyAnalyzing, setIsJourneyAnalyzing] = useState(false);

  // Persist Phase 2 inputs
  useEffect(() => { saveToStorage(projectId, "featureInputs", featureInputs); }, [projectId, featureInputs]);
  useEffect(() => { saveToStorage(projectId, "journeyFeature", journeyFeature); }, [projectId, journeyFeature]);

  const analyzeFeatures = async () => {
    setIsFeatureAnalyzing(true);
    setFeatureResult(null);

    const userId = projectId || 'default-user';

    try {
      // Parse feature list from text input
      const features = featureInputs.featureList
        .split(/[,\n]/)
        .map(f => f.trim())
        .filter(f => f && !f.startsWith('•'))
        .map(f => f.replace(/^[•\-\*]\s*/, '')); // Remove bullet points

      // Map skill level to API format
      const skillLevelMap: Record<string, 'solo-developer' | 'junior-team' | 'senior-team' | 'mixed-experience'> = {
        'solo': 'solo-developer',
        'junior': 'junior-team',
        'senior': 'senior-team',
        'mixed': 'mixed-experience',
      };

      const response = await analyzePhase2Features(userId, {
        desired_features: features,
        mvp_goal: featureInputs.mvpGoal,
        deadline: featureInputs.deadline || null,
        team_skill_level: featureInputs.skillLevel ? skillLevelMap[featureInputs.skillLevel] || null : null,
        additional_constraints: featureInputs.constraints || null,
      });

      const transformedResult = transformPhase2FeatureResponse(response);
      setFeatureResult(transformedResult);
    } catch (error) {
      console.error('Feature analysis failed:', error);
      const errorMessage = error instanceof APIError
        ? error.message
        : 'Feature analysis failed. Please try again.';
      toast({
        title: "Analysis Failed",
        description: errorMessage,
        variant: "destructive",
      });
      // Fall back to mock data on error
      setFeatureResult(mockAIResponses[2] || null);
    } finally {
      setIsFeatureAnalyzing(false);
    }
  };

  const analyzeJourney = async () => {
    setIsJourneyAnalyzing(true);
    setJourneyResult(null);

    const userId = projectId || 'default-user';

    try {
      // Map journey feature selection to a feature name
      const featureNameMap: Record<string, string> = {
        'ai-feedback': 'AI Essay Feedback',
        'progress-tracker': 'Progress Tracker Dashboard',
        'cv-builder': 'CV Builder',
        'brainstorm': 'Brainstorming Prompts',
        'sample-essays': 'Sample Essays Library',
      };

      const selectedFeatureName = featureNameMap[journeyFeature] || journeyFeature;

      const response = await generatePhase2UserJourney(userId, {
        selected_feature: selectedFeatureName,
      });

      const transformedResult = transformPhase2JourneyResponse(response);
      setJourneyResult(transformedResult);
    } catch (error) {
      console.error('User journey generation failed:', error);
      const errorMessage = error instanceof APIError
        ? error.message
        : 'User journey generation failed. Please try again.';
      toast({
        title: "Analysis Failed",
        description: errorMessage,
        variant: "destructive",
      });
      // Fall back to mock data on error
      setJourneyResult(mockAIResponses[2] || null);
    } finally {
      setIsJourneyAnalyzing(false);
    }
  };

  const renderPhase2 = () => (
    <div className="space-y-8">
      <div className="flex items-start justify-between gap-4 mb-8">
        <div>
          <h2 className="text-3xl font-display font-bold mb-3">Requirements Analysis</h2>
          <p className="text-muted-foreground max-w-2xl">
            Define features, generate requirements, and map user journeys with AI assistance.
          </p>
        </div>
        <CompleteButton />
      </div>

      <Tabs value={phase2Tab} onValueChange={(v) => setPhase2Tab(v as 'feature-analyzer' | 'user-journey')} className="w-full">
        <TabsList className="grid w-full grid-cols-2">
          <TabsTrigger value="feature-analyzer" className="gap-2">
            <ListChecks className="w-4 h-4" />
            Feature Analyzer
          </TabsTrigger>
          <TabsTrigger value="user-journey" className="gap-2">
            <Users className="w-4 h-4" />
            User Journey Generator
          </TabsTrigger>
        </TabsList>

        {/* Feature Analyzer Tab */}
        <TabsContent value="feature-analyzer" className="space-y-6 mt-6">
          <div className="glass-card rounded-xl p-6 space-y-6">
            <div>
              <h3 className="text-lg font-display font-semibold mb-1">Feature Analyzer Inputs</h3>
              <p className="text-sm text-muted-foreground">AI will normalize, deduplicate, and prioritize your features</p>
            </div>

            {/* Required Inputs */}
            <div className="space-y-4">
              <Badge variant="secondary" className="gap-1.5">
                <AlertTriangle className="w-3 h-3" />
                Required
              </Badge>

              {/* Feature List */}
              <div className="space-y-2">
                <Label className="font-display font-semibold flex items-center gap-2">
                  <Layers className="w-4 h-4 text-primary" />
                  List of Desired Features *
                </Label>
                <Textarea
                  placeholder="Enter features (one per line or comma-separated):
• AI essay feedback
• Progress tracker
• CV builder
• Sample essays library
• Deadline reminders"
                  value={featureInputs.featureList}
                  onChange={(e) => setFeatureInputs({ ...featureInputs, featureList: e.target.value })}
                  className="min-h-[140px]"
                />
              </div>

              {/* Primary Persona (locked from Phase 1) */}
              <div className="space-y-2">
                <Label className="font-display font-semibold flex items-center gap-2">
                  <Users className="w-4 h-4 text-primary" />
                  Primary User Persona
                  <Badge variant="outline" className="ml-2 text-xs">From Phase 1</Badge>
                </Label>
                <div className="p-3 rounded-lg bg-muted/50 border border-border flex items-center gap-3">
                  <Users className="w-5 h-5 text-muted-foreground" />
                  <span className="text-sm">{featureInputs.primaryPersona}</span>
                </div>
              </div>

              {/* MVP Goal */}
              <div className="space-y-2">
                <Label className="font-display font-semibold flex items-center gap-2">
                  <Target className="w-4 h-4 text-primary" />
                  MVP Goal (1 sentence) *
                </Label>
                <Input
                  placeholder="e.g., Help students submit their first polished essay draft within 48 hours"
                  value={featureInputs.mvpGoal}
                  onChange={(e) => setFeatureInputs({ ...featureInputs, mvpGoal: e.target.value })}
                />
              </div>
            </div>

            {/* Optional Inputs */}
            <div className="space-y-4">
              <Badge variant="outline" className="gap-1.5">
                <Lightbulb className="w-3 h-3" />
                Optional
              </Badge>

              <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                {/* Constraints */}
                <div className="space-y-2">
                  <Label className="text-sm flex items-center gap-2">
                    <Code className="w-4 h-4 text-muted-foreground" />
                    Constraints
                  </Label>
                  <Input
                    placeholder="e.g., React, 2 developers"
                    value={featureInputs.constraints}
                    onChange={(e) => setFeatureInputs({ ...featureInputs, constraints: e.target.value })}
                  />
                </div>

                {/* Deadline */}
                <div className="space-y-2">
                  <Label className="text-sm flex items-center gap-2">
                    <TrendingUp className="w-4 h-4 text-muted-foreground" />
                    Deadline / Timeline
                  </Label>
                  <Input
                    placeholder="e.g., 8 weeks, Dec 2024"
                    value={featureInputs.deadline}
                    onChange={(e) => setFeatureInputs({ ...featureInputs, deadline: e.target.value })}
                  />
                </div>

                {/* Skill Level */}
                <div className="space-y-2">
                  <Label className="text-sm flex items-center gap-2">
                    <Zap className="w-4 h-4 text-muted-foreground" />
                    Team Skill Level
                  </Label>
                  <Select
                    value={featureInputs.skillLevel}
                    onValueChange={(v) => setFeatureInputs({ ...featureInputs, skillLevel: v })}
                  >
                    <SelectTrigger>
                      <SelectValue placeholder="Select level" />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="solo">Solo Developer</SelectItem>
                      <SelectItem value="junior">Junior Team</SelectItem>
                      <SelectItem value="mixed">Mixed Experience</SelectItem>
                      <SelectItem value="senior">Senior Team</SelectItem>
                    </SelectContent>
                  </Select>
                </div>
              </div>
            </div>
          </div>

          {/* AI Analysis Section */}
          <AIFeedbackPanel
            isLoading={isFeatureAnalyzing}
            result={featureResult}
            onAnalyze={analyzeFeatures}
            title="Feature Analysis"
            description="Normalized features, requirements table, and MoSCoW prioritization"
            disabled={!featureInputs.featureList || !featureInputs.mvpGoal}
            phaseId={2}
            subFunction="feature-analyzer"
          />
        </TabsContent>

        {/* User Journey Tab */}
        <TabsContent value="user-journey" className="space-y-6 mt-6">
          <div className="glass-card rounded-xl p-6 space-y-6">
            <div>
              <h3 className="text-lg font-display font-semibold mb-1">User Journey Generator</h3>
              <p className="text-sm text-muted-foreground">Select a feature to generate a detailed step-by-step user flow</p>
            </div>

            <div className="space-y-4">
              <Label className="font-display font-semibold flex items-center gap-2">
                <Sparkles className="w-4 h-4 text-primary" />
                Main Feature to Map *
              </Label>
              <Select
                value={journeyFeature}
                onValueChange={setJourneyFeature}
              >
                <SelectTrigger className="w-full">
                  <SelectValue placeholder="Select a feature from your list" />
                </SelectTrigger>
                <SelectContent>
                  {featureResult?.phase2Details?.featureAnalysis?.normalizedFeatures?.length &&
                    featureResult.phase2Details.featureAnalysis.normalizedFeatures.length > 0 ? (
                    featureResult.phase2Details.featureAnalysis.normalizedFeatures.map((f) => (
                      <SelectItem key={f.id} value={f.normalized}>
                        {f.normalized}
                      </SelectItem>
                    ))
                  ) : (
                    <>
                      <SelectItem value="ai-feedback">AI Essay Feedback</SelectItem>
                      <SelectItem value="progress-tracker">Progress Tracker Dashboard</SelectItem>
                      <SelectItem value="cv-builder">CV Builder</SelectItem>
                      <SelectItem value="brainstorm">Brainstorming Prompts</SelectItem>
                      <SelectItem value="sample-essays">Sample Essays Library</SelectItem>
                    </>
                  )}
                </SelectContent>
              </Select>

              {journeyFeature && (
                <div className="p-4 rounded-xl bg-gradient-to-r from-primary/10 via-background to-purple-500/10 border border-primary/20">
                  <div className="flex items-center gap-3">
                    <div className="w-10 h-10 rounded-lg bg-primary/20 flex items-center justify-center">
                      <Sparkles className="w-5 h-5 text-primary" />
                    </div>
                    <div>
                      <p className="font-semibold">Ready to generate journey</p>
                      <p className="text-sm text-muted-foreground">
                        AI will create a step-by-step flow with UI elements, emotional states, and edge cases
                      </p>
                    </div>
                  </div>
                </div>
              )}
            </div>
          </div>

          {/* AI Analysis Section */}
          <AIFeedbackPanel
            isLoading={isJourneyAnalyzing}
            result={journeyResult}
            onAnalyze={analyzeJourney}
            title="User Journey Mapping"
            description="Complete user flow with UI elements, emotions, and accessibility notes"
            disabled={!journeyFeature}
            phaseId={2}
            subFunction="user-journey"
          />
        </TabsContent>
      </Tabs>
    </div>
  );

  // Phase 3 state
  const [geographicScope, setGeographicScope] = useState(() => loadFromStorage(projectId, "geographicScope", ""));
  const [industryContext, setIndustryContext] = useState(() => loadFromStorage(projectId, "industryContext", ""));
  const [knownCompetitors, setKnownCompetitors] = useState(() => loadFromStorage(projectId, "knownCompetitors", ""));
  const [isResearchingCompetitors, setIsResearchingCompetitors] = useState(false);
  const [researchedCompetitors, setResearchedCompetitors] = useState<{
    name: string;
    description: string;
    model: string;
    targetCustomers: string;
  }[]>(() => loadFromStorage(projectId, "researchedCompetitors", []));

  // Persist Phase 3 inputs
  useEffect(() => { saveToStorage(projectId, "geographicScope", geographicScope); }, [projectId, geographicScope]);
  useEffect(() => { saveToStorage(projectId, "industryContext", industryContext); }, [projectId, industryContext]);
  useEffect(() => { saveToStorage(projectId, "knownCompetitors", knownCompetitors); }, [projectId, knownCompetitors]);
  useEffect(() => { saveToStorage(projectId, "researchedCompetitors", researchedCompetitors); }, [projectId, researchedCompetitors]);

  const researchCompetitors = async () => {
    setIsResearchingCompetitors(true);
    const userId = projectId || 'default-user';

    try {
      const response = await researchCompetitorsApi(userId, {
        industry: industryContext,
        geographic_scope: geographicScope,
        known_competitors: knownCompetitors
      });

      setResearchedCompetitors(response.competitors.map(c => ({
        name: c.name,
        description: c.description,
        model: c.business_model,
        targetCustomers: c.target_customers
      })));

      // Add new names to the input field if they don't already exist
      setKnownCompetitors(prev => {
        const existingNames = new Set(prev.split(/[,\n]/).map(n => n.trim().toLowerCase()).filter(n => n));
        const newNames = response.competitors
          .map(c => c.name)
          .filter(name => !existingNames.has(name.toLowerCase()));

        if (newNames.length === 0) return prev;

        const combined = prev.trim()
          ? prev.trim() + (prev.includes('\n') ? '\n' : ', ') + newNames.join(', ')
          : newNames.join(', ');
        return combined;
      });

      toast({
        title: "Research Complete",
        description: `Found ${response.competitors.length} relevant competitors.`,
      });
    } catch (error) {
      console.error('Competitor research failed:', error);
      toast({
        title: "Research Failed",
        description: error instanceof APIError ? error.message : "Failed to research competitors",
        variant: "destructive",
      });
    } finally {
      setIsResearchingCompetitors(false);
    }
  };

  const renderPhase3 = () => (
    <div className="space-y-8">
      <div className="flex items-start justify-between gap-4 mb-8">
        <div>
          <h2 className="text-3xl font-display font-bold mb-3">Market Analysis</h2>
          <p className="text-muted-foreground max-w-2xl">
            Research your market, analyze competitors, and identify unique opportunities for your project.
          </p>
        </div>
        <CompleteButton />
      </div>

      {/* Input Section */}
      <div className="glass-card rounded-xl p-6 space-y-6">
        <div className="flex items-center gap-3 mb-2">
          <Globe className="w-5 h-5 text-primary" />
          <h3 className="text-lg font-display font-semibold">Market Research Inputs</h3>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          <div className="space-y-2">
            <Label htmlFor="geographic-scope" className="font-medium">Geographic Scope</Label>
            <Select value={geographicScope} onValueChange={setGeographicScope}>
              <SelectTrigger id="geographic-scope">
                <SelectValue placeholder="Select target market region" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="vietnam">Vietnam</SelectItem>
                <SelectItem value="sea">Southeast Asia (SEA)</SelectItem>
                <SelectItem value="apac">Asia Pacific (APAC)</SelectItem>
                <SelectItem value="global">Global</SelectItem>
              </SelectContent>
            </Select>
          </div>

          <div className="space-y-2">
            <Label htmlFor="industry-context" className="font-medium">Industry Context</Label>
            <Input
              id="industry-context"
              value={industryContext}
              onChange={(e) => setIndustryContext(e.target.value)}
              placeholder="e.g., Education, Software Development, SMEs, Healthcare..."
            />
          </div>
        </div>

        <div className="space-y-2">
          <div className="flex items-center justify-between">
            <Label htmlFor="known-competitors" className="font-medium">Competitors</Label>
            <Button
              variant="outline"
              size="sm"
              onClick={researchCompetitors}
              disabled={isResearchingCompetitors || !industryContext}
              className="gap-2"
            >
              {isResearchingCompetitors ? (
                <>
                  <Loader2 className="w-3 h-3 animate-spin" />
                  Searching...
                </>
              ) : (
                <>
                  <Search className="w-3 h-3" />
                  AI Research Competitors
                </>
              )}
            </Button>
          </div>
          <Textarea
            id="known-competitors"
            value={knownCompetitors}
            onChange={(e) => setKnownCompetitors(e.target.value)}
            placeholder="List any competitors you already know about, separated by commas or new lines..."
            className="min-h-[80px]"
          />
        </div>

        {/* Researched Competitors */}
        {researchedCompetitors.length > 0 && (
          <div className="space-y-3">
            <Label className="font-medium text-sm text-muted-foreground">AI-Researched Competitors</Label>
            <div className="grid gap-3">
              {researchedCompetitors.map((comp, index) => (
                <div key={index} className="p-3 rounded-lg bg-secondary/50 border border-border/50">
                  <div className="flex items-start justify-between">
                    <div>
                      <p className="font-medium text-sm">{comp.name}</p>
                      <p className="text-xs text-muted-foreground mt-1">{comp.description}</p>
                    </div>
                    <Badge variant="secondary" className="text-xs">{comp.model}</Badge>
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>

      {/* AI Analyze Panel */}
      <AIFeedbackPanel
        isLoading={isAnalyzing}
        result={phase3Result}
        onAnalyze={() => analyzeWithAI(3)}
        phaseId={3}
      />

      {/* Results Section - shown after full analysis */}
      {phase3Result && (
        <div className="space-y-6">
          {/* Market Summary - Only if full analysis done */}
          {phase3Result && (
            <div className="glass-card rounded-xl p-6 space-y-4">
              <div className="flex items-center gap-3">
                <Target className="w-5 h-5 text-primary" />
                <h3 className="text-lg font-display font-semibold">Market Research Summary</h3>
              </div>
              <p className="text-muted-foreground leading-relaxed">
                {phase3Result.phase3Details?.marketResearch?.overview || `The ${industryContext || "requested"} market in ${geographicScope || "your target region"} is experiencing significant development according to our AI analysis.`}
              </p>

              <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mt-4">
                {[
                  ...(phase3Result.phase3Details?.marketResearch?.market_size ? [{
                    label: "Market Size",
                    value: phase3Result.phase3Details.marketResearch.market_size.value,
                    source: phase3Result.phase3Details.marketResearch.market_size.source,
                    link: "#"
                  }] : []),
                  ...(phase3Result.phase3Details?.marketResearch?.growth_rate ? [{
                    label: "Growth Rate",
                    value: phase3Result.phase3Details.marketResearch.growth_rate.value,
                    source: phase3Result.phase3Details.marketResearch.growth_rate.source,
                    link: "#"
                  }] : []),
                  ...(phase3Result.phase3Details?.marketResearch?.key_statistics || []).map(stat => ({
                    label: stat.metric,
                    value: stat.value,
                    source: stat.source,
                    link: "#"
                  }))
                ].slice(0, 4).map((stat, index) => (
                  <div key={index} className="p-4 rounded-xl bg-secondary/50 space-y-1">
                    <p className="text-2xl font-bold text-primary">{stat.value}</p>
                    <p className="text-sm text-muted-foreground">{stat.label}</p>
                    <div className="text-xs text-primary flex items-center gap-1 opacity-70">
                      {stat.source}
                    </div>
                  </div>
                ))}
                {(!phase3Result.phase3Details?.marketResearch?.key_statistics?.length && !phase3Result.phase3Details?.marketResearch?.market_size) && (
                  <p className="col-span-4 text-sm text-muted-foreground italic">No detailed statistics available for this segment.</p>
                )}
              </div>
            </div>
          )}

          {/* Porter's Five Forces - Only if full analysis done */}
          {phase3Result && (
            <div className="glass-card rounded-xl p-6 space-y-4">
              <div className="flex items-center gap-3">
                <TrendingUp className="w-5 h-5 text-primary" />
                <h3 className="text-lg font-display font-semibold">Porter's Five Forces Analysis</h3>
              </div>
              <div className="space-y-3">
                {Object.entries(phase3Result.phase3Details?.porterFiveForces || {}).filter(([key]) => key !== 'overall_attractiveness').map(([key, force]: [string, any], index) => (
                  <div key={index} className="p-4 rounded-xl bg-secondary/50 flex items-start justify-between gap-4">
                    <div className="flex-1">
                      <p className="font-medium">{force.force || key.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase())}</p>
                      <p className="text-sm text-muted-foreground mt-1">{force.description || force.analysis}</p>
                    </div>
                    <Badge variant={force.level === "High" ? "destructive" : force.level === "Medium" ? "default" : "secondary"}>
                      {force.level}
                    </Badge>
                  </div>
                ))}
              </div>
              {phase3Result.phase3Details?.porterFiveForces?.overall_attractiveness && (
                <div className="mt-4 p-4 rounded-xl bg-primary/5 border border-primary/10">
                  <p className="text-sm font-medium text-primary uppercase tracking-wider mb-2">Overall Assessment</p>
                  <p className="text-sm text-foreground/80">{phase3Result.phase3Details.porterFiveForces.overall_attractiveness}</p>
                </div>
              )}
            </div>
          )}

          {/* Competitor Analysis with SWOT */}
          <div className="glass-card rounded-xl p-6 space-y-4">
            <div className="flex items-center gap-3">
              <Users className="w-5 h-5 text-primary" />
              <h3 className="text-lg font-display font-semibold">Detailed Competitor Analysis</h3>
            </div>
            <div className="space-y-4">
              {(() => {
                const competitors = phase3Result.phase3Details?.competitorAnalysis?.competitors || [];

                if (competitors.length === 0) {
                  return <p className="text-center text-sm text-muted-foreground italic">No detailed competitor profiles available.</p>;
                }

                return competitors.map((competitor: any, index: number) => (
                  <div key={index} className="p-4 rounded-xl bg-secondary/50 space-y-4">
                    <div>
                      <h4 className="font-semibold text-lg">{competitor.name}</h4>
                      <p className="text-sm text-muted-foreground">{competitor.description}</p>
                    </div>
                    <div className="grid grid-cols-2 gap-4 text-sm">
                      <div>
                        <p className="text-muted-foreground">Business Model</p>
                        <p className="font-medium">{competitor.business_model}</p>
                      </div>
                      <div>
                        <p className="text-muted-foreground">Target Customers</p>
                        <p className="font-medium">{competitor.target_customer}</p>
                      </div>
                    </div>
                    {competitor.strengths?.length > 0 && (
                      <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
                        <div className="p-3 rounded-lg bg-green-500/10 border border-green-500/20">
                          <p className="text-xs font-medium text-green-600 mb-2">Strengths</p>
                          <ul className="text-xs space-y-1">
                            {(competitor.strengths || []).map((s: string, i: number) => <li key={i}>• {s}</li>)}
                          </ul>
                        </div>
                        <div className="p-3 rounded-lg bg-red-500/10 border border-red-500/20">
                          <p className="text-xs font-medium text-red-600 mb-2">Weaknesses</p>
                          <ul className="text-xs space-y-1">
                            {(competitor.weaknesses || []).map((w: string, i: number) => <li key={i}>• {w}</li>)}
                          </ul>
                        </div>
                        <div className="p-3 rounded-lg bg-blue-500/10 border border-blue-500/20">
                          <p className="text-xs font-medium text-blue-600 mb-2">Opportunities</p>
                          <ul className="text-xs space-y-1">
                            {(competitor.opportunities || []).map((o: string, i: number) => <li key={i}>• {o}</li>)}
                          </ul>
                        </div>
                        <div className="p-3 rounded-lg bg-orange-500/10 border border-orange-500/20">
                          <p className="text-xs font-medium text-orange-600 mb-2">Threats</p>
                          <ul className="text-xs space-y-1">
                            {(competitor.threats || []).map((t: string, i: number) => <li key={i}>• {t}</li>)}
                          </ul>
                        </div>
                      </div>
                    )}
                  </div>
                ));
              })()}
            </div>
          </div>

          {/* Unique Selling Points - Only if full analysis done */}
          {phase3Result && (
            <div className="glass-card rounded-xl p-6 space-y-4">
              <div className="flex items-center gap-3">
                <Rocket className="w-5 h-5 text-primary" />
                <h3 className="text-lg font-display font-semibold">Generated Unique Selling Points</h3>
              </div>
              <div className="space-y-4">
                {[
                  ...(phase3Result.phase3Details?.uspGeneration?.primary_usp ? [phase3Result.phase3Details.uspGeneration.primary_usp] : []),
                  ...(phase3Result.phase3Details?.uspGeneration?.secondary_usps || [])
                ].map((usp: any, index: number) => (
                  <div key={index} className="p-4 rounded-xl bg-gradient-to-r from-primary/10 to-primary/5 border border-primary/20 space-y-2">
                    <div className="flex items-start gap-3">
                      <Lightbulb className="w-5 h-5 text-primary mt-0.5" />
                      <div>
                        <h4 className="font-semibold">{usp.usp}</h4>
                        <p className="text-sm text-muted-foreground mt-1">{usp.supporting_evidence}</p>
                        <div className="mt-2 p-2 rounded-lg bg-background/50">
                          <p className="text-xs text-muted-foreground">
                            <span className="font-medium text-primary">Differentiation:</span> {usp.differentiation_level}
                          </p>
                        </div>
                      </div>
                    </div>
                  </div>
                ))}
                {(!phase3Result.phase3Details?.uspGeneration?.primary_usp && !phase3Result.phase3Details?.uspGeneration?.secondary_usps?.length) && (
                  <p className="text-center text-sm text-muted-foreground italic">Execute "Market Analysis" to generate strategic USPs.</p>
                )}
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );

  const analyzeTechStack = () => {
    setIsTechStackAnalyzing(true);
    setTechStackResult(null);
    setTimeout(() => {
      setTechStackResult({
        recommendedStack: {
          frontend: { choice: "React with Next.js", reason: "Best for SEO, server-side rendering, and scalability. Strong ecosystem and community support for your education platform." },
          backend: { choice: "Node.js/Express", reason: "JavaScript everywhere approach reduces context switching. Great for real-time features and API development." },
          database: { choice: "PostgreSQL with Supabase", reason: "Robust relational database with excellent scalability. Supabase provides auth, storage, and real-time subscriptions out of the box." },
          hosting: { choice: "Vercel + Supabase Cloud", reason: "Optimized for Next.js with edge functions. Zero-config deployments and automatic scaling." },
          aiApproach: { choice: "LLM + Retrieval (RAG)", reason: "More flexible than rules-first. Allows personalized responses based on user context and project requirements." }
        },
        architecture: `┌─────────────────────────────────────────────────────────────┐
│                        CLIENT LAYER                          │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐          │
│  │   Next.js   │  │   React     │  │  Tailwind   │          │
│  │   App       │  │ Components  │  │    CSS      │          │
│  └──────┬──────┘  └──────┬──────┘  └─────────────┘          │
│         │                │                                   │
│         └────────┬───────┘                                   │
│                  ▼                                           │
│  ┌─────────────────────────────────────────────────┐        │
│  │              API Layer (Next.js API Routes)      │        │
│  └──────────────────────┬──────────────────────────┘        │
└─────────────────────────┼───────────────────────────────────┘
                          ▼
┌─────────────────────────────────────────────────────────────┐
│                      BACKEND SERVICES                        │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐          │
│  │  Supabase   │  │   Edge      │  │   OpenAI    │          │
│  │  (Auth/DB)  │  │  Functions  │  │   API       │          │
│  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘          │
│         │                │                │                  │
│         └────────┬───────┴────────────────┘                  │
│                  ▼                                           │
│  ┌─────────────────────────────────────────────────┐        │
│  │              PostgreSQL Database                  │        │
│  │  • User profiles  • Projects  • AI chat history  │        │
│  └─────────────────────────────────────────────────┘        │
└─────────────────────────────────────────────────────────────┘

DATA FLOW:
User Input → Next.js → API Route → Supabase/AI → Response → UI Update

AI INTEGRATION POINT:
• Input: User queries, project context, document uploads
• Processing: RAG pipeline with vector embeddings
• Output: Personalized recommendations, analysis, feedback`,
        tradeoffs: [
          {
            decision: "Next.js over plain React",
            tradeoff: "Adds complexity with SSR/SSG concepts, but significantly improves SEO and initial load performance",
            failureMode: "If you use client-side rendering for everything, you'll struggle with SEO and core web vitals scores"
          },
          {
            decision: "PostgreSQL over MongoDB",
            tradeoff: "Requires schema planning upfront, but provides data integrity and complex query support",
            failureMode: "This part will break first under scale: unoptimized queries without proper indexing. Add indexes for frequently queried columns early."
          },
          {
            decision: "LLM + RAG over Rules-First",
            tradeoff: "Higher API costs and latency, but much more flexible and natural responses",
            failureMode: "If you don't implement proper context limits, you'll hit token limits and costs will spike unexpectedly"
          },
          {
            decision: "Supabase over custom auth",
            tradeoff: "Less control over auth flow, but saves 2-3 weeks of development time",
            failureMode: "If you need complex RBAC beyond row-level security, you'll struggle with Supabase's permission model"
          }
        ]
      });
      setIsTechStackAnalyzing(false);
    }, 2500);
  };

  const analyzeWireframes = () => {
    setIsWireframeAnalyzing(true);
    setWireframeResult(null);
    setTimeout(() => {
      const screens = screensList.split('\n').filter(s => s.trim()).map((screen, index) => ({
        name: screen.trim(),
        pageStructure: {
          header: `${screen.trim()} Header`,
          body: [
            "- Main content area with primary functionality",
            "- Input controls for user interaction",
            "- Status indicators and feedback elements",
            "- Action buttons for primary user tasks"
          ],
          footer: "Navigation links | Help | Settings"
        },
        navigationFlow: `From this screen, user can complete the main task and navigate to related features. Back navigation to dashboard or previous screen available.`,
        keyComponents: [
          "Header with title and navigation",
          "Main content container",
          "Form inputs or data display",
          "Primary action button",
          "Loading/success states"
        ]
      }));
      setWireframeResult({ screens });
      setIsWireframeAnalyzing(false);
    }, 2000);
  };

  const renderPhase4 = () => (
    <div className="space-y-8">
      <div className="flex items-start justify-between gap-4 mb-8">
        <div>
          <h2 className="text-3xl font-display font-bold mb-3">Solution Design</h2>
          <p className="text-muted-foreground max-w-2xl">
            Design your solution architecture, choose technology stack, and create wireframe blueprints.
          </p>
        </div>
        <CompleteButton />
      </div>

      <Tabs value={phase4Tab} onValueChange={(v) => setPhase4Tab(v as 'tech-stack' | 'wireframe')} className="w-full">
        <TabsList className="grid w-full grid-cols-2">
          <TabsTrigger value="tech-stack" className="gap-2">
            <Cpu className="w-4 h-4" />
            Tech Stack Analyzer
          </TabsTrigger>
          <TabsTrigger value="wireframe" className="gap-2">
            <Layout className="w-4 h-4" />
            Wireframe Sandbox
          </TabsTrigger>
        </TabsList>

        {/* Tech Stack Analyzer Tab */}
        <TabsContent value="tech-stack" className="space-y-6 mt-6">
          <div className="glass-card rounded-xl p-6 space-y-6">
            <div>
              <h3 className="text-xl font-display font-bold mb-1">Tech Stack Decision Helper</h3>
              <p className="text-sm text-muted-foreground">Choose your technology stack with AI-powered recommendations</p>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              {/* Frontend Framework */}
              <div className="space-y-2">
                <Label className="font-display font-semibold flex items-center gap-2">
                  <Monitor className="w-4 h-4 text-primary" />
                  Frontend Framework
                </Label>
                <Select
                  value={techStackInputs.frontend}
                  onValueChange={(v) => setTechStackInputs({ ...techStackInputs, frontend: v })}
                >
                  <SelectTrigger>
                    <SelectValue placeholder="Select frontend framework" />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="react">React</SelectItem>
                    <SelectItem value="vue">Vue</SelectItem>
                    <SelectItem value="angular">Angular</SelectItem>
                    <SelectItem value="svelte">Svelte</SelectItem>
                    <SelectItem value="nextjs">Next.js</SelectItem>
                    <SelectItem value="nuxt">Nuxt</SelectItem>
                    <SelectItem value="static">None / Static HTML</SelectItem>
                  </SelectContent>
                </Select>
              </div>

              {/* Backend Framework */}
              <div className="space-y-2">
                <Label className="font-display font-semibold flex items-center gap-2">
                  <Server className="w-4 h-4 text-primary" />
                  Backend Framework
                </Label>
                <Select
                  value={techStackInputs.backend}
                  onValueChange={(v) => setTechStackInputs({ ...techStackInputs, backend: v })}
                >
                  <SelectTrigger>
                    <SelectValue placeholder="Select backend framework" />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="node-express">Node.js/Express</SelectItem>
                    <SelectItem value="django">Django</SelectItem>
                    <SelectItem value="flask">Flask</SelectItem>
                    <SelectItem value="fastapi">FastAPI</SelectItem>
                    <SelectItem value="rails">Ruby on Rails</SelectItem>
                    <SelectItem value="spring">Spring Boot</SelectItem>
                    <SelectItem value="dotnet">.NET Core</SelectItem>
                    <SelectItem value="serverless">Serverless (AWS Lambda, etc.)</SelectItem>
                  </SelectContent>
                </Select>
              </div>

              {/* Database */}
              <div className="space-y-2">
                <Label className="font-display font-semibold flex items-center gap-2">
                  <Database className="w-4 h-4 text-primary" />
                  Database
                </Label>
                <Select
                  value={techStackInputs.database}
                  onValueChange={(v) => setTechStackInputs({ ...techStackInputs, database: v })}
                >
                  <SelectTrigger>
                    <SelectValue placeholder="Select database" />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="postgresql">PostgreSQL</SelectItem>
                    <SelectItem value="mysql">MySQL</SelectItem>
                    <SelectItem value="mongodb">MongoDB</SelectItem>
                    <SelectItem value="sqlite">SQLite</SelectItem>
                    <SelectItem value="dynamodb">DynamoDB</SelectItem>
                    <SelectItem value="firebase">Firebase</SelectItem>
                    <SelectItem value="supabase">Supabase</SelectItem>
                    <SelectItem value="redis">Redis (Cache/Store)</SelectItem>
                  </SelectContent>
                </Select>
              </div>

              {/* AI/ML Usage */}
              <div className="space-y-2">
                <Label className="font-display font-semibold flex items-center gap-2">
                  <Cpu className="w-4 h-4 text-primary" />
                  AI/ML Usage
                </Label>
                <Select
                  value={techStackInputs.aiml}
                  onValueChange={(v) => setTechStackInputs({ ...techStackInputs, aiml: v })}
                >
                  <SelectTrigger>
                    <SelectValue placeholder="Select AI/ML usage level" />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="none">None</SelectItem>
                    <SelectItem value="api-integration">API Integration (OpenAI, Gemini, etc.)</SelectItem>
                    <SelectItem value="pretrained">Pre-trained Models</SelectItem>
                    <SelectItem value="custom-ml">Custom ML Training</SelectItem>
                    <SelectItem value="deep-learning">Deep Learning (TensorFlow, PyTorch)</SelectItem>
                  </SelectContent>
                </Select>
              </div>
            </div>
          </div>

          {/* AI Analysis Section */}
          <AIFeedbackPanel
            isLoading={isTechStackAnalyzing}
            result={techStackResult ? { overallScore: 0, breakdown: [], feedback: [], suggestedQuestions: [] } : null}
            onAnalyze={analyzeTechStack}
            title="Tech Stack Analysis"
            description="Get AI recommendations for your technology choices"
            disabled={!techStackInputs.frontend && !techStackInputs.backend && !techStackInputs.database}
            phaseId={4}
            subFunction="tech-stack"
          />

          {/* Tech Stack Results */}
          {techStackResult && (
            <div className="space-y-6 animate-fade-in">
              {/* Recommended Stack */}
              <div className="glass-card rounded-xl p-6 space-y-4">
                <h3 className="text-lg font-display font-bold flex items-center gap-2">
                  <Sparkles className="w-5 h-5 text-primary" />
                  Recommended Stack
                </h3>

                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  {Object.entries(techStackResult.recommendedStack).map(([key, value]: [string, any]) => (
                    <div key={key} className="p-4 rounded-xl bg-secondary/50 space-y-2">
                      <div className="flex items-center gap-2">
                        {key === 'frontend' && <Monitor className="w-4 h-4 text-primary" />}
                        {key === 'backend' && <Server className="w-4 h-4 text-primary" />}
                        {key === 'database' && <Database className="w-4 h-4 text-primary" />}
                        {key === 'hosting' && <Globe className="w-4 h-4 text-primary" />}
                        {key === 'aiApproach' && <Cpu className="w-4 h-4 text-primary" />}
                        <span className="font-semibold capitalize">{key.replace(/([A-Z])/g, ' $1').trim()}</span>
                      </div>
                      <p className="font-medium text-primary">{value.choice}</p>
                      <p className="text-sm text-muted-foreground">{value.reason}</p>
                    </div>
                  ))}
                </div>
              </div>

              {/* Architecture Diagram */}
              <div className="glass-card rounded-xl p-6 space-y-4">
                <h3 className="text-lg font-display font-bold flex items-center gap-2">
                  <Layers className="w-5 h-5 text-primary" />
                  Architecture Diagram
                </h3>
                <div className="p-4 rounded-xl bg-muted/50 overflow-x-auto">
                  <pre className="text-xs md:text-sm font-mono whitespace-pre text-foreground">
                    {techStackResult.architecture}
                  </pre>
                </div>
              </div>

              {/* Trade-offs & Failure Modes */}
              <div className="glass-card rounded-xl p-6 space-y-4">
                <h3 className="text-lg font-display font-bold flex items-center gap-2">
                  <AlertCircle className="w-5 h-5 text-amber-500" />
                  Trade-offs & Failure Modes
                </h3>
                <div className="space-y-4">
                  {techStackResult.tradeoffs.map((item: any, index: number) => (
                    <div key={index} className="p-4 rounded-xl border border-border bg-background space-y-3">
                      <div className="flex items-start gap-3">
                        <Badge variant="outline" className="shrink-0">{item.decision}</Badge>
                      </div>
                      <p className="text-sm text-muted-foreground">{item.tradeoff}</p>
                      <div className="p-3 rounded-lg bg-destructive/10 border border-destructive/20">
                        <p className="text-sm">
                          <span className="font-semibold text-destructive">⚠️ Failure Mode:</span>{" "}
                          <span className="text-muted-foreground">{item.failureMode}</span>
                        </p>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          )}
        </TabsContent>

        {/* Wireframe Sandbox Tab */}
        <TabsContent value="wireframe" className="space-y-6 mt-6">
          <div className="glass-card rounded-xl p-6 space-y-6">
            <div>
              <h3 className="text-xl font-display font-bold mb-1">Wireframe Sandbox</h3>
              <p className="text-sm text-muted-foreground">List your main screens and AI will generate wireframe blueprints</p>
            </div>

            <div className="space-y-3">
              <Label className="font-display font-semibold flex items-center gap-2">
                <Layout className="w-4 h-4 text-primary" />
                List of Main Screens *
              </Label>
              <p className="text-sm text-muted-foreground">Enter one screen name per line</p>
              <Textarea
                placeholder="Example:
Dashboard
AI Essay Generator - Input
AI Essay Generator - Results
Progress Tracker
CV Builder
Settings"
                value={screensList}
                onChange={(e) => setScreensList(e.target.value)}
                className="min-h-[180px] font-mono"
              />
            </div>
          </div>

          {/* AI Analysis Section */}
          <AIFeedbackPanel
            isLoading={isWireframeAnalyzing}
            result={wireframeResult ? { overallScore: 0, breakdown: [], feedback: [], suggestedQuestions: [] } : null}
            onAnalyze={analyzeWireframes}
            title="Wireframe Generation"
            description="Generate detailed wireframe blueprints for each screen"
            disabled={!screensList.trim()}
            phaseId={4}
            subFunction="wireframe"
          />

          {/* Wireframe Results */}
          {wireframeResult && wireframeResult.screens && (
            <div className="space-y-6 animate-fade-in">
              {wireframeResult.screens.map((screen: any, index: number) => (
                <div key={index} className="glass-card rounded-xl p-6 space-y-4">
                  <h3 className="text-lg font-display font-bold text-primary">{screen.name}</h3>

                  {/* Page Structure */}
                  <div className="space-y-2">
                    <h4 className="font-semibold flex items-center gap-2">
                      <Layers className="w-4 h-4" />
                      Page Structure
                    </h4>
                    <div className="p-4 rounded-xl bg-muted/50 space-y-2 font-mono text-sm">
                      <p>Header: '{screen.pageStructure.header}'</p>
                      <p>Body:</p>
                      {screen.pageStructure.body.map((item: string, i: number) => (
                        <p key={i} className="pl-4">{item}</p>
                      ))}
                      <p>Footer: '{screen.pageStructure.footer}'</p>
                    </div>
                  </div>

                  {/* Navigation Flow */}
                  <div className="space-y-2">
                    <h4 className="font-semibold flex items-center gap-2">
                      <ArrowRight className="w-4 h-4" />
                      Navigation Flow
                    </h4>
                    <div className="p-4 rounded-xl bg-muted/50">
                      <p className="text-sm text-muted-foreground">{screen.navigationFlow}</p>
                    </div>
                  </div>

                  {/* Key UI Components */}
                  <div className="space-y-2">
                    <h4 className="font-semibold flex items-center gap-2">
                      <Code className="w-4 h-4" />
                      Key UI Components
                    </h4>
                    <div className="p-4 rounded-xl bg-muted/50">
                      <ul className="list-disc list-inside space-y-1 text-sm text-muted-foreground">
                        {screen.keyComponents.map((comp: string, i: number) => (
                          <li key={i}>{comp}</li>
                        ))}
                      </ul>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          )}
        </TabsContent>
      </Tabs>
    </div>
  );

  // Task Breakdown Analysis
  const analyzeTaskBreakdown = () => {
    setIsTaskBreakdownAnalyzing(true);
    setTaskBreakdownResult(null);

    // Parse features from input to make tasks relevant
    const features = taskBreakdownInputs.priorityFeatures
      .split(/[\n,•]/)
      .map(f => f.trim())
      .filter(f => f.length > 0);

    const featureTasks = features.slice(0, 4).map((feature, idx) => ({
      id: `T2.${idx + 1}`,
      name: `Implement ${feature}`,
      effort: `${8 + idx * 2}h`,
      priority: idx === 0 ? "Critical" : idx === 1 ? "High" : "Medium",
      dependencies: idx === 0 ? ["T1.3", "T1.4"] : [`T2.${idx}`],
      status: "core"
    }));

    setTimeout(() => {
      setTaskBreakdownResult({
        phases: [
          {
            name: "Phase 1: Foundation Setup",
            duration: "Week 1-2",
            tasks: [
              { id: "T1.1", name: "Project setup & environment configuration", effort: "4h", priority: "Critical", dependencies: [], status: "foundation" },
              { id: "T1.2", name: "Database schema design & migration", effort: "6h", priority: "Critical", dependencies: ["T1.1"], status: "foundation" },
              { id: "T1.3", name: "Authentication system implementation", effort: "8h", priority: "High", dependencies: ["T1.2"], status: "foundation" },
              { id: "T1.4", name: "Basic UI layout & navigation skeleton", effort: "5h", priority: "High", dependencies: ["T1.1"], status: "foundation" }
            ]
          },
          {
            name: "Phase 2: Core Features",
            duration: "Week 3-5",
            tasks: featureTasks.length > 0 ? featureTasks : [
              { id: "T2.1", name: "User dashboard implementation", effort: "12h", priority: "High", dependencies: ["T1.4", "T1.3"], status: "core" },
              { id: "T2.2", name: "Core feature module setup", effort: "10h", priority: "Critical", dependencies: ["T1.2"], status: "core" },
              { id: "T2.3", name: "Data management features", effort: "8h", priority: "Medium", dependencies: ["T2.1"], status: "core" },
              { id: "T2.4", name: "User interaction system", effort: "10h", priority: "High", dependencies: ["T2.2"], status: "core" }
            ]
          },
          {
            name: "Phase 3: Enhancement & Polish",
            duration: "Week 6-7",
            tasks: [
              { id: "T3.1", name: "Analytics & reporting dashboard", effort: "6h", priority: "Medium", dependencies: ["T2.1"], status: "enhancement" },
              { id: "T3.2", name: "Export & data sharing features", effort: "5h", priority: "Medium", dependencies: ["T2.3"], status: "enhancement" },
              { id: "T3.3", name: "UI/UX refinements & animations", effort: "8h", priority: "Low", dependencies: ["T2.1"], status: "enhancement" },
              { id: "T3.4", name: "Performance optimization", effort: "4h", priority: "Medium", dependencies: ["T2.4"], status: "enhancement" }
            ]
          },
          {
            name: "Phase 4: Testing & Deployment",
            duration: "Week 8",
            tasks: [
              { id: "T4.1", name: "Unit & integration testing", effort: "8h", priority: "Critical", dependencies: ["T3.4"], status: "testing" },
              { id: "T4.2", name: "User acceptance testing", effort: "6h", priority: "High", dependencies: ["T4.1"], status: "testing" },
              { id: "T4.3", name: "Bug fixes & refinements", effort: "6h", priority: "High", dependencies: ["T4.2"], status: "testing" },
              { id: "T4.4", name: "Production deployment & monitoring", effort: "4h", priority: "Critical", dependencies: ["T4.3"], status: "testing" }
            ]
          }
        ],
        totalEstimate: `${90 + featureTasks.length * 10} hours`,
        criticalPath: ["T1.1", "T1.2", "T2.1", featureTasks[0]?.id || "T2.2", "T3.4", "T4.1", "T4.4"].filter(Boolean),
        riskAreas: [
          { area: features[0] || "Core Feature Integration", risk: "High", mitigation: "Start early, have fallback approaches ready" },
          { area: "Authentication", risk: "Medium", mitigation: "Use battle-tested auth library (Supabase Auth)" },
          { area: features[1] || "User Interface", risk: "Medium", mitigation: "Create design system early, test with users" }
        ]
      });
      setIsTaskBreakdownAnalyzing(false);
    }, 2500);
  };

  // Sprint Planning
  const generateSprintPlan = () => {
    setIsSprintPlanning(true);
    setSprintResult(null);
    const weeks = parseInt(sprintInputs.totalWeeks) || 4;
    setTimeout(() => {
      const sprints = [];
      for (let i = 1; i <= weeks; i++) {
        sprints.push({
          week: i,
          theme: i === 1 ? "Foundation & Setup" :
            i === 2 ? "Core Authentication & Data Layer" :
              i <= Math.ceil(weeks * 0.6) ? "Feature Development" :
                i <= Math.ceil(weeks * 0.85) ? "Integration & Polish" : "Testing & Deployment",
          goals: [
            i === 1 ? "Complete project scaffolding and environment setup" :
              i === 2 ? "Implement user authentication flow" :
                i <= Math.ceil(weeks * 0.6) ? "Build core feature modules" :
                  i <= Math.ceil(weeks * 0.85) ? "Integrate components and refine UX" : "Conduct testing and prepare for launch"
          ],
          deliverables: i === 1 ? ["Development environment ready", "CI/CD pipeline configured", "Database schema deployed"] :
            i === 2 ? ["Login/signup working", "User session management", "Protected routes"] :
              i <= Math.ceil(weeks * 0.6) ? ["Dashboard MVP", "AI integration proof-of-concept", "Core CRUD operations"] :
                i <= Math.ceil(weeks * 0.85) ? ["Polished UI", "Performance optimizations", "Error handling"] :
                  ["Test coverage >80%", "UAT sign-off", "Production deployment"],
          effort: i === 1 ? "15h" : i === 2 ? "18h" : i <= Math.ceil(weeks * 0.6) ? "22h" : "16h",
          status: i === 1 ? "ready" : "upcoming"
        });
      }
      setSprintResult({
        sprints,
        summary: {
          totalSprints: weeks,
          teamCapacity: `${parseInt(sprintInputs.teamSize || "1") * parseInt(sprintInputs.workHoursPerDay || "6") * 5}h/week`,
          bufferIncluded: "15%",
          estimatedCompletion: `${weeks} weeks`
        },
        milestones: [
          { name: "MVP Foundation Complete", week: 2, type: "milestone" },
          { name: "Feature Freeze", week: Math.ceil(weeks * 0.75), type: "milestone" },
          { name: "Beta Release", week: weeks - 1, type: "release" },
          { name: "Production Launch", week: weeks, type: "release" }
        ]
      });
      setIsSprintPlanning(false);
    }, 2000);
  };

  const toggleMilestone = (milestone: string) => {
    setSprintInputs(prev => ({
      ...prev,
      selectedMilestones: prev.selectedMilestones.includes(milestone)
        ? prev.selectedMilestones.filter(m => m !== milestone)
        : [...prev.selectedMilestones, milestone]
    }));
  };

  const renderPhase5 = () => (
    <div className="space-y-8">
      <div className="flex items-start justify-between gap-4 mb-8">
        <div>
          <h2 className="text-3xl font-display font-bold mb-3">Prototype Development</h2>
          <p className="text-muted-foreground max-w-2xl">
            Break down your project into actionable tasks and plan your sprints for effective execution.
          </p>
        </div>
        <CompleteButton />
      </div>

      <Tabs value={phase5Tab} onValueChange={(v) => setPhase5Tab(v as 'task-breakdown' | 'sprint-planning')} className="w-full">
        <TabsList className="grid w-full grid-cols-2">
          <TabsTrigger value="task-breakdown" className="gap-2">
            <ListChecks className="w-4 h-4" />
            Task Breakdown
          </TabsTrigger>
          <TabsTrigger value="sprint-planning" className="gap-2">
            <Rocket className="w-4 h-4" />
            Sprint Planning
          </TabsTrigger>
        </TabsList>

        {/* Task Breakdown Tab */}
        <TabsContent value="task-breakdown" className="space-y-6 mt-6">
          <div className="glass-card rounded-xl p-6 space-y-6">
            <div>
              <h3 className="text-lg font-display font-semibold mb-1">Task Breakdown Generator</h3>
              <p className="text-sm text-muted-foreground">AI will analyze your requirements and create a step-by-step implementation roadmap</p>
            </div>

            <div className="space-y-4">
              {/* Project Scope */}
              <div className="space-y-2">
                <div className="flex items-center justify-between">
                  <Label className="font-display font-semibold flex items-center gap-2">
                    <Target className="w-4 h-4 text-primary" />
                    Project Scope Summary
                  </Label>
                  <Button
                    variant="outline"
                    size="sm"
                    onClick={() => {
                      // Load from Phase 2 requirements stored in localStorage
                      const storedFeatureInputs = loadFromStorage(projectId, "featureInputs", { featureList: "", mvpGoal: "" });
                      const scope = storedFeatureInputs.mvpGoal
                        ? `${storedFeatureInputs.mvpGoal}${storedFeatureInputs.featureList ? `\n\nFeatures:\n${storedFeatureInputs.featureList}` : ''}`
                        : storedFeatureInputs.featureList || "";
                      if (scope) {
                        setTaskBreakdownInputs(prev => ({ ...prev, projectScope: scope }));
                      }
                    }}
                    className="gap-1.5"
                  >
                    <FileCheck className="w-3.5 h-3.5" />
                    From Requirements
                  </Button>
                </div>
                <Textarea
                  placeholder="Summarize your project scope from Requirements Analysis phase...
Example: AI-powered essay feedback platform with progress tracking, CV builder, and sample essays library"
                  value={taskBreakdownInputs.projectScope}
                  onChange={(e) => setTaskBreakdownInputs({ ...taskBreakdownInputs, projectScope: e.target.value })}
                  className="min-h-[100px]"
                />
              </div>

              {/* Complexity Level */}
              <div className="space-y-2">
                <Label className="font-display font-semibold flex items-center gap-2">
                  <Layers className="w-4 h-4 text-primary" />
                  Project Complexity
                </Label>
                <Select
                  value={taskBreakdownInputs.complexity}
                  onValueChange={(v) => setTaskBreakdownInputs({ ...taskBreakdownInputs, complexity: v })}
                >
                  <SelectTrigger>
                    <SelectValue placeholder="Select complexity level" />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="simple">Simple (Single feature, basic CRUD)</SelectItem>
                    <SelectItem value="moderate">Moderate (Multiple features, auth, API integrations)</SelectItem>
                    <SelectItem value="complex">Complex (AI/ML, real-time features, advanced UX)</SelectItem>
                    <SelectItem value="enterprise">Enterprise (Multi-tenant, scalability, compliance)</SelectItem>
                  </SelectContent>
                </Select>
              </div>

              {/* Priority Features */}
              <div className="space-y-2">
                <div className="flex items-center justify-between">
                  <Label className="font-display font-semibold flex items-center gap-2">
                    <Zap className="w-4 h-4 text-primary" />
                    Must-Have Features for MVP
                  </Label>
                  <Button
                    variant="outline"
                    size="sm"
                    onClick={() => {
                      // Load features from Phase 2
                      const storedFeatureInputs = loadFromStorage(projectId, "featureInputs", { featureList: "" });
                      if (storedFeatureInputs.featureList) {
                        setTaskBreakdownInputs(prev => ({ ...prev, priorityFeatures: storedFeatureInputs.featureList }));
                      }
                    }}
                    className="gap-1.5"
                  >
                    <FileCheck className="w-3.5 h-3.5" />
                    From Requirements
                  </Button>
                </div>
                <Textarea
                  placeholder="List your top 3-5 must-have features (one per line):
• User authentication
• AI feedback engine
• Progress dashboard"
                  value={taskBreakdownInputs.priorityFeatures}
                  onChange={(e) => setTaskBreakdownInputs({ ...taskBreakdownInputs, priorityFeatures: e.target.value })}
                  className="min-h-[100px]"
                />
              </div>
            </div>

            {/* Generate Button */}
            <Button
              onClick={analyzeTaskBreakdown}
              disabled={isTaskBreakdownAnalyzing || !taskBreakdownInputs.projectScope}
              className="w-full aiba-button-primary gap-2"
            >
              {isTaskBreakdownAnalyzing ? (
                <>
                  <Loader2 className="w-4 h-4 animate-spin" />
                  Generating Task Breakdown...
                </>
              ) : (
                <>
                  <Sparkles className="w-4 h-4" />
                  Generate Task Breakdown
                </>
              )}
            </Button>
          </div>

          {/* Task Breakdown Results */}
          {taskBreakdownResult && (
            <div className="space-y-6 animate-fade-in">
              {/* Summary Stats */}
              <div className="grid grid-cols-3 gap-4">
                <div className="glass-card rounded-xl p-4 text-center">
                  <p className="text-2xl font-bold text-primary">{taskBreakdownResult.totalEstimate}</p>
                  <p className="text-sm text-muted-foreground">Total Effort</p>
                </div>
                <div className="glass-card rounded-xl p-4 text-center">
                  <p className="text-2xl font-bold text-primary">{taskBreakdownResult.phases.reduce((acc: number, p: any) => acc + p.tasks.length, 0)}</p>
                  <p className="text-sm text-muted-foreground">Total Tasks</p>
                </div>
                <div className="glass-card rounded-xl p-4 text-center">
                  <p className="text-2xl font-bold text-primary">{taskBreakdownResult.phases.length}</p>
                  <p className="text-sm text-muted-foreground">Phases</p>
                </div>
              </div>

              {/* Task Phases */}
              {taskBreakdownResult.phases.map((phase: any, phaseIdx: number) => (
                <div key={phaseIdx} className="glass-card rounded-xl p-6 space-y-4">
                  <div className="flex items-center justify-between">
                    <div className="flex items-center gap-3">
                      <div className={`w-10 h-10 rounded-lg flex items-center justify-center ${phase.name.includes("Foundation") ? "bg-blue-500/20 text-blue-500" :
                        phase.name.includes("Core") ? "bg-green-500/20 text-green-500" :
                          phase.name.includes("Enhancement") ? "bg-purple-500/20 text-purple-500" :
                            "bg-orange-500/20 text-orange-500"
                        }`}>
                        {phaseIdx + 1}
                      </div>
                      <div>
                        <h4 className="font-display font-bold">{phase.name}</h4>
                        <p className="text-sm text-muted-foreground">{phase.duration}</p>
                      </div>
                    </div>
                    <Badge variant="secondary">{phase.tasks.length} tasks</Badge>
                  </div>

                  <div className="space-y-2">
                    {phase.tasks.map((task: any, taskIdx: number) => (
                      <div key={taskIdx} className="flex items-center gap-4 p-4 rounded-xl bg-muted/30 hover:bg-muted/50 transition-all border border-border/50">
                        <div className={`min-w-[52px] h-9 px-3 rounded-lg flex items-center justify-center text-xs font-semibold ${phase.name.includes("Foundation") ? "bg-blue-500/15 text-blue-600 dark:text-blue-400 border border-blue-500/20" :
                          phase.name.includes("Core") ? "bg-green-500/15 text-green-600 dark:text-green-400 border border-green-500/20" :
                            phase.name.includes("Enhancement") ? "bg-purple-500/15 text-purple-600 dark:text-purple-400 border border-purple-500/20" :
                              "bg-orange-500/15 text-orange-600 dark:text-orange-400 border border-orange-500/20"
                          }`}>
                          {task.id}
                        </div>
                        <div className="flex-1 min-w-0">
                          <p className="font-medium text-sm">{task.name}</p>
                          {task.dependencies.length > 0 && (
                            <p className="text-xs text-muted-foreground mt-0.5">
                              Depends on: {task.dependencies.join(", ")}
                            </p>
                          )}
                        </div>
                        <Badge variant={
                          task.priority === "Critical" ? "destructive" :
                            task.priority === "High" ? "default" : "secondary"
                        } className="text-xs shrink-0">
                          {task.priority}
                        </Badge>
                        <div className="text-sm font-mono text-muted-foreground bg-secondary/50 px-2 py-1 rounded shrink-0">
                          {task.effort}
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              ))}

              {/* Critical Path */}
              <div className="glass-card rounded-xl p-6 space-y-4">
                <h4 className="font-display font-bold flex items-center gap-2">
                  <AlertCircle className="w-5 h-5 text-destructive" />
                  Critical Path
                </h4>
                <div className="flex flex-wrap items-center gap-2">
                  {taskBreakdownResult.criticalPath.map((taskId: string, idx: number) => (
                    <div key={taskId} className="flex items-center gap-2">
                      <Badge variant="outline" className="font-mono">{taskId}</Badge>
                      {idx < taskBreakdownResult.criticalPath.length - 1 && (
                        <ArrowRight className="w-4 h-4 text-muted-foreground" />
                      )}
                    </div>
                  ))}
                </div>
                <p className="text-sm text-muted-foreground">
                  These tasks are on the critical path. Delays in any of these will directly impact your project timeline.
                </p>
              </div>

              {/* Risk Areas */}
              <div className="glass-card rounded-xl p-6 space-y-4">
                <h4 className="font-display font-bold flex items-center gap-2">
                  <AlertTriangle className="w-5 h-5 text-yellow-500" />
                  Risk Areas & Mitigation
                </h4>
                <div className="space-y-3">
                  {taskBreakdownResult.riskAreas.map((risk: any, idx: number) => (
                    <div key={idx} className="flex items-start gap-4 p-3 rounded-lg bg-yellow-500/10 border border-yellow-500/20">
                      <Badge variant={risk.risk === "High" ? "destructive" : "secondary"} className="mt-0.5">
                        {risk.risk}
                      </Badge>
                      <div>
                        <p className="font-medium text-sm">{risk.area}</p>
                        <p className="text-sm text-muted-foreground mt-1">{risk.mitigation}</p>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          )}
        </TabsContent>

        {/* Sprint Planning Tab */}
        <TabsContent value="sprint-planning" className="space-y-6 mt-6">
          <div className="glass-card rounded-xl p-6 space-y-6">
            <div>
              <h3 className="text-lg font-display font-semibold mb-1">Sprint Planning Generator</h3>
              <p className="text-sm text-muted-foreground">Configure your timeline and get a detailed weekly sprint plan with goals and deliverables</p>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              {/* Total Timeline */}
              <div className="space-y-2">
                <Label className="font-display font-semibold flex items-center gap-2">
                  <BarChart3 className="w-4 h-4 text-primary" />
                  Total Timeline *
                </Label>
                <Select
                  value={sprintInputs.totalWeeks}
                  onValueChange={(v) => setSprintInputs({ ...sprintInputs, totalWeeks: v })}
                >
                  <SelectTrigger>
                    <SelectValue placeholder="How many weeks?" />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="4">4 weeks</SelectItem>
                    <SelectItem value="6">6 weeks</SelectItem>
                    <SelectItem value="8">8 weeks</SelectItem>
                    <SelectItem value="10">10 weeks</SelectItem>
                    <SelectItem value="12">12 weeks</SelectItem>
                  </SelectContent>
                </Select>
              </div>

              {/* Team Size */}
              <div className="space-y-2">
                <Label className="font-display font-semibold flex items-center gap-2">
                  <Users className="w-4 h-4 text-primary" />
                  Team Size
                </Label>
                <Select
                  value={sprintInputs.teamSize}
                  onValueChange={(v) => setSprintInputs({ ...sprintInputs, teamSize: v })}
                >
                  <SelectTrigger>
                    <SelectValue placeholder="Team members" />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="1">Solo Developer</SelectItem>
                    <SelectItem value="2">2 people</SelectItem>
                    <SelectItem value="3">3 people</SelectItem>
                    <SelectItem value="4">4+ people</SelectItem>
                  </SelectContent>
                </Select>
              </div>

              {/* Work Hours */}
              <div className="space-y-2">
                <Label className="font-display font-semibold flex items-center gap-2">
                  <Zap className="w-4 h-4 text-primary" />
                  Hours/Day
                </Label>
                <Select
                  value={sprintInputs.workHoursPerDay}
                  onValueChange={(v) => setSprintInputs({ ...sprintInputs, workHoursPerDay: v })}
                >
                  <SelectTrigger>
                    <SelectValue placeholder="Hours per day" />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="4">4 hours (part-time)</SelectItem>
                    <SelectItem value="6">6 hours (focused)</SelectItem>
                    <SelectItem value="8">8 hours (full-time)</SelectItem>
                  </SelectContent>
                </Select>
              </div>
            </div>

            {/* Key Milestones Selection */}
            <div className="space-y-3">
              <Label className="font-display font-semibold flex items-center gap-2">
                <Target className="w-4 h-4 text-primary" />
                Key Milestones to Include
              </Label>
              <div className="flex flex-wrap gap-2">
                {["MVP Demo", "User Testing", "Feature Freeze", "Beta Launch", "Final Presentation", "Production Deployment"].map((milestone) => (
                  <Badge
                    key={milestone}
                    variant={sprintInputs.selectedMilestones.includes(milestone) ? "default" : "outline"}
                    className="cursor-pointer hover:bg-primary/20 transition-colors"
                    onClick={() => toggleMilestone(milestone)}
                  >
                    {sprintInputs.selectedMilestones.includes(milestone) && <CheckCircle2 className="w-3 h-3 mr-1" />}
                    {milestone}
                  </Badge>
                ))}
              </div>
            </div>

            {/* Generate Button */}
            <Button
              onClick={generateSprintPlan}
              disabled={isSprintPlanning || !sprintInputs.totalWeeks}
              className="w-full aiba-button-primary gap-2"
            >
              {isSprintPlanning ? (
                <>
                  <Loader2 className="w-4 h-4 animate-spin" />
                  Generating Sprint Plan...
                </>
              ) : (
                <>
                  <Sparkles className="w-4 h-4" />
                  Generate Sprint Plan
                </>
              )}
            </Button>
          </div>

          {/* Sprint Plan Results */}
          {sprintResult && (
            <div className="space-y-6 animate-fade-in">
              {/* Summary */}
              <div className="grid grid-cols-4 gap-4">
                <div className="glass-card rounded-xl p-4 text-center">
                  <p className="text-2xl font-bold text-primary">{sprintResult.summary.totalSprints}</p>
                  <p className="text-sm text-muted-foreground">Weeks</p>
                </div>
                <div className="glass-card rounded-xl p-4 text-center">
                  <p className="text-2xl font-bold text-primary">{sprintResult.summary.teamCapacity}</p>
                  <p className="text-sm text-muted-foreground">Team Capacity</p>
                </div>
                <div className="glass-card rounded-xl p-4 text-center">
                  <p className="text-2xl font-bold text-primary">{sprintResult.milestones.length}</p>
                  <p className="text-sm text-muted-foreground">Milestones</p>
                </div>
                <div className="glass-card rounded-xl p-4 text-center">
                  <p className="text-2xl font-bold text-green-500">{sprintResult.summary.bufferIncluded}</p>
                  <p className="text-sm text-muted-foreground">Buffer</p>
                </div>
              </div>

              {/* Sprint Timeline Visual */}
              <div className="glass-card rounded-xl p-6">
                <h4 className="font-display font-bold mb-6 flex items-center gap-2">
                  <BarChart3 className="w-5 h-5 text-primary" />
                  Sprint Timeline
                </h4>

                <div className="relative">
                  {/* Timeline line */}
                  <div className="absolute left-6 top-0 bottom-0 w-0.5 bg-gradient-to-b from-primary via-primary/50 to-primary/20" />

                  {/* Sprints */}
                  <div className="space-y-6">
                    {sprintResult.sprints.map((sprint: any, idx: number) => (
                      <div key={idx} className="relative flex gap-6">
                        {/* Week indicator */}
                        <div className={`relative z-10 w-12 h-12 rounded-full flex items-center justify-center font-bold text-sm ${sprint.status === "ready" ? "bg-primary text-primary-foreground" :
                          "bg-muted border-2 border-primary/30 text-muted-foreground"
                          }`}>
                          W{sprint.week}
                        </div>

                        {/* Sprint content */}
                        <div className="flex-1 pb-6">
                          <div className="p-4 rounded-xl bg-gradient-to-r from-muted to-transparent border border-border hover:border-primary/30 transition-colors">
                            <div className="flex items-center justify-between mb-3">
                              <h5 className="font-display font-bold text-lg">{sprint.theme}</h5>
                              <div className="flex items-center gap-2">
                                <Badge variant="outline" className="font-mono">{sprint.effort}</Badge>
                                {sprint.status === "ready" && (
                                  <Badge className="bg-green-500/20 text-green-500 border-green-500/30">Ready</Badge>
                                )}
                              </div>
                            </div>

                            <div className="grid grid-cols-2 gap-4">
                              {/* Goals */}
                              <div>
                                <p className="text-xs uppercase tracking-wider text-muted-foreground mb-2 flex items-center gap-1">
                                  <Target className="w-3 h-3" /> Sprint Goals
                                </p>
                                <ul className="space-y-1">
                                  {sprint.goals.map((goal: string, gIdx: number) => (
                                    <li key={gIdx} className="text-sm flex items-start gap-2">
                                      <CheckCircle2 className="w-4 h-4 text-primary mt-0.5 shrink-0" />
                                      <span>{goal}</span>
                                    </li>
                                  ))}
                                </ul>
                              </div>

                              {/* Deliverables */}
                              <div>
                                <p className="text-xs uppercase tracking-wider text-muted-foreground mb-2 flex items-center gap-1">
                                  <FileCheck className="w-3 h-3" /> Deliverables
                                </p>
                                <ul className="space-y-1">
                                  {sprint.deliverables.map((del: string, dIdx: number) => (
                                    <li key={dIdx} className="text-sm flex items-start gap-2">
                                      <div className="w-1.5 h-1.5 rounded-full bg-primary mt-2 shrink-0" />
                                      <span className="text-muted-foreground">{del}</span>
                                    </li>
                                  ))}
                                </ul>
                              </div>
                            </div>

                            {/* Milestone indicator */}
                            {sprintResult.milestones.filter((m: any) => m.week === sprint.week).map((m: any, mIdx: number) => (
                              <div key={mIdx} className="mt-4 pt-3 border-t border-border">
                                <Badge className={`${m.type === "release" ? "bg-green-500/20 text-green-500" : "bg-primary/20 text-primary"
                                  } gap-1`}>
                                  <Rocket className="w-3 h-3" />
                                  {m.name}
                                </Badge>
                              </div>
                            ))}
                          </div>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              </div>

              {/* Milestones Summary */}
              <div className="glass-card rounded-xl p-6">
                <h4 className="font-display font-bold mb-4 flex items-center gap-2">
                  <Rocket className="w-5 h-5 text-primary" />
                  Key Milestones
                </h4>
                <div className="flex flex-wrap gap-3">
                  {sprintResult.milestones.map((milestone: any, idx: number) => (
                    <div key={idx} className={`p-3 rounded-lg border ${milestone.type === "release"
                      ? "bg-green-500/10 border-green-500/30"
                      : "bg-primary/10 border-primary/30"
                      }`}>
                      <div className="flex items-center gap-2 mb-1">
                        {milestone.type === "release" ? (
                          <Rocket className="w-4 h-4 text-green-500" />
                        ) : (
                          <Target className="w-4 h-4 text-primary" />
                        )}
                        <span className="font-semibold text-sm">{milestone.name}</span>
                      </div>
                      <p className="text-xs text-muted-foreground">Week {milestone.week}</p>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          )}
        </TabsContent>
      </Tabs>
    </div>
  );

  const renderPhase6 = () => (
    <div className="space-y-8">
      <div className="flex items-start justify-between gap-4 mb-8">
        <div>
          <h2 className="text-3xl font-display font-bold mb-3">Testing & Validation</h2>
          <p className="text-muted-foreground max-w-2xl">
            Create comprehensive test plans and validation criteria for your solution.
          </p>
        </div>
        <CompleteButton />
      </div>

      <Tabs defaultValue="test-cases" className="w-full">
        <TabsList className="grid w-full grid-cols-3">
          <TabsTrigger value="test-cases">Test Cases</TabsTrigger>
          <TabsTrigger value="acceptance">Acceptance Criteria</TabsTrigger>
          <TabsTrigger value="risk">Risk Assessment</TabsTrigger>
        </TabsList>

        <TabsContent value="test-cases" className="space-y-4 mt-6">
          <div className="glass-card rounded-xl p-6 space-y-4">
            <Label className="text-lg font-display font-semibold">Test Case Generator</Label>
            <p className="text-sm text-muted-foreground">AI will generate test cases based on your functional requirements</p>
            <Button className="w-full aiba-button-primary gap-2">
              <Sparkles className="w-4 h-4" />
              Generate Test Cases
            </Button>
          </div>
        </TabsContent>

        <TabsContent value="acceptance" className="space-y-4 mt-6">
          <div className="glass-card rounded-xl p-6 space-y-4">
            <Label className="text-lg font-display font-semibold">Acceptance Criteria</Label>
            <Textarea placeholder="Define acceptance criteria for each major feature..." className="min-h-[150px]" />
            <Button variant="outline" className="w-full gap-2">
              <Sparkles className="w-4 h-4" />
              Generate from Requirements
            </Button>
          </div>
        </TabsContent>

        <TabsContent value="risk" className="space-y-4 mt-6">
          <div className="glass-card rounded-xl p-6 space-y-4">
            <Label className="text-lg font-display font-semibold">Risk Assessment Matrix</Label>
            <div className="grid grid-cols-3 gap-4">
              {["Technical", "Schedule", "Resource"].map((risk) => (
                <div key={risk} className="p-4 rounded-xl bg-secondary">
                  <Label className="font-medium">{risk} Risk</Label>
                  <Select>
                    <SelectTrigger className="mt-2">
                      <SelectValue placeholder="Level" />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="low">Low</SelectItem>
                      <SelectItem value="medium">Medium</SelectItem>
                      <SelectItem value="high">High</SelectItem>
                    </SelectContent>
                  </Select>
                </div>
              ))}
            </div>
          </div>
          <AIFeedbackPanel
            isLoading={isAnalyzing}
            result={aiResult}
            onAnalyze={() => analyzeWithAI(6)}
            title="Testing & Validation Analysis"
            description="Get AI suggestions for your test strategy"
          />
        </TabsContent>
      </Tabs>
    </div>
  );

  // Phase 7 states
  const [selectedDocStyle, setSelectedDocStyle] = useState<string | null>(() =>
    loadFromStorage(projectId, 'selectedDocStyle', null)
  );
  const [projectTitle, setProjectTitle] = useState<string>(() =>
    loadFromStorage(projectId, 'projectTitle', '')
  );
  const [customSectionName, setCustomSectionName] = useState('');
  const [isGeneratingDoc, setIsGeneratingDoc] = useState(false);

  // Documentation sections with toggles
  const getDocumentSections = (style: string) => {
    const baseSections = {
      'academic-report': [
        { id: 'abstract', title: 'Abstract & Introduction', required: true },
        { id: 'literature', title: 'Literature Review', required: false },
        { id: 'methodology', title: 'Methodology', required: true },
        { id: 'problem', title: 'Problem Statement', required: true },
        { id: 'solution', title: 'Proposed Solution', required: true },
        { id: 'implementation', title: 'Implementation Details', required: false },
        { id: 'results', title: 'Results & Discussion', required: true },
        { id: 'conclusion', title: 'Conclusion', required: true },
        { id: 'references', title: 'References', required: false },
      ],
      'software-engineering': [
        { id: 'executive', title: 'Executive Summary', required: true },
        { id: 'problem', title: 'Problem Statement', required: true },
        { id: 'personas', title: 'User Research & Personas', required: false },
        { id: 'requirements', title: 'Requirement Specification', required: true },
        { id: 'market', title: 'Market Analysis', required: false },
        { id: 'design', title: 'Solution Design', required: true },
        { id: 'architecture', title: 'System Architecture', required: false },
        { id: 'prototype', title: 'Prototype Description', required: true },
        { id: 'testing', title: 'Testing & Validation', required: true },
        { id: 'limitations', title: 'Limitations & Constraints', required: false },
        { id: 'future', title: 'Future Improvements', required: false },
        { id: 'conclusion', title: 'Conclusion', required: false },
      ],
      'business-proposal': [
        { id: 'executive', title: 'Executive Summary', required: true },
        { id: 'problem', title: 'Problem & Opportunity', required: true },
        { id: 'market', title: 'Market Analysis', required: true },
        { id: 'value', title: 'Value Proposition', required: true },
        { id: 'solution', title: 'Solution Overview', required: true },
        { id: 'roadmap', title: 'Product Roadmap', required: false },
        { id: 'roi', title: 'ROI Analysis', required: false },
        { id: 'timeline', title: 'Implementation Timeline', required: false },
        { id: 'conclusion', title: 'Conclusion & Next Steps', required: true },
      ],
      'startup-pitch': [
        { id: 'problem', title: 'Problem & Solution', required: true },
        { id: 'market', title: 'Market Opportunity', required: true },
        { id: 'demo', title: 'Product Demo', required: true },
        { id: 'traction', title: 'Traction & Validation', required: false },
        { id: 'business', title: 'Business Model', required: false },
        { id: 'competition', title: 'Competitive Landscape', required: false },
        { id: 'team', title: 'Team', required: false },
        { id: 'ask', title: 'The Ask', required: true },
      ],
    };
    return baseSections[style as keyof typeof baseSections] || [];
  };

  const [docSections, setDocSections] = useState<{ id: string; title: string; required: boolean; enabled: boolean }[]>([]);
  const [customSections, setCustomSections] = useState<string[]>(() =>
    loadFromStorage(projectId, 'customSections', [])
  );

  // Update sections when style changes
  useEffect(() => {
    if (selectedDocStyle) {
      const sections = getDocumentSections(selectedDocStyle).map(s => ({ ...s, enabled: true }));
      setDocSections(sections);
      saveToStorage(projectId, 'selectedDocStyle', selectedDocStyle);
    }
  }, [selectedDocStyle, projectId]);

  useEffect(() => {
    saveToStorage(projectId, 'projectTitle', projectTitle);
  }, [projectTitle, projectId]);

  useEffect(() => {
    saveToStorage(projectId, 'customSections', customSections);
  }, [customSections, projectId]);

  const toggleSection = (sectionId: string) => {
    setDocSections(prev => prev.map(s =>
      s.id === sectionId ? { ...s, enabled: !s.enabled } : s
    ));
  };

  // Drag and drop state
  const [draggedIndex, setDraggedIndex] = useState<number | null>(null);
  const [dragOverIndex, setDragOverIndex] = useState<number | null>(null);

  const handleDragStart = (index: number) => {
    setDraggedIndex(index);
  };

  const handleDragOver = (e: React.DragEvent, index: number) => {
    e.preventDefault();
    setDragOverIndex(index);
  };

  const handleDragEnd = () => {
    if (draggedIndex !== null && dragOverIndex !== null && draggedIndex !== dragOverIndex) {
      // Reorder sections
      const newSections = [...docSections];
      const [draggedItem] = newSections.splice(draggedIndex, 1);
      newSections.splice(dragOverIndex, 0, draggedItem);
      setDocSections(newSections);
    }
    setDraggedIndex(null);
    setDragOverIndex(null);
  };

  const handleCustomDragStart = (index: number) => {
    setDraggedIndex(docSections.length + index);
  };

  const handleCustomDragOver = (e: React.DragEvent, index: number) => {
    e.preventDefault();
    setDragOverIndex(docSections.length + index);
  };

  const handleCustomDragEnd = () => {
    const customStartIdx = docSections.length;
    if (draggedIndex !== null && dragOverIndex !== null && draggedIndex !== dragOverIndex) {
      // Only handle custom section reordering within custom sections
      if (draggedIndex >= customStartIdx && dragOverIndex >= customStartIdx) {
        const customDragIdx = draggedIndex - customStartIdx;
        const customDropIdx = dragOverIndex - customStartIdx;
        const newCustom = [...customSections];
        const [draggedItem] = newCustom.splice(customDragIdx, 1);
        newCustom.splice(customDropIdx, 0, draggedItem);
        setCustomSections(newCustom);
      }
    }
    setDraggedIndex(null);
    setDragOverIndex(null);
  };

  const addCustomSection = () => {
    if (customSectionName.trim() && !customSections.includes(customSectionName.trim())) {
      setCustomSections(prev => [...prev, customSectionName.trim()]);
      setCustomSectionName('');
    }
  };

  const removeCustomSection = (name: string) => {
    setCustomSections(prev => prev.filter(s => s !== name));
  };

  const generateDocumentation = async () => {
    setIsGeneratingDoc(true);
    const userId = projectId || 'default-user';

    try {
      const result = await generatePhase7Documentation(userId, {
        document_type: selectedDocStyle as any,
        project_title: projectTitle,
        author_name: 'AI Assistant User',
        additional_context: customSections.join(", ")
      });

      if (result.download_url) {
        // Construct full URL
        // Use the configured API base URL or fallback
        const API_BASE = (import.meta as any).env?.VITE_API_BASE_URL || 'http://localhost:8000';
        const downloadUrl = `${API_BASE}${result.download_url}`;

        // Trigger download
        window.open(downloadUrl, '_blank');

        toast({
          title: "Documentation Generated",
          description: "Your document is downloading now.",
        });
      } else {
        toast({
          title: "Documentation Generated",
          description: "Document content generated successfully (preview mode).",
        });
      }
    } catch (error) {
      console.error("Documentation generation failed:", error);
      toast({
        title: "Generation Failed",
        description: error instanceof Error ? error.message : "Failed to generate documentation",
        variant: "destructive"
      });
    } finally {
      setIsGeneratingDoc(false);
    }
  };

  const enabledSectionsCount = docSections.filter(s => s.enabled).length + customSections.length;

  const docStyles = [
    {
      id: 'academic-report',
      title: 'Academic Report',
      description: 'Formal structure with sections, figures, citations, and references',
      icon: FileText,
      features: ['Abstract & Introduction', 'Literature Review', 'Methodology'],
      tone: 'Formal and scholarly'
    },
    {
      id: 'software-engineering',
      title: 'Software Engineering Documentation',
      description: 'Technical documentation including SRS, design specs, and testing',
      icon: Code,
      features: ['Requirements Specification', 'System Architecture', 'API Documentation'],
      tone: 'Technical and precise'
    },
    {
      id: 'business-proposal',
      title: 'Business/Product Proposal',
      description: 'Business-focused with value proposition, ROI, and roadmap',
      icon: Building2,
      features: ['Executive Summary', 'Market Analysis', 'Value Proposition'],
      tone: 'Professional and persuasive'
    },
    {
      id: 'startup-pitch',
      title: 'Startup Pitch Document',
      description: 'Investor-ready pitch with problem, solution, market, and demo',
      icon: Rocket,
      features: ['Problem & Solution', 'Market Opportunity', 'Product Demo'],
      tone: 'Concise and compelling'
    },
  ];

  const renderPhase7 = () => (
    <div className="space-y-8">
      <div className="flex items-start justify-between gap-4 mb-8">
        <div>
          <h2 className="text-3xl font-display font-bold mb-3">Final Documentation</h2>
          <p className="text-muted-foreground max-w-2xl">
            Generate comprehensive project documentation from all your previous phases. Export as a professional PDF.
          </p>
        </div>
        <CompleteButton label="Complete Project" />
      </div>

      {!selectedDocStyle ? (
        // Step 1: Choose Documentation Style
        <div className="space-y-6">
          <div>
            <h3 className="text-xl font-display font-bold mb-2">Choose Documentation Style</h3>
            <p className="text-muted-foreground">
              Select the documentation style that best fits your project needs. The AI will adapt the structure, tone, and content accordingly.
            </p>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            {docStyles.map((style) => (
              <div
                key={style.id}
                className="glass-card rounded-xl p-6 space-y-4 hover:border-primary/50 transition-all cursor-pointer group"
                onClick={() => setSelectedDocStyle(style.id)}
              >
                <style.icon className="w-8 h-8 text-foreground" />

                <div>
                  <h4 className="font-display font-bold text-lg">{style.title}</h4>
                  <p className="text-sm text-muted-foreground mt-1">{style.description}</p>
                </div>

                <div className="space-y-2">
                  <p className="text-xs font-medium text-muted-foreground">Key Features:</p>
                  <ul className="space-y-1">
                    {style.features.map((feature, idx) => (
                      <li key={idx} className="text-sm flex items-center gap-2">
                        <span className="w-1.5 h-1.5 rounded-full bg-foreground" />
                        {feature}
                      </li>
                    ))}
                  </ul>
                </div>

                <p className="text-sm text-muted-foreground">
                  <span className="font-medium">Tone:</span> {style.tone}
                </p>

                <Button className="w-full bg-foreground text-background hover:bg-foreground/90 group-hover:translate-y-0 transition-all">
                  Select This Style
                </Button>
              </div>
            ))}
          </div>
        </div>
      ) : (
        // Step 2: Configure Sections
        <div className="space-y-6">
          <div className="flex items-center justify-between">
            <div>
              <h3 className="text-xl font-display font-bold mb-2">Generate Complete Documentation</h3>
              <p className="text-muted-foreground">
                Select which sections to include in your documentation, then click generate to create the complete document with all content automatically.
              </p>
            </div>
            <Button
              variant="outline"
              size="sm"
              onClick={() => setSelectedDocStyle(null)}
              className="gap-2"
            >
              <ChevronLeft className="w-4 h-4" />
              Change Style
            </Button>
          </div>

          {/* Project Title */}
          <div className="space-y-2">
            <Label className="font-display font-semibold">Project Title</Label>
            <Input
              placeholder="Enter your project title..."
              value={projectTitle}
              onChange={(e) => setProjectTitle(e.target.value)}
              className="text-lg"
            />
          </div>

          {/* Documentation Sections */}
          <div className="glass-card rounded-xl p-6 space-y-4">
            <div>
              <h4 className="font-display font-bold">Documentation Sections</h4>
              <p className="text-sm text-muted-foreground">Toggle sections on or off as needed for your documentation.</p>
            </div>

            <div className="space-y-2">
              {docSections.map((section, idx) => (
                <div
                  key={section.id}
                  draggable
                  onDragStart={() => handleDragStart(idx)}
                  onDragOver={(e) => handleDragOver(e, idx)}
                  onDragEnd={handleDragEnd}
                  className={`flex items-center justify-between p-4 rounded-xl border transition-all cursor-move ${draggedIndex === idx
                    ? 'opacity-50 scale-[0.98] bg-primary/10 border-primary/30'
                    : dragOverIndex === idx
                      ? 'bg-primary/5 border-primary/40 shadow-lg'
                      : 'bg-muted/30 border-border/50 hover:bg-muted/50'
                    }`}
                >
                  <div className="flex items-center gap-3">
                    <GripVertical className="w-5 h-5 text-muted-foreground cursor-grab active:cursor-grabbing" />
                    <span className="font-medium">{section.title}</span>
                    {section.required && (
                      <Badge variant="secondary" className="text-xs">Required</Badge>
                    )}
                  </div>
                  <div className="flex items-center gap-3">
                    <Switch
                      checked={section.enabled}
                      onCheckedChange={() => toggleSection(section.id)}
                    />
                    <Eye className={`w-4 h-4 ${section.enabled ? 'text-primary' : 'text-muted-foreground/40'}`} />
                  </div>
                </div>
              ))}

              {/* Custom Sections */}
              {customSections.map((section, idx) => (
                <div
                  key={`custom-${idx}`}
                  draggable
                  onDragStart={() => handleCustomDragStart(idx)}
                  onDragOver={(e) => handleCustomDragOver(e, idx)}
                  onDragEnd={handleCustomDragEnd}
                  className={`flex items-center justify-between p-4 rounded-xl border transition-all cursor-move ${draggedIndex === docSections.length + idx
                    ? 'opacity-50 scale-[0.98] bg-primary/10 border-primary/30'
                    : dragOverIndex === docSections.length + idx
                      ? 'bg-primary/10 border-primary/40 shadow-lg'
                      : 'bg-primary/5 border-primary/20 hover:bg-primary/10'
                    }`}
                >
                  <div className="flex items-center gap-3">
                    <GripVertical className="w-5 h-5 text-muted-foreground cursor-grab active:cursor-grabbing" />
                    <span className="font-medium">{section}</span>
                    <Badge variant="outline" className="text-xs">Custom</Badge>
                  </div>
                  <div className="flex items-center gap-3">
                    <Button
                      variant="ghost"
                      size="sm"
                      onClick={() => removeCustomSection(section)}
                      className="h-8 w-8 p-0 text-muted-foreground hover:text-destructive"
                    >
                      <X className="w-4 h-4" />
                    </Button>
                    <Eye className="w-4 h-4 text-primary" />
                  </div>
                </div>
              ))}
            </div>
          </div>

          {/* Add Custom Section */}
          <div className="glass-card rounded-xl p-6 space-y-4">
            <h4 className="font-display font-bold">Add Custom Section</h4>
            <div className="flex gap-3">
              <Input
                placeholder="Section name (e.g., 'Risk Analysis')"
                value={customSectionName}
                onChange={(e) => setCustomSectionName(e.target.value)}
                onKeyDown={(e) => e.key === 'Enter' && addCustomSection()}
                className="flex-1"
              />
              <Button
                variant="secondary"
                onClick={addCustomSection}
                disabled={!customSectionName.trim()}
                className="gap-2"
              >
                <Plus className="w-4 h-4" />
                Add
              </Button>
            </div>
          </div>

          {/* Generate Button */}
          <Button
            onClick={generateDocumentation}
            disabled={isGeneratingDoc || !projectTitle.trim()}
            className="w-full h-14 text-lg aiba-button-primary gap-3"
          >
            {isGeneratingDoc ? (
              <>
                <Loader2 className="w-5 h-5 animate-spin" />
                Generating Documentation...
              </>
            ) : (
              <>
                <Download className="w-5 h-5" />
                Generate Complete Documentation ({enabledSectionsCount} sections)
              </>
            )}
          </Button>
        </div>
      )}
    </div>
  );

  const phaseRenderers: { [key: number]: () => JSX.Element } = {
    1: renderPhase1,
    2: renderPhase2,
    3: renderPhase3,
    4: renderPhase7
  };

  return phaseRenderers[phaseId] ? phaseRenderers[phaseId]() : <div>Phase not found</div>;
};

export default PhaseContent;
