import { useState, useEffect } from "react";
import { Link, useParams, useNavigate } from "react-router-dom";
import {
  Sparkles, ArrowLeft, Check, Lock, ChevronRight,
  Lightbulb, FileSearch, TrendingUp, Layers, Code, TestTube, FileText, LogOut
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";
import PhaseContent from "@/components/project/PhaseContent";
import { useAuth } from "@/hooks/useAuth";
import { supabase } from "@/integrations/supabase/client";
import { toast } from "sonner";

interface Phase {
  id: number;
  title: string;
  description: string;
  icon: React.ElementType;
  status: "completed" | "current" | "locked";
}

const PHASE_NAMES = [
  "Problem Definition",
  "Requirements Analysis",
  "Market Analysis",
  "Documentation"
];

const ProjectDetail = () => {
  const { id } = useParams();
  const navigate = useNavigate();
  const { user, loading, signOut } = useAuth();
  const [currentPhase, setCurrentPhase] = useState(1);
  const [completedPhases, setCompletedPhases] = useState<number[]>([]);
  const [projectName, setProjectName] = useState("Project");
  const [isLoading, setIsLoading] = useState(true);

  // Save current viewing phase to sessionStorage whenever it changes
  useEffect(() => {
    if (id && currentPhase) {
      sessionStorage.setItem(`project-${id}-viewingPhase`, String(currentPhase));
    }
  }, [id, currentPhase]);

  // Redirect to auth if not logged in
  useEffect(() => {
    if (!loading && !user) {
      navigate("/auth");
    }
  }, [user, loading, navigate]);

  // Fetch project data from Supabase
  useEffect(() => {
    const fetchProject = async () => {
      if (!user || !id) return;

      const { data, error } = await supabase
        .from("projects")
        .select("*")
        .eq("id", id)
        .maybeSingle();

      if (error) {
        toast.error("Failed to load project");
        console.error(error);
        navigate("/dashboard");
        return;
      }

      if (!data) {
        toast.error("Project not found");
        navigate("/dashboard");
        return;
      }

      setProjectName(data.name);
      setCompletedPhases(data.completed_phases || []);

      // Check if there's a saved viewing phase in sessionStorage
      const savedViewingPhase = sessionStorage.getItem(`project-${id}-viewingPhase`);
      if (savedViewingPhase) {
        const savedPhase = parseInt(savedViewingPhase, 10);
        // Make sure saved phase is valid (not locked)
        const maxUnlocked = (data.completed_phases?.length || 0) > 0
          ? Math.max(...(data.completed_phases || [])) + 1
          : 1;
        if (savedPhase >= 1 && savedPhase <= Math.min(maxUnlocked, 4)) {
          setCurrentPhase(savedPhase);
        } else {
          setCurrentPhase(data.current_phase || 1);
        }
      } else {
        setCurrentPhase(data.current_phase || 1);
      }

      setIsLoading(false);
    };

    if (user && id) {
      fetchProject();
    }
  }, [user, id, navigate]);

  // Save progress to Supabase
  const saveProgress = async (newCurrentPhase: number, newCompletedPhases: number[]) => {
    if (!id) return;

    const { error } = await supabase
      .from("projects")
      .update({
        current_phase: newCurrentPhase,
        phase_name: PHASE_NAMES[newCurrentPhase - 1],
        completed_phases: newCompletedPhases
      })
      .eq("id", id);

    if (error) {
      toast.error("Failed to save progress");
      console.error(error);
    }
  };

  const phases: Phase[] = [
    { id: 1, title: "Problem Definition", description: "Clarify the problem and target users", icon: Lightbulb, status: "current" },
    { id: 2, title: "Requirements Analysis", description: "Define functional and non-functional requirements", icon: FileSearch, status: "locked" },
    { id: 3, title: "Market Analysis", description: "Analyze market fit and competitors", icon: TrendingUp, status: "locked" },
    { id: 4, title: "Documentation", description: "Generate technical and user documentation", icon: FileText, status: "locked" }
  ];

  // Calculate the highest unlocked phase (max completed phase + 1, or 1 if none completed)
  const highestUnlockedPhase = completedPhases.length > 0
    ? Math.max(...completedPhases) + 1
    : 1;

  const getPhaseStatus = (phaseId: number): "completed" | "current" | "locked" => {
    if (completedPhases.includes(phaseId)) return "completed";
    if (phaseId === currentPhase) return "current";
    // A phase is unlocked (not locked) if it's <= the highest unlocked phase
    if (phaseId <= highestUnlockedPhase) return "current";
    return "locked";
  };

  const handlePhaseComplete = async (phaseId: number) => {
    if (!completedPhases.includes(phaseId)) {
      const newCompletedPhases = [...completedPhases, phaseId];
      // Updated to stop incrementing after phase 4
      const newCurrentPhase = phaseId < 4 ? phaseId + 1 : phaseId;

      setCompletedPhases(newCompletedPhases);
      setCurrentPhase(newCurrentPhase);

      await saveProgress(newCurrentPhase, newCompletedPhases);
      toast.success("Phase completed!");
    }
  };

  const handlePhaseClick = (phaseId: number) => {
    const status = getPhaseStatus(phaseId);
    if (status !== "locked") {
      setCurrentPhase(phaseId);
    }
  };

  const handleSignOut = async () => {
    await signOut();
    navigate("/");
  };

  const getUserInitial = () => {
    if (!user?.email) return "U";
    return user.email.charAt(0).toUpperCase();
  };

  if (loading || isLoading) {
    return (
      <div className="min-h-screen bg-background flex items-center justify-center">
        <div className="animate-pulse text-muted-foreground">Loading...</div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-background">
      {/* Navigation */}
      <nav className="fixed top-0 left-0 right-0 z-50 bg-background/80 backdrop-blur-xl border-b border-border">
        <div className="container mx-auto px-6 py-4 flex items-center justify-between">
          <Link to="/" className="flex items-center gap-2">
            <Sparkles className="w-6 h-6 text-primary" />
            <span className="font-display font-bold text-xl">AIBA</span>
          </Link>
          <span className="font-medium">{projectName}</span>
          <div className="flex items-center gap-3">
            <Button variant="ghost" size="icon" onClick={handleSignOut} title="Sign Out">
              <LogOut className="w-5 h-5" />
            </Button>
            <div className="w-10 h-10 rounded-full bg-foreground text-background flex items-center justify-center font-medium">
              {getUserInitial()}
            </div>
          </div>
        </div>
      </nav>

      <main className="pt-24 pb-12">
        <div className="container mx-auto px-6">
          {/* Back Button */}
          <Link to="/dashboard" className="inline-flex items-center gap-2 text-muted-foreground hover:text-foreground transition-colors mb-8">
            <ArrowLeft className="w-4 h-4" />
            <span>Back to Dashboard</span>
          </Link>

          <div className="grid lg:grid-cols-[320px_1fr] gap-8">
            {/* Phases Sidebar */}
            <div className="space-y-4">
              <h2 className="text-2xl font-display font-bold mb-6">Project Phases</h2>

              <div className="space-y-3">
                {phases.map((phase) => {
                  const status = getPhaseStatus(phase.id);
                  const isActive = currentPhase === phase.id;

                  return (
                    <button
                      key={phase.id}
                      onClick={() => handlePhaseClick(phase.id)}
                      disabled={status === "locked"}
                      className={cn(
                        "w-full flex items-center gap-4 p-4 rounded-xl text-left transition-all duration-300",
                        isActive && "bg-foreground text-background shadow-lg",
                        !isActive && status === "completed" && "bg-secondary hover:bg-secondary/80",
                        !isActive && status === "locked" && "opacity-50 cursor-not-allowed bg-muted"
                      )}
                    >
                      <div className={cn(
                        "w-10 h-10 rounded-full flex items-center justify-center flex-shrink-0 transition-all",
                        isActive && "bg-background",
                        !isActive && status === "completed" && "bg-primary/10",
                        !isActive && status === "locked" && "bg-muted-foreground/10"
                      )}>
                        {status === "completed" ? (
                          <Check className="w-5 h-5 text-primary" />
                        ) : status === "locked" ? (
                          <Lock className="w-4 h-4 text-muted-foreground" />
                        ) : (
                          <phase.icon className={cn(
                            "w-5 h-5",
                            isActive ? "text-primary" : "text-muted-foreground"
                          )} />
                        )}
                      </div>

                      <div className="flex-1 min-w-0">
                        <p className={cn(
                          "font-medium truncate",
                          isActive && "text-background",
                          !isActive && "text-foreground"
                        )}>
                          {phase.title}
                        </p>
                        <p className={cn(
                          "text-sm truncate",
                          isActive ? "text-background/70" : "text-muted-foreground"
                        )}>
                          {phase.description}
                        </p>
                      </div>

                      {isActive && (
                        <ChevronRight className="w-5 h-5 text-background/70" />
                      )}
                    </button>
                  );
                })}
              </div>
            </div>

            {/* Phase Content */}
            <div className="glass-card rounded-2xl p-8">
              <PhaseContent
                phaseId={currentPhase}
                onComplete={() => handlePhaseComplete(currentPhase)}
                isCompleted={completedPhases.includes(currentPhase)}
                projectId={id}
              />
            </div>
          </div>
        </div>
      </main>
    </div>
  );
};

export default ProjectDetail;