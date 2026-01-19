import { useState, useEffect } from "react";
import { Link, useNavigate } from "react-router-dom";
import { Sparkles, Plus, Clock, MessageSquare, FileText, Trash2, Calendar, TrendingUp, LogOut } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Progress } from "@/components/ui/progress";
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogTrigger } from "@/components/ui/dialog";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Textarea } from "@/components/ui/textarea";
import { ProjectTimelineChart } from "@/components/dashboard/ProjectTimelineChart";
import { PhaseDistributionChart } from "@/components/dashboard/PhaseDistributionChart";
import { ProgressTrendChart } from "@/components/dashboard/ProgressTrendChart";
import { useAuth } from "@/hooks/useAuth";
import { supabase } from "@/integrations/supabase/client";
import { toast } from "sonner";

interface Project {
  id: string;
  name: string;
  description: string;
  currentPhase: number;
  totalPhases: number;
  phaseName: string;
  updatedAt: string;
  startDate: string;
  endDate: string;
}

const PHASE_NAMES = [
  "Problem Definition",
  "Requirements Analysis",
  "Market Analysis",
  "Solution Design",
  "Prototype Development",
  "Testing & Validation",
  "Documentation"
];

const Dashboard = () => {
  const { user, loading, signOut } = useAuth();
  const navigate = useNavigate();
  const [projects, setProjects] = useState<Project[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [isDialogOpen, setIsDialogOpen] = useState(false);
  const [newProject, setNewProject] = useState({
    name: "",
    description: "",
    startDate: "",
    endDate: ""
  });

  // Redirect to auth if not logged in
  useEffect(() => {
    if (!loading && !user) {
      navigate("/auth");
    }
  }, [user, loading, navigate]);

  // Fetch projects from Supabase
  useEffect(() => {
    const fetchProjects = async () => {
      if (!user) return;
      
      const { data, error } = await supabase
        .from("projects")
        .select("*")
        .order("updated_at", { ascending: false });

      if (error) {
        toast.error("Failed to fetch projects");
        console.error(error);
      } else {
        const formattedProjects: Project[] = (data || []).map((p) => ({
          id: p.id,
          name: p.name,
          description: p.description || "",
          currentPhase: p.current_phase || 1,
          totalPhases: p.total_phases || 7,
          phaseName: p.phase_name || PHASE_NAMES[0],
          updatedAt: formatDate(p.updated_at),
          startDate: p.start_date || "",
          endDate: p.end_date || ""
        }));
        setProjects(formattedProjects);
      }
      setIsLoading(false);
    };

    if (user) {
      fetchProjects();
    }
  }, [user]);

  const formatDate = (dateString: string) => {
    const date = new Date(dateString);
    const now = new Date();
    const diffMs = now.getTime() - date.getTime();
    const diffHours = Math.floor(diffMs / (1000 * 60 * 60));
    const diffDays = Math.floor(diffMs / (1000 * 60 * 60 * 24));

    if (diffHours < 1) return "just now";
    if (diffHours < 24) return `${diffHours} hours ago`;
    if (diffDays === 1) return "yesterday";
    if (diffDays < 7) return `${diffDays} days ago`;
    return date.toLocaleDateString();
  };

  const handleCreateProject = async () => {
    if (!newProject.name || !user) return;
    
    const { data, error } = await supabase
      .from("projects")
      .insert({
        user_id: user.id,
        name: newProject.name,
        description: newProject.description || null,
        start_date: newProject.startDate || null,
        end_date: newProject.endDate || null,
        current_phase: 1,
        total_phases: 7,
        phase_name: PHASE_NAMES[0],
        completed_phases: []
      })
      .select()
      .single();

    if (error) {
      toast.error("Failed to create project");
      console.error(error);
      return;
    }

    const project: Project = {
      id: data.id,
      name: data.name,
      description: data.description || "",
      currentPhase: data.current_phase || 1,
      totalPhases: data.total_phases || 7,
      phaseName: data.phase_name || PHASE_NAMES[0],
      updatedAt: "just now",
      startDate: data.start_date || "",
      endDate: data.end_date || ""
    };
    
    setProjects([project, ...projects]);
    setNewProject({ name: "", description: "", startDate: "", endDate: "" });
    setIsDialogOpen(false);
    toast.success("Project created!");
  };

  const deleteProject = async (id: string) => {
    const { error } = await supabase
      .from("projects")
      .delete()
      .eq("id", id);

    if (error) {
      toast.error("Failed to delete project");
      console.error(error);
      return;
    }

    setProjects(projects.filter(p => p.id !== id));
    toast.success("Project deleted");
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
          <span className="font-medium">Dashboard</span>
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

      {/* Main Content */}
      <main className="pt-24 pb-12 relative">
        <div className="absolute inset-0 grid-pattern opacity-30" />
        <div className="absolute bottom-0 left-0 right-0 h-64 bg-gradient-to-t from-primary/5 to-transparent" />
        
        <div className="container mx-auto px-6 relative">
          {/* Header */}
          <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 mb-8">
            <div>
              <h1 className="text-4xl font-display font-bold gradient-text mb-2">Your Projects</h1>
              <p className="text-muted-foreground">Track and manage your engineering projects</p>
            </div>
            
            <Dialog open={isDialogOpen} onOpenChange={setIsDialogOpen}>
              <DialogTrigger asChild>
                <Button className="aiba-button-primary gap-2">
                  <Plus className="w-4 h-4" />
                  New Project
                </Button>
              </DialogTrigger>
              <DialogContent className="sm:max-w-md">
                <DialogHeader>
                  <DialogTitle className="font-display text-xl">Create New Project</DialogTitle>
                </DialogHeader>
                <div className="space-y-4 py-4">
                  <div className="space-y-2">
                    <Label htmlFor="name">Project Name</Label>
                    <Input
                      id="name"
                      placeholder="e.g., COMP3080 Capstone"
                      value={newProject.name}
                      onChange={(e) => setNewProject({ ...newProject, name: e.target.value })}
                    />
                  </div>
                  <div className="space-y-2">
                    <Label htmlFor="description">Description</Label>
                    <Textarea
                      id="description"
                      placeholder="Brief description of your project..."
                      value={newProject.description}
                      onChange={(e) => setNewProject({ ...newProject, description: e.target.value })}
                    />
                  </div>
                  <div className="grid grid-cols-2 gap-4">
                    <div className="space-y-2">
                      <Label htmlFor="startDate">Start Date</Label>
                      <Input
                        id="startDate"
                        type="date"
                        value={newProject.startDate}
                        onChange={(e) => setNewProject({ ...newProject, startDate: e.target.value })}
                      />
                    </div>
                    <div className="space-y-2">
                      <Label htmlFor="endDate">End Date</Label>
                      <Input
                        id="endDate"
                        type="date"
                        value={newProject.endDate}
                        onChange={(e) => setNewProject({ ...newProject, endDate: e.target.value })}
                      />
                    </div>
                  </div>
                  <Button onClick={handleCreateProject} className="w-full aiba-button-primary">
                    Create Project
                  </Button>
                </div>
              </DialogContent>
            </Dialog>
          </div>

          {/* Empty State */}
          {projects.length === 0 && (
            <div className="glass-card rounded-2xl p-12 text-center mb-10">
              <div className="w-20 h-20 mx-auto mb-6 rounded-full bg-secondary flex items-center justify-center">
                <Sparkles className="w-10 h-10 text-primary" />
              </div>
              <h2 className="text-2xl font-display font-bold mb-3">Welcome to AIBA!</h2>
              <p className="text-muted-foreground mb-6 max-w-md mx-auto">
                Start your first AI-powered engineering project. AIBA will guide you through each phase from problem definition to documentation.
              </p>
              <Button onClick={() => setIsDialogOpen(true)} className="aiba-button-primary gap-2">
                <Plus className="w-4 h-4" />
                Create Your First Project
              </Button>
            </div>
          )}

          {/* Stats Overview */}
          {projects.length > 0 && (
            <div className="grid grid-cols-2 lg:grid-cols-4 gap-4 mb-10">
              {[
                { icon: FileText, label: "Total Projects", value: projects.length },
                { icon: TrendingUp, label: "In Progress", value: projects.filter(p => p.currentPhase < 7).length },
                { icon: Calendar, label: "Completed", value: projects.filter(p => p.currentPhase === 7).length },
                { icon: Clock, label: "This Month", value: projects.length }
              ].map((stat, index) => (
                <div key={index} className="glass-card rounded-xl p-4">
                  <div className="flex items-center gap-3">
                    <div className="w-10 h-10 rounded-lg bg-secondary flex items-center justify-center">
                      <stat.icon className="w-5 h-5 text-primary" />
                    </div>
                    <div>
                      <p className="text-2xl font-display font-bold">{stat.value}</p>
                      <p className="text-xs text-muted-foreground">{stat.label}</p>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          )}

          {/* Recent Projects */}
          {projects.length > 0 && (
            <div className="space-y-4 mb-10">
              <h2 className="text-2xl font-display font-bold gradient-text">Recent Projects</h2>
              
              <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-6">
                {projects.map((project) => (
                  <div 
                    key={project.id}
                    className="glass-card rounded-2xl p-6 hover:shadow-lg transition-all duration-300 group"
                  >
                    <div className="mb-4">
                      <h3 className="text-xl font-display font-bold gradient-text mb-1">{project.name}</h3>
                      <div className="flex items-center gap-2 text-sm text-muted-foreground">
                        <Clock className="w-4 h-4" />
                        <span>Updated {project.updatedAt}</span>
                      </div>
                    </div>
                    
                    <div className="space-y-3 mb-6">
                      <div className="flex items-center justify-between text-sm">
                        <span className="text-muted-foreground">Progress</span>
                        <span className="font-medium gradient-text">Phase {project.currentPhase}/{project.totalPhases}</span>
                      </div>
                      <Progress value={(project.currentPhase / project.totalPhases) * 100} className="h-2" />
                      <div className="flex items-center gap-2 text-sm">
                        <div className="w-2 h-2 rounded-full bg-primary" />
                        <span className="text-muted-foreground">{project.phaseName}</span>
                      </div>
                    </div>
                    
                    <div className="flex items-center gap-2">
                      <Link to={`/project/${project.id}`} className="flex-1">
                        <Button className="w-full aiba-button-primary gap-2">
                          <MessageSquare className="w-4 h-4" />
                          Continue
                        </Button>
                      </Link>
                      <Button variant="outline" size="icon">
                        <FileText className="w-4 h-4" />
                      </Button>
                      <Button 
                        variant="outline" 
                        size="icon"
                        onClick={() => deleteProject(project.id)}
                        className="hover:bg-destructive hover:text-destructive-foreground hover:border-destructive"
                      >
                        <Trash2 className="w-4 h-4" />
                      </Button>
                    </div>
                  </div>
                ))}
                
                {/* Add New Card */}
                {projects.length < 6 && (
                  <button
                    onClick={() => setIsDialogOpen(true)}
                    className="glass-card rounded-2xl p-6 border-2 border-dashed border-border hover:border-primary/50 transition-all duration-300 flex flex-col items-center justify-center gap-4 min-h-[250px] group"
                  >
                    <div className="w-14 h-14 rounded-2xl bg-secondary flex items-center justify-center group-hover:bg-primary/10 transition-colors">
                      <Plus className="w-7 h-7 text-primary" />
                    </div>
                    <p className="font-medium text-muted-foreground group-hover:text-foreground transition-colors">
                      Create New Project
                    </p>
                  </button>
                )}
              </div>
            </div>
          )}

          {/* Analytics Charts */}
          {projects.length > 0 && (
            <div className="grid md:grid-cols-2 gap-6">
              <ProjectTimelineChart projects={projects} />
              <PhaseDistributionChart projects={projects} />
              <ProgressTrendChart projects={projects} />
            </div>
          )}
        </div>
      </main>
    </div>
  );
};

export default Dashboard;