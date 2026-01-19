import { Link } from "react-router-dom";
import { Sparkles, ArrowRight, MessageSquare, BarChart3, FileText, CheckCircle2, Zap, Shield, Users } from "lucide-react";
import { Button } from "@/components/ui/button";

const LandingPage = () => {
  const features = [
    {
      icon: MessageSquare,
      title: "Smart Chat Interface",
      description: "Not just a chatbot - an intelligent guide that adapts to each phase of your project development."
    },
    {
      icon: BarChart3,
      title: "Progress Dashboard",
      description: "Track your project phases with visual diagrams, timelines, and completion metrics in real-time."
    },
    {
      icon: FileText,
      title: "Auto Documentation",
      description: "Export professional reports and documentation that understand your current work context automatically."
    }
  ];

  const phases = [
    { number: 1, title: "Problem Definition", description: "Clarify the problem and target users" },
    { number: 2, title: "Requirements Analysis", description: "Define functional and non-functional requirements" },
    { number: 3, title: "Market Analysis", description: "Analyze market fit and competitors" },
    { number: 4, title: "Solution Design", description: "Design architecture and tech stack" },
    { number: 5, title: "Prototype Development", description: "Plan implementation approach" },
    { number: 6, title: "Testing & Validation", description: "Create test plans and validation criteria" },
    { number: 7, title: "Documentation", description: "Generate technical and user documentation" }
  ];

  const stats = [
    { value: "500+", label: "Students" },
    { value: "1000+", label: "Projects Created" },
    { value: "98%", label: "Success Rate" },
    { value: "7", label: "AI-Guided Phases" }
  ];

  return (
    <div className="min-h-screen bg-background">
      {/* Navigation */}
      <nav className="fixed top-0 left-0 right-0 z-50 bg-background/80 backdrop-blur-xl border-b border-border">
        <div className="container mx-auto px-6 py-4 flex items-center justify-between">
          <Link to="/" className="flex items-center gap-2">
            <Sparkles className="w-6 h-6 text-primary" />
            <span className="font-display font-bold text-xl">AIBA</span>
          </Link>
          <div className="hidden md:flex items-center gap-8">
            <a href="#features" className="text-muted-foreground hover:text-foreground transition-colors">Features</a>
            <a href="#process" className="text-muted-foreground hover:text-foreground transition-colors">Process</a>
            <a href="#faq" className="text-muted-foreground hover:text-foreground transition-colors">FAQs</a>
          </div>
          <div className="flex items-center gap-4">
            <Link to="/auth">
              <Button variant="ghost">Sign In</Button>
            </Link>
            <Link to="/auth">
              <Button className="aiba-button-primary">Sign Up</Button>
            </Link>
          </div>
        </div>
      </nav>

      {/* Hero Section */}
      <section className="relative pt-32 pb-20 overflow-hidden bg-gradient-to-b from-background to-muted/20">
        {/* Grid Pattern */}
        <div className="absolute inset-0 opacity-60">
          <div 
            className="absolute inset-0" 
            style={{
              backgroundImage: `
                linear-gradient(to right, hsl(var(--border) / 0.3) 1px, transparent 1px),
                linear-gradient(to bottom, hsl(var(--border) / 0.3) 1px, transparent 1px)
              `,
              backgroundSize: '60px 60px'
            }}
          />
        </div>
        
        <div className="container mx-auto px-6 relative">
          <div className="grid lg:grid-cols-2 gap-12 items-center">
            <div className="space-y-8 animate-slide-up">
              <div className="inline-flex items-center gap-2 px-4 py-2 rounded-full bg-background border border-border shadow-sm">
                <Sparkles className="w-4 h-4 text-primary" />
                <span className="text-sm font-medium">AI-Powered Business Analysis</span>
              </div>
              
              <h1 className="text-5xl lg:text-6xl font-display font-bold leading-tight">
                Something that introduce{" "}
                <span className="gradient-text">our product</span>
              </h1>
              
              <p className="text-xl text-muted-foreground max-w-lg">
                Empower your team to brainstorm better and faster with AIBA
              </p>
              
              <div className="flex flex-wrap gap-4">
                <Link to="/dashboard">
                  <Button size="lg" className="aiba-button-primary gap-2">
                    Get Started <ArrowRight className="w-4 h-4" />
                  </Button>
                </Link>
                <Button size="lg" variant="outline" className="bg-background">
                  Watch Demo
                </Button>
              </div>
              
              <div className="flex items-center gap-8 pt-4">
                <div className="space-y-1">
                  <p className="text-3xl font-display font-bold">500+</p>
                  <p className="text-sm text-muted-foreground">Students</p>
                </div>
                <div className="h-12 w-px bg-border" />
                <div className="space-y-1">
                  <p className="text-3xl font-display font-bold">1000+</p>
                  <p className="text-sm text-muted-foreground">Projects Created</p>
                </div>
              </div>
            </div>
            
            {/* Chat Preview Card with Glow */}
            <div className="relative animate-float" style={{ animationDelay: "0.2s" }}>
              {/* Cyan gradient glow behind the card */}
              <div className="absolute -inset-8 bg-gradient-to-br from-primary/30 via-primary/20 to-transparent rounded-[40px] blur-3xl" />
              
              {/* Glowing border wrapper */}
              <div className="relative p-[2px] rounded-2xl bg-gradient-to-br from-primary via-primary/50 to-primary/20 shadow-[0_0_40px_-10px_hsl(var(--primary))]">
                {/* Corner accents */}
                <div className="absolute -top-2 -right-2 w-4 h-4 rounded-full bg-primary shadow-[0_0_12px_hsl(var(--primary))]" />
                <div className="absolute -bottom-2 -right-2 w-3 h-3 rounded-full border-2 border-primary bg-background" />
                
                {/* Card content */}
                <div className="bg-background rounded-2xl p-6">
                  <div className="flex items-center gap-2 mb-6">
                    <span className="text-lg font-medium">Chatbot</span>
                    <Sparkles className="w-5 h-5 text-primary ml-auto" />
                  </div>
                  
                  <div className="bg-foreground text-background rounded-xl p-4 mb-4">
                    <p>What do you have in mind?</p>
                  </div>
                  
                  <div className="flex items-center gap-3 p-3 border border-border rounded-xl bg-background">
                    <input 
                      type="text" 
                      placeholder="Type your idea here..." 
                      className="flex-1 bg-transparent outline-none text-sm"
                      readOnly
                    />
                    <Button size="sm" className="aiba-button-primary gap-2">
                      Brainstorm <Sparkles className="w-3 h-3" />
                    </Button>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* Features Section */}
      <section id="features" className="py-24 relative">
        <div className="absolute inset-0 grid-pattern opacity-30" />
        <div className="container mx-auto px-6 relative">
          <div className="text-center mb-16 space-y-4">
            <h2 className="text-4xl lg:text-5xl font-display font-bold">
              Built for <span className="gradient-text">Engineering Students</span>
            </h2>
            <p className="text-xl text-muted-foreground max-w-2xl mx-auto">
              Everything you need to transform your ideas into comprehensive project plans
            </p>
          </div>
          
          <div className="grid md:grid-cols-3 gap-8">
            {features.map((feature, index) => (
              <div 
                key={index}
                className="glass-card rounded-2xl p-8 hover:shadow-xl transition-all duration-300 hover:-translate-y-1"
                style={{ animationDelay: `${index * 0.1}s` }}
              >
                <div className="w-14 h-14 rounded-xl bg-secondary flex items-center justify-center mb-6">
                  <feature.icon className="w-7 h-7 text-primary" />
                </div>
                <h3 className="text-xl font-display font-bold mb-3">{feature.title}</h3>
                <p className="text-muted-foreground">{feature.description}</p>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* 7-Step Process Section */}
      <section id="process" className="py-24 bg-muted/30">
        <div className="container mx-auto px-6">
          <div className="text-center mb-16 space-y-4">
            <h2 className="text-4xl lg:text-5xl font-display font-bold">
              The <span className="gradient-text">7-Phase</span> Methodology
            </h2>
            <p className="text-xl text-muted-foreground max-w-2xl mx-auto">
              Our structured approach guides you through every stage of project development
            </p>
          </div>
          
          <div className="relative">
            {/* Connection Line */}
            <div className="hidden lg:block absolute top-1/2 left-0 right-0 h-0.5 bg-gradient-to-r from-transparent via-primary/30 to-transparent -translate-y-1/2" />
            
            <div className="grid sm:grid-cols-2 lg:grid-cols-7 gap-6">
              {phases.map((phase, index) => (
                <div 
                  key={index}
                  className="relative group"
                >
                  <div className="glass-card rounded-2xl p-6 h-full hover:shadow-lg transition-all duration-300 hover:-translate-y-2 text-center">
                    <div className="w-12 h-12 rounded-full bg-primary text-primary-foreground flex items-center justify-center mx-auto mb-4 font-display font-bold text-lg group-hover:scale-110 transition-transform">
                      {phase.number}
                    </div>
                    <h3 className="font-display font-bold mb-2 text-sm">{phase.title}</h3>
                    <p className="text-xs text-muted-foreground">{phase.description}</p>
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>
      </section>

      {/* Benefits Section */}
      <section className="py-24">
        <div className="container mx-auto px-6">
          <div className="grid lg:grid-cols-2 gap-16 items-center">
            <div className="space-y-8">
              <h2 className="text-4xl lg:text-5xl font-display font-bold">
                Why choose <span className="gradient-text">AIBA</span>?
              </h2>
              
              <div className="space-y-6">
                {[
                  { icon: Zap, title: "AI-Powered Insights", description: "Get intelligent suggestions and analysis at every phase of your project" },
                  { icon: Shield, title: "Professional Framework", description: "Based on real-world business analysis methodologies used by companies" },
                  { icon: Users, title: "Collaborative", description: "Work together with your team and share project progress seamlessly" }
                ].map((benefit, index) => (
                  <div key={index} className="flex gap-4">
                    <div className="w-12 h-12 rounded-xl bg-secondary flex items-center justify-center flex-shrink-0">
                      <benefit.icon className="w-6 h-6 text-primary" />
                    </div>
                    <div>
                      <h3 className="font-display font-bold mb-1">{benefit.title}</h3>
                      <p className="text-muted-foreground">{benefit.description}</p>
                    </div>
                  </div>
                ))}
              </div>
            </div>
            
            <div className="relative">
              <div className="glass-card rounded-2xl p-8">
                <div className="space-y-4">
                  {phases.slice(0, 4).map((phase, index) => (
                    <div 
                      key={index}
                      className={`flex items-center gap-4 p-4 rounded-xl transition-all ${
                        index === 0 ? "bg-foreground text-background" : "bg-secondary"
                      }`}
                    >
                      <div className={`w-10 h-10 rounded-full flex items-center justify-center ${
                        index === 0 ? "bg-background" : "bg-primary/10"
                      }`}>
                        {index < 3 ? (
                          <CheckCircle2 className={`w-5 h-5 ${index === 0 ? "text-primary" : "text-primary"}`} />
                        ) : (
                          <span className="text-sm font-medium text-muted-foreground">{phase.number}</span>
                        )}
                      </div>
                      <div>
                        <p className="font-medium">{phase.title}</p>
                        <p className={`text-sm ${index === 0 ? "text-background/70" : "text-muted-foreground"}`}>
                          {phase.description}
                        </p>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* Stats Section */}
      <section className="py-16 bg-foreground text-background">
        <div className="container mx-auto px-6">
          <div className="grid grid-cols-2 lg:grid-cols-4 gap-8">
            {stats.map((stat, index) => (
              <div key={index} className="text-center">
                <p className="text-4xl lg:text-5xl font-display font-bold mb-2">{stat.value}</p>
                <p className="text-background/60">{stat.label}</p>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* CTA Section */}
      <section className="py-24 relative overflow-hidden">
        <div className="absolute inset-0 grid-pattern opacity-30" />
        <div className="absolute top-1/2 left-1/2 w-[600px] h-[600px] bg-primary/10 rounded-full blur-3xl -translate-x-1/2 -translate-y-1/2" />
        
        <div className="container mx-auto px-6 relative text-center">
          <h2 className="text-4xl lg:text-5xl font-display font-bold mb-6">
            Ready to start your <span className="gradient-text">project journey</span>?
          </h2>
          <p className="text-xl text-muted-foreground mb-10 max-w-2xl mx-auto">
            Join thousands of students who have transformed their ideas into professional project plans with AIBA.
          </p>
          <Link to="/dashboard">
            <Button size="lg" className="aiba-button-primary gap-2 px-8">
              Get Started Free <ArrowRight className="w-5 h-5" />
            </Button>
          </Link>
        </div>
      </section>

      {/* Footer */}
      <footer className="border-t border-border py-12">
        <div className="container mx-auto px-6">
          <div className="flex flex-col md:flex-row items-center justify-between gap-6">
            <div className="flex items-center gap-2">
              <Sparkles className="w-5 h-5 text-primary" />
              <span className="font-display font-bold">AIBA</span>
            </div>
            <p className="text-sm text-muted-foreground">
              © 2024 AIBA. All rights reserved.
            </p>
            <div className="flex items-center gap-6">
              <a href="#" className="text-sm text-muted-foreground hover:text-foreground transition-colors">Privacy</a>
              <a href="#" className="text-sm text-muted-foreground hover:text-foreground transition-colors">Terms</a>
              <a href="#" className="text-sm text-muted-foreground hover:text-foreground transition-colors">Contact</a>
            </div>
          </div>
        </div>
      </footer>
    </div>
  );
};

export default LandingPage;
