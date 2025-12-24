'use client';

import { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { ChevronLeft, ChevronRight } from 'lucide-react';
import Link from 'next/link';
import Card from '@/components/ui/Card';
import Badge from '@/components/ui/Badge';
import Button from '@/components/ui/Button';

const caseStudies = [
  {
    id: 1,
    title: 'AI-Powered Customer Service',
    problem: 'E-commerce company struggling with 10,000+ daily customer inquiries',
    solution: 'Implemented custom AI chatbot with natural language processing',
    results: '80% reduction in response time, 95% customer satisfaction',
    serviceType: 'AI',
    metrics: { time: '-80%', satisfaction: '95%', cost: '-60%' }
  },
  {
    id: 2,
    title: 'Enterprise Resource Planning System',
    problem: 'Manufacturing company with fragmented legacy systems',
    solution: 'Built integrated ERP platform with real-time analytics',
    results: '40% improvement in operational efficiency',
    serviceType: 'SaaS',
    metrics: { efficiency: '+40%', costs: '-25%', visibility: '100%' }
  },
  {
    id: 3,
    title: 'Mobile-First Healthcare Platform',
    problem: 'Healthcare provider needed HIPAA-compliant patient portal',
    solution: 'Developed secure mobile app with AI-powered diagnostics support',
    results: '200K+ active users, 4.8★ app store rating',
    serviceType: 'Integrated',
    metrics: { users: '200K+', rating: '4.8★', engagement: '+150%' }
  },
];

export default function CaseStudyCarousel() {
  const [currentIndex, setCurrentIndex] = useState(0);
  
  const next = () => {
    setCurrentIndex((prev) => (prev + 1) % caseStudies.length);
  };
  
  const prev = () => {
    setCurrentIndex((prev) => (prev - 1 + caseStudies.length) % caseStudies.length);
  };
  
  const currentStudy = caseStudies[currentIndex];
  
  return (
    <section className="py-20 bg-[var(--color-card-background)]">
      <div className="container mx-auto px-4">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          transition={{ duration: 0.6 }}
          className="text-center mb-12"
        >
          <h2 className="text-4xl md:text-5xl font-bold mb-4 text-[var(--color-foreground)]">
            Featured Success Stories
          </h2>
          <p className="text-xl text-[var(--color-muted)] max-w-2xl mx-auto">
            Real results from real businesses
          </p>
        </motion.div>
        
        <div className="max-w-4xl mx-auto">
          <div className="relative">
            <AnimatePresence mode="wait">
              <motion.div
                key={currentStudy.id}
                initial={{ opacity: 0, x: 100 }}
                animate={{ opacity: 1, x: 0 }}
                exit={{ opacity: 0, x: -100 }}
                transition={{ duration: 0.3 }}
              >
                <Card className="p-8">
                  <div className="flex items-center justify-between mb-6">
                    <h3 className="text-2xl font-bold text-[var(--color-foreground)]">
                      {currentStudy.title}
                    </h3>
                    <Badge variant={currentStudy.serviceType === 'AI' ? 'info' : currentStudy.serviceType === 'SaaS' ? 'success' : 'warning'}>
                      {currentStudy.serviceType}
                    </Badge>
                  </div>
                  
                  <div className="space-y-4 mb-6">
                    <div>
                      <h4 className="font-semibold text-[var(--color-foreground)] mb-2">Problem</h4>
                      <p className="text-[var(--color-muted)]">{currentStudy.problem}</p>
                    </div>
                    
                    <div>
                      <h4 className="font-semibold text-[var(--color-foreground)] mb-2">Solution</h4>
                      <p className="text-[var(--color-muted)]">{currentStudy.solution}</p>
                    </div>
                    
                    <div>
                      <h4 className="font-semibold text-[var(--color-foreground)] mb-2">Results</h4>
                      <p className="text-[var(--color-primary)] font-semibold">{currentStudy.results}</p>
                    </div>
                  </div>
                  
                  <div className="grid grid-cols-3 gap-4 p-4 bg-[var(--color-background)] rounded-lg">
                    {Object.entries(currentStudy.metrics).map(([key, value]) => (
                      <div key={key} className="text-center">
                        <div className="text-2xl font-bold text-[var(--color-primary)]">{value}</div>
                        <div className="text-sm text-[var(--color-muted)] capitalize">{key}</div>
                      </div>
                    ))}
                  </div>
                </Card>
              </motion.div>
            </AnimatePresence>
            
            {/* Navigation */}
            <div className="flex items-center justify-between mt-8">
              <button
                onClick={prev}
                className="p-2 rounded-lg bg-[var(--color-card-background)] border border-[var(--color-border)] hover:bg-[var(--color-background)] transition-colors"
                aria-label="Previous case study"
              >
                <ChevronLeft className="w-6 h-6 text-[var(--color-foreground)]" />
              </button>
              
              <div className="flex gap-2">
                {caseStudies.map((_, index) => (
                  <button
                    key={index}
                    onClick={() => setCurrentIndex(index)}
                    className={`w-2 h-2 rounded-full transition-all ${
                      index === currentIndex
                        ? 'bg-[var(--color-primary)] w-8'
                        : 'bg-[var(--color-muted)]'
                    }`}
                    aria-label={`Go to case study ${index + 1}`}
                  />
                ))}
              </div>
              
              <button
                onClick={next}
                className="p-2 rounded-lg bg-[var(--color-card-background)] border border-[var(--color-border)] hover:bg-[var(--color-background)] transition-colors"
                aria-label="Next case study"
              >
                <ChevronRight className="w-6 h-6 text-[var(--color-foreground)]" />
              </button>
            </div>
          </div>
          
          <div className="text-center mt-8">
            <Link href="/case-studies">
              <Button variant="outline" size="lg">
                View All Case Studies
              </Button>
            </Link>
          </div>
        </div>
      </div>
    </section>
  );
}
