'use client';

import { motion } from 'framer-motion';
import Link from 'next/link';
import Button from '@/components/ui/Button';

export default function Hero() {
  return (
    <section className="relative min-h-screen flex items-center justify-center overflow-hidden bg-gradient-to-br from-[var(--color-background)] via-[var(--color-background)] to-[var(--color-card-background)]">
      {/* Animated Background */}
      <div className="absolute inset-0 opacity-20">
        <div className="absolute top-1/4 left-1/4 w-96 h-96 bg-[var(--color-primary)] rounded-full blur-3xl animate-pulse" />
        <div className="absolute bottom-1/4 right-1/4 w-96 h-96 bg-[var(--color-info)] rounded-full blur-3xl animate-pulse delay-1000" />
      </div>
      
      <div className="container mx-auto px-4 relative z-10">
        <div className="grid lg:grid-cols-2 gap-12 items-center">
          {/* Left: Text Content */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6 }}
          >
            <h1 className="text-5xl md:text-6xl lg:text-7xl font-bold mb-6 text-[var(--color-foreground)]">
              AI That <span className="text-[var(--color-primary)]">Actually Works</span>
            </h1>
            <p className="text-xl md:text-2xl text-[var(--color-muted)] mb-8">
              Custom AI solutions powered by cutting-edge technology.
              Transform your business with intelligent automation.
            </p>
            <div className="flex flex-col sm:flex-row gap-4">
              <Link href="/ai-solutions">
                <Button variant="primary" size="lg" className="w-full sm:w-auto">
                  Explore AI Solutions
                </Button>
              </Link>
              <Link href="/contact">
                <Button variant="outline" size="lg" className="w-full sm:w-auto">
                  Request Demo
                </Button>
              </Link>
            </div>
          </motion.div>
          
          {/* Right: Interactive Demo Placeholder */}
          <motion.div
            initial={{ opacity: 0, scale: 0.9 }}
            animate={{ opacity: 1, scale: 1 }}
            transition={{ duration: 0.6, delay: 0.2 }}
            className="relative"
          >
            <div className="relative aspect-square max-w-lg mx-auto">
              {/* Neural Network Visualization */}
              <svg
                viewBox="0 0 400 400"
                className="w-full h-full"
                xmlns="http://www.w3.org/2000/svg"
              >
                {/* Nodes */}
                {[...Array(3)].map((_, layer) => (
                  <g key={layer}>
                    {[...Array(layer === 1 ? 5 : 3)].map((_, node) => (
                      <motion.circle
                        key={`${layer}-${node}`}
                        cx={100 + layer * 100}
                        cy={100 + node * 60}
                        r="15"
                        fill="var(--color-primary)"
                        initial={{ opacity: 0 }}
                        animate={{ opacity: [0.3, 1, 0.3] }}
                        transition={{
                          duration: 2,
                          repeat: Infinity,
                          delay: layer * 0.2 + node * 0.1,
                        }}
                      />
                    ))}
                  </g>
                ))}
                
                {/* Connections */}
                {[...Array(3)].map((_, i) => (
                  <g key={`layer-${i}`}>
                    {[...Array(i === 0 ? 5 : 3)].map((_, j) => (
                      <motion.line
                        key={`${i}-${j}`}
                        x1={100 + i * 100}
                        y1={100 + j * 60}
                        x2={200 + i * 100}
                        y2={100 + (j + 1) * 60}
                        stroke="var(--color-primary)"
                        strokeWidth="2"
                        opacity="0.2"
                        initial={{ pathLength: 0 }}
                        animate={{ pathLength: 1 }}
                        transition={{
                          duration: 2,
                          repeat: Infinity,
                          delay: i * 0.3,
                        }}
                      />
                    ))}
                  </g>
                ))}
              </svg>
              
              {/* Floating Cards */}
              <motion.div
                className="absolute top-0 right-0 bg-[var(--color-card-background)] p-4 rounded-lg shadow-lg border border-[var(--color-border)]"
                animate={{ y: [0, -10, 0] }}
                transition={{ duration: 3, repeat: Infinity }}
              >
                <div className="text-sm text-[var(--color-muted)]">AI Accuracy</div>
                <div className="text-2xl font-bold text-[var(--color-primary)]">98.7%</div>
              </motion.div>
              
              <motion.div
                className="absolute bottom-0 left-0 bg-[var(--color-card-background)] p-4 rounded-lg shadow-lg border border-[var(--color-border)]"
                animate={{ y: [0, 10, 0] }}
                transition={{ duration: 3, repeat: Infinity, delay: 0.5 }}
              >
                <div className="text-sm text-[var(--color-muted)]">Processing Speed</div>
                <div className="text-2xl font-bold text-[var(--color-success)]">2.3ms</div>
              </motion.div>
            </div>
          </motion.div>
        </div>
      </div>
      
      {/* Scroll Indicator */}
      <motion.div
        className="absolute bottom-8 left-1/2 transform -translate-x-1/2"
        animate={{ y: [0, 10, 0] }}
        transition={{ duration: 2, repeat: Infinity }}
      >
        <div className="w-6 h-10 border-2 border-[var(--color-primary)] rounded-full flex justify-center">
          <div className="w-1 h-3 bg-[var(--color-primary)] rounded-full mt-2 animate-pulse" />
        </div>
      </motion.div>
    </section>
  );
}
