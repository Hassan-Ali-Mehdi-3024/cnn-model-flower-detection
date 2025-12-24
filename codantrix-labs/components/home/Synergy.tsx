'use client';

import { motion } from 'framer-motion';

export default function Synergy() {
  return (
    <section className="py-20 bg-[var(--color-card-background)]">
      <div className="container mx-auto px-4">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          transition={{ duration: 0.6 }}
          className="max-w-3xl mx-auto text-center"
        >
          <h2 className="text-4xl md:text-5xl font-bold mb-6 text-[var(--color-foreground)]">
            Built on Solid <span className="text-[var(--color-primary)]">SaaS Foundation</span>
          </h2>
          <p className="text-xl text-[var(--color-muted)] mb-8">
            Our AI solutions are powered by robust, scalable infrastructure. 
            We combine artificial intelligence with enterprise-grade software engineering 
            to deliver solutions that are not just smart, but reliable, secure, and built to scale.
          </p>
          
          {/* Visual Divider */}
          <div className="flex items-center justify-center gap-4 mt-12">
            <div className="h-1 w-20 bg-gradient-to-r from-transparent to-[var(--color-primary)] rounded-full" />
            <div className="w-3 h-3 bg-[var(--color-primary)] rounded-full animate-pulse" />
            <div className="h-1 w-20 bg-gradient-to-l from-transparent to-[var(--color-primary)] rounded-full" />
          </div>
        </motion.div>
      </div>
    </section>
  );
}
