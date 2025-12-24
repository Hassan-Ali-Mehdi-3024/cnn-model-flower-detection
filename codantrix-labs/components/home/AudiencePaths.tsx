'use client';

import { motion } from 'framer-motion';
import Link from 'next/link';
import { Briefcase, Code2, GraduationCap, Handshake } from 'lucide-react';
import Card from '@/components/ui/Card';

const paths = [
  {
    icon: Briefcase,
    title: 'For Businesses',
    description: 'Transform your operations with AI-powered solutions and custom software.',
    href: '/ai-solutions',
    color: 'text-[var(--color-primary)]'
  },
  {
    icon: Code2,
    title: 'For Developers',
    description: 'Access our documentation, APIs, and resources to build amazing things.',
    href: '/resources',
    color: 'text-[var(--color-info)]'
  },
  {
    icon: GraduationCap,
    title: 'For Students',
    description: 'Join our team and learn from industry experts while building the future.',
    href: '/careers',
    color: 'text-[var(--color-success)]'
  },
  {
    icon: Handshake,
    title: 'For Partners',
    description: 'Collaborate with us to deliver cutting-edge solutions to your clients.',
    href: '/services',
    color: 'text-[var(--color-warning)]'
  },
];

export default function AudiencePaths() {
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
            Find Your Path
          </h2>
          <p className="text-xl text-[var(--color-muted)] max-w-2xl mx-auto">
            Whatever your goals, we have something for you
          </p>
        </motion.div>
        
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
          {paths.map((path, index) => (
            <motion.div
              key={path.title}
              initial={{ opacity: 0, y: 20 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true }}
              transition={{ duration: 0.4, delay: index * 0.1 }}
            >
              <Link href={path.href}>
                <Card hoverable className="h-full text-center p-6">
                  <div className={`w-16 h-16 rounded-full bg-[var(--color-background)] flex items-center justify-center mx-auto mb-4`}>
                    <path.icon className={`w-8 h-8 ${path.color}`} />
                  </div>
                  <h3 className="text-xl font-bold mb-2 text-[var(--color-foreground)]">
                    {path.title}
                  </h3>
                  <p className="text-[var(--color-muted)]">
                    {path.description}
                  </p>
                </Card>
              </Link>
            </motion.div>
          ))}
        </div>
      </div>
    </section>
  );
}
