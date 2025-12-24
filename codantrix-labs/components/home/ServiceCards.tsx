'use client';

import { motion } from 'framer-motion';
import Link from 'next/link';
import { Bot, Code, Smartphone, Building2, Link as LinkIcon, Headphones } from 'lucide-react';
import Card from '@/components/ui/Card';

const services = [
  {
    icon: Bot,
    title: 'Custom AI Solutions',
    description: 'Tailored artificial intelligence systems designed to solve your unique business challenges.',
    href: '/ai-solutions',
    color: 'text-[var(--color-primary)]'
  },
  {
    icon: Code,
    title: 'Web Development',
    description: 'Modern, responsive web applications built with cutting-edge technologies.',
    href: '/services#web',
    color: 'text-[var(--color-info)]'
  },
  {
    icon: Smartphone,
    title: 'Mobile Apps',
    description: 'Native and cross-platform mobile applications for iOS and Android.',
    href: '/services#mobile',
    color: 'text-[var(--color-success)]'
  },
  {
    icon: Building2,
    title: 'Enterprise Software',
    description: 'Scalable enterprise solutions designed for complex business operations.',
    href: '/services#enterprise',
    color: 'text-[var(--color-warning)]'
  },
  {
    icon: LinkIcon,
    title: 'Integration Ecosystem',
    description: 'Seamless integration with your existing tools and platforms.',
    href: '/services#integration',
    color: 'text-[var(--color-error)]'
  },
  {
    icon: Headphones,
    title: 'Support & Services',
    description: '24/7 dedicated support and maintenance for your peace of mind.',
    href: '/services#support',
    color: 'text-[var(--color-info)]'
  },
];

export default function ServiceCards() {
  return (
    <section className="py-20 bg-[var(--color-background)]">
      <div className="container mx-auto px-4">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          transition={{ duration: 0.6 }}
          className="text-center mb-12"
        >
          <h2 className="text-4xl md:text-5xl font-bold mb-4 text-[var(--color-foreground)]">
            Our Services
          </h2>
          <p className="text-xl text-[var(--color-muted)] max-w-2xl mx-auto">
            Comprehensive technology solutions for businesses of all sizes
          </p>
        </motion.div>
        
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-8">
          {services.map((service, index) => (
            <motion.div
              key={service.title}
              initial={{ opacity: 0, y: 20 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true }}
              transition={{ duration: 0.4, delay: index * 0.1 }}
            >
              <Link href={service.href}>
                <Card hoverable className="h-full">
                  <service.icon className={`w-12 h-12 mb-4 ${service.color}`} />
                  <h3 className="text-xl font-bold mb-2 text-[var(--color-foreground)]">
                    {service.title}
                  </h3>
                  <p className="text-[var(--color-muted)]">
                    {service.description}
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
