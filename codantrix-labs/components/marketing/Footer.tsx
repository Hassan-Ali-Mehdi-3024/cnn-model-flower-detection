'use client';

import Link from 'next/link';
import { Github, Twitter, Linkedin, Mail } from 'lucide-react';
import Button from '@/components/ui/Button';
import Input from '@/components/ui/Input';
import { useState } from 'react';

export default function Footer() {
  const [email, setEmail] = useState('');
  const [subscribed, setSubscribed] = useState(false);
  
  const handleSubscribe = (e: React.FormEvent) => {
    e.preventDefault();
    // TODO: Implement newsletter subscription
    setSubscribed(true);
    setTimeout(() => setSubscribed(false), 3000);
  };
  
  const footerLinks = {
    'AI Solutions': [
      { name: 'Custom AI Development', href: '/ai-solutions#custom' },
      { name: 'Machine Learning', href: '/ai-solutions#ml' },
      { name: 'AI Integration', href: '/ai-solutions#integration' },
    ],
    'SaaS Services': [
      { name: 'Web Development', href: '/services#web' },
      { name: 'Mobile Apps', href: '/services#mobile' },
      { name: 'Enterprise Solutions', href: '/services#enterprise' },
    ],
    'Company': [
      { name: 'About Us', href: '/about' },
      { name: 'Careers', href: '/careers' },
      { name: 'Case Studies', href: '/case-studies' },
      { name: 'Contact', href: '/contact' },
    ],
    'Resources': [
      { name: 'Blog', href: '/blog' },
      { name: 'Documentation', href: '/docs' },
      { name: 'Support', href: '/support' },
      { name: 'Privacy Policy', href: '/privacy' },
    ],
  };
  
  return (
    <footer className="bg-[var(--color-card-background)] border-t border-[var(--color-border)] mt-20">
      <div className="container mx-auto px-4 py-12">
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-6 gap-8">
          {/* Company Info */}
          <div className="lg:col-span-2">
            <Link href="/" className="flex items-center space-x-2 mb-4">
              <div className="w-10 h-10 bg-[var(--color-primary)] rounded-lg flex items-center justify-center">
                <span className="text-white font-bold text-xl">C</span>
              </div>
              <span className="text-xl font-bold text-[var(--color-foreground)]">
                Codantrix Labs
              </span>
            </Link>
            <p className="text-[var(--color-muted)] mb-4">
              Custom AI solutions and enterprise SaaS services for modern businesses.
              Building the future with cutting-edge technology.
            </p>
            <div className="flex space-x-4">
              <a 
                href="https://github.com" 
                target="_blank" 
                rel="noopener noreferrer"
                className="text-[var(--color-muted)] hover:text-[var(--color-primary)] transition-colors"
              >
                <Github className="w-5 h-5" />
              </a>
              <a 
                href="https://twitter.com" 
                target="_blank" 
                rel="noopener noreferrer"
                className="text-[var(--color-muted)] hover:text-[var(--color-primary)] transition-colors"
              >
                <Twitter className="w-5 h-5" />
              </a>
              <a 
                href="https://linkedin.com" 
                target="_blank" 
                rel="noopener noreferrer"
                className="text-[var(--color-muted)] hover:text-[var(--color-primary)] transition-colors"
              >
                <Linkedin className="w-5 h-5" />
              </a>
              <a 
                href="mailto:hello@codantrix.com"
                className="text-[var(--color-muted)] hover:text-[var(--color-primary)] transition-colors"
              >
                <Mail className="w-5 h-5" />
              </a>
            </div>
          </div>
          
          {/* Links */}
          {Object.entries(footerLinks).map(([category, links]) => (
            <div key={category}>
              <h3 className="font-semibold text-[var(--color-foreground)] mb-4">
                {category}
              </h3>
              <ul className="space-y-2">
                {links.map((link) => (
                  <li key={link.name}>
                    <Link
                      href={link.href}
                      className="text-[var(--color-muted)] hover:text-[var(--color-primary)] transition-colors"
                    >
                      {link.name}
                    </Link>
                  </li>
                ))}
              </ul>
            </div>
          ))}
        </div>
        
        {/* Newsletter */}
        <div className="mt-12 pt-8 border-t border-[var(--color-border)]">
          <div className="max-w-md">
            <h3 className="font-semibold text-[var(--color-foreground)] mb-2">
              Subscribe to our newsletter
            </h3>
            <p className="text-[var(--color-muted)] text-sm mb-4">
              Get the latest insights on AI and technology delivered to your inbox.
            </p>
            <form onSubmit={handleSubscribe} className="flex gap-2">
              <Input
                type="email"
                placeholder="Enter your email"
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                required
                className="flex-1"
              />
              <Button type="submit" variant="primary">
                {subscribed ? 'Subscribed!' : 'Subscribe'}
              </Button>
            </form>
          </div>
        </div>
        
        {/* Copyright */}
        <div className="mt-8 pt-8 border-t border-[var(--color-border)] text-center text-[var(--color-muted)] text-sm">
          <p>© {new Date().getFullYear()} Codantrix Labs. All rights reserved.</p>
        </div>
      </div>
    </footer>
  );
}
