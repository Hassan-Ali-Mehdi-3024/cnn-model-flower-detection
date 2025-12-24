'use client';

import { useState, useEffect } from 'react';
import Link from 'next/link';
import { Menu, X, Sun, Moon, ChevronDown } from 'lucide-react';
import { useTheme } from '@/lib/theme';
import { cn } from '@/lib/utils';
import Button from '@/components/ui/Button';

export default function Navbar() {
  const [isOpen, setIsOpen] = useState(false);
  const [isScrolled, setIsScrolled] = useState(false);
  const { theme, toggleTheme } = useTheme();
  
  useEffect(() => {
    const handleScroll = () => {
      setIsScrolled(window.scrollY > 10);
    };
    
    window.addEventListener('scroll', handleScroll);
    return () => window.removeEventListener('scroll', handleScroll);
  }, []);
  
  const navLinks = [
    { 
      name: 'AI Solutions', 
      href: '/ai-solutions',
      dropdown: [
        { name: 'Custom AI Development', href: '/ai-solutions#custom' },
        { name: 'Machine Learning', href: '/ai-solutions#ml' },
        { name: 'AI Integration', href: '/ai-solutions#integration' },
      ]
    },
    { 
      name: 'SaaS Services', 
      href: '/services',
      dropdown: [
        { name: 'Web Development', href: '/services#web' },
        { name: 'Mobile Apps', href: '/services#mobile' },
        { name: 'Enterprise Solutions', href: '/services#enterprise' },
      ]
    },
    { name: 'Products', href: '/products' },
    { name: 'Case Studies', href: '/case-studies' },
    { 
      name: 'Resources', 
      href: '/resources',
      dropdown: [
        { name: 'Blog', href: '/blog' },
        { name: 'Documentation', href: '/docs' },
      ]
    },
    { name: 'About', href: '/about' },
    { name: 'Careers', href: '/careers' },
  ];
  
  return (
    <nav
      className={cn(
        'fixed top-0 left-0 right-0 z-50 transition-all duration-300',
        isScrolled 
          ? 'bg-[var(--color-background)]/95 backdrop-blur-md shadow-md' 
          : 'bg-transparent'
      )}
    >
      <div className="container mx-auto px-4">
        <div className="flex items-center justify-between h-16">
          {/* Logo */}
          <Link href="/" className="flex items-center space-x-2">
            <div className="w-10 h-10 bg-[var(--color-primary)] rounded-lg flex items-center justify-center">
              <span className="text-white font-bold text-xl">C</span>
            </div>
            <span className="text-xl font-bold text-[var(--color-foreground)]">
              Codantrix Labs
            </span>
          </Link>
          
          {/* Desktop Navigation */}
          <div className="hidden lg:flex items-center space-x-8">
            {navLinks.map((link) => (
              <div key={link.name} className="relative group">
                <Link
                  href={link.href}
                  className="text-[var(--color-foreground)] hover:text-[var(--color-primary)] transition-colors flex items-center"
                >
                  {link.name}
                  {link.dropdown && <ChevronDown className="ml-1 w-4 h-4" />}
                </Link>
                
                {link.dropdown && (
                  <div className="absolute top-full left-0 mt-2 w-48 bg-[var(--color-card-background)] border border-[var(--color-border)] rounded-lg shadow-lg opacity-0 invisible group-hover:opacity-100 group-hover:visible transition-all duration-200">
                    {link.dropdown.map((item) => (
                      <Link
                        key={item.name}
                        href={item.href}
                        className="block px-4 py-2 text-[var(--color-foreground)] hover:bg-[var(--color-background)] hover:text-[var(--color-primary)] transition-colors first:rounded-t-lg last:rounded-b-lg"
                      >
                        {item.name}
                      </Link>
                    ))}
                  </div>
                )}
              </div>
            ))}
          </div>
          
          {/* Right Side Actions */}
          <div className="flex items-center space-x-4">
            {/* Theme Toggle */}
            <button
              onClick={toggleTheme}
              className="p-2 rounded-lg hover:bg-[var(--color-card-background)] transition-colors"
              aria-label="Toggle theme"
            >
              {theme === 'light' ? (
                <Moon className="w-5 h-5 text-[var(--color-foreground)]" />
              ) : (
                <Sun className="w-5 h-5 text-[var(--color-foreground)]" />
              )}
            </button>
            
            {/* Contact CTA */}
            <Link href="/contact" className="hidden lg:block">
              <Button variant="primary" size="md">
                Contact
              </Button>
            </Link>
            
            {/* Mobile Menu Toggle */}
            <button
              onClick={() => setIsOpen(!isOpen)}
              className="lg:hidden p-2 rounded-lg hover:bg-[var(--color-card-background)] transition-colors"
              aria-label="Toggle menu"
            >
              {isOpen ? (
                <X className="w-6 h-6 text-[var(--color-foreground)]" />
              ) : (
                <Menu className="w-6 h-6 text-[var(--color-foreground)]" />
              )}
            </button>
          </div>
        </div>
        
        {/* Mobile Menu */}
        {isOpen && (
          <div className="lg:hidden py-4 border-t border-[var(--color-border)]">
            {navLinks.map((link) => (
              <div key={link.name} className="py-2">
                <Link
                  href={link.href}
                  className="block text-[var(--color-foreground)] hover:text-[var(--color-primary)] transition-colors"
                  onClick={() => setIsOpen(false)}
                >
                  {link.name}
                </Link>
                {link.dropdown && (
                  <div className="ml-4 mt-2 space-y-2">
                    {link.dropdown.map((item) => (
                      <Link
                        key={item.name}
                        href={item.href}
                        className="block text-sm text-[var(--color-muted)] hover:text-[var(--color-primary)] transition-colors"
                        onClick={() => setIsOpen(false)}
                      >
                        {item.name}
                      </Link>
                    ))}
                  </div>
                )}
              </div>
            ))}
            <div className="mt-4">
              <Link href="/contact" onClick={() => setIsOpen(false)}>
                <Button variant="primary" size="md" className="w-full">
                  Contact
                </Button>
              </Link>
            </div>
          </div>
        )}
      </div>
    </nav>
  );
}
