'use client';

import { useEffect, useState } from 'react';
import { useRouter, usePathname } from 'next/navigation';
import Link from 'next/link';
import { LayoutDashboard, Settings, LogOut } from 'lucide-react';
import { isAuthenticated, logout, getAdminEmail } from '@/lib/auth';
import Button from '@/components/ui/Button';

export default function AdminLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  const router = useRouter();
  const pathname = usePathname();
  const [mounted, setMounted] = useState(false);
  const [adminEmail, setAdminEmail] = useState('');
  
  useEffect(() => {
    setMounted(true);
    
    // Skip auth check for login page
    if (pathname === '/admin/login') {
      return;
    }
    
    // Check authentication
    if (!isAuthenticated()) {
      router.push('/admin/login');
    } else {
      setAdminEmail(getAdminEmail() || '');
    }
  }, [pathname, router]);
  
  const handleLogout = () => {
    logout();
    router.push('/admin/login');
  };
  
  // Don't render layout for login page
  if (pathname === '/admin/login') {
    return <>{children}</>;
  }
  
  // Don't render until mounted (avoid hydration mismatch)
  if (!mounted || !isAuthenticated()) {
    return null;
  }
  
  const navItems = [
    { href: '/admin/dashboard', label: 'Dashboard', icon: LayoutDashboard },
    { href: '/admin/settings', label: 'Settings', icon: Settings },
  ];
  
  return (
    <div className="min-h-screen bg-[var(--color-background)]">
      {/* Admin Navbar */}
      <nav className="bg-[var(--color-card-background)] border-b border-[var(--color-border)]">
        <div className="container mx-auto px-4">
          <div className="flex items-center justify-between h-16">
            <div className="flex items-center space-x-8">
              <Link href="/admin/dashboard" className="flex items-center space-x-2">
                <div className="w-8 h-8 bg-[var(--color-primary)] rounded-lg flex items-center justify-center">
                  <span className="text-white font-bold">C</span>
                </div>
                <span className="font-bold text-[var(--color-foreground)]">Admin</span>
              </Link>
              
              <div className="hidden md:flex items-center space-x-1">
                {navItems.map((item) => (
                  <Link
                    key={item.href}
                    href={item.href}
                    className={`flex items-center gap-2 px-4 py-2 rounded-lg transition-colors ${
                      pathname === item.href
                        ? 'bg-[var(--color-primary)] text-white'
                        : 'text-[var(--color-foreground)] hover:bg-[var(--color-background)]'
                    }`}
                  >
                    <item.icon className="w-4 h-4" />
                    {item.label}
                  </Link>
                ))}
              </div>
            </div>
            
            <div className="flex items-center gap-4">
              <span className="text-sm text-[var(--color-muted)] hidden sm:block">
                {adminEmail}
              </span>
              <Button
                variant="outline"
                size="sm"
                onClick={handleLogout}
                className="flex items-center gap-2"
              >
                <LogOut className="w-4 h-4" />
                Logout
              </Button>
            </div>
          </div>
        </div>
      </nav>
      
      {/* Main Content */}
      <main className="container mx-auto px-4 py-8">
        {children}
      </main>
    </div>
  );
}
