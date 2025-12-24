'use client';

import { useState, useEffect } from 'react';
import { useRouter } from 'next/navigation';
import { motion } from 'framer-motion';
import { Lock } from 'lucide-react';
import Card from '@/components/ui/Card';
import Button from '@/components/ui/Button';
import Input from '@/components/ui/Input';
import { login, isAuthenticated } from '@/lib/auth';

export default function AdminLogin() {
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);
  const router = useRouter();
  
  useEffect(() => {
    // Redirect if already authenticated
    if (isAuthenticated()) {
      router.push('/admin/dashboard');
    }
  }, [router]);
  
  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError('');
    setLoading(true);
    
    try {
      const success = login(email, password);
      
      if (success) {
        router.push('/admin/dashboard');
      } else {
        setError('Invalid email or password');
      }
    } catch (err) {
      setError('An error occurred. Please try again.');
    } finally {
      setLoading(false);
    }
  };
  
  return (
    <div className="min-h-screen flex items-center justify-center bg-gradient-to-br from-[var(--color-background)] to-[var(--color-card-background)]">
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.6 }}
        className="w-full max-w-md px-4"
      >
        <Card className="p-8">
          <div className="text-center mb-8">
            <div className="w-16 h-16 bg-[var(--color-primary)]/10 rounded-full flex items-center justify-center mx-auto mb-4">
              <Lock className="w-8 h-8 text-[var(--color-primary)]" />
            </div>
            <h1 className="text-3xl font-bold text-[var(--color-foreground)] mb-2">
              Admin Login
            </h1>
            <p className="text-[var(--color-muted)]">
              Sign in to access the admin dashboard
            </p>
          </div>
          
          <form onSubmit={handleSubmit} className="space-y-6">
            <Input
              label="Email"
              type="email"
              placeholder="admin@codantrix.com"
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              required
            />
            
            <Input
              label="Password"
              type="password"
              placeholder="Enter your password"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              required
            />
            
            {error && (
              <div className="p-3 bg-[var(--color-error)]/10 border border-[var(--color-error)] rounded-lg text-[var(--color-error)] text-sm">
                {error}
              </div>
            )}
            
            <Button
              type="submit"
              variant="primary"
              size="lg"
              className="w-full"
              loading={loading}
            >
              Sign In
            </Button>
          </form>
          
          <div className="mt-6 text-center text-sm text-[var(--color-muted)]">
            <p>Default credentials:</p>
            <p>Email: admin@codantrix.com</p>
            <p>Password: admin123</p>
          </div>
        </Card>
      </motion.div>
    </div>
  );
}
