'use client';

import { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import { Rocket } from 'lucide-react';
import Card from '@/components/ui/Card';
import Button from '@/components/ui/Button';
import Input from '@/components/ui/Input';
import Spinner from '@/components/ui/Spinner';
import { supabase } from '@/lib/supabase';

export default function ProductsPage() {
  const [productsVisible, setProductsVisible] = useState(false);
  const [loading, setLoading] = useState(true);
  const [email, setEmail] = useState('');
  const [subscribed, setSubscribed] = useState(false);
  
  useEffect(() => {
    fetchProductsVisibility();
  }, []);
  
  const fetchProductsVisibility = async () => {
    try {
      const { data, error } = await supabase
        .from('admin_settings')
        .select('setting_value')
        .eq('setting_key', 'products_visible')
        .single();
      
      if (error) {
        console.log('Products visibility setting not found, defaulting to false');
        setProductsVisible(false);
      } else {
        setProductsVisible(data?.setting_value?.visible || false);
      }
    } catch (err) {
      console.error('Error fetching products visibility:', err);
      setProductsVisible(false);
    } finally {
      setLoading(false);
    }
  };
  
  const handleWaitlist = async (e: React.FormEvent) => {
    e.preventDefault();
    // TODO: Implement waitlist signup to Supabase
    setSubscribed(true);
    setTimeout(() => {
      setSubscribed(false);
      setEmail('');
    }, 3000);
  };
  
  if (loading) {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <Spinner size="lg" />
      </div>
    );
  }
  
  return (
    <div className="min-h-screen py-20">
      <div className="container mx-auto px-4">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6 }}
          className="max-w-4xl mx-auto"
        >
          <h1 className="text-5xl md:text-6xl font-bold mb-6 text-center text-[var(--color-foreground)]">
            Our <span className="text-[var(--color-primary)]">Products</span>
          </h1>
          
          {!productsVisible ? (
            <div className="mt-12">
              <Card className="text-center p-12">
                <div className="w-24 h-24 bg-[var(--color-primary)]/10 rounded-full flex items-center justify-center mx-auto mb-6">
                  <Rocket className="w-12 h-12 text-[var(--color-primary)]" />
                </div>
                
                <h2 className="text-3xl font-bold mb-4 text-[var(--color-foreground)]">
                  Coming Soon
                </h2>
                <p className="text-xl text-[var(--color-muted)] mb-8 max-w-2xl mx-auto">
                  We're building amazing products that will revolutionize how you work with AI and technology. 
                  Be the first to know when they launch by joining our waitlist.
                </p>
                
                <div className="max-w-md mx-auto">
                  <form onSubmit={handleWaitlist} className="flex gap-2">
                    <Input
                      type="email"
                      placeholder="Enter your email"
                      value={email}
                      onChange={(e) => setEmail(e.target.value)}
                      required
                      className="flex-1"
                    />
                    <Button type="submit" variant="primary" size="lg">
                      {subscribed ? 'Joined!' : 'Join Waitlist'}
                    </Button>
                  </form>
                  
                  {subscribed && (
                    <motion.p
                      initial={{ opacity: 0, y: 10 }}
                      animate={{ opacity: 1, y: 0 }}
                      className="mt-4 text-[var(--color-success)]"
                    >
                      ✓ You're on the list! We'll notify you when products launch.
                    </motion.p>
                  )}
                </div>
              </Card>
              
              {/* What to Expect */}
              <div className="mt-16 grid grid-cols-1 md:grid-cols-3 gap-8">
                <Card className="text-center p-6">
                  <div className="text-4xl mb-4">🤖</div>
                  <h3 className="text-xl font-bold mb-2 text-[var(--color-foreground)]">AI-Powered Tools</h3>
                  <p className="text-[var(--color-muted)]">
                    Intelligent solutions that learn and adapt to your needs
                  </p>
                </Card>
                
                <Card className="text-center p-6">
                  <div className="text-4xl mb-4">⚡</div>
                  <h3 className="text-xl font-bold mb-2 text-[var(--color-foreground)]">Lightning Fast</h3>
                  <p className="text-[var(--color-muted)]">
                    Optimized for speed and performance at scale
                  </p>
                </Card>
                
                <Card className="text-center p-6">
                  <div className="text-4xl mb-4">🔒</div>
                  <h3 className="text-xl font-bold mb-2 text-[var(--color-foreground)]">Enterprise Security</h3>
                  <p className="text-[var(--color-muted)]">
                    Built with security and compliance in mind
                  </p>
                </Card>
              </div>
            </div>
          ) : (
            <div className="mt-12">
              <p className="text-center text-xl text-[var(--color-muted)] mb-8">
                Explore our innovative products and solutions
              </p>
              
              {/* Placeholder for actual products grid */}
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-8">
                <Card hoverable className="p-6">
                  <div className="text-4xl mb-4">🚀</div>
                  <h3 className="text-xl font-bold mb-2 text-[var(--color-foreground)]">Product 1</h3>
                  <p className="text-[var(--color-muted)] mb-4">Product description coming soon...</p>
                  <Button variant="outline" size="sm">Learn More</Button>
                </Card>
              </div>
            </div>
          )}
        </motion.div>
      </div>
    </div>
  );
}
