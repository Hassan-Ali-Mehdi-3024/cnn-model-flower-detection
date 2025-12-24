'use client';

import { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import Link from 'next/link';
import { Rocket } from 'lucide-react';
import Card from '@/components/ui/Card';
import Button from '@/components/ui/Button';
import Input from '@/components/ui/Input';
import Spinner from '@/components/ui/Spinner';
import { supabase } from '@/lib/supabase';

export default function ProductsTeaser() {
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
    // TODO: Implement waitlist signup
    setSubscribed(true);
    setTimeout(() => {
      setSubscribed(false);
      setEmail('');
    }, 3000);
  };
  
  if (loading) {
    return (
      <section className="py-20 bg-[var(--color-background)]">
        <div className="container mx-auto px-4">
          <Spinner size="lg" className="mx-auto" />
        </div>
      </section>
    );
  }
  
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
            Products We're Building
          </h2>
          <p className="text-xl text-[var(--color-muted)] max-w-2xl mx-auto">
            {productsVisible 
              ? 'Explore our innovative solutions'
              : 'Exciting new products launching soon'
            }
          </p>
        </motion.div>
        
        {!productsVisible ? (
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            transition={{ duration: 0.6, delay: 0.2 }}
            className="max-w-2xl mx-auto"
          >
            <Card className="text-center p-12">
              <div className="w-20 h-20 bg-[var(--color-primary)]/10 rounded-full flex items-center justify-center mx-auto mb-6">
                <Rocket className="w-10 h-10 text-[var(--color-primary)]" />
              </div>
              
              <h3 className="text-2xl font-bold mb-4 text-[var(--color-foreground)]">
                Coming Soon
              </h3>
              <p className="text-[var(--color-muted)] mb-8">
                We're building amazing products that will revolutionize how you work. 
                Join our waitlist to be notified when they launch.
              </p>
              
              <form onSubmit={handleWaitlist} className="max-w-md mx-auto">
                <div className="flex gap-2">
                  <Input
                    type="email"
                    placeholder="Enter your email"
                    value={email}
                    onChange={(e) => setEmail(e.target.value)}
                    required
                    className="flex-1"
                  />
                  <Button type="submit" variant="primary">
                    {subscribed ? 'Joined!' : 'Join Waitlist'}
                  </Button>
                </div>
              </form>
            </Card>
          </motion.div>
        ) : (
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-8">
            {/* Placeholder for actual products when visible */}
            <Card hoverable className="text-center p-8">
              <div className="text-4xl mb-4">🚀</div>
              <h3 className="text-xl font-bold mb-2 text-[var(--color-foreground)]">Product 1</h3>
              <p className="text-[var(--color-muted)]">Coming soon...</p>
            </Card>
          </div>
        )}
        
        <div className="text-center mt-12">
          <Link href="/products">
            <Button variant="outline" size="lg">
              Explore Products
            </Button>
          </Link>
        </div>
      </div>
    </section>
  );
}
