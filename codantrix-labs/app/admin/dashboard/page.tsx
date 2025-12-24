'use client';

import { motion } from 'framer-motion';
import Link from 'next/link';
import { Settings, Users, MessageSquare, Eye } from 'lucide-react';
import Card from '@/components/ui/Card';
import Button from '@/components/ui/Button';
import { getAdminEmail } from '@/lib/auth';

const stats = [
  { label: 'Total Inquiries', value: '24', icon: MessageSquare, color: 'text-[var(--color-primary)]' },
  { label: 'New Messages', value: '5', icon: Users, color: 'text-[var(--color-success)]' },
  { label: 'Page Views (Month)', value: '12.5K', icon: Eye, color: 'text-[var(--color-info)]' },
];

const quickActions = [
  { label: 'Manage Settings', href: '/admin/settings', icon: Settings },
  { label: 'View Site', href: '/', icon: Eye },
];

export default function AdminDashboard() {
  const adminEmail = getAdminEmail();
  const adminName = adminEmail?.split('@')[0] || 'Admin';
  
  return (
    <div className="max-w-7xl mx-auto">
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.6 }}
      >
        <h1 className="text-4xl font-bold mb-2 text-[var(--color-foreground)]">
          Welcome back, {adminName}!
        </h1>
        <p className="text-[var(--color-muted)] mb-8">
          Here's what's happening with your site today.
        </p>
        
        {/* Stats Grid */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-8">
          {stats.map((stat, index) => (
            <motion.div
              key={stat.label}
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.4, delay: index * 0.1 }}
            >
              <Card className="p-6">
                <div className="flex items-center justify-between">
                  <div>
                    <p className="text-sm text-[var(--color-muted)] mb-1">{stat.label}</p>
                    <p className="text-3xl font-bold text-[var(--color-foreground)]">{stat.value}</p>
                  </div>
                  <div className={`w-12 h-12 rounded-full bg-[var(--color-background)] flex items-center justify-center`}>
                    <stat.icon className={`w-6 h-6 ${stat.color}`} />
                  </div>
                </div>
              </Card>
            </motion.div>
          ))}
        </div>
        
        {/* Quick Actions */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6, delay: 0.3 }}
        >
          <h2 className="text-2xl font-bold mb-4 text-[var(--color-foreground)]">Quick Actions</h2>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {quickActions.map((action) => (
              <Link key={action.label} href={action.href}>
                <Card hoverable className="p-6">
                  <div className="flex items-center gap-4">
                    <div className="w-12 h-12 rounded-full bg-[var(--color-primary)]/10 flex items-center justify-center">
                      <action.icon className="w-6 h-6 text-[var(--color-primary)]" />
                    </div>
                    <span className="text-lg font-semibold text-[var(--color-foreground)]">
                      {action.label}
                    </span>
                  </div>
                </Card>
              </Link>
            ))}
          </div>
        </motion.div>
        
        {/* Recent Activity */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6, delay: 0.4 }}
          className="mt-8"
        >
          <h2 className="text-2xl font-bold mb-4 text-[var(--color-foreground)]">Recent Activity</h2>
          <Card className="p-6">
            <div className="text-center py-8 text-[var(--color-muted)]">
              <p>No recent activity to display</p>
              <p className="text-sm mt-2">Activity will appear here as users interact with your site</p>
            </div>
          </Card>
        </motion.div>
      </motion.div>
    </div>
  );
}
