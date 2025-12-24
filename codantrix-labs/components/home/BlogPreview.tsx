'use client';

import { motion } from 'framer-motion';
import Link from 'next/link';
import { Calendar, User, ArrowRight } from 'lucide-react';
import Card from '@/components/ui/Card';
import Button from '@/components/ui/Button';

const blogPosts = [
  {
    id: 1,
    title: 'The Future of AI in Business: Trends to Watch in 2024',
    excerpt: 'Explore the latest AI trends that are transforming how businesses operate and compete in the digital age.',
    author: 'Sarah Johnson',
    date: '2024-01-15',
    image: '🤖',
    slug: 'future-of-ai-business-2024'
  },
  {
    id: 2,
    title: 'Building Scalable SaaS Applications: Best Practices',
    excerpt: 'Learn the essential patterns and practices for building SaaS applications that scale effortlessly.',
    author: 'Michael Chen',
    date: '2024-01-12',
    image: '🚀',
    slug: 'building-scalable-saas-applications'
  },
  {
    id: 3,
    title: 'How Machine Learning Can Reduce Operational Costs',
    excerpt: 'Discover practical ways to leverage ML to optimize operations and significantly reduce business costs.',
    author: 'Emily Rodriguez',
    date: '2024-01-10',
    image: '💡',
    slug: 'ml-reduce-operational-costs'
  },
];

export default function BlogPreview() {
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
            Latest Insights
          </h2>
          <p className="text-xl text-[var(--color-muted)] max-w-2xl mx-auto">
            Expert perspectives on AI, technology, and business innovation
          </p>
        </motion.div>
        
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-8">
          {blogPosts.map((post, index) => (
            <motion.div
              key={post.id}
              initial={{ opacity: 0, y: 20 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true }}
              transition={{ duration: 0.4, delay: index * 0.1 }}
            >
              <Link href={`/blog/${post.slug}`}>
                <Card hoverable className="h-full flex flex-col">
                  {/* Featured Image Placeholder */}
                  <div className="w-full h-48 bg-gradient-to-br from-[var(--color-primary)]/20 to-[var(--color-info)]/20 rounded-lg mb-4 flex items-center justify-center text-6xl">
                    {post.image}
                  </div>
                  
                  <h3 className="text-xl font-bold mb-3 text-[var(--color-foreground)] line-clamp-2">
                    {post.title}
                  </h3>
                  
                  <p className="text-[var(--color-muted)] mb-4 flex-grow line-clamp-3">
                    {post.excerpt}
                  </p>
                  
                  <div className="flex items-center justify-between text-sm text-[var(--color-muted)] pt-4 border-t border-[var(--color-border)]">
                    <div className="flex items-center gap-2">
                      <User className="w-4 h-4" />
                      <span>{post.author}</span>
                    </div>
                    <div className="flex items-center gap-2">
                      <Calendar className="w-4 h-4" />
                      <span>{new Date(post.date).toLocaleDateString()}</span>
                    </div>
                  </div>
                  
                  <div className="mt-4">
                    <span className="text-[var(--color-primary)] font-medium inline-flex items-center gap-2 group-hover:gap-3 transition-all">
                      Read more
                      <ArrowRight className="w-4 h-4" />
                    </span>
                  </div>
                </Card>
              </Link>
            </motion.div>
          ))}
        </div>
        
        <div className="text-center mt-12">
          <Link href="/blog">
            <Button variant="outline" size="lg">
              View All Articles
            </Button>
          </Link>
        </div>
      </div>
    </section>
  );
}
