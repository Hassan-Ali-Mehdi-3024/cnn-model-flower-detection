import Hero from '@/components/home/Hero';
import Synergy from '@/components/home/Synergy';
import ServiceCards from '@/components/home/ServiceCards';
import CaseStudyCarousel from '@/components/home/CaseStudyCarousel';
import ProductsTeaser from '@/components/home/ProductsTeaser';
import AudiencePaths from '@/components/home/AudiencePaths';
import BlogPreview from '@/components/home/BlogPreview';
import FinalCTA from '@/components/home/FinalCTA';

export default function Home() {
  return (
    <div className="overflow-x-hidden">
      <Hero />
      <Synergy />
      <ServiceCards />
      <CaseStudyCarousel />
      <ProductsTeaser />
      <AudiencePaths />
      <BlogPreview />
      <FinalCTA />
    </div>
  );
}
