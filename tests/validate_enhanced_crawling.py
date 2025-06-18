#!/usr/bin/env python3
"""
Real-world validation script for enhanced crawling functionality.

This script tests the enhanced crawler against known problematic documentation sites
to validate improvements in content quality and extraction performance.
"""

import asyncio
import os
import json
import time
from pathlib import Path
from typing import Dict, List, Any
from dataclasses import dataclass

# Add src to path for imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from enhanced_crawler_config import detect_framework, DocumentationFramework
from content_quality import calculate_content_quality, ContentQualityMetrics
from smart_crawler_factory import EnhancedCrawler


@dataclass
class ValidationResult:
    """Results from enhanced crawling validation."""
    url: str
    framework_detected: str
    extraction_successful: bool
    quality_metrics: Dict[str, Any]
    performance_metrics: Dict[str, Any]
    improvement_over_baseline: Dict[str, Any]
    errors: List[str]


class EnhancedCrawlingValidator:
    """Validator for enhanced crawling functionality."""
    
    def __init__(self):
        self.test_urls = {
            "n8n_docs": [
                "https://docs.n8n.io/workflows/",
                "https://docs.n8n.io/nodes/",
                "https://docs.n8n.io/integrations/"
            ],
            "virustotal_docs": [
                "https://developers.virustotal.com/reference/overview",
                "https://developers.virustotal.com/reference/files",
                "https://developers.virustotal.com/reference/urls"
            ],
            "other_docs": [
                "https://docs.github.com/en/get-started",
                "https://kubernetes.io/docs/concepts/",
                "https://reactjs.org/docs/getting-started.html"
            ]
        }
        
        self.expected_improvements = {
            "content_navigation_ratio": 0.6,  # Target 60%+ content vs navigation
            "max_link_density": 0.3,         # Target <30% link density
            "min_quality_score": 0.5,        # Target 50%+ overall quality
            "max_extraction_time": 5.0,      # Target <5 seconds per page
        }
    
    async def validate_framework_detection(self) -> Dict[str, Any]:
        """Validate framework detection accuracy."""
        print("🔍 Validating framework detection...")
        
        results = {
            "total_sites": 0,
            "successful_detections": 0,
            "framework_breakdown": {},
            "detection_times": [],
            "errors": []
        }
        
        all_urls = []
        for category, urls in self.test_urls.items():
            all_urls.extend(urls)
        
        results["total_sites"] = len(all_urls)
        
        for url in all_urls:
            try:
                start_time = time.time()
                framework = detect_framework(url)
                detection_time = (time.time() - start_time) * 1000
                
                results["detection_times"].append(detection_time)
                results["successful_detections"] += 1
                
                framework_name = framework.value
                results["framework_breakdown"][framework_name] = \
                    results["framework_breakdown"].get(framework_name, 0) + 1
                
                print(f"✅ {url} -> {framework_name} ({detection_time:.1f}ms)")
                
            except Exception as e:
                results["errors"].append(f"{url}: {str(e)}")
                print(f"❌ {url} -> Error: {str(e)}")
        
        # Calculate statistics
        if results["detection_times"]:
            results["avg_detection_time_ms"] = sum(results["detection_times"]) / len(results["detection_times"])
            results["max_detection_time_ms"] = max(results["detection_times"])
        
        return results
    
    async def validate_enhanced_extraction(self) -> Dict[str, Any]:
        """Validate enhanced extraction quality."""
        print("\n📊 Validating enhanced extraction quality...")
        
        results = {
            "total_extractions": 0,
            "successful_extractions": 0,
            "quality_distribution": {"excellent": 0, "good": 0, "fair": 0, "poor": 0},
            "average_metrics": {},
            "performance_stats": {},
            "validation_results": [],
            "errors": []
        }
        
        # Test on a subset of URLs to avoid overwhelming the servers
        test_urls = [
            "https://docs.n8n.io/workflows/",
            "https://developers.virustotal.com/reference/overview"
        ]
        
        total_quality_scores = []
        total_content_ratios = []
        total_link_densities = []
        total_extraction_times = []
        
        for url in test_urls:
            try:
                print(f"🔄 Testing enhanced extraction: {url}")
                
                start_time = time.time()
                
                async with EnhancedCrawler() as crawler:
                    result = await crawler.crawl_single_page_enhanced(url)
                
                extraction_time = time.time() - start_time
                total_extraction_times.append(extraction_time)
                
                results["total_extractions"] += 1
                
                if result.success and result.quality_metrics:
                    results["successful_extractions"] += 1
                    
                    # Collect quality metrics
                    quality = result.quality_metrics
                    total_quality_scores.append(quality.overall_quality_score)
                    total_content_ratios.append(quality.content_to_navigation_ratio)
                    total_link_densities.append(quality.link_density)
                    
                    # Track quality distribution
                    results["quality_distribution"][quality.quality_category] += 1
                    
                    # Validate against expectations
                    validation_result = ValidationResult(
                        url=url,
                        framework_detected=result.framework.value,
                        extraction_successful=True,
                        quality_metrics={
                            "overall_score": quality.overall_quality_score,
                            "content_navigation_ratio": quality.content_to_navigation_ratio,
                            "link_density": quality.link_density,
                            "word_count": quality.word_count,
                            "category": quality.quality_category
                        },
                        performance_metrics={
                            "extraction_time_seconds": extraction_time,
                            "framework_detection_ms": result.framework_detection_time_ms,
                            "quality_analysis_ms": result.quality_analysis_time_ms,
                            "extraction_attempts": result.extraction_attempts,
                            "used_fallback": result.used_fallback
                        },
                        improvement_over_baseline=self._calculate_improvements(quality),
                        errors=[]
                    )
                    
                    results["validation_results"].append(validation_result)
                    
                    print(f"✅ {url}")
                    print(f"   Framework: {result.framework.value}")
                    print(f"   Quality: {quality.quality_category} ({quality.overall_quality_score:.3f})")
                    print(f"   Content/Nav ratio: {quality.content_to_navigation_ratio:.3f}")
                    print(f"   Extraction time: {extraction_time:.2f}s")
                    
                else:
                    error_msg = f"Extraction failed for {url}"
                    results["errors"].append(error_msg)
                    print(f"❌ {error_msg}")
                
            except Exception as e:
                error_msg = f"{url}: {str(e)}"
                results["errors"].append(error_msg)
                print(f"❌ {error_msg}")
        
        # Calculate average metrics
        if total_quality_scores:
            results["average_metrics"] = {
                "quality_score": sum(total_quality_scores) / len(total_quality_scores),
                "content_navigation_ratio": sum(total_content_ratios) / len(total_content_ratios),
                "link_density": sum(total_link_densities) / len(total_link_densities),
            }
        
        if total_extraction_times:
            results["performance_stats"] = {
                "avg_extraction_time_seconds": sum(total_extraction_times) / len(total_extraction_times),
                "max_extraction_time_seconds": max(total_extraction_times),
                "min_extraction_time_seconds": min(total_extraction_times)
            }
        
        return results
    
    def _calculate_improvements(self, quality_metrics: Any) -> Dict[str, Any]:
        """Calculate improvements over baseline expectations."""
        improvements = {}
        
        # Content-to-navigation ratio improvement
        if quality_metrics.content_to_navigation_ratio >= self.expected_improvements["content_navigation_ratio"]:
            improvements["content_ratio"] = "✅ IMPROVED"
        else:
            improvements["content_ratio"] = "❌ BELOW_TARGET"
        
        # Link density improvement  
        if quality_metrics.link_density <= self.expected_improvements["max_link_density"]:
            improvements["link_density"] = "✅ IMPROVED"
        else:
            improvements["link_density"] = "❌ ABOVE_TARGET"
        
        # Overall quality improvement
        if quality_metrics.overall_quality_score >= self.expected_improvements["min_quality_score"]:
            improvements["overall_quality"] = "✅ IMPROVED"
        else:
            improvements["overall_quality"] = "❌ BELOW_TARGET"
        
        return improvements
    
    async def validate_fallback_mechanisms(self) -> Dict[str, Any]:
        """Validate fallback mechanisms work correctly."""
        print("\n🔄 Validating fallback mechanisms...")
        
        results = {
            "total_tests": 0,
            "fallback_triggered": 0,
            "fallback_successful": 0,
            "errors": []
        }
        
        # Test with a generic site that should trigger fallbacks
        test_url = "https://example.com"  # Simple site, should use generic framework
        
        try:
            async with EnhancedCrawler(max_fallback_attempts=3) as crawler:
                result = await crawler.crawl_single_page_enhanced(test_url)
                
                results["total_tests"] = 1
                
                if result.used_fallback:
                    results["fallback_triggered"] = 1
                    
                    if result.success:
                        results["fallback_successful"] = 1
                        print(f"✅ Fallback mechanism worked for {test_url}")
                        print(f"   Attempts: {result.extraction_attempts}")
                        print(f"   Framework: {result.framework.value}")
                    else:
                        print(f"❌ Fallback triggered but extraction failed for {test_url}")
                else:
                    print(f"ℹ️  No fallback needed for {test_url}")
                
        except Exception as e:
            results["errors"].append(f"{test_url}: {str(e)}")
            print(f"❌ Error testing fallback: {str(e)}")
        
        return results
    
    def generate_report(self, framework_results: Dict, extraction_results: Dict, 
                       fallback_results: Dict) -> str:
        """Generate a comprehensive validation report."""
        report = "\n" + "="*80 + "\n"
        report += "Enhanced Crawling Validation Report\n"
        report += "="*80 + "\n\n"
        
        # Framework Detection Results
        report += "🔍 FRAMEWORK DETECTION RESULTS\n"
        report += f"Total sites tested: {framework_results['total_sites']}\n"
        report += f"Successful detections: {framework_results['successful_detections']}\n"
        
        if framework_results.get('avg_detection_time_ms'):
            report += f"Average detection time: {framework_results['avg_detection_time_ms']:.1f}ms\n"
        
        report += "Framework distribution:\n"
        for framework, count in framework_results['framework_breakdown'].items():
            report += f"  - {framework}: {count} sites\n"
        
        if framework_results['errors']:
            report += f"Errors: {len(framework_results['errors'])}\n"
        
        # Enhanced Extraction Results  
        report += "\n📊 ENHANCED EXTRACTION RESULTS\n"
        report += f"Total extractions: {extraction_results['total_extractions']}\n"
        report += f"Successful extractions: {extraction_results['successful_extractions']}\n"
        
        if extraction_results.get('average_metrics'):
            metrics = extraction_results['average_metrics']
            report += f"Average quality score: {metrics['quality_score']:.3f}\n"
            report += f"Average content/nav ratio: {metrics['content_navigation_ratio']:.3f}\n"
            report += f"Average link density: {metrics['link_density']:.3f}\n"
        
        if extraction_results.get('performance_stats'):
            perf = extraction_results['performance_stats']
            report += f"Average extraction time: {perf['avg_extraction_time_seconds']:.2f}s\n"
        
        report += "Quality distribution:\n"
        for category, count in extraction_results['quality_distribution'].items():
            report += f"  - {category}: {count} extractions\n"
        
        # Improvement Analysis
        report += "\n🎯 IMPROVEMENT ANALYSIS\n"
        for result in extraction_results.get('validation_results', []):
            report += f"\n{result.url}:\n"
            report += f"  Framework: {result.framework_detected}\n"
            report += f"  Quality: {result.quality_metrics['category']} ({result.quality_metrics['overall_score']:.3f})\n"
            
            for metric, status in result.improvement_over_baseline.items():
                report += f"  {metric}: {status}\n"
        
        # Fallback Results
        report += "\n🔄 FALLBACK MECHANISM RESULTS\n"
        report += f"Tests run: {fallback_results['total_tests']}\n"
        report += f"Fallback triggered: {fallback_results['fallback_triggered']}\n"
        report += f"Fallback successful: {fallback_results['fallback_successful']}\n"
        
        # Overall Assessment
        report += "\n📈 OVERALL ASSESSMENT\n"
        
        success_rate = extraction_results['successful_extractions'] / max(extraction_results['total_extractions'], 1)
        report += f"Extraction success rate: {success_rate:.1%}\n"
        
        if extraction_results.get('average_metrics'):
            avg_metrics = extraction_results['average_metrics']
            
            # Check if targets are met
            targets_met = 0
            total_targets = 3
            
            if avg_metrics['content_navigation_ratio'] >= self.expected_improvements['content_navigation_ratio']:
                targets_met += 1
                report += "✅ Content/navigation ratio target MET\n"
            else:
                report += "❌ Content/navigation ratio target MISSED\n"
            
            if avg_metrics['link_density'] <= self.expected_improvements['max_link_density']:
                targets_met += 1
                report += "✅ Link density target MET\n"
            else:
                report += "❌ Link density target MISSED\n"
            
            if avg_metrics['quality_score'] >= self.expected_improvements['min_quality_score']:
                targets_met += 1
                report += "✅ Quality score target MET\n"
            else:
                report += "❌ Quality score target MISSED\n"
            
            report += f"\nTargets achieved: {targets_met}/{total_targets} ({targets_met/total_targets:.1%})\n"
        
        if extraction_results['errors'] or framework_results['errors']:
            total_errors = len(extraction_results['errors']) + len(framework_results['errors'])
            report += f"\nTotal errors encountered: {total_errors}\n"
        
        report += "\n" + "="*80 + "\n"
        
        return report


async def main():
    """Run the enhanced crawling validation."""
    print("🚀 Starting Enhanced Crawling Validation")
    print("This may take a few minutes to complete...\n")
    
    validator = EnhancedCrawlingValidator()
    
    # Run validation tests
    framework_results = await validator.validate_framework_detection()
    extraction_results = await validator.validate_enhanced_extraction()
    fallback_results = await validator.validate_fallback_mechanisms()
    
    # Generate and display report
    report = validator.generate_report(framework_results, extraction_results, fallback_results)
    print(report)
    
    # Save report to file
    report_file = Path(__file__).parent / "enhanced_crawling_validation_report.txt"
    with open(report_file, 'w') as f:
        f.write(report)
    
    print(f"📄 Full report saved to: {report_file}")
    
    # Save detailed results as JSON
    results_file = Path(__file__).parent / "enhanced_crawling_validation_results.json"
    detailed_results = {
        "framework_detection": framework_results,
        "enhanced_extraction": extraction_results,
        "fallback_mechanisms": fallback_results,
        "timestamp": time.time()
    }
    
    with open(results_file, 'w') as f:
        json.dump(detailed_results, f, indent=2, default=str)
    
    print(f"📊 Detailed results saved to: {results_file}")


if __name__ == "__main__":
    # Set environment variable for testing
    os.environ["USE_ENHANCED_CRAWLING"] = "true"
    
    asyncio.run(main())