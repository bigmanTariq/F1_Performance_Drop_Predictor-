#!/usr/bin/env python3
"""
Code Quality and Documentation Review Script

This script performs comprehensive code quality checks:
1. Docstring coverage and quality
2. Type hint coverage
3. Error handling validation
4. Code complexity analysis
5. Requirements compliance verification
"""

import ast
import os
import sys
import re
import json
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'code_quality_review_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class CodeQualityReviewer:
    """Comprehensive code quality and documentation reviewer"""
    
    def __init__(self):
        self.project_root = Path.cwd()
        self.src_dir = self.project_root / "src"
        self.review_results = {}
        
    def analyze_file(self, filepath: Path) -> Dict[str, Any]:
        """Analyze a single Python file for code quality metrics"""
        
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Parse AST
            tree = ast.parse(content)
            
            # Initialize metrics
            metrics = {
                'filepath': str(filepath),
                'lines_of_code': len(content.splitlines()),
                'functions': [],
                'classes': [],
                'imports': [],
                'docstring_coverage': 0,
                'type_hint_coverage': 0,
                'error_handling_score': 0,
                'complexity_score': 0,
                'issues': []
            }
            
            # Analyze AST nodes
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    func_info = self.analyze_function(node, content)
                    metrics['functions'].append(func_info)
                elif isinstance(node, ast.ClassDef):
                    class_info = self.analyze_class(node, content)
                    metrics['classes'].append(class_info)
                elif isinstance(node, (ast.Import, ast.ImportFrom)):
                    metrics['imports'].append(self.get_import_info(node))
            
            # Calculate coverage metrics
            metrics['docstring_coverage'] = self.calculate_docstring_coverage(metrics)
            metrics['type_hint_coverage'] = self.calculate_type_hint_coverage(metrics)
            metrics['error_handling_score'] = self.calculate_error_handling_score(content)
            metrics['complexity_score'] = self.calculate_complexity_score(metrics)
            
            # Check for common issues
            metrics['issues'] = self.find_code_issues(content, metrics)
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error analyzing {filepath}: {e}")
            return {'filepath': str(filepath), 'error': str(e)}
    
    def analyze_function(self, node: ast.FunctionDef, content: str) -> Dict[str, Any]:
        """Analyze a function for quality metrics"""
        
        func_info = {
            'name': node.name,
            'line_number': node.lineno,
            'has_docstring': ast.get_docstring(node) is not None,
            'docstring_quality': 0,
            'has_type_hints': False,
            'parameter_count': len(node.args.args),
            'has_return_annotation': node.returns is not None,
            'complexity': 1  # Base complexity
        }
        
        # Check docstring quality
        docstring = ast.get_docstring(node)
        if docstring:
            func_info['docstring_quality'] = self.assess_docstring_quality(docstring)
        
        # Check type hints
        func_info['has_type_hints'] = self.check_function_type_hints(node)
        
        # Calculate cyclomatic complexity
        func_info['complexity'] = self.calculate_function_complexity(node)
        
        return func_info
    
    def analyze_class(self, node: ast.ClassDef, content: str) -> Dict[str, Any]:
        """Analyze a class for quality metrics"""
        
        class_info = {
            'name': node.name,
            'line_number': node.lineno,
            'has_docstring': ast.get_docstring(node) is not None,
            'docstring_quality': 0,
            'method_count': 0,
            'methods': []
        }
        
        # Check docstring quality
        docstring = ast.get_docstring(node)
        if docstring:
            class_info['docstring_quality'] = self.assess_docstring_quality(docstring)
        
        # Analyze methods
        for item in node.body:
            if isinstance(item, ast.FunctionDef):
                method_info = self.analyze_function(item, content)
                class_info['methods'].append(method_info)
                class_info['method_count'] += 1
        
        return class_info
    
    def get_import_info(self, node: ast.AST) -> Dict[str, Any]:
        """Extract import information"""
        
        if isinstance(node, ast.Import):
            return {
                'type': 'import',
                'modules': [alias.name for alias in node.names]
            }
        elif isinstance(node, ast.ImportFrom):
            return {
                'type': 'from_import',
                'module': node.module,
                'names': [alias.name for alias in node.names]
            }
    
    def assess_docstring_quality(self, docstring: str) -> int:
        """Assess docstring quality on a scale of 1-5"""
        
        if not docstring:
            return 0
        
        score = 1  # Base score for having a docstring
        
        # Check for description
        if len(docstring.strip()) > 20:
            score += 1
        
        # Check for Args section
        if 'Args:' in docstring or 'Parameters:' in docstring:
            score += 1
        
        # Check for Returns section
        if 'Returns:' in docstring or 'Return:' in docstring:
            score += 1
        
        # Check for examples or detailed explanation
        if 'Example:' in docstring or len(docstring) > 100:
            score += 1
        
        return min(score, 5)
    
    def check_function_type_hints(self, node: ast.FunctionDef) -> bool:
        """Check if function has proper type hints"""
        
        # Check parameter type hints
        for arg in node.args.args:
            if not arg.annotation:
                return False
        
        # Check return type hint (except for __init__)
        if node.name != '__init__' and not node.returns:
            return False
        
        return True
    
    def calculate_function_complexity(self, node: ast.FunctionDef) -> int:
        """Calculate cyclomatic complexity of a function"""
        
        complexity = 1  # Base complexity
        
        for child in ast.walk(node):
            if isinstance(child, (ast.If, ast.While, ast.For, ast.AsyncFor)):
                complexity += 1
            elif isinstance(child, ast.ExceptHandler):
                complexity += 1
            elif isinstance(child, (ast.And, ast.Or)):
                complexity += 1
        
        return complexity
    
    def calculate_docstring_coverage(self, metrics: Dict[str, Any]) -> float:
        """Calculate percentage of functions/classes with docstrings"""
        
        total_items = len(metrics['functions']) + len(metrics['classes'])
        if total_items == 0:
            return 100.0
        
        documented_items = 0
        
        for func in metrics['functions']:
            if func['has_docstring']:
                documented_items += 1
        
        for cls in metrics['classes']:
            if cls['has_docstring']:
                documented_items += 1
        
        return (documented_items / total_items) * 100
    
    def calculate_type_hint_coverage(self, metrics: Dict[str, Any]) -> float:
        """Calculate percentage of functions with type hints"""
        
        total_functions = len(metrics['functions'])
        if total_functions == 0:
            return 100.0
        
        typed_functions = sum(1 for func in metrics['functions'] if func['has_type_hints'])
        return (typed_functions / total_functions) * 100
    
    def calculate_error_handling_score(self, content: str) -> int:
        """Calculate error handling score based on try/except usage"""
        
        try_count = content.count('try:')
        except_count = content.count('except')
        logger_count = content.count('logger.')
        
        # Basic scoring
        score = 0
        if try_count > 0:
            score += 2
        if except_count >= try_count:
            score += 2
        if logger_count > 0:
            score += 1
        
        return min(score, 5)
    
    def calculate_complexity_score(self, metrics: Dict[str, Any]) -> int:
        """Calculate overall complexity score"""
        
        if not metrics['functions']:
            return 5
        
        avg_complexity = sum(func['complexity'] for func in metrics['functions']) / len(metrics['functions'])
        
        if avg_complexity <= 3:
            return 5
        elif avg_complexity <= 5:
            return 4
        elif avg_complexity <= 8:
            return 3
        elif avg_complexity <= 12:
            return 2
        else:
            return 1
    
    def find_code_issues(self, content: str, metrics: Dict[str, Any]) -> List[str]:
        """Find common code quality issues"""
        
        issues = []
        
        # Check for long lines
        lines = content.splitlines()
        for i, line in enumerate(lines, 1):
            if len(line) > 120:
                issues.append(f"Line {i}: Line too long ({len(line)} characters)")
        
        # Check for missing error handling in critical functions
        critical_patterns = ['open(', 'pd.read_csv', 'json.load', 'requests.']
        for pattern in critical_patterns:
            if pattern in content and 'try:' not in content:
                issues.append(f"Missing error handling for {pattern}")
        
        # Check for hardcoded paths
        if '/Users/' in content or 'C:\\' in content:
            issues.append("Hardcoded file paths detected")
        
        # Check for print statements (should use logging)
        if 'print(' in content and 'src/serve.py' not in metrics['filepath']:
            issues.append("Print statements found (should use logging)")
        
        # Check for TODO/FIXME comments
        todo_count = content.upper().count('TODO')
        fixme_count = content.upper().count('FIXME')
        if todo_count > 0:
            issues.append(f"{todo_count} TODO comments found")
        if fixme_count > 0:
            issues.append(f"{fixme_count} FIXME comments found")
        
        return issues
    
    def review_requirements_compliance(self) -> Dict[str, Any]:
        """Check compliance with project requirements"""
        
        logger.info("Reviewing requirements compliance...")
        
        compliance = {
            'requirement_5_5': {'status': 'checking', 'details': []},
            'requirement_6_5': {'status': 'checking', 'details': []}
        }
        
        # Requirement 5.5: Well-commented code and Python best practices
        src_files = list(self.src_dir.glob("*.py"))
        
        total_docstring_coverage = 0
        total_type_hint_coverage = 0
        total_files = 0
        
        for filepath in src_files:
            if filepath.name.startswith('__'):
                continue
            
            metrics = self.analyze_file(filepath)
            if 'error' not in metrics:
                total_docstring_coverage += metrics['docstring_coverage']
                total_type_hint_coverage += metrics['type_hint_coverage']
                total_files += 1
        
        if total_files > 0:
            avg_docstring_coverage = total_docstring_coverage / total_files
            avg_type_hint_coverage = total_type_hint_coverage / total_files
            
            compliance['requirement_5_5']['details'].append(f"Average docstring coverage: {avg_docstring_coverage:.1f}%")
            compliance['requirement_5_5']['details'].append(f"Average type hint coverage: {avg_type_hint_coverage:.1f}%")
            
            if avg_docstring_coverage >= 80 and avg_type_hint_coverage >= 70:
                compliance['requirement_5_5']['status'] = 'passed'
            else:
                compliance['requirement_5_5']['status'] = 'needs_improvement'
        
        # Requirement 6.5: All requirements met and documented
        required_files = [
            'src/data_prep.py',
            'src/features.py',
            'src/train.py',
            'src/predict.py',
            'src/serve.py',
            'README.md',
            'GRADER_WRITEUP.md',
            'peer_review_instructions.md'
        ]
        
        missing_files = []
        for file_path in required_files:
            if not (self.project_root / file_path).exists():
                missing_files.append(file_path)
        
        if not missing_files:
            compliance['requirement_6_5']['status'] = 'passed'
            compliance['requirement_6_5']['details'].append("All required files present")
        else:
            compliance['requirement_6_5']['status'] = 'failed'
            compliance['requirement_6_5']['details'].append(f"Missing files: {missing_files}")
        
        return compliance
    
    def generate_quality_report(self) -> Dict[str, Any]:
        """Generate comprehensive code quality report"""
        
        logger.info("Generating code quality report...")
        
        # Analyze all Python files in src/
        src_files = list(self.src_dir.glob("*.py"))
        file_analyses = {}
        
        for filepath in src_files:
            if filepath.name.startswith('__'):
                continue
            
            logger.info(f"Analyzing {filepath.name}...")
            analysis = self.analyze_file(filepath)
            file_analyses[filepath.name] = analysis
        
        # Calculate overall metrics
        overall_metrics = self.calculate_overall_metrics(file_analyses)
        
        # Check requirements compliance
        compliance = self.review_requirements_compliance()
        
        # Generate report
        report = {
            'review_timestamp': datetime.now().isoformat(),
            'project_root': str(self.project_root),
            'files_analyzed': len(file_analyses),
            'overall_metrics': overall_metrics,
            'file_analyses': file_analyses,
            'requirements_compliance': compliance,
            'recommendations': self.generate_recommendations(file_analyses, overall_metrics)
        }
        
        return report
    
    def calculate_overall_metrics(self, file_analyses: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate overall project metrics"""
        
        total_lines = 0
        total_functions = 0
        total_classes = 0
        total_docstring_coverage = 0
        total_type_hint_coverage = 0
        total_issues = 0
        
        valid_files = 0
        
        for filename, analysis in file_analyses.items():
            if 'error' in analysis:
                continue
            
            valid_files += 1
            total_lines += analysis['lines_of_code']
            total_functions += len(analysis['functions'])
            total_classes += len(analysis['classes'])
            total_docstring_coverage += analysis['docstring_coverage']
            total_type_hint_coverage += analysis['type_hint_coverage']
            total_issues += len(analysis['issues'])
        
        if valid_files == 0:
            return {'error': 'No valid files analyzed'}
        
        return {
            'total_lines_of_code': total_lines,
            'total_functions': total_functions,
            'total_classes': total_classes,
            'average_docstring_coverage': total_docstring_coverage / valid_files,
            'average_type_hint_coverage': total_type_hint_coverage / valid_files,
            'total_issues': total_issues,
            'files_with_issues': sum(1 for analysis in file_analyses.values() 
                                   if 'issues' in analysis and analysis['issues']),
            'quality_score': self.calculate_quality_score(file_analyses)
        }
    
    def calculate_quality_score(self, file_analyses: Dict[str, Any]) -> float:
        """Calculate overall quality score (0-100)"""
        
        scores = []
        
        for analysis in file_analyses.values():
            if 'error' in analysis:
                continue
            
            file_score = 0
            
            # Docstring coverage (30%)
            file_score += (analysis['docstring_coverage'] / 100) * 30
            
            # Type hint coverage (25%)
            file_score += (analysis['type_hint_coverage'] / 100) * 25
            
            # Error handling (20%)
            file_score += (analysis['error_handling_score'] / 5) * 20
            
            # Complexity (15%)
            file_score += (analysis['complexity_score'] / 5) * 15
            
            # Issues penalty (10%)
            issue_penalty = min(len(analysis['issues']) * 2, 10)
            file_score += max(0, 10 - issue_penalty)
            
            scores.append(file_score)
        
        return sum(scores) / len(scores) if scores else 0
    
    def generate_recommendations(self, file_analyses: Dict[str, Any], 
                               overall_metrics: Dict[str, Any]) -> List[str]:
        """Generate improvement recommendations"""
        
        recommendations = []
        
        # Docstring recommendations
        if overall_metrics['average_docstring_coverage'] < 80:
            recommendations.append(
                f"Improve docstring coverage from {overall_metrics['average_docstring_coverage']:.1f}% to at least 80%"
            )
        
        # Type hint recommendations
        if overall_metrics['average_type_hint_coverage'] < 70:
            recommendations.append(
                f"Add type hints to improve coverage from {overall_metrics['average_type_hint_coverage']:.1f}% to at least 70%"
            )
        
        # Issue recommendations
        if overall_metrics['total_issues'] > 0:
            recommendations.append(
                f"Address {overall_metrics['total_issues']} code quality issues across {overall_metrics['files_with_issues']} files"
            )
        
        # File-specific recommendations
        for filename, analysis in file_analyses.items():
            if 'error' in analysis:
                continue
            
            if analysis['docstring_coverage'] < 50:
                recommendations.append(f"{filename}: Add docstrings to functions and classes")
            
            if analysis['type_hint_coverage'] < 50:
                recommendations.append(f"{filename}: Add type hints to function parameters and returns")
            
            if analysis['error_handling_score'] < 3:
                recommendations.append(f"{filename}: Improve error handling with try/except blocks")
        
        return recommendations
    
    def save_report(self, report: Dict[str, Any]) -> str:
        """Save the quality report to file"""
        
        report_file = f"code_quality_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Code quality report saved to: {report_file}")
        return report_file
    
    def print_summary(self, report: Dict[str, Any]) -> None:
        """Print a summary of the code quality review"""
        
        print("\n" + "="*70)
        print("CODE QUALITY AND DOCUMENTATION REVIEW SUMMARY")
        print("="*70)
        
        overall = report['overall_metrics']
        
        print(f"Files Analyzed: {report['files_analyzed']}")
        print(f"Total Lines of Code: {overall['total_lines_of_code']:,}")
        print(f"Total Functions: {overall['total_functions']}")
        print(f"Total Classes: {overall['total_classes']}")
        print(f"Overall Quality Score: {overall['quality_score']:.1f}/100")
        
        print("\nCoverage Metrics:")
        print(f"  Docstring Coverage: {overall['average_docstring_coverage']:.1f}%")
        print(f"  Type Hint Coverage: {overall['average_type_hint_coverage']:.1f}%")
        
        print(f"\nIssues Found: {overall['total_issues']}")
        print(f"Files with Issues: {overall['files_with_issues']}")
        
        print("\nRequirements Compliance:")
        compliance = report['requirements_compliance']
        for req, status in compliance.items():
            print(f"  {req}: {status['status'].upper()}")
        
        if report['recommendations']:
            print("\nRecommendations:")
            for i, rec in enumerate(report['recommendations'], 1):
                print(f"  {i}. {rec}")
        
        print("="*70)
    
    def run_review(self) -> bool:
        """Run the complete code quality review"""
        
        logger.info("Starting code quality and documentation review...")
        
        try:
            report = self.generate_quality_report()
            report_file = self.save_report(report)
            self.print_summary(report)
            
            # Determine if review passed
            overall_score = report['overall_metrics']['quality_score']
            compliance_passed = all(
                status['status'] in ['passed', 'checking'] 
                for status in report['requirements_compliance'].values()
            )
            
            if overall_score >= 75 and compliance_passed:
                logger.info("✅ Code quality review PASSED!")
                return True
            else:
                logger.warning("⚠️ Code quality review needs improvement")
                return False
                
        except Exception as e:
            logger.error(f"Code quality review failed: {e}")
            return False

def main():
    """Main execution function"""
    reviewer = CodeQualityReviewer()
    success = reviewer.run_review()
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()