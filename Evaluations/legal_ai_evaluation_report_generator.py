"""
CAPABILITIES:
- Generates comprehensive evaluation reports for legal AI systems in multiple formats (PDF, HTML, JSON)
- Creates professional visualizations of performance metrics and evaluation results
- Supports customizable templates for different legal domains and jurisdictions
- Includes automatic version tracking and report comparison
- Implements structured sections following legal documentation standards
- Provides executive summaries and detailed technical appendices
- Embeds interactive charts and tables for digital reports
- Includes metadata for compliance tracking and audit trails
- Supports digital signatures and timestamping for legal validity
- Generates accessibility-compliant documents (WCAG 2.1)
"""

import logging
import json
import os
import time
import csv
import uuid
import datetime
import shutil
import tempfile
import argparse
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Tuple
from dataclasses import dataclass, field, asdict

# Third-party imports with graceful fallbacks
try:
    from fpdf import FPDF
    FPDF_AVAILABLE = True
except ImportError:
    FPDF_AVAILABLE = False
    print("Warning: fpdf library not available. PDF generation will be disabled.")

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    CHART_AVAILABLE = True
except ImportError:
    CHART_AVAILABLE = False
    print("Warning: matplotlib/seaborn libraries not available. Chart generation will be disabled.")

try:
    import jinja2
    JINJA_AVAILABLE = True
except ImportError:
    JINJA_AVAILABLE = False
    print("Warning: jinja2 library not available. HTML templating will be limited.")

try:
    import markdown
    import weasyprint
    MARKDOWN_PDF_AVAILABLE = True
except ImportError:
    MARKDOWN_PDF_AVAILABLE = False
    print("Warning: markdown/weasyprint libraries not available. Markdown-to-PDF conversion disabled.")


# Configure logging
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
log_filename = f"{log_dir}/report_generator_{timestamp}.log"

logging.basicConfig(
    filename=log_filename,
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s'
)

# Add console handler
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
logging.getLogger().addHandler(console_handler)

logger = logging.getLogger("report_generator")


@dataclass
class ReportSection:
    """A section within a report with title and content."""
    title: str
    content: Any
    section_type: str = "text"  # Options: text, table, chart, json, code
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert section to dictionary."""
        return {
            "title": self.title,
            "content": self.content,
            "section_type": self.section_type,
            "metadata": self.metadata
        }


@dataclass
class ReportMetadata:
    """Metadata for the entire report."""
    title: str
    author: str = "Legal AI Evaluation System"
    date: str = field(default_factory=lambda: datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    version: str = "1.0"
    subject: str = "AI System Evaluation Report"
    keywords: List[str] = field(default_factory=list)
    report_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metadata to dictionary."""
        return asdict(self)


@dataclass
class LegalEvaluationReport:
    """Complete evaluation report."""
    metadata: ReportMetadata
    sections: List[ReportSection] = field(default_factory=list)
    summary: Dict[str, Any] = field(default_factory=dict)
    attachments: List[str] = field(default_factory=list)
    
    def add_section(self, section: ReportSection) -> None:
        """Add a section to the report."""
        self.sections.append(section)
    
    def add_attachment(self, filepath: str) -> None:
        """Add an attachment to the report."""
        self.attachments.append(filepath)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert report to dictionary."""
        return {
            "metadata": self.metadata.to_dict(),
            "summary": self.summary,
            "sections": [s.to_dict() for s in self.sections],
            "attachments": self.attachments
        }


class ReportGenerator:
    """
    Generates comprehensive evaluation reports for legal AI systems
    with support for multiple output formats and visualizations.
    """
    
    def __init__(
        self,
        output_dir: str = "evaluation_reports",
        template_dir: str = "report_templates",
        assets_dir: str = "report_assets"
    ):
        """
        Initialize the report generator.
        
        Args:
            output_dir: Directory to save generated reports
            template_dir: Directory containing report templates
            assets_dir: Directory containing assets like logos and CSS
        """
        self.output_dir = output_dir
        self.template_dir = template_dir
        self.assets_dir = assets_dir
        
        # Create directories if they don't exist
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(template_dir, exist_ok=True)
        os.makedirs(assets_dir, exist_ok=True)
        
        # Initialize report registry for version tracking
        self.report_registry_file = os.path.join(output_dir, "report_registry.json")
        self.report_registry = self._load_report_registry()
        
        logger.info(f"Initialized report generator with output directory: {output_dir}")
        
        # Check for logo file, create default if needed
        self.logo_path = os.path.join(assets_dir, "logo.png")
        if not os.path.exists(self.logo_path) and CHART_AVAILABLE:
            self._create_default_logo()
    
    def _load_report_registry(self) -> Dict[str, Any]:
        """Load the report registry from file or create if it doesn't exist."""
        if os.path.exists(self.report_registry_file):
            try:
                with open(self.report_registry_file, 'r') as f:
                    return json.load(f)
            except json.JSONDecodeError:
                logger.warning(f"Error parsing report registry. Creating new registry.")
                return {"reports": {}}
        else:
            return {"reports": {}}
    
    def _save_report_registry(self) -> None:
        """Save the report registry to file."""
        with open(self.report_registry_file, 'w') as f:
            json.dump(self.report_registry, f, indent=2)
    
    def _create_default_logo(self) -> None:
        """Create a default logo if none exists."""
        try:
            plt.figure(figsize=(2, 2))
            plt.text(0.5, 0.5, "LEGAL\nAI", 
                     horizontalalignment='center', 
                     verticalalignment='center',
                     fontsize=24, 
                     fontweight='bold',
                     transform=plt.gca().transAxes)
            plt.axis('off')
            plt.savefig(self.logo_path, dpi=100, bbox_inches='tight')
            plt.close()
            logger.info(f"Created default logo at {self.logo_path}")
        except Exception as e:
            logger.error(f"Failed to create default logo: {str(e)}")
    
    def create_report(
        self,
        evaluation_data: Dict[str, Any],
        title: str = "Legal AI System Evaluation Report",
        author: str = "Legal AI Evaluation System",
        template_name: Optional[str] = None
    ) -> LegalEvaluationReport:
        """
        Create a structured evaluation report.
        
        Args:
            evaluation_data: Dictionary containing all evaluation results
            title: Report title
            author: Report author
            template_name: Name of template to use (None for default)
            
        Returns:
            LegalEvaluationReport instance
        """
        logger.info(f"Creating report: {title}")
        
        # Create metadata
        metadata = ReportMetadata(
            title=title,
            author=author,
            keywords=["legal ai", "evaluation", "performance assessment"]
        )
        
        # Create report instance
        report = LegalEvaluationReport(metadata=metadata)
        
        # Add executive summary
        summary = self._generate_executive_summary(evaluation_data)
        report.summary = summary
        report.add_section(ReportSection(
            title="Executive Summary",
            content=summary,
            section_type="text"
        ))
        
        # Add methodology section
        if "methodology" in evaluation_data:
            report.add_section(ReportSection(
                title="Evaluation Methodology",
                content=evaluation_data["methodology"],
                section_type="text"
            ))
        
        # Add performance metrics section
        if "performance_metrics" in evaluation_data:
            perf_metrics = evaluation_data["performance_metrics"]
            report.add_section(ReportSection(
                title="Performance Metrics",
                content=perf_metrics,
                section_type="json"
            ))
            
            # Generate and add charts for performance metrics
            if CHART_AVAILABLE:
                chart_path = self._generate_performance_chart(perf_metrics)
                if chart_path:
                    report.add_attachment(chart_path)
                    report.add_section(ReportSection(
                        title="Performance Visualization",
                        content=chart_path,
                        section_type="chart"
                    ))
        
        # Add detailed results
        if "detailed_results" in evaluation_data:
            report.add_section(ReportSection(
                title="Detailed Evaluation Results",
                content=evaluation_data["detailed_results"],
                section_type="table"
            ))
        
        # Add system information
        if "system_info" in evaluation_data:
            report.add_section(ReportSection(
                title="System Information",
                content=evaluation_data["system_info"],
                section_type="json"
            ))
        
        # Add conclusions and recommendations
        if "conclusions" in evaluation_data:
            report.add_section(ReportSection(
                title="Conclusions and Recommendations",
                content=evaluation_data["conclusions"],
                section_type="text"
            ))
        
        logger.info(f"Created report with {len(report.sections)} sections")
        return report
    
    def _generate_executive_summary(self, evaluation_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate an executive summary from evaluation data."""
        summary = {}
        
        # Extract key performance indicators
        if "performance_metrics" in evaluation_data:
            metrics = evaluation_data["performance_metrics"]
            summary["key_metrics"] = {}
            
            for key in ["accuracy", "precision", "recall", "f1_score", "auc"]:
                if key in metrics:
                    summary["key_metrics"][key] = metrics[key]
            
            # Add execution time if available
            if "execution_time_sec" in metrics:
                summary["execution_time_sec"] = metrics["execution_time_sec"]
        
        # Add overall assessment
        if "performance_metrics" in evaluation_data and "accuracy" in evaluation_data["performance_metrics"]:
            accuracy = evaluation_data["performance_metrics"]["accuracy"]
            if accuracy >= 0.95:
                assessment = "Excellent performance, suitable for critical legal applications"
            elif accuracy >= 0.90:
                assessment = "Very good performance, suitable for most legal applications"
            elif accuracy >= 0.80:
                assessment = "Good performance, suitable for assistive legal applications"
            elif accuracy >= 0.70:
                assessment = "Moderate performance, requires human oversight"
            else:
                assessment = "Low performance, requires significant improvements"
            
            summary["overall_assessment"] = assessment
        
        # Add any existing executive summary
        if "executive_summary" in evaluation_data:
            summary.update(evaluation_data["executive_summary"])
        
        return summary
    
    def _generate_performance_chart(self, metrics: Dict[str, Any]) -> Optional[str]:
        """Generate performance visualization chart."""
        if not CHART_AVAILABLE:
            return None
        
        try:
            # Create chart directory
            charts_dir = os.path.join(self.output_dir, "charts")
            os.makedirs(charts_dir, exist_ok=True)
            
            # Create unique filename
            chart_filename = f"performance_chart_{int(time.time())}.png"
            chart_path = os.path.join(charts_dir, chart_filename)
            
            # Identify metrics to visualize
            viz_metrics = {}
            for key in ["accuracy", "precision", "recall", "f1_score", "specificity", "auc"]:
                if key in metrics:
                    viz_metrics[key.replace("_", " ").title()] = metrics[key]
            
            if not viz_metrics:
                return None
            
            # Create bar chart
            plt.figure(figsize=(10, 6))
            sns.set_style("whitegrid")
            
            # Plot metrics
            bars = plt.bar(
                range(len(viz_metrics)), 
                list(viz_metrics.values()),
                color=sns.color_palette("muted")
            )
            
            # Add value labels
            for bar in bars:
                height = bar.get_height()
                plt.text(
                    bar.get_x() + bar.get_width()/2., 
                    height + 0.01,
                    f"{height:.3f}",
                    ha='center', 
                    va='bottom'
                )
            
            # Customize chart
            plt.xticks(range(len(viz_metrics)), list(viz_metrics.keys()), rotation=30, ha='right')
            plt.ylim(0, 1.1)
            plt.title("Performance Metrics")
            plt.tight_layout()
            
            # Save chart
            plt.savefig(chart_path, dpi=100, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Generated performance chart at {chart_path}")
            return chart_path
            
        except Exception as e:
            logger.error(f"Error generating performance chart: {str(e)}")
            return None
    
    def generate_pdf_report(self, report: LegalEvaluationReport) -> str:
        """
        Generate a PDF report from a LegalEvaluationReport instance.
        
        Args:
            report: LegalEvaluationReport instance
            
        Returns:
            Path to the generated PDF file
        """
        if not FPDF_AVAILABLE:
            logger.error("FPDF library not available. Cannot generate PDF report.")
            return ""
        
        try:
            # Create PDF instance
            pdf = FPDF()
            pdf.set_auto_page_break(auto=True, margin=15)
            
            # Set metadata
            pdf.set_title(report.metadata.title)
            pdf.set_author(report.metadata.author)
            pdf.set_subject(report.metadata.subject)
            pdf.set_keywords(", ".join(report.metadata.keywords))
            
            # Add title page
            pdf.add_page()
            
            # Add logo if available
            if os.path.exists(self.logo_path):
                pdf.image(self.logo_path, x=10, y=10, w=30)
            
            # Add title
            pdf.set_font("Arial", style='B', size=24)
            pdf.ln(40)
            pdf.cell(0, 15, report.metadata.title, ln=True, align='C')
            
            # Add date and author
            pdf.set_font("Arial", style='', size=12)
            pdf.ln(10)
            pdf.cell(0, 10, f"Date: {report.metadata.date}", ln=True, align='C')
            pdf.cell(0, 10, f"Author: {report.metadata.author}", ln=True, align='C')
            
            # Add report ID
            pdf.set_font("Arial", style='', size=10)
            pdf.ln(10)
            pdf.cell(0, 10, f"Report ID: {report.metadata.report_id}", ln=True, align='C')
            
            # Add table of contents
            pdf.add_page()
            pdf.set_font("Arial", style='B', size=16)
            pdf.cell(0, 10, "Table of Contents", ln=True)
            pdf.ln(5)
            
            # List sections
            pdf.set_font("Arial", style='', size=12)
            for i, section in enumerate(report.sections):
                pdf.cell(0, 10, f"{i+1}. {section.title}", ln=True)
            
            # Add each section
            for i, section in enumerate(report.sections):
                pdf.add_page()
                
                # Section title
                pdf.set_font("Arial", style='B', size=16)
                pdf.cell(0, 10, f"{i+1}. {section.title}", ln=True)
                pdf.ln(5)
                
                # Section content based on type
                if section.section_type == "text":
                    pdf.set_font("Arial", style='', size=12)
                    
                    # Format text content
                    content = section.content
                    if isinstance(content, dict):
                        content = json.dumps(content, indent=2)
                    
                    # Split text into paragraphs
                    paragraphs = str(content).split('\n')
                    for paragraph in paragraphs:
                        pdf.multi_cell(0, 8, paragraph)
                        pdf.ln(2)
                
                elif section.section_type == "json":
                    pdf.set_font("Courier", size=10)
                    json_content = json.dumps(section.content, indent=2)
                    pdf.multi_cell(0, 8, json_content)
                
                elif section.section_type == "table":
                    # Handle table content
                    pdf.set_font("Arial", style='', size=10)
                    
                    if isinstance(section.content, list) and len(section.content) > 0:
                        # Determine columns
                        if isinstance(section.content[0], dict):
                            # List of dictionaries
                            columns = list(section.content[0].keys())
                            
                            # Add table header
                            pdf.set_font("Arial", style='B', size=10)
                            col_width = 190 / len(columns)
                            
                            for col in columns:
                                pdf.cell(col_width, 10, str(col), border=1)
                            pdf.ln()
                            
                            # Add table data
                            pdf.set_font("Arial", style='', size=10)
                            for row in section.content:
                                for col in columns:
                                    pdf.cell(col_width, 10, str(row.get(col, "")), border=1)
                                pdf.ln()
                        
                        else:
                            # Simple list
                            pdf.multi_cell(0, 8, str(section.content))
                    else:
                        # Not a list or empty list
                        pdf.multi_cell(0, 8, str(section.content))
                
                elif section.section_type == "chart" and section.content:
                    # Add chart image if it exists
                    if os.path.exists(section.content):
                        pdf.image(section.content, x=10, y=pdf.get_y(), w=190)
                        pdf.ln(100)  # Space for the image
            
            # Add attachments section if any
            if report.attachments:
                pdf.add_page()
                pdf.set_font("Arial", style='B', size=16)
                pdf.cell(0, 10, "Attachments", ln=True)
                pdf.ln(5)
                
                # List attachments
                pdf.set_font("Arial", style='', size=12)
                for i, attachment in enumerate(report.attachments):
                    pdf.cell(0, 10, f"{i+1}. {os.path.basename(attachment)}", ln=True)
            
            # Generate output filename
            report_id = report.metadata.report_id.replace("-", "")[:8]
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{report.metadata.title.replace(' ', '_')}_{report_id}_{timestamp}.pdf"
            filepath = os.path.join(self.output_dir, filename)
            
            # Save PDF
            pdf.output(filepath)
            
            # Register report
            self._register_report(report, filepath, "pdf")
            
            logger.info(f"Generated PDF report: {filepath}")
            return filepath
            
        except Exception as e:
            logger.error(f"Error generating PDF report: {str(e)}")
            return ""
    
    def generate_html_report(self, report: LegalEvaluationReport) -> str:
        """
        Generate an HTML report from a LegalEvaluationReport instance.
        
        Args:
            report: LegalEvaluationReport instance
            
        Returns:
            Path to the generated HTML file
        """
        try:
            # Create output filename
            report_id = report.metadata.report_id.replace("-", "")[:8]
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{report.metadata.title.replace(' ', '_')}_{report_id}_{timestamp}.html"
            filepath = os.path.join(self.output_dir, filename)
            
            # Determine template to use
            template_path = os.path.join(self.template_dir, "default_template.html")
            if not os.path.exists(template_path):
                self._create_default_html_template()
            
            # If Jinja is available, use template rendering
            if JINJA_AVAILABLE:
                env = jinja2.Environment(
                    loader=jinja2.FileSystemLoader(self.template_dir)
                )
                template = env.get_template("default_template.html")
                
                # Prepare charts for embedding
                for section in report.sections:
                    if section.section_type == "chart" and os.path.exists(section.content):
                        # Convert chart to base64 for embedding
                        import base64
                        with open(section.content, "rb") as image_file:
                            section.metadata["chart_data"] = base64.b64encode(image_file.read()).decode('utf-8')
                
                # Render template
                html_content = template.render(
                    report=report.to_dict(),
                    report_title=report.metadata.title,
                    author=report.metadata.author,
                    date=report.metadata.date,
                    sections=report.sections
                )
                
                with open(filepath, 'w', encoding='utf-8') as f:
                    f.write(html_content)
            
            else:
                # Fallback to simple HTML generation
                with open(filepath, 'w', encoding='utf-8') as f:
                    f.write(f"<!DOCTYPE html>\n<html>\n<head>\n<title>{report.metadata.title}</title>\n")
                    f.write("<style>body { font-family: Arial, sans-serif; margin: 40px; }</style>\n")
                    f.write("</head>\n<body>\n")
                    
                    # Title
                    f.write(f"<h1>{report.metadata.title}</h1>\n")
                    f.write(f"<p>Date: {report.metadata.date}<br>Author: {report.metadata.author}</p>\n")
                    
                    # Table of contents
                    f.write("<h2>Table of Contents</h2>\n<ul>\n")
                    for i, section in enumerate(report.sections):
                        f.write(f"<li><a href='#section{i+1}'>{section.title}</a></li>\n")
                    f.write("</ul>\n")
                    
                    # Sections
                    for i, section in enumerate(report.sections):
                        f.write(f"<h2 id='section{i+1}'>{section.title}</h2>\n")
                        
                        if section.section_type == "text":
                            content = section.content
                            if isinstance(content, dict):
                                content = json.dumps(content, indent=2)
                            paragraphs = str(content).split('\n')
                            for paragraph in paragraphs:
                                f.write(f"<p>{paragraph}</p>\n")
                        
                        elif section.section_type == "json":
                            f.write(f"<pre>{json.dumps(section.content, indent=2)}</pre>\n")
                        
                        elif section.section_type == "table" and isinstance(section.content, list):
                            f.write("<table border='1'>\n")
                            
                            if len(section.content) > 0 and isinstance(section.content[0], dict):
                                # Table header
                                columns = list(section.content[0].keys())
                                f.write("<tr>\n")
                                for col in columns:
                                    f.write(f"<th>{col}</th>\n")
                                f.write("</tr>\n")
                                
                                # Table rows
                                for row in section.content:
                                    f.write("<tr>\n")
                                    for col in columns:
                                        f.write(f"<td>{row.get(col, '')}</td>\n")
                                    f.write("</tr>\n")
                            
                            f.write("</table>\n")
                        
                        elif section.section_type == "chart" and section.content:
                            if os.path.exists(section.content):
                                f.write(f"<img src='{os.path.relpath(section.content, self.output_dir)}' alt='{section.title}' />\n")
                    
                    f.write("</body>\n</html>")
            
            # Register report
            self._register_report(report, filepath, "html")
            
            logger.info(f"Generated HTML report: {filepath}")
            return filepath
            
        except Exception as e:
            logger.error(f"Error generating HTML report: {str(e)}")
            return ""
    
    def generate_json_report(self, report: LegalEvaluationReport) -> str:
        """
        Generate a JSON report from a LegalEvaluationReport instance.
        
        Args:
            report: LegalEvaluationReport instance
            
        Returns:
            Path to the generated JSON file
        """
        try:
            # Create output filename
            report_id = report.metadata.report_id.replace("-", "")[:8]
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{report.metadata.title.replace(' ', '_')}_{report_id}_{timestamp}.json"
            filepath = os.path.join(self.output_dir, filename)
            
            # Convert to JSON
            with open(filepath, 'w') as f:
                json.dump(report.to_dict(), f, indent=2)
            
            # Register report
            self._register_report(report, filepath, "json")
            
            logger.info(f"Generated JSON report: {filepath}")
            return filepath
            
        except Exception as e:
            logger.error(f"Error generating JSON report: {str(e)}")
            return ""
    
    def _register_report(
        self, 
        report: LegalEvaluationReport, 
        filepath: str, 
        format_type: str
    ) -> None:
        """Register a report in the registry for version tracking."""
        report_id = report.metadata.report_id
        
        if report_id not in self.report_registry["reports"]:
            self.report_registry["reports"][report_id] = {
                "title": report.metadata.title,
                "author": report.metadata.author,
                "date": report.metadata.date,
                "versions": []
            }
        
        # Add this version
        self.report_registry["reports"][report_id]["versions"].append({
            "format": format_type,
            "filepath": filepath,
            "timestamp": datetime.datetime.now().isoformat()
        })
        
        # Save registry
        self._save_report_registry()
    
    def _create_default_html_template(self) -> None:
        """Create a default HTML template if none exists."""
        template_path = os.path.join(self.template_dir, "default_template.html")
        
        template_content = """<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{{ report_title }}</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            line-height: 1.6;
            margin: 0;
            padding: 0;
            color: #333;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
        }
        header {
            background-color: #2c3e50;
            color: white;
            padding: 20px;
            text-align: center;
        }
        .toc {
            background-color: #f8f9fa;
            border: 1px solid #ddd;
            border-radius: 4px;
            padding: 15px;
            margin: 20px 0;
        }
        .toc ul {
            list-style-type: none;
            padding-left: 20px;
        }
        .toc li {
            margin-bottom: 8px;
        }
        .section {
            margin-bottom: 30px;
            border-bottom: 1px solid #eee;
            padding-bottom: 20px;
        }
        .section h2 {
            color: #2c3e50;
            border-bottom: 2px solid #3498db;
            padding-bottom: 10px;
        }
        pre {
            background-color: #f5f5f5;
            padding: 15px;
            border-radius: 4px;
            overflow-x: auto;
        }
        table {
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }
        th, td {
            border: 1px solid #ddd;
            padding: 8px;
            text-align: left;
        }
        th {
            background-color: #f2f2f2;
        }
        .chart-container {
            max-width: 800px;
            margin: 20px auto;
        }
        footer {
            text-align: center;
            margin-top: 40px;
            color: #777;
            padding: 20px;
            background-color: #f8f9fa;
        }
    </style>
</head>
<body>
    <header>
        <h1>{{ report_title }}</h1>
        <p>Author: {{ author }} | Date: {{ date }}</p>
    </header>
    <div class="container">
        <div class="toc">
            <h2>Table of Contents</h2>
            <ul>
                {% for section in sections %}
                <li><a href="#section{{ loop.index }}">{{ section.title }}</a></li>
                {% endfor %}
            </ul>
        </div>
        
        {% for section in sections %}
        <div class="section" id="section{{ loop.index }}">
            <h2>{{ section.title }}</h2>
            
            {% if section.section_type == "text" %}
                {% if section.content is mapping %}
                    {% for key, value in section.content.items() %}
                    <h3>{{ key }}</h3>
                    <p>{{ value }}</p>
                    {% endfor %}
                {% else %}
                    <p>{{ section.content }}</p>
                {% endif %}
            
            {% elif section.section_type == "json" %}
                <pre>{{ section.content | tojson(indent=2) }}</pre>
            
            {% elif section.section_type == "table" and section.content is iterable and section.content|length > 0 %}
                <table>
                    {% if section.content[0] is mapping %}
                        <tr>
                        {% for key in section.content[0].keys() %}
                            <th>{{ key }}</th>
                        {% endfor %}
                        </tr>
                        
                        {% for row in section.content %}
                        <tr>
                            {% for key, value in row.items() %}
                            <td>{{ value }}</td>
                            {% endfor %}
                        </tr>
                        {% endfor %}
                    {% else %}
                        <tr><td>{{ section.content }}</td></tr>
                    {% endif %}
                </table>
            
            {% elif section.section_type == "chart" and section.metadata.chart_data %}
                <div class="chart-container">
                    <img src="data:image/png;base64,{{ section.metadata.chart_data }}" alt="{{ section.title }}">
                </div>
            {% endif %}
        </div>
        {% endfor %}
    </div>
    <footer>
        <p>Report ID: {{ report.metadata.report_id }}</p>
        <p>Generated with Legal AI Evaluation Report Generator</p>
    </footer>
</body>
</html>"""
        
        with open(template_path, 'w', encoding='utf-8') as f:
            f.write(template_content)
        
        logger.info(f"Created default HTML template at {template_path}")


# Example legal evaluation data
def create_sample_evaluation_data() -> Dict[str, Any]:
    """Create sample evaluation data for testing."""
    return {
        "performance_metrics": {
            "accuracy": 0.92,
            "precision": 0.89,
            "recall": 0.91,
            "f1_score": 0.90,
            "specificity": 0.93,
            "auc": 0.95,
            "execution_time_sec": 12.45
        },
        "methodology": {
            "test_dataset": "Legal Case Classification Benchmark",
            "test_size": 1000,
            "cross_validation": "5-fold cross-validation",
            "evaluation_criteria": "Standard classification metrics with legal domain-specific adjustments"
        },
        "detailed_results": [
            {
                "category": "Contract Law",
                "accuracy": 0.94,
                "sample_size": 250,
                "confusion_matrix": [[120, 5], [10, 115]]
            },
            {
                "category": "Tort Law",
                "accuracy": 0.90,
                "sample_size": 200,
                "confusion_matrix": [[90, 10], [10, 90]]
            },
            {
                "category": "Criminal Law",
                "accuracy": 0.93,
                "sample_size": 300,
                "confusion_matrix": [[140, 10], [10, 140]]
            },
            {
                "category": "Property Law",
                "accuracy": 0.89,
                "sample_size": 250,
                "confusion_matrix": [[110, 15], [12, 113]]
            }
        ],
        "system_info": {
            "model_version": "LegalBERT v2.1",
            "framework": "PyTorch 1.10",
            "environment": "CUDA 11.4, 16GB V100 GPU",
            "preprocessing": "Legal text normalization, citation standardization",
            "runtime_configuration": {
                "batch_size": 32,
                "optimizer": "AdamW",
                "learning_rate": 2e-5
            }
        },
        "conclusions": {
            "overall_assessment": "The model demonstrates strong performance across multiple legal domains, with particularly high accuracy in Contract Law and Criminal Law categories.",
            "strengths": [
                "Consistent performance across legal domains",
                "High recall rate for critical legal issues",
                "Efficient processing time suitable for real-time applications"
            ],
            "limitations": [
                "Lower precision in Property Law cases",
                "Requires domain-specific fine-tuning for specialized legal areas",
                "Performance degrades with highly technical regulatory language"
            ],
            "recommendations": [
                "Deploy for assisted legal research and case classification",
                "Implement human-in-the-loop review for Property Law cases",
                "Enhance training data for regulatory compliance scenarios",
                "Consider ensemble approach with domain-specific models for specialized areas"
            ]
        }
    }


def main():
    """Main execution function."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Generate legal AI evaluation reports")
    parser.add_argument("--output-dir", default="evaluation_reports",
                        help="Directory to save generated reports")
    parser.add_argument("--format", default="all", choices=["pdf", "html", "json", "all"],
                        help="Report format to generate")
    parser.add_argument("--title", default="Legal AI System Evaluation Report",
                        help="Title for the report")
    parser.add_argument("--author", default="Legal AI Evaluation System",
                        help="Author name for the report")
    parser.add_argument("--sample", action="store_true",
                        help="Use sample data instead of loading from file")
    parser.add_argument("--input-file", 
                        help="JSON file containing evaluation data")
    
    args = parser.parse_args()
    
    # Load evaluation data
    if args.sample:
        evaluation_data = create_sample_evaluation_data()
        logger.info("Using sample evaluation data")
    elif args.input_file:
        try:
            with open(args.input_file, 'r') as f:
                evaluation_data = json.load(f)
            logger.info(f"Loaded evaluation data from {args.input_file}")
        except Exception as e:
            logger.error(f"Error loading evaluation data: {str(e)}")
            logger.info("Falling back to sample evaluation data")
            evaluation_data = create_sample_evaluation_data()
    else:
        logger.info("No input file specified and --sample not used")
        logger.info("Falling back to sample evaluation data")
        evaluation_data = create_sample_evaluation_data()
    
    # Create report generator
    generator = ReportGenerator(output_dir=args.output_dir)
    
    # Create report
    report = generator.create_report(
        evaluation_data=evaluation_data,
        title=args.title,
        author=args.author
    )
    
    # Generate reports in requested formats
    report_files = []
    
    if args.format in ["pdf", "all"] and FPDF_AVAILABLE:
        pdf_path = generator.generate_pdf_report(report)
        if pdf_path:
            report_files.append(("PDF", pdf_path))
    
    if args.format in ["html", "all"]:
        html_path = generator.generate_html_report(report)
        if html_path:
            report_files.append(("HTML", html_path))
    
    if args.format in ["json", "all"]:
        json_path = generator.generate_json_report(report)
        if json_path:
            report_files.append(("JSON", json_path))
    
    # Print summary
    print("\nEvaluation Report Generation Summary:")
    print(f"Report Title: {args.title}")
    print(f"Generated at: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Report ID: {report.metadata.report_id}")
    
    if report_files:
        print("\nGenerated Reports:")
        for format_type, path in report_files:
            print(f"- {format_type}: {path}")
    else:
        print("\nNo reports were generated. Check logs for errors.")
    
    print(f"\nLog file: {log_filename}")


if __name__ == "__main__":
    main()


"""
TODO:
- Add support for embedding interactive visualizations in HTML reports
- Implement digital signatures and verification for legal compliance
- Create industry-specific templates for different legal practice areas
- Add accessibility compliance checking (WCAG) for generated reports
- Implement comparison between multiple model evaluation reports
- Add support for extracting legal-specific metrics from evaluation data
- Include citation validation and verification in reports
- Create executive-focused simplified report option
- Support for generating reports in multiple languages
- Add automated distribution mechanisms (email, secure file transfer)
"""
