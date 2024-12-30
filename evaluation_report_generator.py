import json
from typing import Dict, Any
from fpdf import FPDF
from pydantic import BaseModel

class ReportConfig(BaseModel):
    """
    Configuration for the evaluation report generator.
    """
    benchmark_results_path: str  # Path to the JSON file containing benchmark results
    output_report_path: str  # Path to save the generated report

class EvaluationReportGenerator:
    """
    Generates evaluation reports summarizing system performance and explainability insights.
    """

    def __init__(self, config: ReportConfig):
        """
        Initialize the report generator with the provided configuration.

        Args:
            config (ReportConfig): Configuration for the report generator.
        """
        self.config = config
        self.pdf = FPDF()
        self.pdf.set_auto_page_break(auto=True, margin=15)

    def load_results(self) -> Dict[str, Any]:
        """
        Load benchmark results from the JSON file.

        Returns:
            Dict[str, Any]: Dictionary containing benchmark results.
        """
        with open(self.config.benchmark_results_path, "r") as file:
            results = json.load(file)
        print(f"Loaded benchmark results from {self.config.benchmark_results_path}")
        return results

    def add_title(self, title: str):
        """
        Add a title to the PDF report.

        Args:
            title (str): Title text for the report.
        """
        self.pdf.add_page()
        self.pdf.set_font("Arial", size=16, style="B")
        self.pdf.cell(200, 10, txt=title, ln=True, align="C")
        print("Added title to the report.")

    def add_section(self, section_title: str, content: str):
        """
        Add a section to the PDF report.

        Args:
            section_title (str): Title of the section.
            content (str): Content of the section.
        """
        self.pdf.set_font("Arial", size=14, style="B")
        self.pdf.cell(200, 10, txt=section_title, ln=True)
        self.pdf.set_font("Arial", size=12)
        self.pdf.multi_cell(0, 10, txt=content)
        print(f"Added section: {section_title}")

    def generate_report(self):
        """
        Generate the PDF report based on benchmark results.
        """
        results = self.load_results()

        # Add report title
        self.add_title("Evaluation Report: System Performance")

        # Add latency results
        latency_content = (
            f"Average Latency: {results['latency']['average_latency']:.2f} seconds\n"
            f"Median Latency: {results['latency']['median_latency']:.2f} seconds\n"
            f"Detailed Latencies: {results['latency']['latencies']}\n"
        )
        self.add_section("Latency Results", latency_content)

        # Add accuracy results
        accuracy_content = f"System Accuracy: {results['accuracy']:.2f}%"
        self.add_section("Accuracy Results", accuracy_content)

        # Add consensus reliability
        consensus_content = (
            f"Average Document Overlap: {results['consensus']['average_overlap']:.2f}%\n"
            f"Overlap Details: {results['consensus']['overlaps']}\n"
        )
        self.add_section("Consensus Reliability", consensus_content)

        # Save the report
        self.pdf.output(self.config.output_report_path)
        print(f"Report generated and saved to {self.config.output_report_path}")

if __name__ == "__main__":
    """
    Entry point for the evaluation report generator script.

    What We Did:
    - Loaded benchmark results from a JSON file.
    - Generated a PDF report summarizing latency, accuracy, and consensus reliability.
    - Saved the report to the specified location.

    What's Next:
    - Add visualizations (e.g., charts or tables) to enhance report readability.
    - Include explainability insights with SHAP explanations.
    - Automate the reporting pipeline for continuous evaluation.
    """
    # Configuration
    config = ReportConfig(
        benchmark_results_path="./benchmark_results.json",
        output_report_path="./evaluation_report.pdf"
    )

    # Generate the report
    report_generator = EvaluationReportGenerator(config)
    report_generator.generate_report()
