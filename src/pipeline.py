import fitz
from pathlib import Path
from typing import List, Dict, Optional
from dataclasses import dataclass

# Import all our components
from .utils import Utils, PipelineConfig
from .pdf_analyzer import PDFAnalyzer, PageAnalysis
from .text_extractor import TextExtractor
from .vision_processor import VisionProcessor
from .content_integrator import ContentIntegrator
from .markdown_generator import MarkdownGenerator

@dataclass
class ConversionResult:
    """Result of PDF conversion"""
    pages: List[str]  # Markdown content for each page
    metadata: Dict
    success: bool
    errors: List[str] = None

class PDFToMarkdownPipeline:
    """Main orchestrator for the conversion process"""
    
    def __init__(self, ollama_model: str, ollama_base_url: str, config: Optional[PipelineConfig] = None):
        """Initialize the pipeline with all components"""
        self.config = config or PipelineConfig()
        
        # Initialize all components
        self.analyzer = PDFAnalyzer()
        self.text_extractor = TextExtractor()
        self.vision_processor = VisionProcessor(
            model_name=ollama_model,
            base_url=ollama_base_url,
            temperature=self.config.vision_model_temp
        )
        self.integrator = ContentIntegrator()
        self.markdown_generator = MarkdownGenerator(self.config)
        
        # Pipeline metadata
        self.metadata = {
            "ollama_model": ollama_model,
            "ollama_base_url": ollama_base_url,
            "config": self.config
        }
    
    def convert_pdf(self, pdf_path: str) -> ConversionResult:
        """Main conversion pipeline"""
        try:
            pdf_path = Path(pdf_path)
            if not pdf_path.exists():
                return ConversionResult(
                    pages=[],
                    metadata=self.metadata,
                    success=False,
                    errors=[f"PDF file not found: {pdf_path}"]
                )
            
            # Open PDF document
            doc = fitz.open(str(pdf_path))
            pages_markdown = []
            page_analyses = []
            errors = []
            
            print(f"Processing PDF with {doc.page_count} pages...")
            
            for page_num in range(doc.page_count):
                try:
                    page = doc[page_num]
                    print(f"Processing page {page_num + 1}/{doc.page_count}")
                    
                    # Convert single page
                    page_markdown = self.convert_page(page)
                    pages_markdown.append(page_markdown)
                    
                except Exception as e:
                    error_msg = f"Error processing page {page_num + 1}: {str(e)}"
                    errors.append(error_msg)
                    print(f"Warning: {error_msg}")
                    pages_markdown.append(f"<!-- Error processing page {page_num + 1}: {str(e)} -->")
            
            doc.close()
            
            # Compile metadata
            result_metadata = {
                **self.metadata,
                "total_pages": len(pages_markdown),
                "successful_pages": len(pages_markdown) - len(errors),
                "pdf_path": str(pdf_path),
                "strategies_used": page_analyses
            }
            
            return ConversionResult(
                pages=pages_markdown,
                metadata=result_metadata,
                success=len(errors) == 0,
                errors=errors if errors else None
            )
            
        except Exception as e:
            return ConversionResult(
                pages=[],
                metadata=self.metadata,
                success=False,
                errors=[f"Pipeline error: {str(e)}"]
            )
    
    def convert_page(self, page: fitz.Page) -> str:
        """Process single page through pipeline"""
        
        # 1. Analyze PDF page structure
        print(f"  Analyzing page structure...")
        analysis = self.analyzer.analyze_page_content(page)
        strategy = analysis.strategy.value
        print(f"  Strategy: {strategy} (confidence: {analysis.confidence:.2f})")
        
        # 2. Extract content based on strategy
        text_content = None
        vision_content = []
        
        if strategy in ["text_only", "hybrid"]:
            print(f"  Extracting text content...")
            text_content = self._extract_text_content(page, analysis)
        
        if strategy in ["vision_only", "hybrid", "complex_layout"]:
            print(f"  Processing with vision model...")
            vision_content = self._process_with_vision(page, analysis)
        
        # 3. Integrate content streams
        print(f"  Integrating content...")
        integrated_content = self.integrator.merge_content_streams(
            text_content, vision_content, strategy
        )
        
        # 4. Generate final markdown
        print(f"  Generating markdown...")
        markdown = integrated_content.markdown
        
        return markdown
    
    def _extract_text_content(self, page: fitz.Page, analysis: PageAnalysis):
        """Extract text content from page"""
        try:
            if self.config.text_extraction_priority and analysis.has_extractable_text:
                # Use structured text extraction
                extracted_content = self.text_extractor.extract_structured_text(page)
                
                if self.config.preserve_formatting:
                    formatted_text = self.text_extractor.preserve_formatting(extracted_content.elements)
                    return formatted_text
                else:
                    return extracted_content.raw_text
            else:
                # Fallback to simple text extraction
                return page.get_text()
                
        except Exception as e:
            print(f"    Text extraction failed: {e}")
            return page.get_text()  # Fallback
    
    def _process_with_vision(self, page: fitz.Page, analysis: PageAnalysis) -> List:
        """Process page content with vision model"""
        vision_results = []
        
        try:
            # Convert page to image
            page_image = Utils.extract_page_image(page, self.config.dpi)
            
            # Determine what type of content to extract
            if analysis.has_tables:
                print(f"    Extracting tables...")
                table_result = self.vision_processor.extract_table_data(page_image)
                vision_results.append(table_result)
            
            if analysis.has_formulas:
                print(f"    Extracting formulas...")
                formula_result = self.vision_processor.extract_formulas(page_image)
                vision_results.append(formula_result)
            
            if analysis.has_images:
                print(f"    Describing diagrams...")
                diagram_result = self.vision_processor.describe_diagrams(page_image)
                vision_results.append(diagram_result)
            
            # If no specific content detected, use general extraction
            if not vision_results:
                print(f"    General content extraction...")
                general_result = self.vision_processor.process_image_content(page_image, "general")
                vision_results.append(general_result)
                
        except Exception as e:
            print(f"    Vision processing failed: {e}")
            # Return empty results rather than failing completely
            vision_results = []
        
        return vision_results
    
    def save_results(self, result: ConversionResult, output_dir: str = "./output") -> List[Path]:
        """Save conversion results to files"""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        saved_files = []
        
        # Save each page as separate markdown file
        for i, page_content in enumerate(result.pages):
            page_file = output_path / f"page_{i+1:03d}.md"
            page_file.write_text(page_content, encoding='utf-8')
            saved_files.append(page_file)
        
        # Save combined document
        combined_content = "\n\n---\n\n".join(result.pages)
        combined_file = output_path / "combined_document.md"
        combined_file.write_text(combined_content, encoding='utf-8')
        saved_files.append(combined_file)
        
        # Save metadata
        import json
        metadata_file = output_path / "conversion_metadata.json"
        
        # Convert non-serializable objects to strings
        serializable_metadata = self._make_serializable(result.metadata)
        
        with open(metadata_file, 'w') as f:
            json.dump(serializable_metadata, f, indent=2)
        saved_files.append(metadata_file)
        
        return saved_files
    
    def _make_serializable(self, obj) -> dict:
        """Convert objects to JSON-serializable format"""
        if hasattr(obj, '__dict__'):
            return {k: self._make_serializable(v) for k, v in obj.__dict__.items()}
        elif isinstance(obj, dict):
            return {k: self._make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [self._make_serializable(item) for item in obj]
        elif isinstance(obj, (str, int, float, bool, type(None))):
            return obj
        else:
            return str(obj)
    
    def get_pipeline_info(self) -> Dict:
        """Get information about the pipeline configuration"""
        return {
            "components": {
                "analyzer": type(self.analyzer).__name__,
                "text_extractor": type(self.text_extractor).__name__,
                "vision_processor": type(self.vision_processor).__name__,
                "integrator": type(self.integrator).__name__,
                "markdown_generator": type(self.markdown_generator).__name__
            },
            "config": self.config.__dict__,
            "vision_model": {
                "model": self.vision_processor.model_name,
                "base_url": self.vision_processor.base_url
            }
        }


# Convenience function for simple usage
def convert_pdf_to_markdown(pdf_path: str, ollama_model: str = "llama3.2-vision:11b", 
                          ollama_base_url: str = "http://localhost:11434",
                          output_dir: str = "./output") -> ConversionResult:
    """
    Simple function to convert a PDF to markdown with default settings
    
    Args:
        pdf_path: Path to the PDF file
        ollama_model: Ollama model name for vision processing
        ollama_base_url: Ollama server URL
        output_dir: Directory to save output files
    
    Returns:
        ConversionResult with pages and metadata
    """
    pipeline = PDFToMarkdownPipeline(ollama_model, ollama_base_url)
    result = pipeline.convert_pdf(pdf_path)
    
    if result.success:
        saved_files = pipeline.save_results(result, output_dir)
        print(f"Conversion completed! Files saved to {output_dir}")
        for file in saved_files:
            print(f"  - {file}")
    else:
        print("Conversion failed:")
        for error in result.errors or []:
            print(f"  - {error}")
    
    return result