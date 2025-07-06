import fitz
from pathlib import Path
from typing import List, Dict, Optional
from dataclasses import dataclass
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage
import base64
import io
from PIL import Image
from src.logger_config import setup_logging, get_logger

@dataclass
class ConversionResult:
    """Result of PDF conversion"""
    pages: List[str]  # Markdown content for each page
    metadata: Dict
    success: bool
    errors: List[str] = None

class VisionProcessor:
    """Simple vision processor for PDF content"""
    
    def __init__(self, model_name: str, base_url: str, temperature: float = 0.1):
        self.logger = get_logger(__name__)
        self.model_name = model_name
        self.base_url = base_url
        self.temperature = temperature
        self.chat_model = self._init_ollama(model_name, base_url)
        
        self.logger.info(f"VisionProcessor initialized with model: {model_name}")
    
    def _init_ollama(self, model_name: str, base_url: str) -> ChatOllama:
        """Initialize ChatOllama"""
        try:
            chat_model = ChatOllama(
                model=model_name,
                base_url=base_url,
                temperature=self.temperature
            )
            self.logger.info(f"Connected to Ollama at {base_url}")
            return chat_model
        except Exception as e:
            self.logger.error(f"Failed to initialize ChatOllama: {e}")
            raise ConnectionError(f"Failed to initialize ChatOllama: {e}")
    
    def process_page(self, image_base64: str) -> str:
        """Process page image and return markdown"""
        
        prompt = """Extract and convert all content from this PDF page to clean markdown format.

INSTRUCTIONS:
1. Extract all visible text accurately
2. Convert tables to proper markdown table format with | separators and --- headers
3. Convert mathematical formulas to LaTeX notation ($...$ for inline, $$...$$ for display)
4. Describe any diagrams, charts, or images with appropriate markdown formatting
5. Maintain document structure with proper headers (#, ##, ###)
6. Preserve lists and formatting
7. Ensure LaTeX formulas use correct syntax (e.g., P_i not P_i_i)

Return only the markdown content, no additional commentary."""

        try:
            content = [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_base64}"}}
            ]
            
            message = HumanMessage(content=content)
            response = self.chat_model.invoke([message])
            
            self.logger.debug(f"Vision model response received, length: {len(response.content)}")
            return response.content
            
        except Exception as e:
            self.logger.error(f"Vision processing failed: {e}")
            raise RuntimeError(f"Vision processing failed: {e}")

class Utils:
    """Simple utility functions"""
    
    @staticmethod
    def optimize_image_for_vision(image_data: bytes, max_size: tuple = (1024, 1024)) -> str:
        """Prepare images for vision model processing"""
        try:
            img = Image.open(io.BytesIO(image_data))
            
            # Resize if too large while maintaining aspect ratio
            img.thumbnail(max_size, Image.Resampling.LANCZOS)
            
            # Convert to RGB if necessary
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            # Convert to base64
            buffer = io.BytesIO()
            img.save(buffer, format='PNG', optimize=True)
            img_str = base64.b64encode(buffer.getvalue()).decode()
            
            return img_str
            
        except Exception as e:
            raise ValueError(f"Failed to optimize image: {e}")
    
    @staticmethod
    def extract_page_image(page: fitz.Page, dpi: int = 300) -> str:
        """Convert PDF page to optimized base64 image"""
        # Create transformation matrix for desired DPI
        zoom = dpi / 72
        matrix = fitz.Matrix(zoom, zoom)
        
        # Render page as image
        pix = page.get_pixmap(matrix=matrix, alpha=False)
        img_data = pix.tobytes("png")
        pix = None
        
        return Utils.optimize_image_for_vision(img_data)

class SimplePDFToMarkdownPipeline:
    """Simple vision-only PDF to Markdown pipeline"""
    
    def __init__(self, ollama_model: str, ollama_base_url: str, dpi: int = 300):
        """Initialize the pipeline"""
        self.logger = get_logger(__name__)
        self.dpi = dpi
        
        # Initialize vision processor
        self.vision_processor = VisionProcessor(
            model_name=ollama_model,
            base_url=ollama_base_url
        )
        
        # Pipeline metadata
        self.metadata = {
            "ollama_model": ollama_model,
            "ollama_base_url": ollama_base_url,
            "dpi": dpi,
            "strategy": "vision_only"
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
            with fitz.open(str(pdf_path)) as doc:
                pages_markdown = []
                errors = []
                
                self.logger.info(f"Processing PDF with {doc.page_count} pages...")
                
                for page_num in range(doc.page_count):
                    try:
                        page = doc[page_num]
                        self.logger.info(f"Processing page {page_num + 1}/{doc.page_count}")
                        
                        # Convert page to image
                        page_image = Utils.extract_page_image(page, self.dpi)
                        
                        # Process with vision model
                        markdown = self.vision_processor.process_page(page_image)
                        pages_markdown.append(markdown)
                        
                    except Exception as e:
                        error_msg = f"Error processing page {page_num + 1}: {str(e)}"
                        errors.append(error_msg)
                        self.logger.warning(f"Warning: {error_msg}")
                        pages_markdown.append(f"<!-- Error processing page {page_num + 1}: {str(e)} -->")

            # Compile metadata
            result_metadata = {
                **self.metadata,
                "total_pages": len(pages_markdown),
                "successful_pages": len(pages_markdown) - len(errors),
                "pdf_path": str(pdf_path)
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
    
    def save_results(self, result: ConversionResult, output_dir: str = "./output") -> List[Path]:
        """Save conversion results to files"""
        self.logger.info(f"Saving results to {output_dir}")
        
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        saved_files = []
        
        # Save each page as separate markdown file
        for i, page_content in enumerate(result.pages):
            page_file = output_path / f"page_{i+1:03d}.md"
            page_file.write_text(page_content, encoding='utf-8')
            saved_files.append(page_file)
            self.logger.debug(f"Saved page {i+1} to {page_file}")
        
        # Save combined document
        combined_content = "\n\n---\n\n".join(result.pages)
        combined_file = output_path / "combined_document.md"
        combined_file.write_text(combined_content, encoding='utf-8')
        saved_files.append(combined_file)
        self.logger.debug(f"Saved combined document to {combined_file}")
        
        # Save metadata
        import json
        metadata_file = output_path / "conversion_metadata.json"
        
        with open(metadata_file, 'w') as f:
            json.dump(result.metadata, f, indent=2)
        saved_files.append(metadata_file)
        self.logger.debug(f"Saved metadata to {metadata_file}")
        
        self.logger.info(f"Successfully saved {len(saved_files)} files to {output_dir}")
        
        return saved_files


# Convenience function for simple usage
def convert_pdf_to_markdown_simple(pdf_path: str,
                                  ollama_model: str = "llama3.2-vision:11b", 
                                  ollama_base_url: str = "http://localhost:11434",
                                  output_dir: str = "./output",
                                  dpi: int = 300,
                                  log_level: str = "INFO") -> ConversionResult:
    """
    Simple function to convert a PDF to markdown using vision-only approach
    
    Args:
        pdf_path: Path to the PDF file
        ollama_model: Ollama model name for vision processing
        ollama_base_url: Ollama server URL
        output_dir: Directory to save output files
        dpi: Image resolution for processing
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
    
    Returns:
        ConversionResult with pages and metadata
    """
    # Set up logging
    logger = setup_logging(log_level=log_level)
    
    pipeline = SimplePDFToMarkdownPipeline(ollama_model, ollama_base_url, dpi)
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