import fitz
import re
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
from src.logger_config import get_logger

class ProcessingStrategy(Enum):
    TEXT_ONLY = "text_only"           # Pure text extraction
    VISION_ONLY = "vision_only"       # Vision model for everything
    HYBRID = "hybrid"                 # Combine text + vision

@dataclass
class PageAnalysis:
    """Analysis results for a single page"""
    has_extractable_text: bool
    text_coverage: float  # 0.0 to 1.0
    has_images: bool
    has_tables: bool
    has_formulas: bool
    strategy: ProcessingStrategy
    confidence: float
    metadata: Dict

class PDFAnalyzer:
    """Analyze PDF structure to determine optimal processing strategy"""
    
    def __init__(self):
        self.logger = get_logger(__name__)
        self.text_threshold = 0.7      # Minimum text coverage for text-only
        self.complexity_threshold = 0.6 # Threshold for complex layout detection
        
    def analyze_page_content(self, page: fitz.Page) -> PageAnalysis:
        """Detect text, images, tables, formulas, diagrams"""
        
        # Check text extractability
        has_text = self._has_extractable_text(page)
        self.logger.debug(f"Has text:{has_text}")
        text_coverage = self._calculate_text_coverage(page)
        self.logger.debug(f"Text_coverage:{text_coverage}")

        # Detect content types
        has_images = self._detect_images(page)
        self.logger.debug(f"Has images:{has_images}")
        has_tables = self._detect_tables(page)
        self.logger.debug(f"Has tables:{has_tables}")
        has_formulas = self._detect_formulas(page)
        self.logger.debug(f"Has formulas:{has_formulas}")
                          
        # Determine strategy
        strategy, confidence = self._determine_strategy(
            has_text, text_coverage, has_images, has_tables, 
            has_formulas
        )
        
        return PageAnalysis(
            has_extractable_text=has_text,
            text_coverage=text_coverage,
            has_images=has_images,
            has_tables=has_tables,
            has_formulas=has_formulas,
            strategy=strategy,
            confidence=confidence,
            metadata={
                "page_number": page.number,
                "page_size": (page.rect.width, page.rect.height),
                "image_count": len(page.get_images()),
                "text_blocks": len([b for b in page.get_text("dict")["blocks"] if "lines" in b])
            }
        )
    
    def get_processing_strategy(self, analysis: PageAnalysis) -> ProcessingStrategy:
        """Decide between text extraction vs vision model based on content"""
        return analysis.strategy
    
    def analyze_document(self, pdf_path: str) -> Dict[int, PageAnalysis]:
        """Analyze entire PDF document"""
        with fitz.open(pdf_path) as doc:
            analyses = {}
            
            for page_num in range(doc.page_count):
                page = doc[page_num]
                analyses[page_num] = self.analyze_page_content(page)
        
        return analyses
    
    def _has_extractable_text(self, page: fitz.Page) -> bool:
        """Check if page has meaningful extractable text"""
        text = page.get_text().strip()
        # Filter out common OCR artifacts and very short text
        clean_text = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', text)
        return len(clean_text) > 20
    
    def _calculate_text_coverage(self, page: fitz.Page) -> float:
        """Estimate how much of the page is covered by text"""
        text_dict = page.get_text("dict")
        page_area = page.rect.width * page.rect.height
        
        if page_area == 0:
            return 0.0
        
        text_area = 0
        for block in text_dict["blocks"]:
            if "lines" in block:
                bbox = block["bbox"]
                block_area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
                text_area += block_area
        
        return min(text_area / page_area, 1.0)
    
    def _detect_images(self, page: fitz.Page) -> bool:
        """Detect if page contains significant images"""
        images = page.get_images()
        if not images:
            return False
        
        # Check for images larger than thumbnails
        for img in images:
            try:
                pix = fitz.Pixmap(page.parent, img[0])
                if pix.width > 100 and pix.height > 100:
                    pix = None
                    return True
                pix = None
            except:
                continue
        return False
    
    def _detect_tables(self, page: fitz.Page) -> bool:
        """Detect table-like structures"""
        text_dict = page.get_text("dict")
        
        # Look for grid-like text patterns
        lines_with_multiple_columns = 0
        total_text_lines = 0
        
        for block in text_dict["blocks"]:
            if "lines" not in block:
                continue
                
            for line in block["lines"]:
                total_text_lines += 1
                
                # Check for multiple distinct x-positions (columns)
                x_positions = [span["bbox"][0] for span in line["spans"]]
                unique_x = len(set(x_positions))
                
                if unique_x >= 3:  # At least 3 columns
                    lines_with_multiple_columns += 1
        
        if total_text_lines == 0:
            return False
        
        # If >30% of lines have multiple columns, likely a table
        return (lines_with_multiple_columns / total_text_lines) > 0.3
    
    def _detect_formulas(self, page: fitz.Page) -> bool:
        """Detect mathematical formulas"""
        text = page.get_text()
        
        # Mathematical symbols and patterns
        math_indicators = [
            r'[∫∑∏√±×÷≤≥≠≈∞∂∇]',      # Math symbols
            r'\$[^$]+\ ',                    # LaTeX inline
            r'\\[a-zA-Z]+\{',               # LaTeX commands
            r'[a-zA-Z][_^][0-9a-zA-Z]',     # Sub/superscripts
            r'\\begin\{',                    # LaTeX environments
            r'\b(sin|cos|tan|log|ln|exp|lim|int|sum|prod)\b'  # Math functions
        ]
        
        return any(re.search(pattern, text) for pattern in math_indicators)
    
    def _determine_strategy(self, has_text: bool, text_coverage: float, 
                          has_images: bool, has_tables: bool, 
                          has_formulas: bool) -> Tuple[ProcessingStrategy, float]:
        """Determine optimal processing strategy with confidence"""
        
        # High confidence text-only conditions
        if (has_text and text_coverage > self.text_threshold and 
            not has_images and not has_tables):
            return ProcessingStrategy.TEXT_ONLY, 0.9
        
        # Vision-only conditions
        if (not has_text or text_coverage < 0.1):
            return ProcessingStrategy.VISION_ONLY, 0.8
        
        # Default to hybrid for mixed content
        confidence = 0.6
        if has_formulas or has_tables:
            confidence = 0.7
        
        return ProcessingStrategy.HYBRID, confidence