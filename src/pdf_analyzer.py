import fitz
import re
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum

class ProcessingStrategy(Enum):
    TEXT_ONLY = "text_only"           # Pure text extraction
    VISION_ONLY = "vision_only"       # Vision model for everything
    HYBRID = "hybrid"                 # Combine text + vision
    COMPLEX_LAYOUT = "complex_layout" # Vision-first for complex layouts

@dataclass
class PageAnalysis:
    """Analysis results for a single page"""
    has_extractable_text: bool
    text_coverage: float  # 0.0 to 1.0
    has_images: bool
    has_tables: bool
    has_formulas: bool
    layout_complexity: float  # 0.0 to 1.0
    strategy: ProcessingStrategy
    confidence: float
    metadata: Dict

class PDFAnalyzer:
    """Analyze PDF structure to determine optimal processing strategy"""
    
    def __init__(self):
        self.text_threshold = 0.7      # Minimum text coverage for text-only
        self.complexity_threshold = 0.6 # Threshold for complex layout detection
        
    def analyze_page_content(self, page: fitz.Page) -> PageAnalysis:
        """Detect text, images, tables, formulas, diagrams"""
        
        # Check text extractability
        has_text = self._has_extractable_text(page)
        text_coverage = self._calculate_text_coverage(page)
        
        # Detect content types
        has_images = self._detect_images(page)
        has_tables = self._detect_tables(page)
        has_formulas = self._detect_formulas(page)
        
        # Calculate layout complexity
        complexity = self._calculate_layout_complexity(page)
        
        # Determine strategy
        strategy, confidence = self._determine_strategy(
            has_text, text_coverage, has_images, has_tables, 
            has_formulas, complexity
        )
        
        return PageAnalysis(
            has_extractable_text=has_text,
            text_coverage=text_coverage,
            has_images=has_images,
            has_tables=has_tables,
            has_formulas=has_formulas,
            layout_complexity=complexity,
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
        doc = fitz.open(pdf_path)
        analyses = {}
        
        for page_num in range(doc.page_count):
            page = doc[page_num]
            analyses[page_num] = self.analyze_page_content(page)
        
        doc.close()
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
    
    def _calculate_layout_complexity(self, page: fitz.Page) -> float:
        """Calculate layout complexity score (0.0 to 1.0)"""
        text_dict = page.get_text("dict")
        
        complexity_factors = {
            "column_count": 0,
            "font_variety": 0,
            "size_variety": 0,
            "position_scatter": 0,
            "block_count": 0
        }
        
        # Analyze text blocks
        fonts = set()
        sizes = set()
        x_positions = []
        
        text_blocks = [b for b in text_dict["blocks"] if "lines" in b]
        complexity_factors["block_count"] = len(text_blocks)
        
        for block in text_blocks:
            for line in block["lines"]:
                for span in line["spans"]:
                    fonts.add(span["font"])
                    sizes.add(span["size"])
                    x_positions.append(span["bbox"][0])
        
        # Font variety (normalized)
        complexity_factors["font_variety"] = min(len(fonts) / 5, 1.0)
        
        # Size variety (normalized)
        complexity_factors["size_variety"] = min(len(sizes) / 5, 1.0)
        
        # Column detection
        if x_positions:
            unique_x = len(set(round(x) for x in x_positions))
            complexity_factors["column_count"] = min(unique_x / 5, 1.0)
        
        # Position scatter (how spread out text is)
        if len(x_positions) > 1:
            x_range = max(x_positions) - min(x_positions)
            page_width = page.rect.width
            complexity_factors["position_scatter"] = min(x_range / page_width, 1.0)
        
        # Weighted average
        weights = {
            "column_count": 0.3,
            "font_variety": 0.2,
            "size_variety": 0.2,
            "position_scatter": 0.2,
            "block_count": 0.1
        }
        
        # Normalize block count
        complexity_factors["block_count"] = min(complexity_factors["block_count"] / 10, 1.0)
        
        complexity = sum(weights[k] * complexity_factors[k] for k in weights)
        return min(complexity, 1.0)
    
    def _determine_strategy(self, has_text: bool, text_coverage: float, 
                          has_images: bool, has_tables: bool, 
                          has_formulas: bool, complexity: float) -> Tuple[ProcessingStrategy, float]:
        """Determine optimal processing strategy with confidence"""
        
        # High confidence text-only conditions
        if (has_text and text_coverage > self.text_threshold and 
            not has_images and not has_tables and 
            complexity < 0.4):
            return ProcessingStrategy.TEXT_ONLY, 0.9
        
        # Vision-only conditions
        if (not has_text or text_coverage < 0.3 or 
            complexity > 0.8):
            return ProcessingStrategy.VISION_ONLY, 0.8
        
        # Complex layout conditions
        if (complexity > self.complexity_threshold and 
            (has_tables or has_images)):
            return ProcessingStrategy.COMPLEX_LAYOUT, 0.7
        
        # Default to hybrid for mixed content
        confidence = 0.6
        if has_formulas or has_tables:
            confidence = 0.7
        
        return ProcessingStrategy.HYBRID, confidence