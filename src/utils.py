import fitz  # PyMuPDF
import base64
import io
from PIL import Image
import re
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum

class ContentType(Enum):
    TEXT = "text"
    IMAGE = "image"
    TABLE = "table"
    FORMULA = "formula"
    DIAGRAM = "diagram"

@dataclass
class ContentRegion:
    """Represents a content region on a PDF page"""
    bbox: Tuple[float, float, float, float]  # x0, y0, x1, y1
    content_type: ContentType
    confidence: float
    metadata: Dict = None

class PipelineConfig:
    """Configuration for the entire pipeline"""
    
    def __init__(self):
        self.dpi = 300
        self.vision_model_temp = 0.1
        self.text_extraction_priority = True
        self.image_embed_mode = "base64"  # "base64" or "file"
        self.preserve_formatting = True
        self.table_detection_threshold = 0.7
        self.formula_detection_threshold = 0.8
        self.min_image_size = (50, 50)  # width, height in pixels

class Utils:
    """Helper functions for the pipeline"""
    
    @staticmethod
    def is_text_extractable(page: fitz.Page) -> bool:
        """Check if page has extractable text"""
        try:
            text = page.get_text()
            # Check if meaningful text exists (not just whitespace/garbage)
            clean_text = re.sub(r'\s+', ' ', text).strip()
            return len(clean_text) > 10  # Arbitrary threshold
        except:
            return False
    
    @staticmethod
    def detect_content_types(page: fitz.Page) -> List[ContentRegion]:
        """Identify different content types on page"""
        regions = []
        
        # Detect text blocks
        text_blocks = page.get_text("dict")["blocks"]
        for block in text_blocks:
            if "lines" in block:  # Text block
                bbox = block["bbox"]
                regions.append(ContentRegion(
                    bbox=bbox,
                    content_type=ContentType.TEXT,
                    confidence=0.9,
                    metadata={"block": block}
                ))
        
        # Detect images
        image_list = page.get_images()
        for img_index, img in enumerate(image_list):
            # Get image bbox (approximate)
            pix = fitz.Pixmap(page.parent, img[0])
            if pix.width > 50 and pix.height > 50:  # Filter small images
                # Note: Getting exact bbox for images is complex, using page area
                regions.append(ContentRegion(
                    bbox=(0, 0, page.rect.width, page.rect.height),  # Placeholder
                    content_type=ContentType.IMAGE,
                    confidence=0.8,
                    metadata={"img_index": img_index, "size": (pix.width, pix.height)}
                ))
            pix = None
        
        # Detect potential tables (simple heuristic)
        tables = Utils._detect_tables_heuristic(page)
        regions.extend(tables)
        
        # Detect potential formulas
        formulas = Utils._detect_formulas_heuristic(page)
        regions.extend(formulas)
        
        return regions
    
    @staticmethod
    def _detect_tables_heuristic(page: fitz.Page) -> List[ContentRegion]:
        """Simple table detection using text alignment patterns"""
        tables = []
        text_dict = page.get_text("dict")
        
        # Look for regular spacing patterns that might indicate tables
        for block in text_dict["blocks"]:
            if "lines" in block:
                lines = block["lines"]
                if len(lines) >= 3:  # Need at least 3 rows
                    # Check for consistent column patterns
                    x_positions = []
                    for line in lines:
                        for span in line["spans"]:
                            x_positions.append(span["bbox"][0])  # x0 position
                    
                    # Simple heuristic: if we have multiple consistent x positions
                    unique_x = sorted(set(x_positions))
                    if len(unique_x) >= 3:  # At least 3 columns
                        tables.append(ContentRegion(
                            bbox=block["bbox"],
                            content_type=ContentType.TABLE,
                            confidence=0.6,
                            metadata={"columns": len(unique_x)}
                        ))
        
        return tables
    
    @staticmethod
    def _detect_formulas_heuristic(page: fitz.Page) -> List[ContentRegion]:
        """Simple formula detection using common math symbols"""
        formulas = []
        text = page.get_text()
        
        # Look for mathematical symbols and patterns
        math_patterns = [
            r'[∫∑∏√±×÷≤≥≠≈∞∂∇]',  # Math symbols
            r'\$.*?\$',  # LaTeX inline math
            r'\\\[.*?\\\]',  # LaTeX display math
            r'[a-zA-Z]\^[0-9]',  # Superscripts
            r'[a-zA-Z]_[0-9]',  # Subscripts
            r'\b(sin|cos|tan|log|ln|exp|lim|int)\b',  # Math functions
        ]
        
        for pattern in math_patterns:
            matches = re.finditer(pattern, text, re.DOTALL)
            for match in matches:
                # This is a simplified approach - in reality, we'd need
                # to map text positions back to page coordinates
                formulas.append(ContentRegion(
                    bbox=(0, 0, 100, 20),  # Placeholder bbox
                    content_type=ContentType.FORMULA,
                    confidence=0.7,
                    metadata={"pattern": pattern, "text": match.group()}
                ))
        
        return formulas
    
    @staticmethod
    def optimize_image_for_vision(image_data: bytes, max_size: Tuple[int, int] = (1024, 1024)) -> str:
        """Prepare images for vision model processing"""
        try:
            # Convert to PIL Image
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
    
    @staticmethod
    def get_text_with_formatting(page: fitz.Page) -> Dict:
        """Extract text with formatting information preserved"""
        text_dict = page.get_text("dict")
        
        formatted_content = {
            "blocks": [],
            "fonts": set(),
            "styles": {}
        }
        
        for block in text_dict["blocks"]:
            if "lines" in block:  # Text block
                block_content = {
                    "bbox": block["bbox"],
                    "lines": []
                }
                
                for line in block["lines"]:
                    line_content = {
                        "bbox": line["bbox"],
                        "spans": []
                    }
                    
                    for span in line["spans"]:
                        span_info = {
                            "text": span["text"],
                            "bbox": span["bbox"],
                            "font": span["font"],
                            "size": span["size"],
                            "flags": span["flags"],  # Bold, italic, etc.
                            "color": span.get("color", 0)
                        }
                        line_content["spans"].append(span_info)
                        formatted_content["fonts"].add(span["font"])
                    
                    block_content["lines"].append(line_content)
                formatted_content["blocks"].append(block_content)
        
        return formatted_content